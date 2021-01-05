import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_toolbelt.inference.tiles import (
    CudaTileMerger,
    ImageSlicer,
    compute_pyramid_patch_weight_loss,
    compute_pyramid_patch_weight_loss_2d,
)
from pytorch_toolbelt.utils.torch_utils import rgb_image_from_tensor, tensor_from_rgb_image, to_numpy

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda is not available")


def test_pyramid_loss(width: int = 21, height: int = 21):
    out_general = compute_pyramid_patch_weight_loss(width, height)
    out_2d = compute_pyramid_patch_weight_loss_2d(width, height)
    for og, o2 in zip(out_general, out_2d):
        np.testing.assert_equal(og, o2)


def test_tiles_split_merge():
    image = np.random.random((500, 500, 3)).astype(np.uint8)
    tiler = ImageSlicer(image.shape, tile_size=51, tile_step=26, weight="mean")
    tiles = tiler.split(image)
    merged = tiler.merge(tiles, dtype=np.uint8)
    np.testing.assert_equal(merged, image)


def test_tiles_split_merge_non_dividable():
    image = np.random.random((563, 512, 3)).astype(np.uint8)
    tiler = ImageSlicer(image.shape, tile_size=(128, 128), tile_step=(128, 128), weight="mean")
    tiles = tiler.split(image)
    merged = tiler.merge(tiles, dtype=np.uint8)
    np.testing.assert_equal(merged, image)


def test_tiles_split_merge_non_dividable_3d():
    image = np.random.random((563, 512, 273, 1)).astype(np.uint8)
    tiler = ImageSlicer(image.shape, tile_size=128, tile_step=128, weight="mean")
    tiles = tiler.split(image)
    merged = tiler.merge(tiles, dtype=np.uint8)
    np.testing.assert_equal(merged, image)


@skip_if_no_cuda
def test_tiles_split_merge_non_dividable_cuda():

    image = np.random.random((5632, 5120, 3)).astype(np.uint8)
    tiler = ImageSlicer(image.shape, tile_size=(1280, 1280), tile_step=(1280, 1280), weight="mean")
    tiles = tiler.split(image)

    merger = CudaTileMerger(tiler.target_shape, channels=image.shape[-1], weight=tiler.weight)
    for tile, coordinates in zip(tiles, tiler.crops):
        # Integrate as batch of size 1
        merger.integrate_batch(tensor_from_rgb_image(tile).unsqueeze(0).float().cuda(), coordinates[None, ...])

    merged = merger.merge()
    merged = rgb_image_from_tensor(merged, mean=0, std=1, max_pixel_value=1)
    merged = tiler.crop_to_original_size(merged)

    np.testing.assert_equal(merged, image)


@skip_if_no_cuda
def test_tiles_split_merge_non_dividable_cuda_3d():

    image = np.random.random((563, 512, 563, 1)).astype(np.uint8)
    tiler = ImageSlicer(image.shape, tile_size=128, tile_step=128, weight="mean")
    tiles = tiler.split(image)

    merger = CudaTileMerger(tiler.target_shape, channels=image.shape[-1], weight=tiler.weight)
    for tile, coordinates in zip(tiles, tiler.crops):
        # Integrate as batch of size 1
        merger.integrate_batch(
            torch.from_numpy(np.moveaxis(tile, -1, 0)).unsqueeze(0).float().cuda(), coordinates[None, ...]
        )

    merged = merger.merge()
    merged = rgb_image_from_tensor(merged, mean=0, std=1, max_pixel_value=1)
    merged = tiler.crop_to_original_size(merged)

    np.testing.assert_equal(merged, image)


def test_tiles_split_merge_2():
    image = np.random.random((5000, 5000, 3)).astype(np.uint8)
    tiler = ImageSlicer(image.shape, tile_size=(512, 512), tile_step=(256, 256), weight="pyramid")

    np.testing.assert_allclose(tiler.weight, tiler.weight.T)

    tiles = tiler.split(image)
    merged = tiler.merge(tiles, dtype=np.uint8)
    np.testing.assert_equal(merged, image)


@skip_if_no_cuda
def test_tiles_split_merge_cuda():
    class MaxChannelIntensity(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            max_channel, _ = torch.max(input, dim=1, keepdim=True)
            return max_channel

    image = np.random.random((5000, 5000, 3)).astype(np.uint8)
    tiler = ImageSlicer(image.shape, tile_size=(512, 512), tile_step=(256, 256), weight="pyramid")
    tiles = [tensor_from_rgb_image(tile) for tile in tiler.split(image)]

    model = MaxChannelIntensity().eval().cuda()

    merger = CudaTileMerger(tiler.target_shape, 1, tiler.weight)
    for tiles_batch, coords_batch in DataLoader(list(zip(tiles, tiler.crops)), batch_size=8, pin_memory=True):
        tiles_batch = tiles_batch.float().cuda()
        pred_batch = model(tiles_batch)

        merger.integrate_batch(pred_batch, coords_batch)

    merged = np.moveaxis(to_numpy(merger.merge()), 0, -1).astype(np.uint8)
    merged = tiler.crop_to_original_size(merged)

    np.testing.assert_equal(merged, image.max(axis=2, keepdims=True))
