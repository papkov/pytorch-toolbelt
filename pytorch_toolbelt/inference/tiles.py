"""Implementation of tile-based inference allowing to predict huge images that does not fit into GPU memory entirely
in a sliding-window fashion and merging prediction mask back to full-resolution.
"""
import math
from copy import copy
from functools import reduce
from itertools import product
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch import Tensor

Array = np.ndarray
Ints = Union[int, List[int], Tuple[int, int]]

from pytorch_toolbelt.utils import pytorch_toolbelt_deprecated

__all__ = [
    "ImageSlicer",
    "TileMerger",
    "CudaTileMerger",
    "compute_pyramid_patch_weight_loss",
    "compute_pyramid_patch_weight_loss_2d",
]


def compute_pyramid_patch_weight_loss(*dims: int) -> Tuple[Array, Array, Array]:
    """
    Compute a weight matrix that assigns bigger weight on pixels in center and
    less weight to pixels on image boundary.
    This weight matrix then used for merging individual tile predictions and helps dealing
    with prediction artifacts on tile boundaries.

    :param dims: tile dimensions (any number)
    :return: tuple of arrays with given dimensionality
        weight,
        d_circle,
        d_ladder
    """

    dims = np.array(dims)
    dims_center = dims * 0.5
    dims_start = np.zeros_like(dims)
    dims_end = dims.copy()

    d_circle = [np.square(np.arange(d) - c + 0.5) for d, c in zip(dims, dims_center)]
    d_circle = np.sqrt(reduce(lambda x, y: x[..., np.newaxis] + y, d_circle))

    d_ladder_start = [np.square(np.arange(dim) - start + 0.5) + np.square(0.5) for dim, start in zip(dims, dims_start)]
    d_ladder_end = [np.square(np.arange(dim) - end + 0.5) + np.square(0.5) for dim, end in zip(dims, dims_end)]

    d_ladder = [np.sqrt(np.minimum(s, e)) for s, e in zip(d_ladder_start, d_ladder_end)]
    d_ladder = reduce(lambda x, y: np.minimum(x[..., np.newaxis], y), d_ladder)

    alpha = np.prod(dims) / np.sum(np.divide(d_ladder, np.add(d_circle, d_ladder)))
    weight = alpha * np.divide(d_ladder, np.add(d_circle, d_ladder))

    return weight, d_circle, d_ladder


def compute_pyramid_patch_weight_loss_2d(width: int, height: int) -> Tuple[Array, Array, Array]:
    """
    Original for `compute_pyramid_patch_weight_loss` in a specific 2D case
    Compute a weight matrix that assigns bigger weight on pixels in center and
    less weight to pixels on image boundary.
    This weight matrix then used for merging individual tile predictions and helps dealing
    with prediction artifacts on tile boundaries.

    :param width: Tile width
    :param height: Tile height
    :return: Since-channel image [Width x Height]
    """
    xc = width * 0.5
    yc = height * 0.5
    xl = 0
    xr = width
    yb = 0
    yt = height
    Dc = np.zeros((width, height))
    De = np.zeros((width, height))

    Dcx = np.square(np.arange(width) - xc + 0.5)
    Dcy = np.square(np.arange(height) - yc + 0.5)
    Dc = np.sqrt(Dcx[np.newaxis].transpose() + Dcy)

    De_l = np.square(np.arange(width) - xl + 0.5) + np.square(0.5)
    De_r = np.square(np.arange(width) - xr + 0.5) + np.square(0.5)
    De_b = np.square(0.5) + np.square(np.arange(height) - yb + 0.5)
    De_t = np.square(0.5) + np.square(np.arange(height) - yt + 0.5)

    De_x = np.sqrt(np.minimum(De_l, De_r))
    De_y = np.sqrt(np.minimum(De_b, De_t))
    De = np.minimum(De_x[np.newaxis].transpose(), De_y)

    alpha = (width * height) / np.sum(np.divide(De, np.add(Dc, De)))
    W = alpha * np.divide(De, np.add(Dc, De))
    return W, Dc, De


def make_tuple(numbers: Ints, n_dims: Optional[int] = None):
    if isinstance(numbers, (tuple, list)):
        numbers = tuple(map(int, numbers))
    else:
        assert n_dims is not None
        numbers = (int(numbers),) * n_dims
    return numbers


class ImageSlicer:
    """
    Helper class to slice image into tiles and merge them back
    """

    def __init__(
        self,
        image_shape: Ints,
        tile_size: Ints,
        tile_step: Ints = 0,
        image_margin: int = 0,
        weight: str = "mean",
        is_channels: bool = True,
    ):
        """

        :param image_shape: Shape of the source image (H, W)
        :param tile_size: Tile size (Scalar or tuple (H, W)
        :param tile_step: Step in pixels between tiles (Scalar or tuple (H, W))
        :param image_margin:
        :param weight: Fusion algorithm. 'mean' - averaging
        """
        self.image_shape = image_shape
        # self.image_height = image_shape[0]
        # self.image_width = image_shape[1]

        # Convert tile_size and tile_step to tuples of ints
        n_dims = len(image_shape)
        self.channels = None
        if is_channels:
            n_dims -= 1
            self.channels = self.image_shape[-1]
        self.tile_size = make_tuple(tile_size, n_dims=n_dims)
        self.tile_step = make_tuple(tile_step, n_dims=n_dims)

        weights = {"mean": self._mean, "pyramid": self._pyramid}
        self.weight = weight if isinstance(weight, np.ndarray) else weights[weight]()

        for step, size in zip(self.tile_step, self.tile_size):
            if step < 1 or step > size:
                raise ValueError()

        overlap = [size - step for step, size in zip(self.tile_step, self.tile_size)]

        self.margin_start = np.zeros_like(self.tile_size)
        self.margin_end = np.zeros_like(self.tile_size)

        # self.margin_left = 0
        # self.margin_right = 0
        # self.margin_top = 0
        # self.margin_bottom = 0

        if image_margin == 0:
            # In case margin is not set, we compute it manually
            nd = [
                max(1, math.ceil((dim - over) / step))
                for dim, over, step in zip(self.image_shape, overlap, self.tile_step)
            ]

            # nw = max(1, math.ceil((self.image_width - overlap[1]) / self.tile_step[1]))
            # nh = max(1, math.ceil((self.image_height - overlap[0]) / self.tile_step[0]))

            extra = np.array(
                [step * n - (dim - over) for n, dim, over, step in zip(nd, self.image_shape, overlap, self.tile_step)]
            )

            # extra_w = self.tile_step[1] * nw - (self.image_width - overlap[1])
            # extra_h = self.tile_step[0] * nh - (self.image_height - overlap[0])

            self.margin_start = np.floor_divide(extra, 2)
            self.margin_end = extra - self.margin_start

            # self.margin_left = extra_w // 2
            # self.margin_right = extra_w - self.margin_left
            # self.margin_top = extra_h // 2
            # self.margin_bottom = extra_h - self.margin_top

        else:
            for dim, over, step in zip(self.image_shape, overlap, self.tile_step):
                if (dim - over + 2 * image_margin) % step != 0:
                    raise ValueError()

            # if (self.image_height - overlap[0] + 2 * image_margin) % self.tile_step[0] != 0:
            #     raise ValueError()

            self.margin_start = np.zeros_like(self.tile_size).fill(image_margin)
            self.margin_end = np.zeros_like(self.tile_size).fill(image_margin)

            # self.margin_left = image_margin
            # self.margin_right = image_margin
            # self.margin_top = image_margin
            # self.margin_bottom = image_margin

        crops_product = []
        bbox_crops_product = []
        for shape, margin_start, margin_end, size, step in zip(
            self.image_shape, self.margin_start, self.margin_end, self.tile_size, self.tile_step
        ):
            crops_product_x = []
            bbox_crops_product_x = []
            for x in range(0, shape + margin_start + margin_end - size + 1, step):
                crops_product_x.append(x)
                bbox_crops_product_x.append(x - margin_start)
            crops_product.append(crops_product_x)
            bbox_crops_product.append(bbox_crops_product_x)

        crops = list(product(*crops_product))
        bbox_crops = list(product(*bbox_crops_product))

        # crops = []
        # bbox_crops = []
        #
        # for y in range(
        #     0, self.image_height + self.margin_top + self.margin_bottom - self.tile_size[0] + 1, self.tile_step[0]
        # ):
        #     for x in range(
        #         0, self.image_width + self.margin_left + self.margin_right - self.tile_size[1] + 1, self.tile_step[1]
        #     ):
        #         crops.append((x, y, self.tile_size[1], self.tile_size[0]))
        #         bbox_crops.append((x - self.margin_left, y - self.margin_top, self.tile_size[1], self.tile_size[0]))

        self.crops = np.array(crops)
        self.bbox_crops = np.array(bbox_crops)

    # def pad(self, image, border_type: int = cv2.BORDER_CONSTANT, value: int = 0):
    #     assert image.shape == self.image_shape
    #     # assert image.shape[0] == self.image_height
    #     # assert image.shape[1] == self.image_width
    #
    #     orig_shape_len = len(image.shape)
    #     image = cv2.copyMakeBorder(
    #         image,
    #         self.margin_top,
    #         self.margin_bottom,
    #         self.margin_left,
    #         self.margin_right,
    #         borderType=border_type,
    #         value=value,
    #     )
    #
    #     # This check recovers possible lack of last dummy dimension for single-channel images
    #     if len(image.shape) != orig_shape_len:
    #         image = np.expand_dims(image, axis=-1)
    #
    #     return image

    def split(self, image, border_type=cv2.BORDER_CONSTANT, value=0):
        # image = self.pad(image, border_type, value)

        tiles = []
        for i, crop in enumerate(self.crops):
            tile = self.cut_patch_no_pad(image, slice_index=i)
            # tile = image[y : y + tile_height, x : x + tile_width].copy()
            for j, (tsr, tse) in enumerate(zip(tile.shape, self.tile_size)):
                assert tsr == tse, f"resulted tile size does not match expected in dim {j} {tsr} != {tse}"
            # assert tile.shape[0] == self.tile_size[0]
            # assert tile.shape[1] == self.tile_size[1]

            tiles.append(tile)

        return tiles

    def cut_patch(self, image: np.ndarray, slice_index: int, border_type=cv2.BORDER_CONSTANT, value=0):
        """
        Deprecated, because cv2.copyMakeBorder creates a second image
        """
        return self.cut_patch_no_pad(image, slice_index)
        # image = self.pad(image, border_type, value)
        #
        # x, y, tile_width, tile_height = self.crops[slice_index]
        #
        # tile = image[y : y + tile_height, x : x + tile_width].copy()
        # assert tile.shape[0] == self.tile_size[0]
        # assert tile.shape[1] == self.tile_size[1]
        # return tile

    def crop_no_pad(
        self, slice_index: int, random_crop: bool = False
    ) -> Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int]]]:
        """
        Move padded crop coordinates to original
        :param slice_index: index in self.crops
        :param random_crop: if sample x and y randomly
        :return:
            coordinates in original image (ix0, ix1, iy0, iy1)
            coordinates in tile (tx0, tx1, ty0, ty1)
        """
        # x, y, tile_width, tile_height = self.crops[slice_index]
        coords = self.crops[slice_index]
        if random_crop:
            coords = [np.random.randint(0, shape - ts - 1) for shape, ts in zip(self.image_shape, self.tile_size)]
            # x = np.random.randint(0, self.image_width - tile_width - 1)
            # y = np.random.randint(0, self.image_height - tile_height - 1)

        # Get original coordinates with padding
        # may be negative
        c0 = [c - start for c, start in zip(coords, self.margin_start)]
        # x0 = x - self.margin_left
        # y0 = y - self.margin_top
        # may overflow image size
        c1 = [c + ts for c, ts in zip(c0, self.tile_size)]
        # x1 = x0 + tile_width
        # y1 = y0 + tile_height

        # Restrict coordinated by image size
        ic0 = [max(c, 0) for c in c0]
        # ix0 = max(x0, 0)
        # iy0 = max(y0, 0)
        ic1 = [min(c, shape) for c, shape in zip(c1, self.image_shape)]
        # ix1 = min(x1, self.image_width)
        # iy1 = min(y1, self.image_height)

        # Set shifts for the tile
        tc0 = [ic - c for ic, c in zip(ic0, c0)]  # >= 0
        # tx0 = ix0 - x0  # >= 0
        # ty0 = iy0 - y0  # >= 0
        tc1 = [ts + ic - c for ts, ic, c in zip(self.tile_size, ic1, c1)]
        # tx1 = tile_width + ix1 - x1  # <= tile_width
        # ty1 = tile_height + iy1 - y1  # <= tile_height

        # return (ix0, ix1, iy0, iy1), (tx0, tx1, ty0, ty1)
        return (ic0, ic1), (tc0, tc1)

    def cut_patch_no_pad(self, image: Array, slice_index: int, random_crop: bool = False) -> Array:
        """
        Memory efficient version of ImageSlicer.cut_patch with zero padding
        TODO add padding options (currently only zero padding)
        :param image:
        :param slice_index:
        :param random_crop: if sample x and y randomly
        """
        (ic0, ic1), (tc0, tc1) = self.crop_no_pad(slice_index, random_crop)
        # (ix0, ix1, iy0, iy1), (tx0, tx1, ty0, ty1) = self.crop_no_pad(slice_index, random_crop)
        # print((x0, x1, y0, y1), (ix0, ix1, iy0, iy1), (tx0, tx1, ty0, ty1))

        # Allocate tile
        # shape = (self.tile_size[0], self.tile_size[1])
        shape = copy(self.tile_size)
        if len(image.shape) > len(shape):
            # TODO add channel first option
            shape += (image.shape[-1],)
        tile = np.zeros(shape, dtype=image.dtype)

        image_slice = tuple(slice(c0, c1) for c0, c1 in zip(ic0, ic1))
        tile_slice = tuple(slice(c0, c1) for c0, c1 in zip(tc0, tc1))
        tile[tile_slice] = image[image_slice]
        # tile[ty0:ty1, tx0:tx1] = image[iy0:iy1, ix0:ix1]
        return tile

    @property
    def target_shape(self) -> Tuple[int, ...]:
        """
        Target shape without the last (channel) dimension
        """

        target_shape = tuple(
            image_shape + margin_start + margin_end
            for image_shape, margin_start, margin_end in zip(self.image_shape, self.margin_start, self.margin_end)
        )

        # target_shape = (
        #     self.image_height + self.margin_bottom + self.margin_top,
        #     self.image_width + self.margin_right + self.margin_left,
        # )
        return target_shape

    def merge(self, tiles: List[np.ndarray], dtype=np.float32):
        if len(tiles) != len(self.crops):
            raise ValueError

        # TODO add channel first option
        channels = 1 if len(tiles[0].shape) == len(self.tile_size) else tiles[0].shape[-1]
        # target_shape = (
        #     self.image_height + self.margin_bottom + self.margin_top,
        #     self.image_width + self.margin_right + self.margin_left,
        #     channels,
        # )

        target_shape = self.target_shape + (channels,)

        image = np.zeros(target_shape, dtype=np.float64)
        norm_mask = np.zeros(target_shape, dtype=np.uint8)

        w = np.stack([self.weight] * channels, axis=-1)

        for tile, crop in zip(tiles, self.crops):
            image_slice = tuple(slice(x, x + ts) for x, ts in zip(crop, self.tile_size))
            image[image_slice] += tile * w
            norm_mask[image_slice] += w

        # for tile, (x, y, tile_width, tile_height) in zip(tiles, self.crops):
        #     # print(x, y, tile_width, tile_height, image.shape)
        #     image[y : y + tile_height, x : x + tile_width] += tile * w
        #     norm_mask[y : y + tile_height, x : x + tile_width] += w

        # print(norm_mask.min(), norm_mask.max())
        # TODO is clip necessary with uint?
        norm_mask = np.clip(norm_mask, a_min=0, a_max=None)
        normalized = np.divide(image, norm_mask).astype(dtype)
        crop = self.crop_to_original_size(normalized)
        return crop

    def crop_to_original_size(self, image):
        for i, (ims, ts) in enumerate(zip(image.shape, self.target_shape)):
            assert ims == ts, f"in dim {i} image shape does not match target shape {ims} != {ts}"
        # assert image.shape[0] == self.target_shape[0]
        # assert image.shape[1] == self.target_shape[1]

        image_slice = tuple(
            slice(start, -end if end != 0 else None) for start, end in zip(self.margin_start, self.margin_end)
        )
        crop = image[image_slice]
        # crop = image[
        #     self.margin_top : self.image_height + self.margin_top,
        #     self.margin_left : self.image_width + self.margin_left,
        # ]
        for i, (cs, ims) in enumerate(zip(crop.shape[:-1], self.image_shape[:-1])):
            assert cs == ims, f"in dim {i} crop shape does not match image shape {cs} != {ims}"
        # assert crop.shape[0] == self.image_height
        # assert crop.shape[1] == self.image_width
        return crop

    def _mean(self, tile_size: Optional[Ints] = None):
        if tile_size is None:
            tile_size = self.tile_size
        return np.ones(tile_size, dtype=np.uint8)

    def _pyramid(self, tile_size: Optional[Ints] = None):
        if tile_size is None:
            tile_size = self.tile_size
        w, _, _ = compute_pyramid_patch_weight_loss(*tile_size)
        # quantize weight for memory efficiency
        n_steps = min(
            63 - 1, min(tile_size) // 2
        )  # TODO calculate not to exceed 255 in uint8 anyhow (even with step 1)
        w = ((w - np.min(w)) / np.max(w) * n_steps + 1).astype(np.uint8)
        return w


class TileMerger:
    """
    Helper class to merge final image on GPU. This generally faster than moving individual tiles to CPU.
    """

    def __init__(
        self, image_shape: Tuple[int, ...], channels: int, weight: Array, device="cpu", default_value: float = -99.0
    ):
        """
        :param image_shape: Shape of the source image
        :param channels: Number of channels
        :param weight: Weighting matrix
        :param device: Device for memory allocation
        :param default_value: Negative value to fill image by default
                              in case we predict only some tiles ond need zeros
                              for other areas in final prediction
        """
        self.image_shape = image_shape
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.channels = channels
        self.default_value = default_value

        # Make weight and norm_mask uint8 for memory efficiency
        self.weight = torch.from_numpy(np.expand_dims(weight, axis=0)).to(device).type(torch.uint8)
        self.image = torch.empty((channels,) + self.image_shape, device=device).fill_(default_value).float()
        self.norm_mask = torch.ones((1,) + self.image_shape, device=device, dtype=torch.uint8)

    def accumulate_single(self, tile: Tensor, coords: Tuple[int, ...]):
        """
        Accumulates single element
        :param tile: Predicted image of shape [C,H,W]
        :param coords: Corresponding tile crops (top corner coordinates) w.r.t to original image
        """
        tile = tile.to(device=self.image.device)
        # x, y, tile_width, tile_height = coords

        # Replace default (large negative) value with zero to add predictions
        image_slice = (slice(None),) + tuple(slice(x, x + ts) for x, ts in zip(coords, tile.shape[1:]))
        self.image[image_slice] = torch.where(
            self.image[image_slice] == self.default_value,
            torch.tensor(0.0).float().to(self.image.device),
            self.image[image_slice],
        )
        self.image[image_slice] += tile * self.weight
        self.norm_mask[image_slice] += self.weight

    def integrate_batch(self, batch: Tensor, crop_coords: Array):
        """
        Accumulates batch of tile predictions
        :param batch: Predicted tiles  of shape [B,C,H,W]
        :param crop_coords: Corresponding tile crops w.r.t to original image
        """
        if len(batch) != len(crop_coords):
            raise ValueError("Number of images in batch does not correspond to number of coordinates")

        batch = batch.to(device=self.image.device)
        for tile, coords in zip(batch, crop_coords):
            self.accumulate_single(tile, coords)

    def merge(self) -> Tensor:
        return self.image / self.norm_mask

    def merge_(self) -> None:
        """
        Inplace version of TileMerger.merge() using div_()
        Substitute self.image with (self.image / self.norm_mask)
        :return: None
        """
        self.image.div_(self.norm_mask)

    def threshold_(self, threshold: float = 0.5) -> None:
        """
        Inplace thresholding of TileMerger.image:
        image = sigmoid(image) > threshold
        :return: None
        """
        self.image.sigmoid_()
        self.image.gt_(threshold)
        self.image.type(torch.int8)


@pytorch_toolbelt_deprecated("This class is deprecated and will be removed in 0.5.0. Please use TileMerger instead.")
class CudaTileMerger(TileMerger):
    def __init__(self, image_shape, channels, weight, device="cuda"):
        super().__init__(image_shape, channels, weight, device)
