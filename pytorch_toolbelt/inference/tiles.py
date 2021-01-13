"""Implementation of tile-based inference allowing to predict huge images that does not fit into GPU memory entirely
in a sliding-window fashion and merging prediction mask back to full-resolution.
"""
import math
from functools import reduce
from itertools import product
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

Array = np.ndarray
Ints = Union[int, List[int], Tuple[int, ...]]

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


def make_tuple(numbers: Ints, n_dims: Optional[int] = None) -> Tuple[int, ...]:
    """
    Guarantees tuple of ints from tuple or scalar
    """
    if isinstance(numbers, (tuple, list)):
        numbers = tuple(map(int, numbers))
    else:
        assert n_dims is not None
        numbers = (int(numbers),) * n_dims
    return numbers


def assert_shape(shape_0: Tuple[int, ...], shape_1: Tuple[int, ...]) -> None:
    """
    Assert shape equality for each dim
    """
    for i, (s0, s1) in enumerate(zip(shape_0, shape_1)):
        assert s0 == s1, f"shape_0 does not match shape_1 in dim {i} {s0} != {s1}"


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

        :param image_shape: Shape of the source image (scalar or tuple)
        :param tile_size: Tile size (scalar or tuple)
        :param tile_step: Step in pixels between tiles (scalar or tuple)
        :param image_margin:
        :param weight: Fusion algorithm. 'mean' - simple averaging, 'pyramid' - weighted by position
        """
        self.image_shape = image_shape

        # Convert tile_size and tile_step to tuples of ints
        n_dims = len(image_shape)
        self.channels = None
        if is_channels:
            n_dims -= 1
            self.channels = self.image_shape[-1]
        self.tile_size = make_tuple(tile_size, n_dims=n_dims)
        self.tile_step = make_tuple(tile_step, n_dims=n_dims)

        # Calculate weight for tile fusion (or assign provided)
        weights = {"mean": self._mean, "pyramid": self._pyramid}
        self.weight = weight if isinstance(weight, np.ndarray) else weights[weight]()

        # Check tile step and size correctness
        for step, size in zip(self.tile_step, self.tile_size):
            if step < 1 or step > size:
                raise ValueError()

        # Calculate overlap between tiles
        overlap = [size - step for step, size in zip(self.tile_step, self.tile_size)]

        # Calculate margins (arrays of `self.tile_size` shape)
        if image_margin == 0:
            # In case margin is not set, we compute it manually
            nd = [
                max(1, math.ceil((dim - over) / step))
                for dim, over, step in zip(self.image_shape, overlap, self.tile_step)
            ]

            extra = np.array(
                [step * n - (dim - over) for n, dim, over, step in zip(nd, self.image_shape, overlap, self.tile_step)]
            )

            self.margin_start = np.floor_divide(extra, 2)
            self.margin_end = extra - self.margin_start

        else:
            # If margin is precalculated
            for dim, over, step in zip(self.image_shape, overlap, self.tile_step):
                if (dim - over + 2 * image_margin) % step != 0:
                    raise ValueError()

            self.margin_start = np.zeros_like(self.tile_size).fill(image_margin)
            self.margin_end = np.zeros_like(self.tile_size).fill(image_margin)

        # Calculate crop coordinates
        crops_product = []
        for shape, margin_start, margin_end, size, step in zip(
            self.image_shape, self.margin_start, self.margin_end, self.tile_size, self.tile_step
        ):
            # For each dimension add top corner coordinate to the list
            # No need to store tile size, since it is the same for every patch
            crops_product_x = np.arange(0, shape + margin_start + margin_end - size + 1, step)
            # Append a list of corner coordinates to other lists
            crops_product.append(crops_product_x)

        # Combine coordinates with a Cartesian product; each inner list consists of `n_dims` elements
        self.crops = np.array(list(product(*crops_product)))

    def split(self, image: Array, mode: str = "constant", **kwargs: Any):
        """
        Split image into tiles
        """
        assert_shape(image.shape, self.image_shape)

        tiles = []
        for i, crop in enumerate(self.crops):
            tile, crop = self.crop_tile(image, i, random_crop=False, mode=mode, **kwargs)
            assert_shape(tile.shape, self.tile_size)
            tiles.append(tile)

        return tiles

    def project_crop_to_tile(
        self,
        crop: List[int],
    ) -> Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int]], List[int]]:
        """
        Project crop coordinates in padded image `self.crops` to both the original image and the tile
        :param crop: list of ints, corner coordinates of a crop
        :return:
            coordinates in original image ([ix0, iy0, ...], [ix1, iy1, ...])
            coordinates in tile ([tx0, ty0, ...], [tx1, ty1, ...])
        """

        # Get original coordinates with padding
        c0 = [c - start for c, start in zip(crop, self.margin_start)]  # may be negative
        c1 = [c + ts for c, ts in zip(c0, self.tile_size)]  # may overflow image size

        # Restrict coordinated by image size (image coordinate = ic)
        ic0 = [max(c, 0) for c in c0]
        ic1 = [min(c, shape) for c, shape in zip(c1, self.image_shape)]

        # Set shifts for the tile (tile coordinate = tc)
        tc0 = [ic - c for ic, c in zip(ic0, c0)]  # >= 0
        tc1 = [ts + ic - c for ts, ic, c in zip(self.tile_size, ic1, c1)]
        return (ic0, ic1), (tc0, tc1), crop

    def crop_tile(
        self,
        image: Array,
        crop_index: Optional[int] = None,
        crop: Optional[List[int]] = None,
        random_crop: bool = False,
        mode: str = "constant",
        **kwargs: Any,
    ) -> Tuple[Array, List[int]]:
        """
        Memory efficient version of ImageSlicer.cut_patch with zero padding
        :param image: image to cut a tile from
        :param crop: list of ints, corner coordinates of a crop
        :param crop_index: alternatively, crop index in self.crops
        :param random_crop: if sample crop coordinates randomly
        :param mode: padding mode for np.pad
                    {constant, edge, linear_ramp, maximum, mean, median, minimum, reflect, symmetric, wrap, empty}
        :param kwargs: kwargs for np.pad
        :returns:
            tile, cropped array
            crop, list of crop corner coordinates
        """
        assert_shape(image.shape[:-1], self.image_shape[:-1])

        if crop is None:
            if random_crop:
                crop = [
                    np.random.randint(0, max(1, shape - ts - 1)) for shape, ts in zip(self.image_shape, self.tile_size)
                ]
            else:
                crop = self.crops[crop_index]

        # Get image slice (image coordinate = ic, tile coordinate = tc)
        (ic0, ic1), (tc0, tc1), crop = self.project_crop_to_tile(crop)
        image_slice = tuple(slice(c0, c1) for c0, c1 in zip(ic0, ic1))

        # Assume channel last in padding
        # [(before_0, after_0), (before_1, after_1), ...]
        pad_width = [(c0, ts - c1) for ts, c0, c1 in zip(self.tile_size, tc0, tc1)] + [(0, 0)]

        # Create tile by padding image slice to the tile size
        tile = np.pad(image[image_slice], pad_width=pad_width, mode=mode, **kwargs)
        assert_shape(tile.shape, self.tile_size)
        return tile, crop

    @property
    def target_shape(self) -> Tuple[int, ...]:
        """
        Target shape without the last (channel) dimension
        """
        target_shape = tuple(
            image_shape + margin_start + margin_end
            for image_shape, margin_start, margin_end in zip(self.image_shape, self.margin_start, self.margin_end)
        )
        return target_shape

    def merge(self, tiles: List[Array], dtype=np.float32) -> Array:
        """
        Merge tiles to the original shape
        """
        if len(tiles) != len(self.crops):
            raise ValueError

        # TODO add channel first option
        channels = 1 if len(tiles[0].shape) == len(self.tile_size) else tiles[0].shape[-1]
        target_shape = self.target_shape + (channels,)  # self.target shape is without channel dim

        image = np.zeros(target_shape, dtype=np.float64)
        norm_mask = np.zeros(target_shape, dtype=np.uint8)

        w = np.stack([self.weight] * channels, axis=-1)

        for tile, crop in zip(tiles, self.crops):
            image_slice = tuple(slice(x, x + ts) for x, ts in zip(crop, self.tile_size))
            image[image_slice] += tile * w
            norm_mask[image_slice] += w

        # TODO is clip necessary with uint?
        norm_mask = np.clip(norm_mask, a_min=0, a_max=None)
        normalized = np.divide(image, norm_mask).astype(dtype)
        crop = self.crop_to_original_size(normalized)
        return crop

    def crop_to_original_size(self, image: Array) -> Array:
        """
        Crops an image from target shape to original shape
        """
        assert_shape(image.shape, self.target_shape)
        image_slice = tuple(
            slice(start, -end if end != 0 else None) for start, end in zip(self.margin_start, self.margin_end)
        )
        crop = image[image_slice]
        assert_shape(crop.shape[:-1], self.image_shape[:-1])
        return crop

    def _mean(self, tile_size: Optional[Ints] = None) -> Array:
        """
        Compute patch weight loss with respect to tile size
        """
        if tile_size is None:
            tile_size = self.tile_size
        return np.ones(tile_size, dtype=np.uint8)

    def _pyramid(self, tile_size: Optional[Ints] = None) -> Array:
        """
        Compute pyramid patch weight loss with respect to tile size
        """
        if tile_size is None:
            tile_size = self.tile_size
        w, _, _ = compute_pyramid_patch_weight_loss(*tile_size)
        # quantize weight for memory efficiency
        n_steps = min(
            63 - 1, min(tile_size) // 2
        )  # TODO calculate not to exceed 255 in uint8 anyhow (even with step 1)
        w = ((w - np.min(w)) / np.max(w) * n_steps + 1).astype(np.uint8)
        return w

    def __len__(self):
        return len(self.crops)


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

    def accumulate_single(self, tile: Tensor, coords: Tuple[int, ...]) -> None:
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

    def integrate_batch(self, batch: Tensor, crop_coords: Array) -> None:
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
