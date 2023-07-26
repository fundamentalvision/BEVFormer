# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2021 Toyota Research Institute.  All rights reserved.
import colorsys
import os

import cv2
import matplotlib.colors as mplc
import numpy as np
from PIL import Image, ImageDraw


def fill_color_polygon(image, polygon, color, alpha=0.5):
    """Color interior of polygon with alpha-blending. This function modified input in place.
    """
    _mask = Image.new('L', (image.shape[1], image.shape[0]), 0)
    ImageDraw.Draw(_mask).polygon(polygon, outline=1, fill=1)
    mask = np.array(_mask, np.bool)
    for c in range(3):
        channel = image[:, :, c]
        channel[mask] = channel[mask] * (1. - alpha) + color[c] * alpha


def change_color_brightness(color, brightness_factor):
    """
    Copied from detectron2.utils.visualizer.py
    -------------------------------------------

    Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
    less or more saturation than the original color.

    Args:
        color: color of the polygon. Refer to `matplotlib.colors` for a full list of
            formats that are accepted.
        brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
            0 will correspond to no change, a factor in [-1.0, 0) range will result in
            a darker color and a factor in (0, 1.0] range will result in a lighter color.

    Returns:
        modified_color (tuple[double]): a tuple containing the RGB values of the
            modified color. Each value in the tuple is in the [0.0, 1.0] range.
    """
    assert brightness_factor >= -1.0 and brightness_factor <= 1.0
    color = mplc.to_rgb(color)
    polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
    modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
    modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
    modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
    modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
    return modified_color


def draw_text(ax, text, position, *, font_size, color="g", horizontal_alignment="center", rotation=0):
    """
    Copied from Visualizer.draw_text()
    -----------------------------------

    Args:
        text (str): class label
        position (tuple): a tuple of the x and y coordinates to place text on image.
        font_size (int, optional): font of the text. If not provided, a font size
            proportional to the image width is calculated and used.
        color: color of the text. Refer to `matplotlib.colors` for full list
            of formats that are accepted.
        horizontal_alignment (str): see `matplotlib.text.Text`
        rotation: rotation angle in degrees CCW

    Returns:
        output (VisImage): image object with text drawn.
    """
    # since the text background is dark, we don't want the text to be dark
    color = np.maximum(list(mplc.to_rgb(color)), 0.2)
    color[np.argmax(color)] = max(0.8, np.max(color))

    x, y = position
    ax.text(
        x,
        y,
        text,
        size=font_size,
        family="sans-serif",
        bbox={
            "facecolor": "black",
            "alpha": 0.8,
            "pad": 0.7,
            "edgecolor": "none"
        },
        verticalalignment="top",
        horizontalalignment=horizontal_alignment,
        color=color,
        zorder=10,
        rotation=rotation,
    )
    return ax


def float_to_uint8_color(float_clr):
    assert all([c >= 0. for c in float_clr])
    assert all([c <= 1. for c in float_clr])
    return [int(c * 255.) for c in float_clr]


def mosaic(items, scale=1.0, pad=3, grid_width=None):
    """Creates a mosaic from list of images.

    Parameters
    ----------
    items: list of np.ndarray
        List of images to mosaic.

    scale: float, default=1.0
        Scale factor applied to images. scale > 1.0 enlarges images.

    pad: int, default=3
        Padding size of the images before mosaic

    grid_width: int, default=None
        Mosaic width or grid width of the mosaic

    Returns
    -------
    image: np.array of shape (H, W, 3)
        Image mosaic
    """
    # Determine tile width and height
    N = len(items)
    assert N > 0, 'No items to mosaic!'
    grid_width = grid_width if grid_width else np.ceil(np.sqrt(N)).astype(int)
    grid_height = np.ceil(N * 1. / grid_width).astype(np.int)
    input_size = items[0].shape[:2]
    target_shape = (int(input_size[1] * scale), int(input_size[0] * scale))
    mosaic_items = []
    for j in range(grid_width * grid_height):
        if j < N:
            # Only the first image is scaled, the rest are re-shaped
            # to the same size as the previous image in the mosaic
            im = cv2.resize(items[j], dsize=target_shape)
            mosaic_items.append(im)
        else:
            mosaic_items.append(np.zeros_like(mosaic_items[-1]))

    # Stack W tiles horizontally first, then vertically
    im_pad = lambda im: cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
    mosaic_items = [im_pad(im) for im in mosaic_items]
    hstack = [np.hstack(mosaic_items[j:j + grid_width]) for j in range(0, len(mosaic_items), grid_width)]
    mosaic_viz = np.vstack(hstack) if len(hstack) > 1 \
        else hstack[0]
    return mosaic_viz
