# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""affine_grid and grid_sample operators in python"""
import math
import numpy as np


def affine_grid_python(data, target_shape):
    yv, xv = np.meshgrid(np.arange(target_shape[0]), np.arange(target_shape[1]))
    yv = yv.T * 2 / (target_shape[0] - 1) - 1
    xv = xv.T * 2 / (target_shape[1] - 1) - 1
    ones = np.ones_like(xv)
    grid = np.stack([xv, yv, ones]).reshape(3, -1)
    return data.reshape(-1, 3).dot(grid).reshape(data.shape[0], 2, *target_shape)


def grid_sample_2d(
    data: np.ndarray,
    grid: np.ndarray,
    method="bilinear",
    layout="NCHW",
    padding_mode="zeros",
    align_corners=True,
):
    r"""grid_sample_2d for NCHW layout"""

    assert method in ("bilinear", "nearest", "bicubic"), f"{method} is not supported"
    assert layout == "NCHW"
    assert padding_mode in ("zeros", "border", "reflection"), f"{padding_mode} is not supported"
    assert len(data.shape) == len(grid.shape) == 4

    batch, channel = data.shape[:2]
    in_height, in_width = data.shape[2:]
    out_height, out_width = grid.shape[2:]
    out_shape = [batch, channel, out_height, out_width]
    out = np.zeros(out_shape)

    def _get_pixel(b, c, h, w):
        if 0 <= h <= in_height - 1 and 0 <= w <= in_width - 1:
            return data[b, c, h, w]
        return 0

    def _unnormalize(h, w):
        if align_corners:
            new_h = (h + 1) * (in_height - 1) / 2
            new_w = (w + 1) * (in_width - 1) / 2
        else:
            new_h = -0.5 + (h + 1) * in_height / 2
            new_w = -0.5 + (w + 1) * in_width / 2
        return (new_h, new_w)

    def _clip_coordinates(x, size):
        return min(max(x, 0), size - 1)

    def _reflect_coordinates(i, size):
        def __refelection(i, size, corner_start):
            def __reflect(index, size, corner_start):
                index_align_corner = abs(corner_start - index)
                size_times = index_align_corner // size
                even = size_times % 2 == 0
                extra = index_align_corner - size_times * size
                return extra + corner_start if even else size - extra + corner_start

            if corner_start <= i <= size + corner_start:
                new_i = i
            else:
                new_i = __reflect(i, size, corner_start)
            return new_i

        if align_corners:
            x = __refelection(i, size - 1, 0)
        else:
            x = __refelection(i, size, -0.5)
        return x

    def _compute_source_index(b, h, w):
        y = grid[b, 1, h, w]
        x = grid[b, 0, h, w]
        y, x = _unnormalize(y, x)

        if padding_mode == "reflection":
            y = _reflect_coordinates(y, in_height)
            x = _reflect_coordinates(x, in_width)
            y = _clip_coordinates(y, in_height)
            x = _clip_coordinates(x, in_width)
        elif padding_mode == "border":
            y = _clip_coordinates(y, in_height)
            x = _clip_coordinates(x, in_width)

        return (y, x)

    def _nearest_sample():
        for _b in range(batch):
            for _c in range(channel):
                for _h in range(out_height):
                    for _w in range(out_width):
                        y, x = _compute_source_index(_b, _h, _w)
                        # python round is not used here,
                        # beacause it is done toward the even choice
                        new_y = int(y + 0.5) if y > 0 else int(y - 0.5)
                        new_x = int(x + 0.5) if x > 0 else int(x - 0.5)
                        out[_b, _c, _h, _w] = _get_pixel(_b, _c, new_y, new_x)

    def _bilinear_sample():
        for _b in range(batch):
            for _c in range(channel):
                for _h in range(out_height):
                    for _w in range(out_width):
                        y, x = _compute_source_index(_b, _h, _w)
                        y0 = int(math.floor(y))
                        x0 = int(math.floor(x))
                        y1 = y0 + 1
                        x1 = x0 + 1

                        out[_b, _c, _h, _w] = (
                            _get_pixel(_b, _c, y0, x0) * (1.0 - (y - y0)) * (1.0 - (x - x0))
                            + _get_pixel(_b, _c, y0, x1) * (1.0 - (y - y0)) * (x - x0)
                            + _get_pixel(_b, _c, y1, x0) * (y - y0) * (1.0 - (x - x0))
                            + _get_pixel(_b, _c, y1, x1) * (y - y0) * (x - x0)
                        )

    def _bicubic_sample():
        A = -0.75

        def cubic_weight_1(x_fraction):
            return ((A + 2) * x_fraction - (A + 3)) * x_fraction * x_fraction + 1

        def cubic_weight_2(x_fraction):
            return ((A * x_fraction - 5 * A) * x_fraction + 8 * A) * x_fraction - 4 * A

        def cubic_interp_1d(pixel_0, pixel_1, pixel_2, pixel_3, x_fraction):
            weights = [0] * 4
            weights[0] = cubic_weight_2(x_fraction + 1)
            weights[1] = cubic_weight_1(x_fraction)
            weights[2] = cubic_weight_1(1 - x_fraction)
            weights[3] = cubic_weight_2(2 - x_fraction)

            return (
                pixel_0 * weights[0]
                + pixel_1 * weights[1]
                + pixel_2 * weights[2]
                + pixel_3 * weights[3]
            )

        def coefficients_along_x(x_floor, y_floor, x_fraction):
            coefficients = [0] * 4

            for i in range(4):
                y_ = y_floor - 1 + i
                x_0 = x_floor - 1
                x_1 = x_floor + 0
                x_2 = x_floor + 1
                x_3 = x_floor + 2

                if padding_mode == "border":
                    y_ = _clip_coordinates(y_, in_height)
                    x_0 = _clip_coordinates(x_0, in_width)
                    x_1 = _clip_coordinates(x_1, in_width)
                    x_2 = _clip_coordinates(x_2, in_width)
                    x_3 = _clip_coordinates(x_3, in_width)

                elif padding_mode == "reflection":
                    y_ = _reflect_coordinates(y_, in_height)
                    x_0 = _reflect_coordinates(x_0, in_width)
                    x_1 = _reflect_coordinates(x_1, in_width)
                    x_2 = _reflect_coordinates(x_2, in_width)
                    x_3 = _reflect_coordinates(x_3, in_width)

                    y_ = int(_clip_coordinates(y_, in_height))
                    x_0 = int(_clip_coordinates(x_0, in_width))
                    x_1 = int(_clip_coordinates(x_1, in_width))
                    x_2 = int(_clip_coordinates(x_2, in_width))
                    x_3 = int(_clip_coordinates(x_3, in_width))

                coefficients[i] = cubic_interp_1d(
                    _get_pixel(_b, _c, y_, x_0),
                    _get_pixel(_b, _c, y_, x_1),
                    _get_pixel(_b, _c, y_, x_2),
                    _get_pixel(_b, _c, y_, x_3),
                    x_fraction,
                )
            return coefficients

        for _b in range(batch):
            for _c in range(channel):
                for _h in range(out_height):
                    for _w in range(out_width):
                        y = grid[_b, 1, _h, _w]
                        x = grid[_b, 0, _h, _w]
                        y, x = _unnormalize(y, x)
                        y_floor = int(math.floor(y))
                        x_floor = int(math.floor(x))
                        y_fraction = y - y_floor
                        x_fraction = x - x_floor

                        coefficients = coefficients_along_x(x_floor, y_floor, x_fraction)

                        out[_b, _c, _h, _w] = cubic_interp_1d(
                            coefficients[0],
                            coefficients[1],
                            coefficients[2],
                            coefficients[3],
                            y_fraction,
                        )

    if method == "bilinear":
        _bilinear_sample()
    elif method == "nearest":
        _nearest_sample()
    else:  # mode == "bicubic":
        _bicubic_sample()

    return out


def grid_sample_3d(
    data: np.ndarray,
    grid: np.ndarray,
    method="bilinear",
    layout="NCDHW",
    padding_mode="zeros",
    align_corners=True,
):
    r"""grid_sample_3d for NCDHW layout"""

    assert method in ("bilinear", "nearest"), f"{method} is not supported"
    assert layout == "NCDHW"
    assert padding_mode in ("zeros", "border", "reflection"), f"{padding_mode} is not supported"
    assert len(data.shape) == len(grid.shape) == 5

    batch, channel = data.shape[:2]
    in_depth, in_height, in_width = data.shape[2:]
    out_depth, out_height, out_width = grid.shape[2:]
    out_shape = [batch, channel, out_depth, out_height, out_width]
    out = np.zeros(out_shape)

    def _get_pixel(b, c, d, h, w):
        if 0 <= d <= in_depth - 1 and 0 <= h <= in_height - 1 and 0 <= w <= in_width - 1:
            return data[b, c, d, h, w]
        return 0

    def _unnormalize(d, h, w):
        if align_corners:
            new_d = (d + 1) * (in_depth - 1) / 2
            new_h = (h + 1) * (in_height - 1) / 2
            new_w = (w + 1) * (in_width - 1) / 2
        else:
            new_d = -0.5 + (d + 1) * in_depth / 2
            new_h = -0.5 + (h + 1) * in_height / 2
            new_w = -0.5 + (w + 1) * in_width / 2
        return (new_d, new_h, new_w)

    def _clip_coordinates(x, size):
        return min(max(x, 0), size - 1)

    def _reflect_coordinates(i, size):
        def __refelection(i, size, corner_start):
            def __reflect(index, size, corner_start):
                index_align_corner = abs(corner_start - index)
                size_times = index_align_corner // size
                even = size_times % 2 == 0
                extra = index_align_corner - size_times * size
                return extra + corner_start if even else size - extra + corner_start

            if corner_start <= i <= size + corner_start:
                new_i = i
            else:
                new_i = __reflect(i, size, corner_start)
            return new_i

        if align_corners:
            x = __refelection(i, size - 1, 0)
        else:
            x = __refelection(i, size, -0.5)
        return x

    def _compute_source_index(b, d, h, w):
        z = grid[b, 2, d, h, w]
        y = grid[b, 1, d, h, w]
        x = grid[b, 0, d, h, w]
        z, y, x = _unnormalize(z, y, x)

        if padding_mode == "reflection":
            z = _reflect_coordinates(z, in_depth)
            y = _reflect_coordinates(y, in_height)
            x = _reflect_coordinates(x, in_width)
            z = _clip_coordinates(z, in_depth)
            y = _clip_coordinates(y, in_height)
            x = _clip_coordinates(x, in_width)
        elif padding_mode == "border":
            z = _clip_coordinates(z, in_depth)
            y = _clip_coordinates(y, in_height)
            x = _clip_coordinates(x, in_width)
        return (z, y, x)

    def _nearest_sample():
        for _b in range(batch):
            for _c in range(channel):
                for _d in range(out_depth):
                    for _h in range(out_height):
                        for _w in range(out_width):
                            z, y, x = _compute_source_index(_b, _d, _h, _w)
                            # python round is not used here,
                            # beacause it is done toward the even choice
                            new_z = int(z + 0.5) if z > 0 else int(z - 0.5)
                            new_y = int(y + 0.5) if y > 0 else int(y - 0.5)
                            new_x = int(x + 0.5) if x > 0 else int(x - 0.5)
                            out[_b, _c, _d, _h, _w] = _get_pixel(_b, _c, new_z, new_y, new_x)

    def _triilinear_sample():
        for _b in range(batch):
            for _c in range(channel):
                for _d in range(out_depth):
                    for _h in range(out_height):
                        for _w in range(out_width):
                            z, y, x = _compute_source_index(_b, _d, _h, _w)
                            z0 = int(math.floor(z))
                            y0 = int(math.floor(y))
                            x0 = int(math.floor(x))
                            z1 = z0 + 1
                            y1 = y0 + 1
                            x1 = x0 + 1

                            out[_b, _c, _d, _h, _w] = (
                                _get_pixel(_b, _c, z0, y0, x0)
                                * (1 - (x - x0))
                                * (1 - (y - y0))
                                * (1 - (z - z0))
                                + _get_pixel(_b, _c, z0, y0, x1)
                                * (x - x0)
                                * (1 - (y - y0))
                                * (1 - (z - z0))
                                + _get_pixel(_b, _c, z1, y1, x0)
                                * (1 - (x - x0))
                                * (y - y0)
                                * (z - z0)
                                + _get_pixel(_b, _c, z1, y1, x1) * (x - x0) * (y - y0) * (z - z0)
                                + _get_pixel(_b, _c, z0, y1, x0)
                                * (1 - (x - x0))
                                * (y - y0)
                                * (1 - (z - z0))
                                + _get_pixel(_b, _c, z1, y0, x1)
                                * (x - x0)
                                * (1 - (y - y0))
                                * (z - z0)
                                + _get_pixel(_b, _c, z1, y0, x0)
                                * (1 - (x - x0))
                                * (1 - (y - y0))
                                * (z - z0)
                                + _get_pixel(_b, _c, z0, y1, x1)
                                * (x - x0)
                                * (y - y0)
                                * (1 - (z - z0))
                            )

    if method == "bilinear":
        _triilinear_sample()
    else:  # method == "nearest":
        _nearest_sample()

    return out


def grid_sample_python(
    data: np.ndarray,
    grid: np.ndarray,
    method="bilinear",
    layout="NCHW",
    padding_mode="zeros",
    align_corners=True,
):
    r"""grid_sample_3d for NCDHW layout or grid_sample_2d for NCHW layout"""

    if len(data.shape) == 4:
        grid_sample = grid_sample_2d
    elif len(data.shape) == 5:
        grid_sample = grid_sample_3d
    else:
        raise ValueError("invalid shape")

    return grid_sample(data, grid, method, layout, padding_mode, align_corners)
