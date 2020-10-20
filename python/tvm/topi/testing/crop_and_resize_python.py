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
# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals, too-many-nested-blocks
"""crop and resize in python"""
import math
import numpy as np


def crop_and_resize_python(
    image, boxes, box_indices, crop_size, layout, method="bilinear", extrapolation_value=0
):
    """Crop and resize using python"""
    (target_h, target_w) = crop_size

    if layout == "NHWC":
        batch = boxes.shape[0]
        image_height, image_width, channel = image.shape[1], image.shape[2], image.shape[3]
        scaled_image = np.ones((batch, target_h, target_w, channel))
    else:
        batch = boxes.shape[0]
        channel, image_height, image_width = image.shape[1], image.shape[2], image.shape[3]
        scaled_image = np.ones((batch, channel, target_h, target_w))

    for n, box in enumerate(boxes):
        b_in = box_indices[n]
        y1, x1 = boxes[n][0], boxes[n][1]
        y2, x2 = boxes[n][2], boxes[n][3]

        in_h = (image_height - 1) * (y2 - y1)
        in_w = (image_width - 1) * (x2 - x1)
        h_scale = np.float32(in_h) / np.float32(target_h - 1)
        w_scale = np.float32(in_w) / np.float32(target_w - 1)

        for y in range(target_h):

            in_y = y1 * (image_height - 1) + h_scale * y

            if in_y < 0 or in_y > image_height - 1:
                for x in range(target_w):
                    for d in range(channel):
                        if layout == "NHWC":
                            scaled_image[n][y][x][d] = extrapolation_value
                        else:
                            scaled_image[n][d][y][x] = extrapolation_value
                continue

            if method == "bilinear":
                top_y_index = math.floor(in_y)
                bottom_y_index = math.ceil(in_y)
                y_lerp = in_y - top_y_index

                for x in range(target_w):
                    in_x = x1 * (image_width - 1) + x * w_scale
                    if in_x < 0 or in_x > image_width - 1:
                        for d in range(channel):
                            if layout == "NHWC":
                                scaled_image[n][y][x][d] = extrapolation_value
                            else:
                                scaled_image[n][d][y][x] = extrapolation_value
                        continue

                    left_x_index = math.floor(in_x)
                    right_x_index = math.ceil(in_x)
                    x_lerp = in_x - left_x_index

                    for d in range(channel):
                        if layout == "NHWC":
                            top_left = image[b_in][top_y_index][left_x_index][d]
                            top_right = image[b_in][top_y_index][right_x_index][d]
                            bottom_left = image[b_in][bottom_y_index][left_x_index][d]
                            bottom_right = image[b_in][bottom_y_index][right_x_index][d]
                            top = top_left + (top_right - top_left) * x_lerp
                            bottom = bottom_left + (bottom_right - bottom_left) * x_lerp
                            scaled_image[n][y][x][d] = top + (bottom - top) * y_lerp
                        else:
                            top_left = image[b_in][d][top_y_index][left_x_index]
                            top_right = image[b_in][d][top_y_index][right_x_index]
                            bottom_left = image[b_in][d][bottom_y_index][left_x_index]
                            bottom_right = image[b_in][d][bottom_y_index][right_x_index]
                            top = top_left + (top_right - top_left) * x_lerp
                            bottom = bottom_left + (bottom_right - bottom_left) * x_lerp
                            scaled_image[n][d][y][x] = top + (bottom - top) * y_lerp

            elif method == "nearest_neighbor":
                for x in range(target_w):
                    in_x = x1 * (image_width - 1) + x * w_scale
                    if in_x < 0 or in_x > image_width - 1:
                        for d in range(channel):
                            if layout == "NHWC":
                                scaled_image[n][y][x][d] = extrapolation_value
                            else:
                                scaled_image[n][d][y][x] = extrapolation_value
                        continue
                    closest_x_index = np.round(in_x).astype("int32")
                    closest_y_index = np.round(in_y).astype("int32")
                    for d in range(channel):
                        if layout == "NHWC":
                            scaled_image[n][y][x][d] = image[b_in][closest_y_index][
                                closest_x_index
                            ][d]
                        else:
                            scaled_image[n][d][y][x] = image[b_in][d][closest_y_index][
                                closest_x_index
                            ]

    return scaled_image
