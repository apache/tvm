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
# pylint: disable=invalid-name
"""TVM operator input resize compute."""
from __future__ import absolute_import
import tvm
from tvm import te
from tvm.topi.util import nchw_pack_layout, nchw_xc_layout
from .. import tag


def get_2d_indices(indices, layout="NCHW"):
    """ Get 2d indices """
    (cc, inum, ic) = (0, 0, 0)
    if layout == "NHWC":
        n, y, x, c = indices
        cc = None
    elif layout == "NCHW":
        n, c, y, x = indices
        cc = None
    elif nchw_pack_layout(layout):
        n, c, y, x, inum, ic = indices
    else:
        # else must be NCHWxc
        assert nchw_xc_layout(layout)
        n, c, y, x, cc = indices

    return n, c, y, x, cc, inum, ic


def get_2d_pixel(data, layout, boxes, image_height, image_width, n, c, y, x, cc, ib, ic):
    """ Get 2d pixel """
    if boxes is None:
        y = tvm.te.max(tvm.te.min(y, image_height - 1), 0)
        x = tvm.te.max(tvm.te.min(x, image_width - 1), 0)
    if layout == "NHWC":
        return data(n, y, x, c).astype("float")
    if layout == "NCHW":
        return data(n, c, y, x).astype("float")
    if nchw_pack_layout(layout):
        return data(n, c, y, x, ib, ic).astype("float")

    # else must be NCHWxc
    assert nchw_xc_layout(layout)
    return data(n, c, y, x, cc).astype("float")


def resize_nearest_neighbor(
    indices,
    data,
    image_height,
    image_width,
    target_height,
    target_width,
    boxes=None,
    box_indices=None,
    extrapolation_value=None,
    layout="NCHW",
    coordinate_transformation_mode="align_corners",
    out_dtype=None,
):

    """Perform resize operation with nearest neighbor method on the data.
    For details about Nearest-neighbor interpolation please refer to
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation.

    Parameters
    ----------
    indices : tuple
        The indices of input data

    data : tvm.te.Tensor
        inputs is a 4-D tensor with shape
        [batch, channel, in_height, in_width]
        or  [batch, in_height, in_width, channel]

    image_height : integer
        Input image height

    image_width : integer
        Input image width

    target_height : integer
        The target resized image height

    target_width : integer
        The target resized image width

    boxes : tvm.te.Tensor, optional
        A 2-D tensor of shape [num_boxes, 4]. Each row of the tensor specifies
        the coordinates of a box.

    box_indices : tvm.te.Tensor, optional
        A 1-D tensor of shape [num_boxes], box_indices[i] specifies the data that
        the i-th box refers to.

    extrapolation_value: float, optional
        Value used for extrapolation, when applicable.

    layout: string, optional
        "NCHW", "NHWC", or "NCHWc".

    coordinate_transformation_mode: string, optional
        Describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor.
        Refer to the ONNX Resize operator specification for details.
        Available options are "half_pixel", "align_corners" and "asymmetric".

    out_dtype: string, optional
        Type to return. If left None will be same as input type.

    Returns
    -------
    output : out_dtype
        The computed result with type out_dtype
    """

    def _cast_output(value, data_dtype="float32", out_dtype=None):
        if out_dtype:
            dtype = out_dtype
        else:
            dtype = data_dtype
        return value.astype(dtype)

    n, c, y, x, cc, inum, ic = get_2d_indices(indices, layout)
    box_idx = box_indices(n) if box_indices is not None else n
    if boxes is not None:
        y1, x1 = boxes(n, 0), boxes(n, 1)
        y2, x2 = boxes(n, 2), boxes(n, 3)

        in_h = (image_height - 1) * (y2 - y1)
        in_w = (image_width - 1) * (x2 - x1)
        h_scale = in_h.astype("float") / (target_height - 1)
        w_scale = in_w.astype("float") / (target_width - 1)

        in_y = y1 * (image_height - 1) + h_scale * y
        in_x = x1 * (image_width - 1) + w_scale * x
    else:
        if coordinate_transformation_mode == "align_corners":
            h_scale = (image_height - 1).astype("float") / (target_height - 1)
            w_scale = (image_width - 1).astype("float") / (target_width - 1)
        elif coordinate_transformation_mode in ["asymmetric", "half_pixel"]:
            h_scale = image_height.astype("float") / target_height
            w_scale = image_width.astype("float") / target_width
        else:
            raise ValueError(
                "Unsupported coordinate_transformation_mode: {}".format(
                    coordinate_transformation_mode
                )
            )
        in_y = h_scale * y
        in_x = w_scale * x

    if coordinate_transformation_mode == "align_corners" or boxes is not None:
        closest_x_index = te.round(in_x).astype("int32")
        closest_y_index = te.round(in_y).astype("int32")
    else:
        # Add epsilon to floor to prevent gpu rounding errors.
        epsilon = 1e-5
        closest_y_index = te.floor(in_y + epsilon).astype("int32")
        closest_x_index = te.floor(in_x + epsilon).astype("int32")

    value = get_2d_pixel(
        data,
        layout,
        boxes,
        image_height,
        image_width,
        box_idx,
        c,
        closest_y_index,
        closest_x_index,
        cc,
        inum,
        ic,
    )

    if extrapolation_value is not None:
        out = tvm.tir.if_then_else(
            in_y < 0,
            extrapolation_value,
            tvm.tir.if_then_else(in_y > image_height - 1, extrapolation_value, value),
        )
        # use extrapolation_value if in_x is out of boundary
        value = tvm.tir.if_then_else(
            in_x < 0,
            extrapolation_value,
            tvm.tir.if_then_else(in_x > image_width - 1, extrapolation_value, out),
        )
    return _cast_output(value, data.dtype, out_dtype=out_dtype)


def resize_bilinear(
    indices,
    data,
    image_height,
    image_width,
    target_height,
    target_width,
    boxes=None,
    box_indices=None,
    extrapolation_value=None,
    layout="NCHW",
    coordinate_transformation_mode="align_corners",
    out_dtype=None,
):

    """Perform resize operation with bilinear method on the data.
    For details about Bilinear interpolation please refer to
    https://en.wikipedia.org/wiki/Bilinear_interpolation.

    Parameters
    ----------
    indices : tuple
        The indices of input data

    data : tvm.te.Tensor
        inputs is a 4-D tensor with shape
        [batch, channel, in_height, in_width]
        or  [batch, in_height, in_width, channel]

    image_height : integer
        Input image height

    image_width : integer
        Input image width

    target_height : integer
        The target resized image height

    target_width : integer
        The target resized image width

    boxes : tvm.te.Tensor, optional
        A 2-D tensor of shape [num_boxes, 4]. Each row of the tensor specifies
        the coordinates of a box.

    box_indices : tvm.te.Tensor, optional
        A 1-D tensor of shape [num_boxes], box_indices[i] specifies the data that
        the i-th box refers to.

    extrapolation_value: float, optional
        Value used for extrapolation, when applicable.

    layout: string, optional
        "NCHW", "NHWC", or "NCHWc".

    coordinate_transformation_mode: string, optional
        Describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor.
        Refer to the ONNX Resize operator specification for details.
        Available options are "half_pixel", "align_corners" and "asymmetric".

    out_dtype: string, optional
        Type to return. If left None will be same as input type.

    Returns
    -------
    output : out_dtype
        The computed result with type out_dtype
    """

    def _cast_output(value, data_dtype="float32", out_dtype=None):
        if out_dtype:
            dtype = out_dtype
        else:
            dtype = data_dtype
        return value.astype(dtype)

    def _lerp(A, B, t):
        return A * (1.0 - t) + B * t

    n, c, y, x, cc, inum, ic = get_2d_indices(indices, layout=layout)
    box_idx = box_indices(n) if box_indices is not None else n

    if boxes is not None:
        y1, x1 = boxes(n, 0), boxes(n, 1)
        y2, x2 = boxes(n, 2), boxes(n, 3)

        in_h = (image_height - 1) * (y2 - y1)
        in_w = (image_width - 1) * (x2 - x1)
        h_scale = in_h.astype("float") / (target_height - 1)
        w_scale = in_w.astype("float") / (target_width - 1)

        in_y = y1 * (image_height - 1) + h_scale * y
        in_x = x1 * (image_width - 1) + w_scale * x
    else:
        if coordinate_transformation_mode == "align_corners":
            h_scale = (image_height - 1).astype("float") / (target_height - 1)
            w_scale = (image_width - 1).astype("float") / (target_width - 1)
        elif coordinate_transformation_mode in ["asymmetric", "half_pixel"]:
            h_scale = image_height.astype("float") / target_height
            w_scale = image_width.astype("float") / target_width
        else:
            raise ValueError(
                "Unsupported coordinate_transformation_mode: {}".format(
                    coordinate_transformation_mode
                )
            )

        if coordinate_transformation_mode == "half_pixel":
            in_y = h_scale * (y + 0.5) - 0.5
            in_x = w_scale * (x + 0.5) - 0.5
        else:
            in_y = h_scale * y
            in_x = w_scale * x

    top_y_index = te.floor(in_y).astype("int32")
    bottom_y_index = te.ceil(in_y).astype("int32")
    y_lerp = in_y - top_y_index

    left_x_index = te.floor(in_x).astype("int32")
    right_x_index = te.ceil(in_x).astype("int32")
    x_lerp = in_x - left_x_index

    top_left = get_2d_pixel(
        data,
        layout,
        boxes,
        image_height,
        image_width,
        box_idx,
        c,
        top_y_index,
        left_x_index,
        cc,
        inum,
        ic,
    )
    top_right = get_2d_pixel(
        data,
        layout,
        boxes,
        image_height,
        image_width,
        box_idx,
        c,
        top_y_index,
        right_x_index,
        cc,
        inum,
        ic,
    )
    bottom_left = get_2d_pixel(
        data,
        layout,
        boxes,
        image_height,
        image_width,
        box_idx,
        c,
        bottom_y_index,
        left_x_index,
        cc,
        inum,
        ic,
    )
    bottom_right = get_2d_pixel(
        data,
        layout,
        boxes,
        image_height,
        image_width,
        box_idx,
        c,
        bottom_y_index,
        right_x_index,
        cc,
        inum,
        ic,
    )

    top = _lerp(top_left, top_right, x_lerp)
    bottom = _lerp(bottom_left, bottom_right, x_lerp)
    value = _lerp(top, bottom, y_lerp)

    # use extrapolation_value if in_y/in_x is out of boundary
    if extrapolation_value is not None:
        out = tvm.tir.if_then_else(
            in_y < 0,
            extrapolation_value,
            tvm.tir.if_then_else(in_y > image_height - 1, extrapolation_value, value),
        )
        value = tvm.tir.if_then_else(
            in_x < 0,
            extrapolation_value,
            tvm.tir.if_then_else(in_x > image_width - 1, extrapolation_value, out),
        )
    return _cast_output(value, data.dtype, out_dtype=out_dtype)


def resize_bicubic(
    indices,
    data,
    image_height,
    image_width,
    target_height,
    target_width,
    boxes=None,
    box_indices=None,
    extrapolation_value=None,
    layout="NCHW",
    coordinate_transformation_mode="align_corners",
    out_dtype=None,
):
    """Perform resize operation with bicubic method on the data.
    More details about Bicubic interpolation please refer to
    https://en.wikipedia.org/wiki/Bicubic_interpolation.

    Parameters
    ----------
    indices : tuple
        The indices of input data

    data : tvm.te.Tensor
        inputs is a 4-D tensor with shape
        [batch, channel, in_height, in_width]
        or  [batch, in_height, in_width, channel]

    image_height : integer
        Input image height

    image_width : integer
        Input image width

    target_height : integer
        The target resized image height

    target_width : integer
        The target resized image width

    boxes : tvm.te.Tensor, optional
        A 2-D tensor of shape [num_boxes, 4]. Each row of the tensor specifies
        the coordinates of a box.

    box_indices : tvm.te.Tensor, optional
        A 1-D tensor of shape [num_boxes], box_indices[i] specifies the data that
        the i-th box refers to.

    extrapolation_value: float, optional
        Value used for extrapolation, when applicable.

    layout: string, optional
        "NCHW", "NHWC", or "NCHWc".

    coordinate_transformation_mode: string, optional
        Describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor.
        Refer to the ONNX Resize operator specification for details.
        Available options are "half_pixel", "align_corners" and "asymmetric".

    out_dtype: string, optional
        Type to return. If left None will be same as input type.

    Returns
    -------
    output : out_dtype
        The computed result with type out_dtype
    """

    def _cubic_kernel(A, B, C, D, t):
        a = -A / 2.0 + (3.0 * B) / 2.0 - (3.0 * C) / 2.0 + D / 2.0
        b = A - (5.0 * B) / 2.0 + 2.0 * C - D / 2.0
        c = -A / 2.0 + C / 2.0
        d = B
        return a * t * t * t + b * t * t + c * t + d

    def _cast_output(value, data_dtype="float32", out_dtype=None):
        if out_dtype:
            dtype = out_dtype
        else:
            dtype = data_dtype
        return value.astype(dtype)

    n, c, y, x, cc, inum, ic = get_2d_indices(indices, layout)
    box_idx = box_indices(n) if box_indices is not None else n

    if boxes is not None:
        y1, x1 = boxes(n, 0), boxes(n, 1)
        y2, x2 = boxes(n, 2), boxes(n, 3)

        in_h = (image_height - 1) * (y2 - y1)
        in_w = (image_width - 1) * (x2 - x1)
        h_scale = in_h.astype("float") / (target_height - 1)
        w_scale = in_w.astype("float") / (target_width - 1)

        in_y = y1 * (image_height - 1) + h_scale * y
        in_x = x1 * (image_width - 1) + w_scale * x
    else:
        if coordinate_transformation_mode == "align_corners":
            h_scale = (image_height - 1).astype("float") / (target_height - 1)
            w_scale = (image_width - 1).astype("float") / (target_width - 1)
        elif coordinate_transformation_mode in ["asymmetric", "half_pixel"]:
            h_scale = image_height.astype("float") / target_height
            w_scale = image_width.astype("float") / target_width
        else:
            raise ValueError(
                "Unsupported coordinate_transformation_mode: {}".format(
                    coordinate_transformation_mode
                )
            )

        if coordinate_transformation_mode == "half_pixel":
            in_y = h_scale * (y + 0.5) - 0.5
            in_x = w_scale * (x + 0.5) - 0.5
        else:
            in_y = h_scale * y
            in_x = w_scale * x

    xint = te.floor(in_x).astype("int32")
    xfract = in_x - te.floor(in_x)

    yint = te.floor(in_y).astype("int32")
    yfract = in_y - te.floor(in_y)

    # 1st row
    p00 = _get_pixel(
        data, layout, boxes, image_height, image_width, box_idx, c, yint - 1, xint - 1, cc, inum, ic
    )
    p10 = _get_pixel(
        data, layout, boxes, image_height, image_width, box_idx, c, yint - 1, xint + 0, cc, inum, ic
    )
    p20 = _get_pixel(
        data, layout, boxes, image_height, image_width, box_idx, c, yint - 1, xint + 1, cc, inum, ic
    )
    p30 = _get_pixel(
        data, layout, boxes, image_height, image_width, box_idx, c, yint - 1, xint + 2, cc, inum, ic
    )

    # 2nd row
    p01 = _get_pixel(
        data, layout, boxes, image_height, image_width, box_idx, c, yint + 0, xint - 1, cc, inum, ic
    )
    p11 = _get_pixel(
        data, layout, boxes, image_height, image_width, box_idx, c, yint + 0, xint + 0, cc, inum, ic
    )
    p21 = _get_pixel(
        data, layout, boxes, image_height, image_width, box_idx, c, yint + 0, xint + 1, cc, inum, ic
    )
    p31 = _get_pixel(
        data, layout, boxes, image_height, image_width, box_idx, c, yint + 0, xint + 2, cc, inum, ic
    )

    # 3rd row
    p02 = _get_pixel(
        data, layout, boxes, image_height, image_width, box_idx, c, yint + 1, xint - 1, cc, inum, ic
    )
    p12 = _get_pixel(
        data, layout, boxes, image_height, image_width, box_idx, c, yint + 1, xint + 0, cc, inum, ic
    )
    p22 = _get_pixel(
        data, layout, boxes, image_height, image_width, box_idx, c, yint + 1, xint + 1, cc, inum, ic
    )
    p32 = _get_pixel(
        data, layout, boxes, image_height, image_width, box_idx, c, yint + 1, xint + 2, cc, inum, ic
    )

    # 4th row
    p03 = _get_pixel(
        data, layout, boxes, image_height, image_width, box_idx, c, yint + 2, xint - 1, cc, inum, ic
    )
    p13 = _get_pixel(
        data, layout, boxes, image_height, image_width, box_idx, c, yint + 2, xint + 0, cc, inum, ic
    )
    p23 = _get_pixel(
        data, layout, boxes, image_height, image_width, box_idx, c, yint + 2, xint + 1, cc, inum, ic
    )
    p33 = _get_pixel(
        data, layout, boxes, image_height, image_width, box_idx, c, yint + 2, xint + 2, cc, inum, ic
    )

    # Interpolate bicubically
    col0 = _cubic_kernel(p00, p10, p20, p30, xfract)
    col1 = _cubic_kernel(p01, p11, p21, p31, xfract)
    col2 = _cubic_kernel(p02, p12, p22, p32, xfract)
    col3 = _cubic_kernel(p03, p13, p23, p33, xfract)
    value = _cubic_kernel(col0, col1, col2, col3, yfract)

    # use extrapolation_value if in_y/in_x is out of boundary
    if extrapolation_value is not None:
        out = tvm.tir.if_then_else(
            in_y < 0,
            extrapolation_value,
            tvm.tir.if_then_else(in_y > image_height - 1, extrapolation_value, value),
        )
        value = tvm.tir.if_then_else(
            in_x < 0,
            extrapolation_value,
            tvm.tir.if_then_else(in_x > image_width - 1, extrapolation_value, out),
        )
    return _cast_output(value, data.dtype, out_dtype=out_dtype)


def resize(
    data,
    size,
    layout="NCHW",
    method="bilinear",
    coordinate_transformation_mode="half_pixel",
    out_dtype=None,
    output_shape=None,
):
    """Perform resize operation on the data.

    Parameters
    ----------
    data : tvm.te.Tensor
        inputs is a 4-D tensor with shape
        [batch, channel, in_height, in_width]
        or  [batch, in_height, in_width, channel]

    size: Tuple
        Output resolution scale to

    layout: string, optional
        "NCHW", "NHWC", or "NCHWc".

    coordinate_transformation_mode: string, optional
        Describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor.
        Refer to the ONNX Resize operator specification for details.
        Available options are "half_pixel", "align_corners" and "asymmetric".

    method: {"bilinear", "nearest_neighbor", "bicubic"}
        Method to be used for resizing.

    out_dtype: string, optional
        Type to return. If left None will be same as input type.

    output_shape: tvm.tir.container.Array, optional
        Shape to return. If left None will be inferred
        (If shape is determined dynamically, pass out_dtype.shape as output_shape)

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, channel, in_height*scale, in_width*scale]
        or [batch, in_height*scale, in_width*scale, channel]
        or 5-D with shape [batch, channel-major, in_height*scale, in_width*scale, channel-minor]
    """
    method = method.lower()
    if method == "nearest_neighbor" and coordinate_transformation_mode != "asymmetric":
        raise ValueError(
            "Topi Resize does not support the combination of method %s "
            "and coordinate_transformation_mode %s" % (method, coordinate_transformation_mode)
        )
    if layout == "NHWC":
        in_n, in_h, in_w, in_c = data.shape
        if output_shape is None:
            output_shape = [in_n, size[0], size[1], in_c]
    elif layout == "NCHW":
        in_n, in_c, in_h, in_w = data.shape
        if output_shape is None:
            output_shape = [in_n, in_c, size[0], size[1]]
    elif nchw_pack_layout(layout):  # for NCHWinic
        in_n, in_c, in_h, in_w, in_inum, in_ic = data.shape
        if output_shape is None:
            output_shape = [in_n, in_c, size[0], size[1], in_inum, in_ic]
    elif nchw_xc_layout(layout):  # for NCHWxc
        in_n, in_c, in_h, in_w, in_cc = data.shape
        if output_shape is None:
            output_shape = [in_n, in_c, size[0], size[1], in_cc]
    else:
        raise ValueError("%s layout is not supported." % layout)

    def _nearest_neighbor(*indices):
        return resize_nearest_neighbor(
            indices,
            data,
            in_h,
            in_w,
            size[0],
            size[1],
            layout=layout,
            coordinate_transformation_mode=coordinate_transformation_mode,
            out_dtype=out_dtype,
        )

    def _bilinear(*indices):
        return resize_bilinear(
            indices,
            data,
            in_h,
            in_w,
            size[0],
            size[1],
            layout=layout,
            coordinate_transformation_mode=coordinate_transformation_mode,
            out_dtype=out_dtype,
        )

    def _bicubic(*indices):
        return resize_bicubic(
            indices,
            data,
            in_h,
            in_w,
            size[0],
            size[1],
            layout,
            coordinate_transformation_mode=coordinate_transformation_mode,
            out_dtype=out_dtype,
        )

    # Determine which interpolation method to use then run it.
    if method == "nearest_neighbor":
        compute_func = _nearest_neighbor
    elif method == "bilinear":
        compute_func = _bilinear
    elif method == "bicubic":
        compute_func = _bicubic
    else:
        raise ValueError("%s method is not supported." % method)

    return te.compute(output_shape, compute_func, name="resize", tag=tag.INJECTIVE)


def crop_and_resize(
    data,
    boxes,
    box_indices,
    crop_size,
    layout="NCHW",
    method="bilinear",
    extrapolation_value=0,
    out_dtype=None,
):
    """Perform crop and resize operation on the data.

    Parameters
    ----------
    data : tvm.te.Tensor
        inputs is a 4-D tensor with shape
        [batch, channel, in_height, in_width]
        or  [batch, in_height, in_width, channel]

    boxes : tvm.te.Tensor
        A 2-D tensor of shape [num_boxes, 4]. Each row of the tensor specifies
        the coordinates of a box.

    box_indices : tvm.te.Tensor
        A 1-D tensor of shape [num_boxes], box_indices[i] specifies the data that
        the i-th box refers to.

    crop_size : Tuple
        The target size of each box.

    layout : string, optional
        "NCHW", "NHWC"

    method : {"bilinear", "nearest_neighbor"}
        Method to be used for resizing.

    extrapolation_value: float, optional
        Value used for extrapolation, when applicable.

    out_dtype : string, optional
        Type to return. If left None will be same as input type.

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [num_boxes, channel, crop_height, crop_width]
        or [num_boxes, crop_height, crop_width, channel]
    """
    method = method.lower()
    target_h = crop_size[0]
    target_w = crop_size[1]

    if layout == "NHWC":
        output_shape = [box_indices.shape[0], crop_size[0], crop_size[1], data.shape[3]]
        image_h = data.shape[1].astype("int32")
        image_w = data.shape[2].astype("int32")
    elif layout == "NCHW":
        output_shape = [box_indices.shape[0], data.shape[1], crop_size[0], crop_size[1]]
        image_h = data.shape[2].astype("int32")
        image_w = data.shape[3].astype("int32")
    elif layout.startswith("NCHW"):  # for NCHWxc
        output_shape = [
            box_indices.shape[0],
            data.shape[1],
            crop_size[0],
            crop_size[1],
            data.shape[4],
        ]
        image_h = data.shape[2].astype("int32")
        image_w = data.shape[3].astype("int32")
    else:
        raise ValueError("%s layout is not supported." % layout)

    def _bilinear(*indices):
        return resize_bilinear(
            indices,
            data,
            image_h,
            image_w,
            target_h,
            target_w,
            boxes,
            box_indices,
            extrapolation_value,
            layout,
            out_dtype=out_dtype,
        )

    def _nearest_neighbor(*indices):
        return resize_nearest_neighbor(
            indices,
            data,
            image_h,
            image_w,
            target_h,
            target_w,
            boxes,
            box_indices,
            extrapolation_value,
            layout,
            out_dtype=out_dtype,
        )

    # Determine which interpolation method to use then run it.
    if method == "nearest_neighbor":
        compute_func = _nearest_neighbor
    elif method == "bilinear":
        compute_func = _bilinear
    else:
        raise ValueError("%s method is not supported." % method)

    return te.compute(output_shape, compute_func, name="crop_and_resize", tag=tag.INJECTIVE)


def resize3d(
    data,
    size,
    layout="NCDHW",
    method="nearest_neighbor",
    coordinate_transformation_mode="align_corners",
    out_dtype=None,
):
    """Perform resize operation on the data.

    Parameters
    ----------
    inputs: tvm.te.Tensor
        inputs is a 5-D tensor with shape
        [batch, channel, in_depth, in_height, in_width]
        or  [batch, in_depth, in_height, in_width, channel]

    size: Tuple
        Output resolution scale to

    layout: string, optional
        "NCDHW", "NDHWC", or "NCDHWc".

    coordinate_transformation_mode: string, optional
        Describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor.
        Refer to the ONNX Resize operator specification for details.

        Available options are "half_pixel", "align_corners" and "asymmetric".
    method: {"trilinear", "nearest_neighbor"}
        Method to be used for resizing.

    out_dtype: string, optional
        Type to return. If left None will be same as input type.

    Returns
    -------
    output : tvm.te.Tensor
        5-D with shape [batch, channel, in_depth*scale, in_height*scale, in_width*scale]
        or [batch, in_depth*scale, in_height*scale, in_width*scale, channel]
        or 5-D with shape [batch, channel-major, in_depth*scale, in_height*scale, in_width*scale,
        channel-minor]
    """
    method = method.lower()

    if layout == "NDHWC":
        in_n, in_d, in_h, in_w, in_c = data.shape
        output_shape = [in_n, size[0], size[1], size[2], in_c]
    elif layout == "NCDHW":
        in_n, in_c, in_d, in_h, in_w = data.shape
        output_shape = [in_n, in_c, size[0], size[1], size[2]]
    # Otherwise layout must be NCHWxc
    else:
        in_n, in_c, in_d, in_h, in_w, in_cc = data.shape
        output_shape = [in_n, in_c, size[0], size[1], size[2], in_cc]

    if coordinate_transformation_mode == "align_corners":
        z_ratio = (in_d - 1).astype("float") / (size[0] - 1)
        y_ratio = (in_h - 1).astype("float") / (size[1] - 1)
        x_ratio = (in_w - 1).astype("float") / (size[2] - 1)
    elif coordinate_transformation_mode in ["asymmetric", "half_pixel"]:
        z_ratio = (in_d).astype("float") / (size[0])
        y_ratio = (in_h).astype("float") / (size[1])
        x_ratio = (in_w).astype("float") / (size[2])
    else:
        raise ValueError(
            "Unsupported coordinate_transformation_mode: {}".format(coordinate_transformation_mode)
        )

    def _get_pixel(n, c, z, y, x, cc):
        z = tvm.te.max(tvm.te.min(z, in_d - 1), 0)
        y = tvm.te.max(tvm.te.min(y, in_h - 1), 0)
        x = tvm.te.max(tvm.te.min(x, in_w - 1), 0)
        if layout == "NDHWC":
            return data(n, z, y, x, c).astype("float")
        if layout == "NCDHW":
            return data(n, c, z, y, x).astype("float")
        # else must be NCDHWxc
        return data(n, c, z, y, x, cc).astype("float")

    def _get_indices(*indices):
        if layout == "NDHWC":
            n, z, y, x, c = indices
            cc = None
        elif layout == "NCDHW":
            n, c, z, y, x = indices
            cc = None
        else:
            n, c, z, y, x, cc = indices

        return n, c, z, y, x, cc

    def _cast_output(value):
        if out_dtype:
            dtype = out_dtype
        else:
            dtype = data.dtype
        return value.astype(dtype)

    # Nearest neighbor computation
    def _nearest_neighbor(*indices):
        n, c, z, y, x, cc = _get_indices(*indices)

        in_z = z_ratio * z
        in_y = y_ratio * y
        in_x = x_ratio * x

        if coordinate_transformation_mode == "align_corners":
            zint = te.round(in_z).astype("int32")
            yint = te.round(in_y).astype("int32")
            xint = te.round(in_x).astype("int32")
        elif coordinate_transformation_mode in ["asymmetric", "half_pixel"]:
            # Add epsilon to floor to prevent gpu rounding errors.
            epsilon = 1e-5
            zint = te.floor(in_z + epsilon).astype("int32")
            yint = te.floor(in_y + epsilon).astype("int32")
            xint = te.floor(in_x + epsilon).astype("int32")
        else:
            raise ValueError(
                "Unsupported coordinate_transformation_mode: {}".format(
                    coordinate_transformation_mode
                )
            )

        return _cast_output(_get_pixel(n, c, zint, yint, xint, cc))

    # Trilinear helper functions and computation.
    def _lerp(A, B, t):
        return A * (1.0 - t) + B * t

    def _trilinear(*indices):
        n, c, z, y, x, cc = _get_indices(*indices)

        if coordinate_transformation_mode == "half_pixel":
            in_z = z_ratio * (z + 0.5) - 0.5
            in_y = y_ratio * (y + 0.5) - 0.5
            in_x = x_ratio * (x + 0.5) - 0.5
        else:
            in_z = z_ratio * z
            in_y = y_ratio * y
            in_x = x_ratio * x

        zint = te.floor(in_z).astype("int32")
        zfract = in_z - te.floor(in_z)

        xint = te.floor(in_x).astype("int32")
        xfract = in_x - te.floor(in_x)

        yint = te.floor(in_y).astype("int32")
        yfract = in_y - te.floor(in_y)

        p000 = _get_pixel(n, c, zint, yint, xint, cc)
        p001 = _get_pixel(n, c, zint, yint, xint + 1, cc)
        p010 = _get_pixel(n, c, zint, yint + 1, xint, cc)
        p011 = _get_pixel(n, c, zint, yint + 1, xint + 1, cc)
        p100 = _get_pixel(n, c, zint + 1, yint, xint, cc)
        p101 = _get_pixel(n, c, zint + 1, yint, xint + 1, cc)
        p110 = _get_pixel(n, c, zint + 1, yint + 1, xint, cc)
        p111 = _get_pixel(n, c, zint + 1, yint + 1, xint + 1, cc)

        dep00 = _lerp(p000, p100, zfract)
        dep01 = _lerp(p001, p101, zfract)
        dep10 = _lerp(p010, p110, zfract)
        dep11 = _lerp(p011, p111, zfract)
        col0 = _lerp(dep00, dep01, xfract)
        col1 = _lerp(dep10, dep11, xfract)
        value = _lerp(col0, col1, yfract)
        return _cast_output(value)

    # Determine which interpolation method to use then run it.
    if method == "nearest_neighbor":
        compute_func = _nearest_neighbor
    elif method == "trilinear":
        compute_func = _trilinear
    else:
        raise ValueError("%s method is not supported." % method)

    return te.compute(output_shape, compute_func, name="resize3d", tag=tag.INJECTIVE)
