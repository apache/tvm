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
# pylint: disable=import-self, invalid-name, unused-argument, too-many-lines, len-as-condition, broad-except
# pylint: disable=import-outside-toplevel, redefined-builtin
"""TF: Tensorflow frontend operator inventory"""
import warnings

# Numpy support
import numpy as np
import tvm

from tvm.topi.utils import get_const_tuple

from ... import expr as _expr
from ... import op as _op
from ..common import AttrCvt, get_relay_op
from ..common import infer_type as _infer_type
from ..common import infer_shape as _infer_shape
from ..common import infer_channels as _infer_channels
from ..common import infer_value as _infer_value
from . import tensorflow_util as util


def impl_rsqrt():
    """ Rsqrt """

    def _impl(inputs, attr, params, mod):
        inputs.append(tvm.relay.const(-0.5, attr["T"].name))
        return AttrCvt(op_name="power")(inputs, attr)

    return _impl


def impl_argx(func, func_name):
    """ A common wrapper for argmin and argmax operations """

    def _impl(inputs, attr, params, mod):
        try:
            # In Tensorflow, `axis` argument is a Tensor, not attribute. We
            # support the case where it inputs from a scalar constant.
            axis_input_value = [util.get_num_param(params, inputs[1])]
        except (IndexError, KeyError):
            raise TypeError(
                "Unsupported argument for `{}` : `axis` should be a constant".format(func_name)
            )
        out = func(inputs[0], axis=axis_input_value, keepdims=False)
        dtype = attr["output_type"].name
        if dtype != "int32":
            out = _op.cast(out, dtype=dtype)
        return out

    return _impl


def impl_elemwise(name):
    """ Elementwise op """

    def _impl(inputs, attr, params, mod):
        assert len(inputs) == 2, "{} take 2 inputs, {} given".format(name, len(inputs))
        return get_relay_op(name)(*inputs)

    return _impl


def impl_pool3d(name):
    """ Pool3D """

    def _impl(inputs, attr, params, mod):
        attr["data_format"] = attr["data_format"].decode("utf-8")
        flip_layout = False

        input_shape = _infer_shape(inputs[0], mod)

        if attr["data_format"] == "NDHWC":
            attr["kernel_shape"] = (attr["ksize"][1], attr["ksize"][2], attr["ksize"][3])
            attr["strides"] = (attr["strides"][1], attr["strides"][2], attr["strides"][3])
        elif attr["data_format"] == "NCDHW":
            attr["kernel_shape"] = (attr["ksize"][2], attr["ksize"][3], attr["ksize"][4])
            attr["strides"] = (attr["strides"][2], attr["strides"][3], attr["strides"][4])
        else:
            msg = 'Value {} of attribute "data_format" of operator Pooling ' "is not valid."
            raise tvm.error.OpAttributeInvalid(msg.format(attr["data_format"]))
        if attr["data_format"] == "NDHWC":
            input_shape = [_infer_shape(inputs[0], mod)[i] for i in (0, 4, 1, 2, 3)]
            inputs[0] = _op.transpose(inputs[0], axes=(0, 4, 1, 2, 3))
            attr["data_format"] = "NCDHW"
            flip_layout = True

        attr["padding"] = attr["padding"].decode("utf-8")

        if attr["padding"] == "VALID":
            attr["padding"] = [0, 0, 0, 0, 0, 0]
        elif attr["padding"] == "SAME":
            stride_d, stride_h, stride_w = attr["strides"]
            kernel_d, kernel_h, kernel_w = attr["kernel_shape"]
            if attr["data_format"] == "NDHWC":
                in_d = input_shape[1]
                in_h = input_shape[2]
                in_w = input_shape[3]
            else:
                in_d = input_shape[2]
                in_h = input_shape[3]
                in_w = input_shape[4]
            pad_d = util.get_pad_pair(in_d, kernel_d, stride_d)
            pad_v = util.get_pad_pair(in_h, kernel_h, stride_h)
            pad_h = util.get_pad_pair(in_w, kernel_w, stride_w)

            attr["padding"] = [pad_d[0], pad_v[0], pad_h[0], pad_d[1], pad_v[1], pad_h[1]]
        else:
            msg = 'Value {} in attribute "padding" of operator Pooling is ' "not valid."
            raise tvm.error.OpAttributeInvalid(msg.format(attr["padding"]))

        if name == "avg_pool":
            attr["count_include_pad"] = False
        attr["ceil_mode"] = False
        out = AttrCvt(
            op_name=name,
            transforms={"kernel_shape": "pool_size", "data_format": "layout"},
            ignores=["ksize"],
        )(inputs, attr)
        if flip_layout:
            out = _op.transpose(out, axes=(0, 2, 3, 4, 1))
        return out

    return _impl


def impl_pooling(name):
    """ Pool """

    def _impl(inputs, attr, params, mod):

        attr["data_format"] = attr["data_format"].decode("utf-8")
        flip_layout = False

        input_shape = _infer_shape(inputs[0], mod)

        if attr["data_format"] == "NHWC":
            attr["kernel_shape"] = (attr["ksize"][1], attr["ksize"][2])
            attr["strides"] = (attr["strides"][1], attr["strides"][2])
        elif attr["data_format"] == "NCHW":
            attr["kernel_shape"] = (attr["ksize"][2], attr["ksize"][3])
            attr["strides"] = (attr["strides"][2], attr["strides"][3])
        else:
            msg = 'Value {} of attribute "data_format" of operator Pooling ' "is not valid."
            raise tvm.error.OpAttributeInvalid(msg.format(attr["data_format"]))

        if attr["_target_layout"] == "NCHW" and attr["data_format"] == "NHWC":
            tmp_shape = _infer_shape(inputs[0], mod)
            input_shape = [tmp_shape[ii] for ii in (0, 3, 1, 2)]
            inputs[0] = _op.transpose(inputs[0], axes=(0, 3, 1, 2))
            attr["data_format"] = "NCHW"
            flip_layout = True

        # Fix padding
        attr["padding"] = attr["padding"].decode("utf-8")

        if attr["padding"] == "VALID":
            attr["padding"] = [0, 0]
        elif attr["padding"] == "SAME":
            stride_h, stride_w = attr["strides"]
            kernel_h, kernel_w = attr["kernel_shape"]
            if attr["data_format"] == "NHWC":
                in_h = input_shape[1]
                in_w = input_shape[2]
            else:
                in_h = input_shape[2]
                in_w = input_shape[3]

            pad_v = util.get_pad_pair(in_h, kernel_h, stride_h)
            pad_h = util.get_pad_pair(in_w, kernel_w, stride_w)

            attr["padding"] = [pad_v[0], pad_h[0], pad_v[1], pad_h[1]]
        elif attr["padding"] == "EXPLICIT":
            paddings = attr["explicit_paddings"]
            assert len(paddings) == 8
            if flip_layout or attr["data_format"] == "NHWC":
                attr["padding"] = [paddings[2], paddings[4], paddings[3], paddings[5]]
            else:
                attr["padding"] = [paddings[4], paddings[6], paddings[5], paddings[7]]
        else:
            msg = 'Value {} in attribute "padding" of operator Pooling is ' "not valid."
            raise tvm.error.OpAttributeInvalid(msg.format(attr["padding"]))

        if name == "avg_pool":
            attr["count_include_pad"] = False

        out = AttrCvt(
            op_name=util.dimension_picker(name),
            transforms={"kernel_shape": "pool_size", "data_format": "layout"},
            ignores=["ksize", "explicit_paddings"],
            extras={"ceil_mode": False},
            custom_check=util.dimension_constraint(),
        )(inputs, attr)

        if flip_layout:
            out = _op.transpose(out, axes=(0, 2, 3, 1))

        return out

    return _impl


def impl_conv(opname):
    """ Convolution """

    def _impl(inputs, attr, params, mod):
        attr["data_format"] = attr["data_format"].decode("utf-8")
        flip_layout = False

        if opname == "conv_transpose" and attr["data_format"] == "NHWC":
            # transform to NCHW for TVM backend compatible and set 'flip_layout'
            # to have output flip back to NHWC
            inputs[2] = _op.transpose(inputs[2], axes=(0, 3, 1, 2))
            attr["strides"][1], attr["strides"][2], attr["strides"][3] = (
                attr["strides"][3],
                attr["strides"][1],
                attr["strides"][2],
            )
            attr["data_format"] = "NCHW"

            # Check whether output shapes attribute is set and not None
            if (
                opname == "conv_transpose"
                and len(attr["_output_shapes"]) > 0
                and attr["_output_shapes"][0]
            ):
                tmp_shape = attr["_output_shapes"][0]
                tmp_shape = [tmp_shape[ii] for ii in (0, 3, 1, 2)]
                attr["_output_shapes"][0] = tmp_shape

            flip_layout = True

        inputs_data = inputs[0] if opname != "conv_transpose" else inputs[2]

        # NCHW Layout require weights transpose
        weights_shape = _infer_shape(inputs[1], mod)
        if attr["data_format"] == "NCHW":
            tmp_shape = weights_shape
            if opname in ["conv", "conv_transpose"]:
                tmp_shape = [tmp_shape[ii] for ii in (3, 2, 0, 1)]
                inputs[1] = _op.transpose(inputs[1], axes=(3, 2, 0, 1))
            else:
                tmp_shape = [tmp_shape[ii] for ii in (2, 3, 0, 1)]
                inputs[1] = _op.transpose(inputs[1], axes=(2, 3, 0, 1))
            weights_shape = tmp_shape

        input_shape = _infer_shape(inputs_data, mod)
        if attr["_target_layout"] == "NCHW" and attr["data_format"] == "NHWC":
            input_shape = [input_shape[ii] for ii in (0, 3, 1, 2)]
            inputs_data = _op.transpose(inputs_data, axes=(0, 3, 1, 2))
            if opname in ["conv", "conv_transpose"]:
                weights_shape = [weights_shape[ii] for ii in (3, 2, 0, 1)]
                inputs[1] = _op.transpose(inputs[1], axes=(3, 2, 0, 1))
            else:
                weights_shape = [weights_shape[ii] for ii in (2, 3, 0, 1)]
                inputs[1] = _op.transpose(inputs[1], axes=(2, 3, 0, 1))

            attr["data_format"] = "NCHW"
            attr["strides"] = [attr["strides"][ii] for ii in (0, 3, 1, 2)]
            flip_layout = True

        if attr["data_format"] == "NHWC":
            in_channels = input_shape[3]
            kernel_h, kernel_w, _, depth_mult = weights_shape
            attr["kernel_shape"] = (weights_shape[0], weights_shape[1])
            if opname == "conv":
                attr["channels"] = weights_shape[3]
            elif opname == "conv_transpose":
                attr["channels"] = weights_shape[2]
            else:
                attr["channels"] = input_shape[3] * depth_mult

            if "dilations" in attr:
                attr["dilations"] = (attr["dilations"][1], attr["dilations"][2])
            attr["strides"] = (attr["strides"][1], attr["strides"][2])
        elif attr["data_format"] == "NCHW":
            in_channels = input_shape[1]
            _, depth_mult, kernel_h, kernel_w = weights_shape
            attr["kernel_shape"] = (weights_shape[2], weights_shape[3])
            if opname == "conv":
                attr["channels"] = weights_shape[0]
            elif opname == "conv_transpose":
                attr["channels"] = weights_shape[1]
            else:
                attr["channels"] = input_shape[1] * depth_mult
                if attr["channels"] < 0:
                    attr["channels"] *= -1

            if "dilations" in attr:
                attr["dilations"] = (attr["dilations"][2], attr["dilations"][3])
            attr["strides"] = (attr["strides"][2], attr["strides"][3])
        else:
            msg = 'Value {} in attribute "data_format" of operator Conv is ' "not valid."
            raise tvm.error.OpAttributeInvalid(msg.format(attr["data_format"]))

        if opname == "depthwise":
            attr["groups"] = in_channels

        # Fix padding
        attr["padding"] = attr["padding"].decode("utf-8")

        if attr["padding"] == "VALID":
            attr["padding"] = [0, 0]
        elif attr["padding"] == "SAME":
            stride_h, stride_w = attr["strides"]
            kernel_h, kernel_w = attr["kernel_shape"]

            pdata_shape = input_shape
            # Check whether output shapes attribute is set and not None
            if (
                opname == "conv_transpose"
                and len(attr["_output_shapes"]) > 0
                and attr["_output_shapes"][0]
            ):
                pdata_shape = attr["_output_shapes"][0]

            if attr["data_format"] == "NHWC":
                in_h = pdata_shape[1]
                in_w = pdata_shape[2]
            else:
                in_h = pdata_shape[2]
                in_w = pdata_shape[3]

            dilation_h = attr["dilations"][0]
            dilation_w = attr["dilations"][1]
            dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
            dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
            pad_v = util.get_pad_pair(in_h, dilated_kernel_h, stride_h)
            pad_h = util.get_pad_pair(in_w, dilated_kernel_w, stride_w)

            attr["padding"] = [pad_v[0], pad_h[0], pad_v[1], pad_h[1]]
        elif attr["padding"] == "EXPLICIT":
            paddings = attr["explicit_paddings"]
            assert len(paddings) == 8
            if flip_layout or attr["data_format"] == "NHWC":
                attr["padding"] = [paddings[2], paddings[4], paddings[3], paddings[5]]
            else:
                attr["padding"] = [paddings[4], paddings[6], paddings[5], paddings[7]]
        else:
            msg = 'Value {} in attribute "padding" of operator Conv is not ' "valid."
            raise tvm.error.OpAttributeInvalid(msg.format(attr["padding"]))

        if "kernel_layout" not in attr:
            if opname in ["conv", "conv_transpose"]:
                attr["kernel_layout"] = "HWIO" if attr["data_format"] == "NHWC" else "OIHW"
            else:
                attr["kernel_layout"] = "HWOI" if attr["data_format"] == "NHWC" else "OIHW"

        # Ignore the new attributes from TF2.0, for now.
        out = AttrCvt(
            op_name=util.dimension_picker(
                "conv", surfix="_transpose" if opname == "conv_transpose" else ""
            ),
            ignores=["explicit_paddings"],
            transforms={
                "kernel_shape": "kernel_size",
                "data_format": "data_layout",
                "dilations": ("dilation", (0, 0)),
                "group": ("groups", 1),
            },
            custom_check=util.dimension_constraint(),
        )([inputs_data, inputs[1]], attr)

        if flip_layout:
            out = _op.transpose(out, axes=(0, 2, 3, 1))

        return out

    return _impl


# Dilation2d
def impl_dilation2d():
    """ Dilation 2D """

    def _impl(inputs, attr, params, mod):
        if "data_format" not in attr:
            attr["data_format"] = "NHWC"

        input_shape = _infer_shape(inputs[0], mod)
        weights_shape = _infer_shape(inputs[1], mod)

        if attr["_target_layout"] == "NCHW" and attr["data_format"] == "NHWC":
            input_shape = [input_shape[ii] for ii in (0, 3, 1, 2)]
            inputs[0] = _op.transpose(inputs[0], axes=(0, 3, 1, 2))
            weights_shape = [weights_shape[ii] for ii in (2, 0, 1)]
            inputs[1] = _op.transpose(inputs[1], axes=(2, 0, 1))
            attr["data_format"] = "NCHW"

        if attr["data_format"] in ["NHWC", "NCHW"]:
            if "rates" in attr:
                attr["dilations"] = attr["rates"]
            if "dilations" in attr:
                attr["dilations"] = (attr["dilations"][1], attr["dilations"][2])
            attr["strides"] = (attr["strides"][1], attr["strides"][2])
        else:
            msg = 'Value {} in attribute "data_format" of operator Dilation2D is ' "not valid."
            raise tvm.error.OpAttributeInvalid(msg.format(attr["data_format"]))

        attr["padding"] = attr["padding"].decode("utf-8")
        if attr["padding"] == "VALID":
            attr["padding"] = [0, 0]
        elif attr["padding"] == "SAME":
            stride_h, stride_w = attr["strides"]
            if attr["data_format"] == "NHWC":
                kernel_h, kernel_w = weights_shape[0], weights_shape[1]
            else:
                kernel_h, kernel_w = weights_shape[1], weights_shape[2]
            if attr["data_format"] == "NHWC":
                in_h = input_shape[1]
                in_w = input_shape[2]
            else:
                in_h = input_shape[2]
                in_w = input_shape[3]

            dilation_h = attr["dilations"][0]
            dilation_w = attr["dilations"][1]
            dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
            dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
            pad_v = util.get_pad_pair(in_h, dilated_kernel_h, stride_h)
            pad_h = util.get_pad_pair(in_w, dilated_kernel_w, stride_w)

            if attr["data_format"] == "NHWC":
                inputs[0] = _op.nn.pad(
                    data=inputs[0],
                    pad_width=((0, 0), (pad_v[0], pad_v[1]), (pad_h[0], pad_h[1]), (0, 0)),
                )
            else:
                inputs[0] = _op.nn.pad(
                    data=inputs[0],
                    pad_width=((0, 0), (0, 0), (pad_v[0], pad_v[1]), (pad_h[0], pad_h[1])),
                )

            attr["padding"] = [0, 0]

        else:
            msg = 'Value {} in attribute "padding" of operator Dilation2d is not ' "valid."
            raise tvm.error.OpAttributeInvalid(msg.format(attr["padding"]))

        attr["kernel_layout"] = "HWI" if attr["data_format"] == "NHWC" else "IHW"
        out = AttrCvt(
            op_name="dilation2d",
            ignores=["explicit_paddings", "rates"],
            transforms={
                "data_format": "data_layout",
            },
        )([inputs[0], inputs[1]], attr)
        if attr["_target_layout"] == "NCHW":
            out = _op.transpose(out, axes=(0, 2, 3, 1))
        return out

    return _impl


def impl_conv3d(opname):
    """ Convolution 3D """

    def _impl(inputs, attr, params, mod):
        attr["data_format"] = attr["data_format"].decode("utf-8")
        flip_layout = False

        inputs_data = inputs[0] if opname != "conv_transpose" else inputs[2]

        # NCDHW Layout require weights transpose
        weights_shape = _infer_shape(inputs[1], mod)
        if attr["data_format"] == "NCDHW":
            tmp_shape = weights_shape
            tmp_shape = [tmp_shape[ii] for ii in (4, 3, 0, 1, 2)]
            inputs[1] = _op.transpose(inputs[1], axes=(4, 3, 0, 1, 2))
            weights_shape = tmp_shape

        input_shape = _infer_shape(inputs_data, mod)

        if attr["_target_layout"] == "NCDHW" and attr["data_format"] == "NDHWC":
            input_shape = [input_shape[ii] for ii in (0, 4, 1, 2, 3)]
            inputs_data = _op.transpose(inputs_data, axes=(0, 4, 1, 2, 3))
            weights_shape = [weights_shape[ii] for ii in (4, 3, 0, 1, 2)]
            inputs[1] = _op.transpose(inputs[1], axes=(4, 3, 0, 1, 2))

            attr["data_format"] = "NCDHW"
            attr["strides"] = [attr["strides"][ii] for ii in (0, 4, 1, 2, 3)]
            flip_layout = True

        if attr["data_format"] == "NDHWC":
            kernel_d, kernel_h, kernel_w, _, _ = weights_shape
            attr["kernel_shape"] = (kernel_d, kernel_h, kernel_w)
            if opname == "conv":
                attr["channels"] = weights_shape[4]
            elif opname == "conv_transpose":
                attr["channels"] = weights_shape[3]

            if "dilations" in attr:
                attr["dilations"] = (
                    attr["dilations"][1],
                    attr["dilations"][2],
                    attr["dilations"][3],
                )
            attr["strides"] = (attr["strides"][1], attr["strides"][2], attr["strides"][3])
        elif attr["data_format"] == "NCDHW":
            _, _, kernel_d, kernel_h, kernel_w = weights_shape
            attr["kernel_shape"] = (kernel_d, kernel_h, kernel_w)
            if opname == "conv":
                attr["channels"] = weights_shape[0]
            elif opname == "conv_transpose":
                attr["channels"] = weights_shape[1]

            if "dilations" in attr:
                attr["dilations"] = (
                    attr["dilations"][2],
                    attr["dilations"][3],
                    attr["dilations"][4],
                )
            attr["strides"] = (attr["strides"][2], attr["strides"][3], attr["strides"][4])
        else:
            msg = 'Value {} in attribute "data_format" of operator Conv is ' "not valid."
            raise tvm.error.OpAttributeInvalid(msg.format(attr["data_format"]))

        # Fix padding
        attr["padding"] = attr["padding"].decode("utf-8")

        if attr["padding"] == "VALID":
            attr["padding"] = [0, 0, 0]
        elif attr["padding"] == "SAME":
            stride_d, stride_h, stride_w = attr["strides"]
            kernel_d, kernel_h, kernel_w = attr["kernel_shape"]

            pdata_shape = input_shape
            if opname == "conv_transpose" and len(attr["_output_shapes"]) > 0:
                pdata_shape = attr["_output_shapes"][0]

            if attr["data_format"] == "NDHWC":
                in_d = pdata_shape[1]
                in_h = pdata_shape[2]
                in_w = pdata_shape[3]
            else:
                in_d = pdata_shape[2]
                in_h = pdata_shape[3]
                in_w = pdata_shape[4]

            dilation_d = attr["dilations"][0]
            dilation_h = attr["dilations"][1]
            dilation_w = attr["dilations"][2]
            dilated_kernel_d = (kernel_d - 1) * dilation_d + 1
            dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
            dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
            pad_d = util.get_pad_pair(in_d, dilated_kernel_d, stride_d)
            pad_v = util.get_pad_pair(in_h, dilated_kernel_h, stride_h)
            pad_h = util.get_pad_pair(in_w, dilated_kernel_w, stride_w)

            attr["padding"] = [pad_d[0], pad_v[0], pad_h[0], pad_d[1], pad_v[1], pad_h[1]]
        elif attr["padding"] == "EXPLICIT":
            paddings = attr["explicit_paddings"]
            assert len(paddings) == 10
            if flip_layout or attr["data_format"] == "NDHWC":
                attr["padding"] = [
                    paddings[2],
                    paddings[4],
                    paddings[6],
                    paddings[3],
                    paddings[5],
                    paddings[7],
                ]
            else:
                attr["padding"] = [
                    paddings[4],
                    paddings[6],
                    paddings[8],
                    paddings[5],
                    paddings[7],
                    paddings[9],
                ]
        else:
            msg = 'Value {} in attribute "padding" of operator Conv is not ' "valid."
            raise tvm.error.OpAttributeInvalid(msg.format(attr["padding"]))

        if "kernel_layout" not in attr:
            attr["kernel_layout"] = "DHWIO" if attr["data_format"] == "NDHWC" else "OIDHW"

        use_bias = len(inputs) == (3 if opname != "conv_transpose" else 4)
        channel_axis = 1 if attr["data_format"] == "NCDHW" else 4

        # Ignore the new attributes from TF2.0, for now.
        out = AttrCvt(
            op_name=util.dimension_picker(
                "conv", surfix="_transpose" if opname == "conv_transpose" else ""
            ),
            ignores=["explicit_paddings", "Tshape"],
            transforms={
                "kernel_shape": "kernel_size",
                "data_format": "data_layout",
                "dilations": ("dilation", (0, 0)),
                "group": ("groups", 1),
            },
            custom_check=util.dimension_constraint(),
        )([inputs_data, inputs[1]], attr)

        if use_bias:
            out = _op.nn.bias_add(
                out, inputs[2] if opname != "conv_transpose" else inputs[3], axis=channel_axis
            )

        if flip_layout:
            out = _op.transpose(out, axes=(0, 2, 3, 4, 1))

        return out

    return _impl


def impl_nms(return_scores=False):
    """ NMS """

    def _impl(inputs, attr, params, mod):
        # Get parameter values
        try:
            max_output_size = int(np.atleast_1d(inputs[2].data.asnumpy().astype("int64"))[0])
        except Exception:
            try:
                max_output_size = (
                    _infer_value(inputs[2], params, mod).asnumpy().astype("int64").tolist()[0]
                )
            except Exception:
                max_output_size = inputs[2]
        iou_threshold = np.atleast_1d(inputs[3].data.asnumpy())[0]
        # score_threshold was introduced from V3
        score_threshold = np.atleast_1d(inputs[4].data.asnumpy())[0] if len(inputs) > 4 else 0.0
        pad_output = "pad_to_max_output_size"

        # Generate data with shape (1, num_anchors, 5)
        scores = AttrCvt(
            op_name="expand_dims",
            ignores=["T_threshold", pad_output],
            extras={"axis": -1, "num_newaxis": 1},
        )([inputs[1]], attr)
        data = get_relay_op("concatenate")([scores, inputs[0]], -1)
        data = get_relay_op("expand_dims")(data, 0, 1)

        # reason why using get_valid_counts is for inference performance
        ct, data, indices = get_relay_op("get_valid_counts")(
            data, score_threshold=score_threshold, id_index=-1, score_index=0
        )
        # TensorFlow NMS doesn't have parameter top_k
        top_k = -1
        # TF doesn't have class id for nms input
        score_index = 0
        nms_ret = get_relay_op("non_max_suppression")(
            data=data,
            valid_count=ct,
            indices=indices,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            force_suppress=True,
            top_k=top_k,
            coord_start=1,
            score_index=score_index,
            id_index=-1,
            return_indices=True,
            invalid_to_bottom=False,
        )

        if pad_output in attr and attr[pad_output]:
            return nms_ret
        # squeeze it, TF NMS is not batched
        size = get_relay_op("squeeze")(nms_ret[1], axis=[1])
        data_slice = get_relay_op("squeeze")(nms_ret[0], axis=[0])

        # slice to get the dynamic result
        ret = get_relay_op("strided_slice")(
            data_slice, begin=_expr.const([0]), end=size, slice_mode="size"
        )

        # NonMaxSuppressionV5 returns scores. pad_output is always False for NMSv5.
        if return_scores:
            if "soft_nms_sigma" in attr and attr["soft_nms_sigma"] != 0.0:
                raise tvm.error.OpAttributeUnImplemented(
                    "soft_nms_sigma for NonMaxSuppressionV5 is not supported"
                )
            ret_scores = _op.take(inputs[1], ret, axis=0)
            return _expr.TupleWrapper(_expr.Tuple([ret, ret_scores, size]), 3)

        return ret

    return _impl


def impl_combined_nms():
    """ Combined NMS """

    def _impl(inputs, attr, params, mod):
        # Get parameter values
        boxes = inputs[0]
        scores = inputs[1]
        try:
            max_output_size = int(np.atleast_1d(inputs[2].data.asnumpy().astype("int64"))[0])
        except Exception:
            try:
                max_output_size = (
                    _infer_value(inputs[2], params, mod).asnumpy().astype("int64").tolist()[0]
                )
            except Exception:
                max_output_size = inputs[2]
        max_total_size = inputs[3]
        iou_threshold = np.atleast_1d(inputs[4].data.asnumpy())[0]
        score_threshold = np.atleast_1d(inputs[5].data.asnumpy())[0]
        if attr["pad_per_class"]:
            raise tvm.error.OpAttributeUnImplemented(
                "pad_per_class for CombinedNonMaxSuppression is not supported"
            )
        boxes_shape = _infer_shape(inputs[0], mod)
        scores_shape = _infer_shape(inputs[1], mod)
        batch_size = boxes_shape[0]
        num_anchors = boxes_shape[1]
        q = boxes_shape[2]
        num_classes = scores_shape[2]

        if q != num_classes:
            # When q is 1, it means same box coords are used for all classes.
            boxes = _op.broadcast_to(boxes, (batch_size, num_anchors, num_classes, 4))
        boxes = _op.reshape(boxes, newshape=[batch_size, num_anchors * num_classes, 4])
        scores = _op.reshape(scores, newshape=[batch_size, num_anchors * num_classes, 1])

        # In TF, class is specified by memory layout only.
        ids = _op.arange(_op.const(num_classes, dtype="float32"))
        ids = _op.broadcast_to(ids, (batch_size, num_anchors, num_classes))
        ids = _op.reshape(ids, newshape=[batch_size, num_anchors * num_classes, 1])

        data = _op.concatenate([ids, scores, boxes], -1)
        ct, data, indices = _op.vision.get_valid_counts(
            data, score_threshold=score_threshold, id_index=0, score_index=1
        )
        nms_ret = _op.vision.non_max_suppression(
            data=data,
            valid_count=ct,
            indices=indices,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            force_suppress=False,
            top_k=-1,
            coord_start=2,
            score_index=1,
            id_index=0,
            return_indices=False,
            invalid_to_bottom=True,
        )
        # Dynamic slice to max_total_size
        neg_one = _expr.const([-1])
        slice_end = _op.concatenate(
            [neg_one, _op.expand_dims(max_total_size, axis=0), neg_one], axis=0
        )
        nms_ret = _op.strided_slice(
            nms_ret, begin=[0, 0, 0], end=slice_end, strides=[1, 1, 1], slice_mode="size"
        )

        # Slice output into boxes, scores, classes
        nmsed_boxes = _op.strided_slice(
            nms_ret, begin=[0, 0, 2], end=[-1, -1, 4], slice_mode="size"
        )
        if attr["clip_boxes"]:
            nmsed_boxes = _op.maximum(nmsed_boxes, _expr.const(0, dtype="float32"))
            nmsed_boxes = _op.minimum(nmsed_boxes, _expr.const(1, dtype="float32"))
        nmsed_scores = _op.strided_slice(
            nms_ret, begin=[0, 0, 1], end=[-1, -1, 1], slice_mode="size"
        )
        nmsed_scores = _op.squeeze(nmsed_scores, axis=[2])
        nmsed_classes = _op.strided_slice(
            nms_ret, begin=[0, 0, 0], end=[-1, -1, 1], slice_mode="size"
        )
        nmsed_classes = _op.squeeze(nmsed_classes, axis=[2])
        # Get number of valid boxes
        nms_count = _op.sum(
            _op.cast(_op.greater(nmsed_scores, _expr.const(0, dtype="float32")), "int32"), axis=1
        )

        # TVM uses -1 for invalid outputs while TF uses 0
        box_range = _op.arange(_expr.const(0, dtype="int32"), max_total_size, dtype="int32")
        shape = _op.strided_slice(_op.shape_of(nmsed_boxes), begin=[0], end=[2])
        box_range = _op.broadcast_to(box_range, shape)
        valid_mask = _op.cast(_op.less(box_range, _op.expand_dims(nms_count, axis=1)), "float32")
        nmsed_boxes = nmsed_boxes * _op.expand_dims(valid_mask, axis=2)
        # Could instead use mask for scores, classes if negative values are possible.
        nmsed_scores = _op.maximum(nmsed_scores, _expr.const(0, dtype="float32"))
        nmsed_classes = _op.maximum(nmsed_classes, _expr.const(0, dtype="float32"))

        return _expr.TupleWrapper(
            _expr.Tuple([nmsed_boxes, nmsed_scores, nmsed_classes, nms_count]), 4
        )

    return _impl


def impl_decode_image():
    """ Decode Image """

    def _impl(inputs, attr, params, mod):
        # Image decode wrapper: Expecting user to feed decoded input to next layer drop this layer.
        warnings.warn("DecodeJpeg: It's a pass through, please handle preprocessing before input")
        return inputs[0]

    return _impl


def impl_unravel_index():
    """ Unravel Index """

    def _impl(inputs, attr, params, mod):
        return _op.unravel_index(inputs[0], inputs[1])

    return _impl


def impl_crop_and_resize():
    """ Crop and Resize """

    def _impl(inputs, attr, params, mod):
        # input image is a 4-D tensor of shape [batch, image_height, image_width, depth]
        # boxes is a 2-D tensor of shape [num_boxes, 4], 4 is for [y1, x1, y2, x2]
        try:
            crop_size = util.get_list_param(params, inputs[3])
        except (IndexError, KeyError):
            crop_size = _infer_value(inputs[3], params, mod).asnumpy().tolist()

        method = attr["method"].decode()
        method = "nearest_neighbor" if method == "nearest" else method
        if method not in ["bilinear", "nearest_neighbor"]:
            raise tvm.error.OpAttributeUnImplemented("Method {} is not supported".format(method))
        layout = attr["layout"] if "layout" in attr else "NHWC"
        extrapolation_value = attr["extrapolation_value"]

        return get_relay_op("crop_and_resize")(
            inputs[0], inputs[1], inputs[2], crop_size, layout, method, extrapolation_value
        )

    return _impl


def impl_cast():
    """ Cast """

    def _impl(inputs, attr, params, mod):
        return inputs[0].astype(attr["DstT"].name)

    return _impl


def impl_expand_dims():
    """ Expand Dims """

    def _impl(inputs, attr, params, mod):
        dim_input = inputs.pop(1)
        axis = util.get_num_param(params, dim_input)
        return AttrCvt(
            op_name="expand_dims",
            ignores=["Tdim", "N"],
            extras={"axis": int(axis), "num_newaxis": 1},
        )(inputs, attr)

    return _impl


def impl_expm1():
    """ Expm1 """

    # op description: https://www.tensorflow.org/api_docs/python/tf/math/expm1
    def _impl(inputs, attr, params, mod):
        exp_out = get_relay_op("exp")(inputs[0])
        return exp_out - tvm.relay.const(1.0)

    return _impl


def impl_resize(method):
    """ Resize """

    def _impl(inputs, attr, params, mod):
        if attr["_output_shapes"][0] is not None:
            size = attr["_output_shapes"][0][1:3]
            # Important that the size is defined. If an axis is not, we need to infer what
            # the shape should be.
            if -1 in size:
                size = _infer_value(inputs[1], params, mod).asnumpy().reshape([-1]).tolist()
        else:
            size = _infer_value(inputs[1], params, mod).asnumpy().reshape([-1]).tolist()

        attr["size"] = size
        inputs.pop(1)
        # NHWC
        attr["layout"] = "NHWC"
        if attr.pop("align_corners") is True:
            attr["coordinate_transformation_mode"] = "align_corners"
        else:
            attr["coordinate_transformation_mode"] = "asymmetric"

        # Ignore the new attributes from TF2.0, for now.
        return AttrCvt(
            op_name="resize", ignores=["Tdim", "half_pixel_centers"], extras={"method": method}
        )(inputs, attr)

    return _impl


def impl_check_numerics():
    """ Check Numerics """

    def _impl(inputs, attr, params, mod):
        # Making a copy node assuming no need to verify
        return AttrCvt(op_name="copy", ignores=["message"])(inputs, attr)

    return _impl


def impl_assert():
    """ Assert """

    # ToDo: In general people want asserts to be gone from TensorFlow graphs
    # when they are optimizing them, so converting it to a no-op is
    # reasonable. However, it would be nice to have the option to keep them
    # once Relay gets a Halt or Assert op.
    return impl_no_op()


def impl_no_op():
    """ No Op """

    def _impl(inputs, attr, params, mod):
        # ToDo: This should really be an op that returns nothing, which could
        # be represented as an empty tuple. It turns out that TVM
        # infrastructure doesn't like running functions that return None and
        # also don't like running functions that return an empty tuple. So it
        # doesn't work, but it should be made to work and then this could be
        # improved. In the mean time, it is hard to imagine a case where it
        # matters in any real way that a no-op is converted to a constant 0.
        return tvm.relay.const(0)

    return _impl


def impl_matmul():
    """ MatMul """

    def _impl(inputs, attr, params, mod):
        channels = _infer_channels(inputs[1], not attr["transpose_b"])
        if attr["transpose_a"]:
            inputs[0] = _op.transpose(inputs[0], axes=(1, 0))
        if not attr["transpose_b"]:
            inputs[1] = _op.transpose(inputs[1], axes=(1, 0))
        return AttrCvt(
            op_name="dense", extras={"units": channels}, ignores=["transpose_a", "transpose_b", "T"]
        )(inputs, attr)

    return _impl


def impl_batch_matmul():
    """ Batch MatMul """

    def _impl(inputs, attr, params, mod):
        input_x = inputs[0]
        input_y = inputs[1]
        orig_shape_x = _infer_shape(input_x, mod)
        orig_shape_y = _infer_shape(input_y, mod)

        # reshape n-dimensional batch matmul into 3d
        if len(orig_shape_x) > 3:
            outer_dims = [orig_shape_x[i] for i in range(0, len(orig_shape_x) - 2)]
            num_outer_elts = np.prod(outer_dims)
            new_shape_x = (num_outer_elts, orig_shape_x[-2], orig_shape_x[-1])
            new_shape_y = (num_outer_elts, orig_shape_y[-2], orig_shape_y[-1])
            input_x = _op.reshape(input_x, newshape=new_shape_x)
            input_y = _op.reshape(input_y, newshape=new_shape_y)

        adj_x = attr["adj_x"]
        adj_y = attr["adj_y"]
        input_x = _op.transpose(input_x, axes=[0, 2, 1]) if adj_x else input_x
        input_y = _op.transpose(input_y, axes=[0, 2, 1]) if not adj_y else input_y
        ret = get_relay_op("batch_matmul")(input_x, input_y)

        # reshape result back to n-dimensional
        if len(orig_shape_x) > 3:
            final_shape = list(orig_shape_x)
            final_shape[-2] = orig_shape_x[-1] if adj_x else orig_shape_x[-2]
            final_shape[-1] = orig_shape_y[-2] if adj_y else orig_shape_y[-1]
            ret = _op.reshape(ret, newshape=final_shape)

        return ret

    return _impl


def impl_sparse_tensor_dense_matmul():
    """ Sparse Tensor Dense MatMul """
    def _impl(inputs, attr, params, mod):
        # Loading this by default causes TVM to not be loadable from other languages.
        # Sparse utility from scipy
        from scipy.sparse import csr_matrix

        assert len(inputs) == 4, "There should be 4 input tensors"

        indices_tensor = _infer_value(inputs[0], params, mod).asnumpy()
        values_tensor = _infer_value(inputs[1], params, mod).asnumpy()
        dense_shape_tensor = _infer_value(inputs[2], params, mod).asnumpy()

        data = inputs[3]

        rows = [x[0] for x in indices_tensor]
        cols = [x[1] for x in indices_tensor]

        # Create scipy sparse Tensor(CSR)
        weight_sp = csr_matrix(
            (values_tensor, (rows, cols)), shape=tuple(dense_shape_tensor.tolist())
        )

        # As per tensorflow implementation, we have 4 possible input combination
        # and the first input(A) is always sparse and second input(B) is always dense.
        # Case 1: A , B , adjoint_a=False, adjoint_b=False  --> A * B
        # Case 2: A , B , adjoint_a=True,   adjoint_b=False  --> A.T * B
        # Case 3: A , B , adjoint_a=False, adjoint_b=True    --> A * B.T
        # Case 4: A , B , adjoint_a=True,   adjoint_b=True    --> A.T * B.T
        #
        # Topi implementation for sparse_dense(matmul) has 2 possible input
        # combination where first input(A) is always dense
        # and second input(B) is always sparse.
        # Case 1: A , B, sparse_lhs = False  --> A * B.T
        # Case 2: A , B, sparse_lhs = True    --> B * A.T
        #
        # The mapping would be as below:
        # TF Case 1: A , B , adjoint_a=False, adjoint_b=False
        #           --> In TF: A * B   --> In Topi: A * B.T.T
        #           --> sparse_dense(transpose(B), A, sparse_lhs=True)
        #
        # TF Case 2: A , B , adjoint_a=True, adjoint_b=False
        #           --> In TF: A.T * B   --> In Topi: A.T * B.T.T
        #           --> sparse_dense(transpose(B), transpose(A), sparse_lhs=True)
        #
        # TF Case 3: A , B , adjoint_a=False, adjoint_b=True
        #           --> In TF: A * B.T   --> In Topi: A * B
        #           --> sparse_dense(B, A, sparse_lhs=True)
        #
        # TF Case 4: A , B , adjoint_a=True, adjoint_b=True
        #           --> In TF: A.T * B.T   --> In Topi: (B * A.T).T
        #           --> transpose(sparse_dense(B, transpose(A), sparse_lhs=False))

        # By default, in tensorflow the first input ,i.e., data is sparse
        sparse_lhs = True

        # TF Case 1:
        if not attr.get("adjoint_a") and not attr.get("adjoint_b"):
            data = _op.transpose(data)
        # TF Case 2:
        elif attr.get("adjoint_a") and not attr.get("adjoint_b"):
            data = _op.transpose(data)
            weight_sp = csr_matrix(weight_sp.transpose())
        # TF Case 3:
        elif not attr.get("adjoint_a") and attr.get("adjoint_b"):
            pass
        # TF Case 4:
        # attr.get("adjoint_a") and attr.get("adjoint_b"):
        else:
            sparse_lhs = False
            weight_sp = csr_matrix(weight_sp.transpose())

        weight_data = _expr.const(weight_sp.data, weight_sp.data.dtype)
        weight_indptrs = _expr.const(weight_sp.indptr, weight_sp.indptr.dtype)
        weight_indices = _expr.const(weight_sp.indices, weight_sp.indices.dtype)

        ret = _op.nn.sparse_dense(data, [weight_data, weight_indices, weight_indptrs], sparse_lhs)

        if not sparse_lhs:
            # TF Case 4
            ret = _op.transpose(ret)

        return ret

    return _impl


def impl_sparse_fill_empty_rows():
    """ Sparse Fill Empty Rows """

    def _impl(inputs, attr, params, mod):
        assert len(inputs) == 4, "There should be 4 input tensors"
        sparse_indices = inputs[0]
        sparse_values = inputs[1]
        sparse_indices_num_cols = _infer_shape(sparse_indices, mod)[1]
        first_column = _op.split(sparse_indices, sparse_indices_num_cols, axis=1)[0]
        sorted_indices = _op.argsort(_op.squeeze(first_column))
        sorted_sparse_indices = _op.take(sparse_indices, sorted_indices, axis=0)
        sorted_sparse_values = _op.take(sparse_values, sorted_indices, axis=0)
        new_sparse_indices, new_sparse_values, empty_row_indicator = _op.sparse_fill_empty_rows(
            sorted_sparse_indices, sorted_sparse_values, inputs[2], inputs[3]
        )

        return _expr.TupleWrapper(
            _expr.Tuple([new_sparse_indices, new_sparse_values, empty_row_indicator]),
            3,
        )

    return _impl


def impl_sparse_reshape():
    """ Sparse Reshape """

    def _impl(inputs, attr, params, mod):
        assert len(inputs) == 3, "There should be 3 input tensors"
        new_indices, new_shape = get_relay_op("sparse_reshape")(inputs[0], inputs[1], inputs[2])
        return _expr.TupleWrapper(_expr.Tuple([new_indices, new_shape]), 2)

    return _impl


def impl_identity():
    """ Identity """

    def _impl(inputs, attr, params, mod):
        return inputs[0]

    return _impl


def impl_identityn():
    """ Identityn """

    def _impl(inputs, attr, params, mod):
        return inputs

    return _impl


def impl_concatV2():
    """ ConcatV2 """

    def _impl(inputs, attr, params, mod):
        pop_node = inputs.pop(len(inputs) - 1)
        axis = int(util.get_num_param(params, pop_node))
        return AttrCvt(op_name="concatenate", ignores=["T", "N", "Tidx"], extras={"axis": axis})(
            [inputs], attr
        )

    return _impl


def impl_concat():
    """ Concat """

    def _impl(inputs, attr, params, mod):
        pop_node = inputs.pop(0)
        axis = int(util.get_num_param(params, pop_node))
        return AttrCvt(op_name="concatenate", ignores=["N"], extras={"axis": axis})([inputs], attr)

    return _impl


def impl_pack():
    """ Pack """

    def _impl(inputs, attr, params, mod):
        axis = int(attr["axis"])
        inputs_reshaped = [_op.expand_dims(i, axis=axis, num_newaxis=1) for i in inputs]
        return _op.concatenate(inputs_reshaped, axis)

    return _impl


def impl_tile():
    """ Tile """

    def _impl(inputs, attr, params, mod):
        reps_input = inputs.pop()
        if isinstance(reps_input, _expr.Call):
            np_reps = _infer_value(reps_input, params, mod).asnumpy()
            reps = [np_reps.flatten()[i] for i in range(np_reps.flatten().shape[0])]
        else:
            reps = util.get_list_param(params, reps_input)
        new_input = [inputs.pop(0)]

        return AttrCvt(op_name="tile", extras={"reps": tuple(reps)}, ignores=["Tmultiples"])(
            new_input, attr
        )

    return _impl


def impl_slice():
    """ Slice """

    def _impl(inputs, attr, params, mod):
        try:
            begin = util.get_list_param(params, inputs[1])
        except (IndexError, KeyError, AttributeError):
            # Handle symbolic begin
            try:
                begin = _infer_value(inputs[1], params, mod).asnumpy().tolist()
            except Exception:
                begin = inputs[1]
        try:
            size = util.get_list_param(params, inputs[2])
        except (IndexError, KeyError, AttributeError):
            # Handle symbolic size
            try:
                size = _infer_value(inputs[2], params, mod).asnumpy().tolist()
            except Exception:
                size = inputs[2]

        # Align begin and strides for dynamic shape.
        data_dim = len(_infer_shape(inputs[0], mod))
        strides = [1] * data_dim
        if not isinstance(begin, (_expr.Call, _expr.Var)):
            for _ in range(len(begin), data_dim):
                begin.append(0)
        elif not isinstance(size, (_expr.Call, _expr.Var)):
            for _ in range(len(size), data_dim):
                size.append(-1)
        return _op.strided_slice(
            inputs[0], begin=begin, end=size, strides=strides, slice_mode="size"
        )

    return _impl


def impl_reshape():
    """ Reshape """

    def _impl(inputs, attr, params, mod):
        pop_node = inputs.pop(1)

        try:
            shape_arg = util.get_tuple_param(params, pop_node)
        except AttributeError:
            # Shape operator is already pruned, hence
            # try to infer shape by precompute prune if possible.
            try:
                params_new = _infer_value(pop_node, params, mod)
                shape_arg = tuple(params_new.asnumpy().astype("int32").flatten())
            except Exception:
                # Deal with symbolic shape case.
                if isinstance(pop_node, _expr.Call) and "shape_of" in str(pop_node.op):
                    # shape_of is the direct ancestor.
                    return _op.reshape_like(inputs[0], pop_node.args[0])
                shape_arg = pop_node

        return AttrCvt(op_name="reshape", extras={"newshape": shape_arg}, ignores=["Tshape"])(
            inputs, attr
        )

    return _impl


def impl_depth_to_space():
    """ Depth To Space """

    def _impl(inputs, attr, params, mod):
        block_size = int(attr["block_size"])
        layout = attr["data_format"].decode("utf-8")
        return _op.nn.depth_to_space(inputs[0], block_size, layout)

    return _impl


def impl_space_to_depth():
    """ Space To Depth """

    def _impl(inputs, attr, params, mod):
        block_size = int(attr["block_size"])
        layout = attr["data_format"].decode("utf-8")
        return _op.nn.space_to_depth(inputs[0], block_size, layout)

    return _impl


def impl_sparse_to_dense():
    """ Sparse To Depth """

    def _impl(inputs, attr, params, mod):
        sparse_indices = inputs[0]
        output_shape = inputs[1]
        sparse_values = inputs[2]
        default_value = inputs[3]

        return _op.sparse_to_dense(sparse_indices, output_shape, sparse_values, default_value)

    return _impl


def impl_bias_add():
    """ Bias Add """

    def _impl(inputs, attr, params, mod):
        # Must expand for proper broadcasting in NCHW.
        if "data_format" in attr and attr["data_format"].decode("utf-8") == "NCHW":
            bias = _op.reshape(inputs[1], newshape=(1, -1, 1, 1))
        else:
            bias = inputs[1]
        return _op.add(inputs[0], bias)

    return _impl


def impl_broadcast_to():
    """ Broadcast To """

    def _impl(inputs, attr, params, mod):
        if isinstance(inputs[1], _expr.Var):
            shape = params[inputs[1].name_hint]
        else:
            shape = _infer_value(inputs[1], params, mod)
        shape = list(shape.asnumpy().reshape([-1]))
        return _op.broadcast_to(inputs[0], shape)

    return _impl


def impl_squeeze():
    """ Squeeze """

    def _impl(inputs, attr, params, mod):
        if len(attr["squeeze_dims"]) == 0:
            attr["squeeze_dims"] = None
        return AttrCvt(
            op_name="squeeze", transforms={"squeeze_dims": "axis"}, ignores=["T", "_cloned"]
        )(inputs, attr)

    return _impl


def impl_fused_batch_norm():
    """ Fused Batch Norm """

    def _impl(inputs, attr, params, mod):
        # Tensorflow: (data, gamma, beta, moving_mean, moving_variance)
        # Relay:       (data, gamma, beta, moving_mean, moving_varience)
        assert len(inputs) == 5
        axis = 3
        need_cast = False

        if "data_format" in attr:
            attr["data_format"] = attr["data_format"].decode("utf-8")
            if attr["data_format"] == "NCHW":
                axis = 1
        if "U" in attr and attr["U"].name != attr["T"].name:
            need_cast = True
            inputs[0] = _op.cast(inputs[0], dtype=attr["U"].name)
        # Check if mean and variance are empty
        # If so, replace them with Mean and Variance Ops
        # For run-time calculation
        moving_mean_shape = [int(n) for n in inputs[3].type_annotation.shape]
        moving_variance_shape = [int(n) for n in inputs[4].type_annotation.shape]
        if moving_mean_shape[0] == 0 and moving_variance_shape[0] == 0:
            inputs[3] = _op.mean(inputs[0], axis=axis, keepdims=False, exclude=True)
            inputs[4] = _op.variance(inputs[0], axis=axis, keepdims=False, exclude=True)
        out = AttrCvt(
            op_name="batch_norm",
            transforms={"scale_after_normalization": "scale", "variance_epsilon": "epsilon"},
            extras={"axis": axis},
            ignores=["data_format", "U", "exponential_avg_factor"],
            disables=["momentum"],
        )(inputs, attr)

        if need_cast:
            out = _expr.TupleGetItem(out.astuple(), 0)
            out = _op.cast(out, dtype=attr["T"].name)
        return out

    return _impl


def impl_batch_norm():
    """ Batch Norm """

    def _impl(inputs, attr, params, mod):
        # Rearrange inputs from
        # (data, moving_mean, moving_variance, beta, gamma)
        #     to
        # (data, gamma, beta, moving_mean, moving_var)
        new_inputs = [inputs[0], inputs[4], inputs[3], inputs[1], inputs[2]]

        axis = 3
        if "data_format" in attr:
            attr["data_format"] = attr["data_format"].decode("utf-8")
            if attr["data_format"] == "NCHW":
                axis = 1

        return AttrCvt(
            op_name="batch_norm",
            transforms={"scale_after_normalization": "scale", "variance_epsilon": "epsilon"},
            extras={"axis": axis},
            ignores=["data_format", "exponential_avg_factor"],
            disables=["momentum"],
        )(new_inputs, attr)

    return _impl


def impl_relu6():
    """ Relu6 """

    def _impl(inputs, attr, params, mod):
        return _op.clip(inputs[0], a_min=0, a_max=6)

    return _impl


def impl_shape():
    """ Shape """

    def _impl(inputs, attr, params, mod):
        is_symbolic_shape = False
        input_shape = _infer_shape(inputs[0], mod)
        for axis in input_shape:
            if not isinstance(axis, (int, tvm.tir.IntImm)):
                is_symbolic_shape = True
                break

        if is_symbolic_shape:
            ret = _op.shape_of(inputs[0], dtype=attr["out_type"].name)
        else:
            ret = np.array(input_shape, dtype=attr["out_type"].name)
        return ret

    return _impl


def impl_fill():
    """ Fill """

    def _impl(inputs, attr, params, mod):
        try:
            output_shape = _infer_value(inputs[0], params, mod).asnumpy().tolist()
        except Exception:
            output_shape = inputs[0]

        return _op.full(inputs[1], output_shape, attr["T"].name)

    return _impl


def impl_lrn():
    """ LRN """

    def _impl(inputs, attr, params, mod):
        attr_new = {}
        depth_radius = attr.get("depth_radius", 5)
        size = (depth_radius * 2) + 1
        attr_new["axis"] = 3  # Fix axis, NHWC format
        attr_new["size"] = size
        attr_new["bias"] = attr.get("bias", 1)
        attr_new["alpha"] = attr.get("alpha", 1) * size
        attr_new["beta"] = attr.get("beta", 0.5)
        return AttrCvt(op_name="lrn")(inputs, attr_new)

    return _impl


def impl_sum():
    """ SUM """

    def _impl(inputs, attr, params, mod):
        axis = util.get_tuple_param(params, inputs[1])
        return AttrCvt(
            op_name="sum",
            extras={"axis": axis},
            transforms={"keep_dims": "keepdims"},
            ignores=["name", "Tidx"],
        )([inputs[0]], attr)

    return _impl


def impl_reduce(op):
    """ Reduce """

    def _impl(inputs, attr, params, mod):
        axis = util.get_list_param(params, inputs[1])
        axis = tuple(axis)
        if not axis:
            axis = None
        return AttrCvt(
            op_name=op,
            extras={"axis": axis},
            transforms={"keep_dims": "keepdims"},
            ignores=["name", "Tidx"],
        )([inputs[0]], attr)

    return _impl


def impl_euclidean_norm():
    """ Euclidean Norm """

    def _impl(inputs, attr, params, mod):
        axis = tuple(util.get_list_param(params, inputs[1]))
        keep_dims = bool(attr.get("keep_dims", False))
        return _op.sqrt(
            _op.cast(_op.reduce.sum(_op.multiply(inputs[0], inputs[0]), axis, keep_dims), "float32")
        )

    return _impl


def impl_square():
    """ Square """

    def _impl(inputs, attr, params, mod):
        return _op.multiply(inputs[0], inputs[0])

    return _impl


def impl_gather():
    "GatherV2, Gather"

    def _impl(inputs, attr, params, mod):
        if len(inputs) > 2:
            axis = util.get_num_param(params, inputs.pop(2))
        else:
            axis = 0
        if int(attr.get("batch_dims", 0)) != 0:
            raise tvm.error.OpAttributeUnImplemented("Attribute batch_dims is not supported")
        new_input = inputs[0:2]
        return AttrCvt(
            op_name="take",
            extras={"axis": tvm.tir.const(axis, "int32")},
            ignores=["Tindices", "Tparams", "validate_indices", "Taxis", "_class", "batch_dims"],
        )(new_input, attr)

    return _impl


def impl_gather_nd():
    """GatherNd"""

    def _impl(inputs, attr, params, mod):
        indices_dims = len(_infer_shape(inputs[1], mod))
        indices = _op.transpose(inputs[1], axes=[-1] + list(range(indices_dims - 1)))
        return AttrCvt(op_name="gather_nd", ignores=["Tindices", "Tparams", "Taxis", "_class"])(
            [inputs[0], indices], attr
        )

    return _impl


def impl_stridedSlice():
    """ Strided Slice """

    def _impl(inputs, attr, params, mod):
        """Strided Slice.
        Operator description: https://www.tensorflow.org/api_docs/python/tf/strided_slice
        Tensorflow mask validation: https://github.com/tensorflow/tensorflow/blob/master/
        tensorflow/core/util/strided_slice_op.cc#L147-L368
        """
        begin = util.get_list_param(params, inputs[1])
        end = util.get_list_param(params, inputs[2])
        stride = util.get_list_param(params, inputs[3])

        begin_mask = int(attr.get("begin_mask", 0))
        end_mask = int(attr.get("end_mask", 0))
        ellipsis_mask = int(attr.get("ellipsis_mask", 0))
        new_axis_mask = int(attr.get("new_axis_mask", 0))
        shrink_axis_mask = int(attr.get("shrink_axis_mask", 0))
        in_type = _infer_type(inputs[0], mod)
        data_shape = get_const_tuple(in_type.checked_type.shape)
        data_dim = len(data_shape)
        stride_dim = len(stride)
        if data_dim == 0 and isinstance(inputs[0], _expr.Constant):
            new_data = inputs[0].data.asnumpy().reshape(1)
            return _expr.const(new_data, inputs[0].data.dtype)

        # This is a special routine to handle strided_slice after shape_of.
        # We need this since in some cases we want to do strided_slice on
        # a partial symbolic shape, such as (1, ?), and get a static shape
        # (1,). Directly slice on shape_of will result in fully dynamic shape.
        # TODO(kevinthesun): Can we generalize this process with partial eval?
        if isinstance(inputs[0], _expr.Call) and inputs[0].op == _op.get("shape_of"):
            bg = begin[0]
            ed = end[0]
            st = stride[0]

            if ed <= 0 < st:
                ed += data_shape[0]

            in_shape = _infer_shape(inputs[0].args[0], mod)
            dtype = in_type.checked_type.dtype
            out_data = []
            idx = bg
            while idx < ed:
                if isinstance(in_shape[idx], int):
                    out_data.append(in_shape[idx])
                else:
                    break
                idx += st

            # Only return when in_shape is fully static in the range from begin to end.
            if idx >= ed:
                ret = _expr.const(out_data, dtype)
                if shrink_axis_mask:
                    ret = _op.squeeze(ret)

                return ret

        def _transform_mask(stride_dim, ellipsis_mask):
            """Handle mask inputs to create new begin, end, stride and output shape"""
            m_begin = [0] * data_dim
            m_end = [0] * data_dim
            m_stride = [0] * data_dim
            fshape_indices = []
            # Count new axis after ellipsis_mask, consider while applying ellipsis_mask.
            ellipsis_seen = False
            new_axes_after_ellipsis = 0
            for i in range(stride_dim):
                mask = 1 << i
                if ellipsis_seen and (mask & new_axis_mask) != 0:
                    new_axes_after_ellipsis += 1
                if (mask & ellipsis_mask) != 0:
                    ellipsis_seen = True
            if not ellipsis_seen:
                # Used later for extending the stride attributes in the below loop.
                ellipsis_mask |= 1 << stride_dim
                stride_dim += 1
            final_index = 0
            for index in range(stride_dim):
                mask = 1 << index
                if mask & ellipsis_mask:
                    # Identify the end index for applying ellipsis_mask
                    to_index = min(
                        ((data_dim - (stride_dim - index)) + 1 + new_axes_after_ellipsis), data_dim
                    )
                    for i in range(final_index, to_index):
                        m_begin[final_index] = 0
                        m_end[final_index] = data_shape[final_index]
                        m_stride[final_index] = 1
                        fshape_indices.append(final_index)
                        final_index += 1
                elif mask & new_axis_mask:
                    fshape_indices.append(-1)
                elif not mask & new_axis_mask:
                    if final_index == len(m_begin):
                        break
                    if mask & begin_mask:
                        m_begin[final_index] = -1 if stride[index] < 0 else 0
                    elif begin[index]:
                        m_begin[final_index] = begin[index]
                    if mask & end_mask:
                        m_end[final_index] = (
                            -(data_shape[final_index] + 1)
                            if stride[index] < 0
                            else data_shape[final_index]
                        )
                    elif end[index]:
                        m_end[final_index] = end[index]
                    m_stride[final_index] = stride[index]
                    if mask & shrink_axis_mask:
                        # Tensorflow make axis with shrink_axis_mask as dimension 1
                        m_begin[final_index] = (
                            data_shape[final_index] + begin[index]
                            if begin[index] < 0
                            else begin[index]
                        )
                        m_end[final_index] = begin[index] + 1
                        m_stride[final_index] = 1
                        fshape_indices.append(-2)
                    else:
                        fshape_indices.append(final_index)

                    final_index += 1
            return m_begin, m_end, m_stride, fshape_indices

        fshape_indices = None
        if begin_mask or end_mask or ellipsis_mask or new_axis_mask or shrink_axis_mask:
            begin, end, stride, fshape_indices = _transform_mask(stride_dim, ellipsis_mask)
        out = _op.strided_slice(inputs[0], begin=begin, end=end, strides=stride)
        out_shape = _infer_shape(out, mod=mod)
        if not fshape_indices:
            fshape_indices = range(len(out_shape))

        # Create final output shape.
        final_output = []
        for gather_index in fshape_indices:
            if gather_index == -1:
                final_output.append(1)
            elif gather_index == -2:
                pass
            else:
                final_output.append(out_shape[gather_index])

        if not final_output:
            if not shrink_axis_mask:
                ret = out
            else:
                final_shape = []
                for dim in out_shape:
                    if dim != 1:
                        final_shape.append(dim)
                if len(final_shape) == 0:
                    ret = _op.squeeze(out)
                else:
                    # We need reshape to handle dynamic shape.
                    ret = _op.reshape(out, newshape=tuple(final_shape))
        else:
            ret = _op.reshape(out, newshape=tuple(final_output))
        return ret

    return _impl


def impl_pad(name):
    """ Pad """

    def _impl(inputs, attr, params, mod):
        try:
            padlist = util.get_param(params, inputs[1])
        except (IndexError, KeyError, AttributeError):
            try:
                padlist = _infer_value(inputs[1], params, mod).asnumpy().tolist()
            except Exception:
                padlist = inputs[1]

        if isinstance(padlist, _expr.Expr):
            paddings = padlist
        else:
            paddings = tuple(tuple(l) for l in padlist)
        attr["pad_width"] = paddings
        attr["pad_value"] = 0
        new_inputs = [inputs[0]]
        if name == "PadV2":
            try:
                attr["pad_value"] = util.get_num_param(params, inputs[2])
            except (IndexError, KeyError, AttributeError):
                attr["pad_value"] = inputs[2]
        return AttrCvt(
            op_name="pad",
            ignores=["Tpaddings"],
        )(new_inputs, attr)

    return _impl


def impl_mirror_pad():
    """ Mirror Pad """

    def _impl(inputs, attr, params, mod):
        padlist = util.get_param(params, inputs[1])
        paddings = tuple(tuple(l) for l in padlist)
        attr["pad_width"] = paddings
        mode = attr["mode"].decode("utf-8")
        attr["mode"] = mode
        new_inputs = [inputs[0]]
        return AttrCvt(
            op_name="mirror_pad",
            ignores=["Tpaddings"],
        )(new_inputs, attr)

    return _impl


def impl_transpose():
    """ Transpose """

    def _impl(inputs, attr, params, mod):
        # If perm is not specified, axes is left empty,
        # otherwise its value is get from params
        try:
            axes = util.get_list_param(params, inputs[1])
        except (IndexError, KeyError, AttributeError):
            axes = _infer_value(inputs[1], params, mod).asnumpy().tolist()
        return _op.transpose(inputs[0], axes=axes)

    return _impl


def impl_where():
    """ Where """

    def _impl(inputs, attr, params, mod):
        if len(inputs) == 1:
            return AttrCvt(op_name="argwhere")(inputs, attr)
        return AttrCvt(op_name="where")(inputs, attr)

    return _impl


def impl_clip_by_value():
    """ Clip By Value """

    def _impl(inputs, attr, params, mod):
        a_min = util.get_num_param(params, inputs[1])
        a_max = util.get_num_param(params, inputs[2])
        return _op.clip(inputs[0], a_min=a_min, a_max=a_max)

    return _impl


def impl_reverse_v2():
    """ Reverse V2 """

    def _impl(inputs, attr, params, mod):
        axis = util.get_num_param(params, inputs[1])
        return AttrCvt(op_name="reverse", ignores=["Tidx"], extras={"axis": int(axis)})(
            [inputs[0]], attr
        )

    return _impl


def impl_rank():
    """ Rank """

    def _impl(inputs, attr, params, mod):
        input_shape = _infer_shape(inputs[0], mod)

        name = attr["_node_name"]
        params[name] = tvm.nd.array(np.array([len(input_shape)]).astype("int32"))
        return [_expr.var(name, shape=params[name].shape, dtype="int32")]

    return _impl


def impl_range():
    """ Range """

    def _impl(inputs, attr, params, mod):
        try:
            start = util.get_param(params, inputs[0])[0]
        except (IndexError, KeyError, AttributeError):
            try:
                start = _infer_value(inputs[1], params, mod).asnumpy().tolist()
                start = start if not isinstance(start, list) else start[0]
            except Exception:
                # Symbolic start
                start = inputs[0]

        try:
            limit = (
                util.get_param(params, inputs[1])[0]
                if hasattr(inputs[1], "name_hint") or isinstance(inputs[1], _expr.Constant)
                else params.pop("Rank").asnumpy()[0]
            )
        except (IndexError, KeyError, AttributeError):
            try:
                limit = _infer_value(inputs[1], params, mod).asnumpy().tolist()
                limit = limit if not isinstance(limit, list) else limit[0]
            except Exception:
                limit = inputs[1]

        try:
            delta = util.get_param(params, inputs[2])[0]
        except (IndexError, KeyError, AttributeError):
            try:
                delta = _infer_value(inputs[2], params, mod).asnumpy().tolist()
                delta = delta if not isinstance(delta, list) else delta[0]
            except Exception:
                # Symbolic delta
                delta = inputs[2]

        # if all attributes are constant, evalute the range function and return relay.const
        if all(
            [
                isinstance(start, (np.int32, np.int64, int, np.float32, np.float64, float)),
                isinstance(limit, (np.int32, np.int64, int, np.float32, np.float64, float)),
                isinstance(delta, (np.int32, np.int64, int, np.float32, np.float64, float)),
            ]
        ):
            return tvm.relay.const(list(range(int(start), int(limit), int(delta))))

        dtype = attr["Tidx"].name if "Tidx" in attr else str(start.dtype)
        if isinstance(start, (np.int32, np.int64, int, np.float32, np.float64, float)):
            start = _expr.const(start, dtype=dtype)
        if isinstance(limit, (np.int32, np.int64, int, np.float32, np.float64, float)):
            limit = _expr.const(limit, dtype=dtype)
        if isinstance(delta, (np.int32, np.int64, int, np.float32, np.float64, float)):
            delta = _expr.const(delta, dtype=dtype)

        return AttrCvt(
            op_name="arange",
            ignores=["Tidx", "_class"],
            extras={"start": start, "stop": limit, "step": delta, "dtype": dtype},
        )([], attr)

    return _impl


def impl_elu():
    """ Elu """

    def _impl(inputs, attr, params, mod):
        dtype = attr["T"].name
        alpha = tvm.relay.const(-1.0, dtype)
        return alpha * _op.nn.relu(tvm.relay.const(1, dtype) - _op.exp(inputs[0])) + _op.nn.relu(
            inputs[0]
        )

    return _impl


def impl_selu():
    """ SElu """

    def _impl(inputs, attr, params, mod):
        dtype = attr["T"].name
        alpha = tvm.relay.const(-1.6732632423543772848170429916717, dtype)
        gamma = tvm.relay.const(1.0507009873554804934193349852946, dtype)
        return gamma * (
            alpha * _op.nn.relu(tvm.relay.const(1, dtype) - _op.exp(inputs[0]))
            + _op.nn.relu(inputs[0])
        )

    return _impl


def impl_mean():
    """ Mean """

    def _impl(inputs, attr, params, mod):
        axis = util.get_tuple_param(params, inputs[1])
        return AttrCvt(
            op_name="mean",
            ignores=["Tdim", "Tidx"],
            transforms={"keep_dims": "keepdims"},
            extras={"axis": axis},
        )([inputs[0]], attr)

    return _impl


def impl_broadcast(name):
    """ Broadcast """

    def _impl(inputs, attr, params, mod):
        return AttrCvt(op_name=name, ignores=["name", "incompatible_shape_error", "Tidx"])(
            inputs, attr
        )

    return _impl


def impl_split(has_size_vector):
    """ Split """

    # TF documentation https://www.tensorflow.org/api_docs/python/tf/split
    def _impl(inputs, attr, params, mod):
        try:
            # order and number of inputs are different:
            # if has_size_vector:
            #     https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/split-v
            # else:
            #     https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/split

            # in addition, `axis` and `num_or_size_splits` can be tensors in TensorFlow,
            # we can only support constants
            if has_size_vector:
                input_node_index = 0
                input_axis_index = 2
                size_splits = util.get_param(params, inputs[1])
                section_beginnings = np.cumsum(size_splits)[:-1]
                indices_or_sections = tuple(section_beginnings)
            else:
                input_node_index = 1
                input_axis_index = 0
                indices_or_sections = attr["num_split"]
            input_node = inputs[input_node_index]
            axis_input_value = util.get_num_param(params, inputs[input_axis_index])
        except (IndexError, KeyError, AttributeError):
            raise TypeError(
                "Unsupported argument for split: `axis` and `num_or_size_splits` "
                "should be constants"
            )
        return _op.split(
            input_node, indices_or_sections=indices_or_sections, axis=int(axis_input_value)
        )

    return _impl


def impl_unpack():
    """ UnPack """

    def _impl(inputs, attr, params, mod):
        input_node = inputs[0]
        axis = attr["axis"]
        input_shape = _infer_shape(input_node, mod)
        axis_length = input_shape[axis]
        if axis_length < 0:
            raise TypeError("Unstack with unknown axis length")
        splitted = _op.split(input_node, indices_or_sections=axis_length, axis=axis)
        axis = [axis]
        return _expr.TupleWrapper(
            _expr.Tuple([_op.squeeze(split_item, axis=axis) for split_item in splitted]),
            len(splitted),
        )

    return _impl


def impl_softmax():
    """ Softmax """

    def _impl(inputs, attr, params, mod):
        return AttrCvt(op_name="softmax", transforms={"axis": ("axis", 1)})([inputs[0]], attr)

    return _impl


def impl_softsign():
    """ Softsign """

    # op description: https://www.tensorflow.org/api_docs/python/tf/math/softsign
    def _impl(inputs, attr, params, mod):
        abs_out = get_relay_op("abs")(inputs[0])
        add_out = abs_out + tvm.relay.const(1, attr["T"].name)
        return inputs[0] / add_out

    return _impl


def impl_softplus():
    """ Softplus """

    # op description: https://www.tensorflow.org/api_docs/python/tf/math/softplus
    def _impl(inputs, attr, params, mod):
        exp_out = AttrCvt("exp")(inputs, attr)
        inputs.append(tvm.relay.const(1, attr["T"].name))
        rh = tvm.relay.const(1, attr["T"].name)
        add_out = get_relay_op("add")(exp_out, rh)
        return get_relay_op("log")(add_out)

    return _impl


def impl_topk():
    """ TopK """

    def _impl(inputs, attr, params, mod):
        k_input = inputs.pop(1)
        try:
            k = int(util.get_num_param(params, k_input))
        except (IndexError, KeyError, AttributeError):
            try:
                k = int(_infer_value(k_input, params, mod).asnumpy().tolist())
            except Exception:
                k = k_input
        if isinstance(k, int):
            if k < 1:
                raise tvm.error.OpAttributeInvalid(
                    "Attribute k must be positive in operator TopKV2"
                )
            k = _expr.const(k)
        if attr["sorted"] is False:
            raise tvm.error.OpAttributeUnImplemented(
                "Attribute sorted=False is not supported in operator TopKV2"
            )
        return AttrCvt(
            op_name="topk",
            ignores=["sorted"],
            extras={"k": k, "is_ascend": False, "dtype": "int32"},
        )([inputs[0]], attr)

    return _impl


def impl_floordiv():
    """ Floor Div """

    def _impl(inputs, attr, params, mod):
        assert len(inputs) == 2
        return AttrCvt("floor_divide")(inputs, attr)

    return _impl


def impl_floormod():
    """ Floor Mod """

    def _impl(inputs, attr, params, mod):
        assert len(inputs) == 2
        return AttrCvt("floor_mod")(inputs, attr)

    return _impl


def impl_logical(name):
    """ Logical """

    def _impl(inputs, attr, params, mod):
        return AttrCvt(op_name=name)(inputs, attr)

    return _impl


def impl_space_to_batch_nd():
    """ Space To Batch ND """

    def _impl(inputs, attr, params, mod):
        try:
            block_shape = util.get_list_param(params, inputs[1])
        except (IndexError, KeyError, AttributeError):
            block_shape = _infer_value(inputs[1], params, mod).asnumpy().tolist()

        try:
            paddings = util.get_list_param(params, inputs[2])
        except (IndexError, KeyError, AttributeError):
            paddings = _infer_value(inputs[2], params, mod).asnumpy()
            paddings = np.squeeze(paddings)
            if len(paddings.shape) == 1:
                paddings = np.expand_dims(paddings, axis=0)
            paddings = paddings.tolist()

        attr["block_shape"] = block_shape
        attr["paddings"] = paddings
        out = AttrCvt("space_to_batch_nd", ignores=["Tblock_shape", "Tpaddings"])([inputs[0]], attr)

        return out

    return _impl


def impl_batch_to_space_nd():
    """ Batch To Space ND """

    def _impl(inputs, attr, params, mod):
        try:
            block_shape = util.get_list_param(params, inputs[1])
        except (IndexError, KeyError, AttributeError):
            block_shape = _infer_value(inputs[1], params, mod).asnumpy().tolist()

        try:
            crops = util.get_list_param(params, inputs[2])
        except (IndexError, KeyError, AttributeError):
            crops = _infer_value(inputs[2], params, mod).asnumpy()
            crops = np.squeeze(crops)
            if len(crops.shape) == 1:
                crops = np.expand_dims(crops, axis=0)
            crops = crops.tolist()

        attr["block_shape"] = block_shape
        attr["crops"] = crops
        out = AttrCvt("batch_to_space_nd", ignores=["Tblock_shape", "Tcrops"])([inputs[0]], attr)

        return out

    return _impl


def impl_atan2():
    """ ATan2 """

    def _impl(inputs, attr, params, mod):
        divide = impl_elemwise("divide")(inputs, attr, params, mod)
        return get_relay_op("atan")(divide)

    return _impl


def impl_prod():
    """ Prod """

    def _impl(inputs, attr, params, mod):
        axis = util.get_num_param(params, inputs[1])
        keepdims = attr["keep_dims"]
        return _op.prod(inputs[0], int(axis), keepdims=keepdims)

    return _impl


def impl_log1p():
    """ Log1p """

    # op description: https://www.tensorflow.org/api_docs/python/tf/math/log1p
    def _impl(inputs, attr, params, mod):
        one = tvm.relay.const(1, attr["T"].name)
        add_out = get_relay_op("add")(inputs[0], one)
        return get_relay_op("log")(add_out)

    return _impl


def impl_one_hot():
    """ One Hot """

    def _impl(inputs, attr, params, mod):
        depth = int(util.get_num_param(params, inputs[1]))
        dtype = attr["T"].name

        on_value = util.get_num_param(params, inputs[2])
        off_value = util.get_num_param(params, inputs[3])
        new_inputs = [
            inputs[0],
            tvm.relay.const(on_value, dtype),
            tvm.relay.const(off_value, dtype),
        ]
        return AttrCvt("one_hot", ignores=["TI"], extras={"depth": depth, "dtype": dtype})(
            new_inputs, attr
        )

    return _impl


def impl_squared_difference():
    """ Squared Difference """

    def _impl(inputs, attr, params, mod):
        difference = _op.subtract(inputs[0], inputs[1])
        return _op.multiply(difference, difference)

    return _impl


def impl_size():
    """ Size """

    def _impl(inputs, attr, params, mod):
        new_attr = attr
        new_attr["out_type"] = attr["out_type"].name
        return AttrCvt("ndarray_size", transforms={"out_type": "dtype"})(inputs, new_attr)

    return _impl


def impl_add_n():
    """ AddN """

    def _impl(inputs, attr, params, mod):
        if not isinstance(inputs, tuple):
            inputs = list(inputs)
        assert len(inputs) > 0, "add_n take >=1 inputs, but 0 given."
        _res = inputs[0]
        for each in inputs[1:]:
            _res = _op.add(_res, each)
        return _res

    return _impl


def impl_unique(return_counts=True):
    """ Unique """

    def _impl(inputs, attr, params, mod):
        assert len(inputs) == 1
        data = inputs[0]
        if return_counts:
            [unique, indices, num_uniq, counts] = _op.unique(
                data, is_sorted=False, return_counts=True
            )
            unique_sliced = _op.strided_slice(unique, begin=[0], end=num_uniq, slice_mode="size")
            counts_sliced = _op.strided_slice(counts, begin=[0], end=num_uniq, slice_mode="size")
            return _expr.TupleWrapper(
                _expr.Tuple([unique_sliced, indices, counts_sliced]),
                3,
            )
        [unique, indices, num_uniq] = _op.unique(data, is_sorted=False, return_counts=False)
        unique_sliced = _op.strided_slice(unique, begin=[0], end=num_uniq, slice_mode="size")
        return _expr.TupleWrapper(
            _expr.Tuple([unique_sliced, indices]),
            2,
        )

    return _impl
