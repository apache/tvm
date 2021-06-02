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
"""TF: Tensorflow frontend."""
import warnings
from collections import defaultdict
from collections import deque

# Numpy support
import numpy as np
import tvm

from tvm.ir import IRModule
from tvm.relay.prelude import Prelude, StaticTensorArrayOps, get_tensor_array_shape
from tvm.relay.transform import InferType
from tvm.topi.utils import get_const_tuple

from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .. import op as _op
from ..ty import Any
from ..expr_functor import ExprMutator, ExprVisitor
from .common import AttrCvt, get_relay_op
from .common import infer_type as _infer_type
from .common import infer_shape as _infer_shape
from .common import infer_channels as _infer_channels
from .common import infer_value as _infer_value

__all__ = ["from_tensorflow"]


def check_symbolic_shape(shape):
    return not all([isinstance(dim, (int, tvm.tir.IntImm)) for dim in shape])


def list_shape_of(tensor, ndim):
    shape_tensor = _op.shape_of(tensor)
    return [
        _op.strided_slice(shape_tensor, begin=[i], end=[i + 1], strides=[1]) for i in range(ndim)
    ]


def _get_pad_pair(input1d, kernel1d, stride1d):
    if isinstance(input1d, tvm.tir.Any) and stride1d != 1:
        raise tvm.error.OpAttributeUnImplemented(
            "SAME padding is not supported in combination with dynamic height or width when stride"
            " is not 1."
        )
    if stride1d == 1 or input1d % stride1d == 0:
        pad = max(kernel1d - stride1d, 0)
    else:
        pad = max(kernel1d - (input1d % stride1d), 0)

    pad_before = pad // 2
    pad_after = pad - pad_before

    return [pad_before, pad_after]


def _math_name_picker(surfix):
    def _impl(attr):
        return "broadcast_" + surfix

    return _impl


def _dimension_picker(prefix, surfix=""):
    def _impl(attr):
        kernel = attr["kernel_shape"]
        if len(kernel) == 2:
            return prefix + "2d" + surfix
        if len(kernel) == 3:
            return prefix + "3d" + surfix
        raise tvm.error.OpAttributeInvalid(
            "Only 2D or 3D kernels are supported for operator {}".format(prefix + "2d or 3d")
        )

    return _impl


def _dimension_constraint():
    def _dim_check(attrs):
        if len(attrs["kernel_shape"]) in (2, 3):
            return True
        return False

    return _dim_check, "Only 2d or 3d kernel supported."


def _get_param(params, input_node):
    if isinstance(input_node, _expr.Constant):
        return np.atleast_1d(input_node.data.numpy())
    return params[input_node.name_hint].numpy()


def _get_num_param(params, input_node):
    return _get_param(params, input_node).item()


def _get_list_param(params, input_node, mod):
    try:
        return _get_param(params, input_node).tolist()
    except (IndexError, KeyError, AttributeError):
        return _infer_value(input_node, params, mod).numpy().tolist()


def _get_tuple_param(params, input_node):
    return tuple(_get_param(params, input_node))


def _need_prelude_for_shape_inference(op):
    return "TensorArray" in op


def _get_more_static_shape(shape0, shape1):
    """Compare two shapes with the same rank,
    and return the one with fewer symbolic dimension.
    """
    assert len(shape0) == len(shape1)
    num_sym_dim0 = 0
    num_sym_dim1 = 0
    for dim0, dim1 in zip(list(shape0), list(shape1)):
        if not isinstance(dim0, int):
            num_sym_dim0 += 1
        if not isinstance(dim1, int):
            num_sym_dim1 += 1

    if num_sym_dim0 < num_sym_dim1:
        return shape0
    return shape1


def _rsqrt():
    def _impl(inputs, attr, params, mod):
        inputs.append(tvm.relay.const(-0.5, attr["T"].name))
        return AttrCvt(op_name="power")(inputs, attr)

    return _impl


def _argx(func, func_name):
    """A common wrapper for argmin and argmax operations"""

    def _impl(inputs, attr, params, mod):
        try:
            # In Tensorflow, `axis` argument is a Tensor, not attribute. We
            # support the case where it inputs from a scalar constant.
            axis_input_value = [_get_num_param(params, inputs[1])]
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


def _elemwise(name):
    def _impl(inputs, attr, params, mod):
        assert len(inputs) == 2, "{} take 2 inputs, {} given".format(name, len(inputs))
        return get_relay_op(name)(*inputs)

    return _impl


def _pool3d(name):
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
            pad_d = _get_pad_pair(in_d, kernel_d, stride_d)
            pad_v = _get_pad_pair(in_h, kernel_h, stride_h)
            pad_h = _get_pad_pair(in_w, kernel_w, stride_w)

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


def _pooling(name):
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

            pad_v = _get_pad_pair(in_h, kernel_h, stride_h)
            pad_h = _get_pad_pair(in_w, kernel_w, stride_w)

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
            op_name=_dimension_picker(name),
            transforms={"kernel_shape": "pool_size", "data_format": "layout"},
            ignores=["ksize", "explicit_paddings"],
            extras={"ceil_mode": False},
            custom_check=_dimension_constraint(),
        )(inputs, attr)

        if flip_layout:
            out = _op.transpose(out, axes=(0, 2, 3, 1))

        return out

    return _impl


def _conv(opname):
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
            pad_v = _get_pad_pair(in_h, dilated_kernel_h, stride_h)
            pad_h = _get_pad_pair(in_w, dilated_kernel_w, stride_w)

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
            op_name=_dimension_picker(
                "conv", surfix="_transpose" if opname == "conv_transpose" else ""
            ),
            ignores=["explicit_paddings"],
            transforms={
                "kernel_shape": "kernel_size",
                "data_format": "data_layout",
                "dilations": ("dilation", (0, 0)),
                "group": ("groups", 1),
            },
            custom_check=_dimension_constraint(),
        )([inputs_data, inputs[1]], attr)

        if flip_layout:
            out = _op.transpose(out, axes=(0, 2, 3, 1))

        return out

    return _impl


# Dilation2d
def _dilation2d():
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
            pad_v = _get_pad_pair(in_h, dilated_kernel_h, stride_h)
            pad_h = _get_pad_pair(in_w, dilated_kernel_w, stride_w)

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


def _conv3d(opname):
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
            pad_d = _get_pad_pair(in_d, dilated_kernel_d, stride_d)
            pad_v = _get_pad_pair(in_h, dilated_kernel_h, stride_h)
            pad_h = _get_pad_pair(in_w, dilated_kernel_w, stride_w)

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
            op_name=_dimension_picker(
                "conv", surfix="_transpose" if opname == "conv_transpose" else ""
            ),
            ignores=["explicit_paddings", "Tshape"],
            transforms={
                "kernel_shape": "kernel_size",
                "data_format": "data_layout",
                "dilations": ("dilation", (0, 0)),
                "group": ("groups", 1),
            },
            custom_check=_dimension_constraint(),
        )([inputs_data, inputs[1]], attr)

        if use_bias:
            out = _op.nn.bias_add(
                out, inputs[2] if opname != "conv_transpose" else inputs[3], axis=channel_axis
            )

        if flip_layout:
            out = _op.transpose(out, axes=(0, 2, 3, 4, 1))

        return out

    return _impl


def _nms(return_scores=False):
    def _impl(inputs, attr, params, mod):
        # Get parameter values
        try:
            max_output_size = int(np.atleast_1d(inputs[2].data.numpy().astype("int64"))[0])
        except Exception:
            try:
                max_output_size = (
                    _infer_value(inputs[2], params, mod).numpy().astype("int64").tolist()[0]
                )
            except Exception:
                max_output_size = inputs[2]
        iou_threshold = np.atleast_1d(inputs[3].data.numpy())[0]
        # score_threshold was introduced from V3
        score_threshold = np.atleast_1d(inputs[4].data.numpy())[0] if len(inputs) > 4 else 0.0
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


def convert_combined_nms_with_all_class_nms(
    batch_size,
    max_output_boxes_per_batch,
    num_class,
    boxes,
    scores,
    max_output_boxes_per_class,
    iou_threshold,
    score_threshold,
    max_total_size,
    clip_boxes,
):
    """Converts TF combined_nms using Relay all_class_max_suppression op"""
    (selected_indices, selected_scores, num_detections,) = _op.vision.all_class_non_max_suppression(
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        output_format="tensorflow",
    )
    box_range = _op.arange(
        _op.const(0, dtype="int64"), _op.const(max_total_size, dtype="int64"), dtype="int64"
    )
    assert isinstance(batch_size, int), "dynamic batch size not supported yet."
    tile_batch_reps = _op.const([batch_size, 1])
    box_range_2d = _op.tile(box_range, tile_batch_reps)
    valid_mask = _op.cast(
        _op.less(box_range_2d, _op.expand_dims(num_detections, axis=1)), "float32"
    )

    def select_topk(do_zero_pad):
        def true_branch():
            arange = _op.arange(
                _op.const(0, dtype="int64"),
                _op.const(max_output_boxes_per_batch, dtype="int64"),
                dtype="int64",
            )
            pad = _op.full(
                _op.const(0, dtype="int64"), (max_total_size - max_output_boxes_per_batch,)
            )
            topk_indices = _op.tile(_op.concatenate([arange, pad], 0), tile_batch_reps)
            nmsed_scores = _op.gather(selected_scores, 1, topk_indices)
            nmsed_scores = nmsed_scores * valid_mask
            return nmsed_scores, topk_indices

        def false_branch():
            if isinstance(max_output_boxes_per_class, int):
                # Do topk on smaller input if possible
                slice_mx = _op.const([max_output_boxes_per_class * num_class], dtype="int64")
                selected_scores_slice = _op.strided_slice(
                    selected_scores, begin=_op.const([0], dtype="int64"), end=slice_mx, axes=[1]
                )
            else:
                selected_scores_slice = selected_scores
            return _op.topk(selected_scores_slice, k=max_total_size, axis=1, ret_type="both")

        # TODO(masahi): support dynamic num_boxes
        # return _expr.If(do_zero_pad, true_branch(), false_branch())
        return true_branch() if do_zero_pad else false_branch()

    assert isinstance(max_output_boxes_per_batch, int), "dynamic number of boxes not supported yet."
    nmsed_scores, topk_indices = select_topk(max_output_boxes_per_batch < max_total_size)

    indices = _op.take(selected_indices, topk_indices, axis=1, batch_dims=1)
    nmsed_box_indices = _op.take(indices, _op.const(1), axis=2)
    nmsed_classes = _op.take(indices, _op.const(0), axis=2)
    nmsed_classes = _op.cast(nmsed_classes, "float32")
    nmsed_boxes = _op.take(boxes, nmsed_box_indices, axis=1, batch_dims=1)
    num_detections = _op.minimum(num_detections, _op.const(max_total_size, dtype="int64"))

    if clip_boxes:
        nmsed_boxes = _op.maximum(nmsed_boxes, _expr.const(0, dtype="float32"))
        nmsed_boxes = _op.minimum(nmsed_boxes, _expr.const(1, dtype="float32"))

    nmsed_boxes = nmsed_boxes * _op.expand_dims(valid_mask, axis=2)

    return _expr.TupleWrapper(
        _expr.Tuple([nmsed_boxes, nmsed_scores, nmsed_classes, num_detections]), 4
    )


def _combined_nms():
    def _impl(inputs, attr, params, mod):
        # Get parameter values
        boxes = inputs[0]
        scores = inputs[1]
        try:
            max_output_size = int(np.atleast_1d(inputs[2].data.numpy().astype("int64"))[0])
        except Exception:
            try:
                max_output_size = (
                    _infer_value(inputs[2], params, mod).numpy().astype("int64").tolist()[0]
                )
            except Exception:
                max_output_size = inputs[2]
        max_total_size = inputs[3]
        iou_threshold = np.atleast_1d(inputs[4].data.numpy())[0]
        score_threshold = np.atleast_1d(inputs[5].data.numpy())[0]
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

        assert isinstance(batch_size, int) and isinstance(
            num_anchors, int
        ), "Dynamic inputs not supported yet"

        if q == 1:
            boxes = _op.squeeze(boxes, axis=[2])
            scores_trans = _op.transpose(scores, [0, 2, 1])
            max_output_boxes_per_batch = num_anchors * num_classes
            return convert_combined_nms_with_all_class_nms(
                batch_size,
                max_output_boxes_per_batch,
                num_classes,
                boxes,
                scores_trans,
                max_output_size,
                iou_threshold,
                score_threshold,
                max_total_size.data.numpy().item(),
                attr["clip_boxes"],
            )

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


def _decode_image():
    def _impl(inputs, attr, params, mod):
        # Image decode wrapper: Expecting user to feed decoded input to next layer drop this layer.
        warnings.warn("DecodeJpeg: It's a pass through, please handle preprocessing before input")
        return inputs[0]

    return _impl


def _unravel_index():
    def _impl(inputs, attr, params, mod):
        return _op.unravel_index(inputs[0], inputs[1])

    return _impl


def _crop_and_resize():
    def _impl(inputs, attr, params, mod):
        # input image is a 4-D tensor of shape [batch, image_height, image_width, depth]
        # boxes is a 2-D tensor of shape [num_boxes, 4], 4 is for [y1, x1, y2, x2]
        crop_size = _get_list_param(params, inputs[3], mod)

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


def _cast():
    def _impl(inputs, attr, params, mod):
        return inputs[0].astype(attr["DstT"].name)

    return _impl


def _expand_dims():
    def _impl(inputs, attr, params, mod):
        dim_input = inputs.pop(1)
        axis = _get_num_param(params, dim_input)
        return AttrCvt(
            op_name="expand_dims",
            ignores=["Tdim", "N"],
            extras={"axis": int(axis), "num_newaxis": 1},
        )(inputs, attr)

    return _impl


def _expm1():
    # op description: https://www.tensorflow.org/api_docs/python/tf/math/expm1
    def _impl(inputs, attr, params, mod):
        exp_out = get_relay_op("exp")(inputs[0])
        return exp_out - tvm.relay.const(1.0)

    return _impl


def _resize(method):
    def _impl(inputs, attr, params, mod):
        if attr["_output_shapes"][0] is not None:
            size = attr["_output_shapes"][0][1:3]
            # Important that the size is defined. If an axis is not, we need to infer what
            # the shape should be.
            if -1 in size:
                size = _infer_value(inputs[1], params, mod).numpy().reshape([-1]).tolist()
        else:
            size = _infer_value(inputs[1], params, mod).numpy().reshape([-1]).tolist()

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


def _check_numerics():
    def _impl(inputs, attr, params, mod):
        # Making a copy node assuming no need to verify
        return AttrCvt(op_name="copy", ignores=["message"])(inputs, attr)

    return _impl


def _assert():
    # ToDo: In general people want asserts to be gone from TensorFlow graphs
    # when they are optimizing them, so converting it to a no-op is
    # reasonable. However, it would be nice to have the option to keep them
    # once Relay gets a Halt or Assert op.
    return _no_op()


def _no_op():
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


def _matmul():
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


def _batch_matmul():
    def _impl(inputs, attr, params, mod):
        input_x = inputs[0]
        input_y = inputs[1]
        orig_shape_x = _infer_shape(input_x, mod)
        orig_shape_y = _infer_shape(input_y, mod)
        ndim = len(orig_shape_x)

        is_static = not check_symbolic_shape(orig_shape_x)

        if ndim > 3 and not is_static:
            shape_of_x = list_shape_of(inputs[0], ndim)
            shape_of_y = list_shape_of(inputs[1], ndim)

        # reshape n-dimensional batch matmul into 3d
        if ndim > 3:
            outer_dims = [orig_shape_x[i] for i in range(0, len(orig_shape_x) - 2)]
            if is_static:
                num_outer_elts = np.prod(outer_dims)
                new_shape_x = (num_outer_elts, orig_shape_x[-2], orig_shape_x[-1])
                new_shape_y = (num_outer_elts, orig_shape_y[-2], orig_shape_y[-1])
            else:  # handle dynamic shape (dyn.reshape op)
                # new shape = [prod(shape[:-2]), -2, -1]
                new_shape_x = [_op.const(1), shape_of_x[-2], shape_of_x[-1]]
                new_shape_y = [_op.const(1), shape_of_y[-2], shape_of_y[-1]]
                for i in range(ndim - 2):
                    new_shape_x[0] *= shape_of_x[i]
                    new_shape_y[0] *= shape_of_y[i]
                new_shape_x = _op.concatenate(_op.Tuple(new_shape_x), axis=0)
                new_shape_y = _op.concatenate(_op.Tuple(new_shape_y), axis=0)

            input_x = _op.reshape(input_x, newshape=new_shape_x)
            input_y = _op.reshape(input_y, newshape=new_shape_y)

        adj_x = attr["adj_x"]
        adj_y = attr["adj_y"]
        input_x = _op.transpose(input_x, axes=[0, 2, 1]) if adj_x else input_x
        input_y = _op.transpose(input_y, axes=[0, 2, 1]) if not adj_y else input_y
        ret = get_relay_op("batch_matmul")(input_x, input_y)

        # reshape result back to n-dimensional
        if ndim > 3:
            if is_static:
                final_shape = list(orig_shape_x)
                final_shape[-2] = orig_shape_x[-1] if adj_x else orig_shape_x[-2]
                final_shape[-1] = orig_shape_y[-2] if adj_y else orig_shape_y[-1]
            else:
                # calculate the resulting shape = [shape[:-2], 0, 0]
                final_shape = list(shape_of_x)
                final_shape[-2] = shape_of_x[-1] if adj_x else shape_of_x[-2]
                final_shape[-1] = shape_of_y[-2] if adj_y else shape_of_y[-1]
                final_shape = _op.concatenate(_op.Tuple(final_shape), axis=0)

            ret = _op.reshape(ret, newshape=final_shape)
        return ret

    return _impl


def _sparse_tensor_dense_matmul():
    def _impl(inputs, attr, params, mod):
        # Loading this by default causes TVM to not be loadable from other languages.
        # Sparse utility from scipy
        from scipy.sparse import csr_matrix

        assert len(inputs) == 4, "There should be 4 input tensors"

        indices_tensor = _infer_value(inputs[0], params, mod).numpy()
        values_tensor = _infer_value(inputs[1], params, mod).numpy()
        dense_shape_tensor = _infer_value(inputs[2], params, mod).numpy()

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


def _sparse_fill_empty_rows():
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


def _sparse_reshape():
    def _impl(inputs, attr, params, mod):
        assert len(inputs) == 3, "There should be 3 input tensors"
        new_indices, new_shape = get_relay_op("sparse_reshape")(inputs[0], inputs[1], inputs[2])
        return _expr.TupleWrapper(_expr.Tuple([new_indices, new_shape]), 2)

    return _impl


def _math_segment_sum():
    def _impl(inputs, attr, params, mod):
        assert len(inputs) == 2, "There should be 2 input tensors"
        return get_relay_op("segment_sum")(inputs[0], inputs[1])

    return _impl


def _sparse_segment_sum():
    def _impl(inputs, attr, params, mod):
        assert len(inputs) == 3, "There should be 3 input tensors"
        data = _op.take(inputs[0], inputs[1], axis=0)
        return _op.segment_sum(data, inputs[2])

    return _impl


def _sparse_segment_sum_with_num_segments():
    def _impl(inputs, attr, params, mod):
        assert len(inputs) == 4, "There should be 4 input tensors"
        data = _op.take(inputs[0], inputs[1], axis=0)
        num_segments = int(inputs[3].data.numpy().item())
        return _op.segment_sum(data, inputs[2], num_segments)

    return _impl


def row_wise_divide(multi_dim_tensor, one_dim_vector):
    """
    This function enables row-wise division of multi_dim_tensor and one_dim_vector.
    To achieve this, it is first tiled to the appropriate shape and then elemwise_division
    """
    multi_dim_tensor_offrow_shape = _op.strided_slice(
        _op.shape_of(multi_dim_tensor, "int32"), [1], [-1], slice_mode="size"
    )
    one_dim_vector_tiled_shape = _op.concatenate(
        [_op.reverse(multi_dim_tensor_offrow_shape, 0), _expr.const([1])], axis=0
    )
    one_dim_vector_tiled = _op.transpose(_op.tile(one_dim_vector, one_dim_vector_tiled_shape))
    return _op.divide(multi_dim_tensor, one_dim_vector_tiled)


def count_all_indices(segment_ids, counts_dtype, num_segments=None):
    """
    This snippet calculates the sqrt count of each index among all valid indices
    Valid indices are from 0 to max of [segment ids, num_segments]
    """

    max_segments = _op.reshape(_op.max(segment_ids), -1) + _expr.const([1])
    if num_segments:
        max_segments = _op.maximum(max_segments, _expr.const([num_segments]))
    max_ones = _op.maximum(max_segments, _op.shape_of(segment_ids))
    counts = _op.segment_sum(
        _op.ones(max_ones, counts_dtype), segment_ids, num_segments=num_segments
    )
    real_counts = _op.clip(counts, 1, 2147483647)  # Clip max doesn't work over int32
    return real_counts


def _sparse_segment_sum_sqrtn():
    def _impl(inputs, attr, params, mod):
        assert len(inputs) == 3, "There should be 3 input tensors"
        data = _op.take(inputs[0], inputs[1], axis=0)
        real_counts = count_all_indices(inputs[2], attr["T"].name)
        real_sqrt_counts = _op.sqrt(_op.cast_like(real_counts, data))

        # Calculate regular segment sum
        segment_sum = _op.segment_sum(data, inputs[2])

        return row_wise_divide(segment_sum, real_sqrt_counts)

    return _impl


def _sparse_segment_sum_sqrtn_with_num_segments():
    def _impl(inputs, attr, params, mod):
        assert len(inputs) == 4, "There should be 4 input tensors"
        data = _op.take(inputs[0], inputs[1], axis=0)
        num_segments = int(inputs[3].data.numpy().item())
        real_counts = count_all_indices(inputs[2], attr["T"].name, num_segments=num_segments)
        real_sqrt_counts = _op.sqrt(_op.cast_like(real_counts, data))

        # Calculate regular segment sum
        segment_sum = _op.segment_sum(data, inputs[2], num_segments=num_segments)

        return row_wise_divide(segment_sum, real_sqrt_counts)

    return _impl


def _sparse_segment_mean():
    def _impl(inputs, attr, params, mod):
        assert len(inputs) == 3, "There should be 3 input tensors"
        data = _op.take(inputs[0], inputs[1], axis=0)
        real_counts = count_all_indices(inputs[2], attr["T"].name)

        # Calculate regular segment sum
        segment_sum = _op.segment_sum(data, inputs[2])

        return row_wise_divide(segment_sum, real_counts)

    return _impl


def _sparse_segment_mean_with_num_segments():
    def _impl(inputs, attr, params, mod):
        assert len(inputs) == 4, "There should be 4 input tensors"
        data = _op.take(inputs[0], inputs[1], axis=0)
        num_segments = int(inputs[3].data.numpy().item())
        real_counts = count_all_indices(inputs[2], attr["T"].name, num_segments=num_segments)

        # Calculate regular segment sum
        segment_sum = _op.segment_sum(data, inputs[2], num_segments=num_segments)

        return row_wise_divide(segment_sum, real_counts)

    return _impl


def _sparse_tensor_dense_add():
    # Sparse utility from scipy
    from scipy.sparse import csr_matrix

    def _impl(inputs, attr, params, mod):
        assert (
            len(inputs) == 4
        ), "There should be 4 input tensors [sparse_indices, sparse_values, sparse_shape, dense]."

        indices_tensor = _infer_value(inputs[0], params, mod).numpy()
        values_tensor = _infer_value(inputs[1], params, mod).numpy()
        dense_shape_tensor = _infer_value(inputs[2], params, mod).numpy()

        data = inputs[3]

        rows = [x[0] for x in indices_tensor]
        cols = [x[1] for x in indices_tensor]

        # Create scipy sparse Tensor(CSR)
        weight_sp = csr_matrix(
            (values_tensor, (rows, cols)), shape=tuple(dense_shape_tensor.tolist())
        )

        weight_data = _expr.const(weight_sp.data, weight_sp.data.dtype)
        weight_indptrs = _expr.const(weight_sp.indptr, weight_sp.indptr.dtype)
        weight_indices = _expr.const(weight_sp.indices, weight_sp.indices.dtype)

        ret = _op.nn.sparse_add(data, [weight_data, weight_indices, weight_indptrs])

        return ret

    return _impl


def _identity():
    def _impl(inputs, attr, params, mod):
        return inputs[0]

    return _impl


def _identityn():
    def _impl(inputs, attr, params, mod):
        return inputs

    return _impl


def _concatV2():
    def _impl(inputs, attr, params, mod):
        pop_node = inputs.pop(len(inputs) - 1)
        axis = int(_get_num_param(params, pop_node))
        return AttrCvt(op_name="concatenate", ignores=["T", "N", "Tidx"], extras={"axis": axis})(
            [inputs], attr
        )

    return _impl


def _concat():
    def _impl(inputs, attr, params, mod):
        pop_node = inputs.pop(0)
        axis = int(_get_num_param(params, pop_node))
        return AttrCvt(op_name="concatenate", ignores=["N"], extras={"axis": axis})([inputs], attr)

    return _impl


def _pack():
    def _impl(inputs, attr, params, mod):
        axis = int(attr["axis"])
        inputs_reshaped = [_op.expand_dims(i, axis=axis, num_newaxis=1) for i in inputs]
        return _op.concatenate(inputs_reshaped, axis)

    return _impl


def _tensor_array():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr.get("dtype").name
        assert not attr["dynamic_size"], "Dynamic size tensor array is " "not supported in TVM yet."

        if "shape" in attr:
            shape = attr["shape"]
            static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, shape)
            static_tensor_array_ops.register()
            tensor_array_constructor = static_tensor_array_ops.get_global_var("tensor_array")
            tensor_array = tensor_array_constructor(inputs[0])
        else:
            tensor_array_constructor = prelude.get_global_var("tensor_array", dtype_str)
            tensor_array = tensor_array_constructor(inputs[0])
        return tensor_array

    return _impl


def _tensor_array_scatter():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr.get("T").name
        input_ta = inputs[0]
        input_shape = get_tensor_array_shape(input_ta, dtype_str, prelude)
        values_shape = _infer_shape(inputs[2], prelude.mod)
        input_t_shape = values_shape[1:]
        indices_shape = _infer_shape(inputs[1], prelude.mod)

        if input_shape is None:
            values_rank = len(values_shape)
            unstack_name = "tensor_array_unstack_tensor{}".format(values_rank)
            unstack_function = prelude.get_global_var(unstack_name, dtype_str)
            values = unstack_function(inputs[2])
            tensor_array_scatter_func = prelude.get_global_var("tensor_array_scatter", dtype_str)
        else:
            input_t_shape = _get_more_static_shape(input_t_shape, input_shape)
            values_shape = (values_shape[0],) + input_t_shape
            static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, input_t_shape)
            static_tensor_array_ops.register()
            # Register static indices shape
            if isinstance(indices_shape[0], int):
                static_tensor_array_ops.define_tensor_array_scatter(indices_shape, True)
            tensor_array_scatter_func = prelude.get_global_var_static(
                "tensor_array_scatter", dtype_str, input_t_shape
            )

            static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, values_shape)
            static_tensor_array_ops.register()
            unstack_function = prelude.get_global_var_static(
                "tensor_array_unstack", dtype_str, values_shape
            )
            values = unstack_function(inputs[2])
        ret = tensor_array_scatter_func(input_ta, inputs[1], values)
        return ret

    return _impl


def _tensor_array_gather():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr.get("dtype").name
        input_shape = get_tensor_array_shape(inputs[2], dtype_str, prelude)
        indices_shape = _infer_shape(inputs[1], prelude.mod)

        if input_shape is None:
            gather_func = prelude.get_var("tensor_array_gather", dtype_str)
            out = gather_func(inputs[2], inputs[1])
        else:
            static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, input_shape)
            static_tensor_array_ops.register()

            if not isinstance(indices_shape[0], int):
                gather_function = prelude.get_global_var_static(
                    "tensor_array_gather", dtype_str, input_shape
                )
                out_tensor_t = gather_function(inputs[2], inputs[1])
                out_shape = (indices_shape[0],) + input_shape
                static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, out_shape)
                static_tensor_array_ops.register()

                # Output shape is (indices_shape[0],) + input_shape
                get_data_func = prelude.get_global_var_static(
                    "tensor_get_data", dtype_str, out_shape
                )
                out = get_data_func(out_tensor_t)
            else:
                # For fixed length indices, directly generate static shape output
                read_func = prelude.get_global_var_static(
                    "tensor_array_read", dtype_str, input_shape
                )
                get_data_func = prelude.get_global_var_static(
                    "tensor_get_data", dtype_str, input_shape
                )
                tensor_list = []
                for i in range(indices_shape[0]):
                    index = _op.take(inputs[1], tvm.relay.const(i))
                    out_tensor = get_data_func(read_func(inputs[2], index))
                    tensor_list.append(_op.expand_dims(out_tensor, axis=0))

                if indices_shape[0] > 1:
                    out = _op.concatenate(tensor_list, axis=0)
                else:
                    out = tensor_list[0]

        return out

    return _impl


def _tensor_array_size():
    def _impl(inputs, attr, params, prelude):
        return prelude.length(inputs[0])

    return _impl


def _tensor_array_write():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr.get("T").name
        input_ta = inputs[3]
        input_ta_shape = get_tensor_array_shape(input_ta, dtype_str, prelude)
        input_t_shape = _infer_shape(inputs[2], prelude.mod)
        input_rank = len(input_t_shape)

        if input_ta_shape is None:
            tensor_name = "tensor{}".format(input_rank)
            tensor_func = prelude.get_tensor_ctor(tensor_name, dtype_str)
            v = tensor_func(inputs[2])
            write_func = prelude.get_global_var("tensor_array_write", dtype_str)
        else:
            input_ta_rank = len(input_ta_shape)
            assert input_ta_rank == input_rank, "Shape rank mismatch: {} vs {}".format(
                input_ta_rank, input_rank
            )
            static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, input_ta_shape)
            static_tensor_array_ops.register()
            tensor_func = static_tensor_array_ops.get_ctor("tensor_constructor")
            v = tensor_func(inputs[2])
            # Write tensor with more static shape
            actual_shape = _get_more_static_shape(input_t_shape, input_ta_shape)
            if actual_shape != input_t_shape:
                new_shape = []
                num_any_dim = 0
                for dim in actual_shape:
                    if not isinstance(dim, int):
                        num_any_dim += 1
                    new_shape.append(dim if isinstance(dim, int) else -1)
                if num_any_dim <= 1:
                    v = tensor_func(_op.reshape(inputs[2], new_shape))

            write_func = prelude.get_global_var_static(
                "tensor_array_write", dtype_str, input_ta_shape
            )

        return write_func(input_ta, _op.take(inputs[1], tvm.relay.const(0)), v)

    return _impl


def _tensor_array_read():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr["dtype"].name
        input_shape = get_tensor_array_shape(inputs[2], dtype_str, prelude)

        if input_shape is None:
            read_func = prelude.get_global_var("tensor_array_read", dtype_str)
            out = read_func(inputs[2], _op.take(inputs[1], tvm.relay.const(0)))
        else:
            static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, input_shape)
            static_tensor_array_ops.register()
            read_func = static_tensor_array_ops.get_global_var("tensor_array_read")
            out_tensor = read_func(inputs[2], _op.take(inputs[1], tvm.relay.const(0)))
            get_data_func = static_tensor_array_ops.get_global_var("tensor_get_data")
            out = get_data_func(out_tensor)

        return out

    return _impl


def _tensor_array_split():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr.get("T").name
        input_ta = inputs[0]
        input_ta_shape = get_tensor_array_shape(input_ta, dtype_str, prelude)
        lengths = _op.cast(inputs[2], "int32")
        lengths_shape = _infer_shape(lengths, prelude.mod)
        value_shape = _infer_shape(inputs[1], prelude.mod)
        input_rank = len(value_shape)

        if input_ta_shape is None:
            tensor_name = "tensor{}".format(input_rank)
            tensor_ctor = prelude.get_tensor_ctor(tensor_name, dtype_str)
            v = tensor_ctor(inputs[1])
            split_func = prelude.get_global_var("tensor_array_split", dtype_str)
        else:
            input_ta_rank = len(input_ta_shape)
            assert input_ta_rank == input_rank, "Shape rank mismatch: {} vs {}".format(
                input_ta_rank, input_rank
            )
            static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, input_ta_shape)
            static_tensor_array_ops.register()

            # Check static value/indices shape
            if isinstance(value_shape[0], int) or isinstance(lengths_shape[0], int):
                static_tensor_array_ops.define_tensor_array_split(value_shape, lengths_shape, True)

            static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, value_shape)
            static_tensor_array_ops.register()
            tensor_ctor = static_tensor_array_ops.get_ctor("tensor_constructor")
            v = tensor_ctor(inputs[1])
            split_func = prelude.get_global_var_static(
                "tensor_array_split", dtype_str, input_ta_shape
            )

        return split_func(input_ta, v, lengths)

    return _impl


def _tensor_array_concat():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr["dtype"].name
        input_shape = get_tensor_array_shape(inputs[1], dtype_str, prelude)

        if input_shape is None:
            concat_func = prelude.get_global_var("tensor_array_concat", dtype_str)
            out = concat_func(inputs[1])
        else:
            static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, input_shape)
            static_tensor_array_ops.register()
            concat_func = prelude.get_global_var_static(
                "tensor_array_concat", dtype_str, input_shape
            )
            out_tensor = concat_func(inputs[1])
            out_shape = (Any(),) + input_shape[1:]
            static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, out_shape)
            static_tensor_array_ops.register()
            get_data_func = prelude.get_global_var_static("tensor_get_data", dtype_str, out_shape)
            out = get_data_func(out_tensor)

        return out

    return _impl


def _tile():
    def _impl(inputs, attr, params, mod):
        reps_input = inputs.pop()
        if isinstance(reps_input, _expr.Call):
            np_reps = _infer_value(reps_input, params, mod).numpy()
            reps = [np_reps.flatten()[i] for i in range(np_reps.flatten().shape[0])]
        else:
            reps = _get_list_param(params, reps_input, mod)
        new_input = [inputs.pop(0)]

        return AttrCvt(op_name="tile", extras={"reps": tuple(reps)}, ignores=["Tmultiples"])(
            new_input, attr
        )

    return _impl


def _slice():
    def _impl(inputs, attr, params, mod):
        try:
            begin = _get_list_param(params, inputs[1], mod)
        except Exception:
            # Handle symbolic begin
            begin = inputs[1]
        try:
            size = _get_list_param(params, inputs[2], mod)
        except Exception:
            # Handle symbolic size
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


def _reshape():
    def _impl(inputs, attr, params, mod):
        pop_node = inputs.pop(1)

        try:
            shape_arg = _get_tuple_param(params, pop_node)
        except AttributeError:
            # Shape operator is already pruned, hence
            # try to infer shape by precompute prune if possible.
            try:
                params_new = _infer_value(pop_node, params, mod)
                shape_arg = tuple(params_new.numpy().astype("int32").flatten())
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


def _depth_to_space():
    def _impl(inputs, attr, params, mod):
        block_size = int(attr["block_size"])
        layout = attr["data_format"].decode("utf-8")
        return _op.nn.depth_to_space(inputs[0], block_size, layout)

    return _impl


def _space_to_depth():
    def _impl(inputs, attr, params, mod):
        block_size = int(attr["block_size"])
        layout = attr["data_format"].decode("utf-8")
        return _op.nn.space_to_depth(inputs[0], block_size, layout)

    return _impl


def _sparse_to_dense():
    def _impl(inputs, attr, params, mod):
        sparse_indices = inputs[0]
        output_shape = inputs[1]
        sparse_values = inputs[2]
        default_value = inputs[3]

        return _op.sparse_to_dense(sparse_indices, output_shape, sparse_values, default_value)

    return _impl


def _bias_add():
    def _impl(inputs, attr, params, mod):
        # Must expand for proper broadcasting in NCHW.
        if "data_format" in attr and attr["data_format"].decode("utf-8") == "NCHW":
            bias = _op.reshape(inputs[1], newshape=(1, -1, 1, 1))
        else:
            bias = inputs[1]
        return _op.add(inputs[0], bias)

    return _impl


def _broadcast_args():
    def _impl(inputs, attr, params, mod):
        if isinstance(inputs[0], _expr.Var):
            s0 = params[inputs[0].name_hint]
        else:
            s0 = _infer_value(inputs[0], params, mod)
        if isinstance(inputs[1], _expr.Var):
            s1 = params[inputs[1].name_hint]
        else:
            s1 = _infer_value(inputs[1], params, mod)
        s0 = list(s0.numpy().reshape([-1]))
        s1 = list(s1.numpy().reshape([-1]))
        s0_size, s1_size = len(s0), len(s1)

        out = deque([])
        for i in range(1, min(s0_size, s1_size) + 1):
            if s0[s0_size - i] == s1[s1_size - i]:
                out.appendleft(s0[s0_size - i])
            elif s0[s0_size - i] == 1:
                out.appendleft(s1[s1_size - i])
            else:
                assert s1[s1_size - i] == 1, "Incompatible broadcast type %s and %s" % (
                    s0[s0_size - i],
                    s1[s1_size - i],
                )
                out.appendleft(s0[s0_size - i])
        if s0_size < s1_size:
            for i in range(s0_size + 1, s1_size + 1):
                out.appendleft(s1[s1_size - i])
        if s1_size < s0_size:
            for i in range(s1_size + 1, s0_size + 1):
                out.appendleft(s0[s0_size - i])
        return _expr.const(list(out), attr["T"].name)

    return _impl


def _broadcast_to():
    def _impl(inputs, attr, params, mod):
        if isinstance(inputs[1], _expr.Var):
            shape = params[inputs[1].name_hint]
        else:
            shape = _infer_value(inputs[1], params, mod)
        shape = list(shape.numpy().reshape([-1]))
        return _op.broadcast_to(inputs[0], shape)

    return _impl


def _squeeze():
    def _impl(inputs, attr, params, mod):
        if len(attr["squeeze_dims"]) == 0:
            attr["squeeze_dims"] = None
        return AttrCvt(
            op_name="squeeze", transforms={"squeeze_dims": "axis"}, ignores=["T", "_cloned"]
        )(inputs, attr)

    return _impl


def _fused_batch_norm():
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


def _batch_norm():
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


def _relu6():
    def _impl(inputs, attr, params, mod):
        return _op.clip(inputs[0], a_min=0, a_max=6)

    return _impl


def _shape():
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


def _fill():
    def _impl(inputs, attr, params, mod):
        try:
            output_shape = _infer_value(inputs[0], params, mod).numpy().tolist()
        except Exception:
            output_shape = inputs[0]

        return _op.full(inputs[1], output_shape, attr["T"].name)

    return _impl


def _lrn():
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


def _sum():
    def _impl(inputs, attr, params, mod):
        axis = _get_tuple_param(params, inputs[1])
        return AttrCvt(
            op_name="sum",
            extras={"axis": axis},
            transforms={"keep_dims": "keepdims"},
            ignores=["name", "Tidx"],
        )([inputs[0]], attr)

    return _impl


def _reduce(op):
    def _impl(inputs, attr, params, mod):
        axis = _get_list_param(params, inputs[1], mod)
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


def _euclidean_norm():
    def _impl(inputs, attr, params, mod):
        axis = tuple(_get_list_param(params, inputs[1], mod))
        keep_dims = bool(attr.get("keep_dims", False))
        return _op.sqrt(
            _op.cast(_op.reduce.sum(_op.multiply(inputs[0], inputs[0]), axis, keep_dims), "float32")
        )

    return _impl


def _square():
    def _impl(inputs, attr, params, mod):
        return _op.multiply(inputs[0], inputs[0])

    return _impl


def _gather():
    "GatherV2, Gather"

    def _impl(inputs, attr, params, mod):
        if len(inputs) > 2:
            axis = _get_num_param(params, inputs.pop(2))
        else:
            axis = 0
        batch_dims = 0
        if int(attr.get("batch_dims", 0)) != 0:
            batch_dims = int(attr.get("batch_dims", 0))
        new_input = inputs[0:2]
        op_ = AttrCvt(
            op_name="take",
            extras={
                "axis": tvm.tir.const(axis, "int32"),
                "batch_dims": tvm.tir.const(batch_dims, "int32"),
            },
            ignores=["Tindices", "Tparams", "validate_indices", "Taxis", "_class"],
        )(new_input, attr)
        return op_

    return _impl


def _gather_nd():
    """GatherNd"""

    def _impl(inputs, attr, params, mod):
        indices_dims = len(_infer_shape(inputs[1], mod))
        indices = _op.transpose(inputs[1], axes=[-1] + list(range(indices_dims - 1)))
        return AttrCvt(op_name="gather_nd", ignores=["Tindices", "Tparams", "Taxis", "_class"])(
            [inputs[0], indices], attr
        )

    return _impl


def _stridedSlice():
    def _impl(inputs, attr, params, mod):
        """Strided Slice.
        Operator description: https://www.tensorflow.org/api_docs/python/tf/strided_slice
        Tensorflow mask validation: https://github.com/tensorflow/tensorflow/blob/master/
        tensorflow/core/util/strided_slice_op.cc#L147-L368
        """
        begin = _get_list_param(params, inputs[1], mod)
        end = _get_list_param(params, inputs[2], mod)
        stride = _get_list_param(params, inputs[3], mod)

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
            new_data = inputs[0].data.numpy().reshape(1)
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


def _pad(name):
    def _impl(inputs, attr, params, mod):
        try:
            padlist = _get_param(params, inputs[1])
        except (IndexError, KeyError, AttributeError):
            try:
                padlist = _infer_value(inputs[1], params, mod).numpy().tolist()
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
                attr["pad_value"] = _get_num_param(params, inputs[2])
            except (IndexError, KeyError, AttributeError):
                attr["pad_value"] = inputs[2]
        return AttrCvt(
            op_name="pad",
            ignores=["Tpaddings"],
        )(new_inputs, attr)

    return _impl


def _mirror_pad():
    def _impl(inputs, attr, params, mod):
        padlist = _get_param(params, inputs[1])
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


def _transpose():
    def _impl(inputs, attr, params, mod):
        # If perm is not specified, axes is left empty,
        # otherwise its value is get from params
        axes = _get_list_param(params, inputs[1], mod)
        return _op.transpose(inputs[0], axes=axes)

    return _impl


def _where():
    def _impl(inputs, attr, params, mod):
        if len(inputs) == 1:
            return AttrCvt(op_name="argwhere")(inputs, attr)
        return AttrCvt(op_name="where")(inputs, attr)

    return _impl


def _clip_by_value():
    def _impl(inputs, attr, params, mod):
        a_min = _get_num_param(params, inputs[1])
        a_max = _get_num_param(params, inputs[2])
        return _op.clip(inputs[0], a_min=a_min, a_max=a_max)

    return _impl


def _reverse_v2():
    def _impl(inputs, attr, params, mod):
        axis = _get_num_param(params, inputs[1])
        return AttrCvt(op_name="reverse", ignores=["Tidx"], extras={"axis": int(axis)})(
            [inputs[0]], attr
        )

    return _impl


def _rank():
    def _impl(inputs, attr, params, mod):
        input_shape = _infer_shape(inputs[0], mod)

        name = attr["_node_name"]
        params[name] = tvm.nd.array(np.array([len(input_shape)]).astype("int32"))
        return [_expr.var(name, shape=params[name].shape, dtype="int32")]

    return _impl


def _range():
    def _impl(inputs, attr, params, mod):
        try:
            start = _get_param(params, inputs[0])[0]
        except (IndexError, KeyError, AttributeError):
            try:
                start = _infer_value(inputs[1], params, mod).numpy().tolist()
                start = start if not isinstance(start, list) else start[0]
            except Exception:
                # Symbolic start
                start = inputs[0]

        try:
            limit = (
                _get_param(params, inputs[1])[0]
                if hasattr(inputs[1], "name_hint") or isinstance(inputs[1], _expr.Constant)
                else params.pop("Rank").numpy()[0]
            )
        except (IndexError, KeyError, AttributeError):
            try:
                limit = _infer_value(inputs[1], params, mod).numpy().tolist()
                limit = limit if not isinstance(limit, list) else limit[0]
            except Exception:
                limit = inputs[1]

        try:
            delta = _get_param(params, inputs[2])[0]
        except (IndexError, KeyError, AttributeError):
            try:
                delta = _infer_value(inputs[2], params, mod).numpy().tolist()
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


def _elu():
    def _impl(inputs, attr, params, mod):
        dtype = attr["T"].name
        alpha = tvm.relay.const(-1.0, dtype)
        return alpha * _op.nn.relu(tvm.relay.const(1, dtype) - _op.exp(inputs[0])) + _op.nn.relu(
            inputs[0]
        )

    return _impl


def _selu():
    def _impl(inputs, attr, params, mod):
        dtype = attr["T"].name
        alpha = tvm.relay.const(-1.6732632423543772848170429916717, dtype)
        gamma = tvm.relay.const(1.0507009873554804934193349852946, dtype)
        return gamma * (
            alpha * _op.nn.relu(tvm.relay.const(1, dtype) - _op.exp(inputs[0]))
            + _op.nn.relu(inputs[0])
        )

    return _impl


def _mean():
    def _impl(inputs, attr, params, mod):
        axis = _get_tuple_param(params, inputs[1])
        return AttrCvt(
            op_name="mean",
            ignores=["Tdim", "Tidx"],
            transforms={"keep_dims": "keepdims"},
            extras={"axis": axis},
        )([inputs[0]], attr)

    return _impl


def _broadcast(name):
    def _impl(inputs, attr, params, mod):
        return AttrCvt(op_name=name, ignores=["name", "incompatible_shape_error", "Tidx"])(
            inputs, attr
        )

    return _impl


def _split(has_size_vector):
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
                size_splits = _get_param(params, inputs[1])
                section_beginnings = np.cumsum(size_splits)[:-1]
                indices_or_sections = tuple(section_beginnings)
            else:
                input_node_index = 1
                input_axis_index = 0
                indices_or_sections = attr["num_split"]
            input_node = inputs[input_node_index]
            axis_input_value = _get_num_param(params, inputs[input_axis_index])
        except (IndexError, KeyError, AttributeError):
            raise TypeError(
                "Unsupported argument for split: `axis` and `num_or_size_splits` "
                "should be constants"
            )
        return _op.split(
            input_node, indices_or_sections=indices_or_sections, axis=int(axis_input_value)
        )

    return _impl


def _unpack():
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


def _softmax():
    def _impl(inputs, attr, params, mod):
        return AttrCvt(op_name="softmax", transforms={"axis": ("axis", 1)})([inputs[0]], attr)

    return _impl


def _softsign():
    # op description: https://www.tensorflow.org/api_docs/python/tf/math/softsign
    def _impl(inputs, attr, params, mod):
        abs_out = get_relay_op("abs")(inputs[0])
        add_out = abs_out + tvm.relay.const(1, attr["T"].name)
        return inputs[0] / add_out

    return _impl


def _softplus():
    # op description: https://www.tensorflow.org/api_docs/python/tf/math/softplus
    def _impl(inputs, attr, params, mod):
        exp_out = AttrCvt("exp")(inputs, attr)
        inputs.append(tvm.relay.const(1, attr["T"].name))
        rh = tvm.relay.const(1, attr["T"].name)
        add_out = get_relay_op("add")(exp_out, rh)
        return get_relay_op("log")(add_out)

    return _impl


def _topk():
    def _impl(inputs, attr, params, mod):
        k_input = inputs.pop(1)
        try:
            k = int(_get_num_param(params, k_input))
        except (IndexError, KeyError, AttributeError):
            try:
                k = int(_infer_value(k_input, params, mod).numpy().tolist())
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


def _floordiv():
    def _impl(inputs, attr, params, mod):
        assert len(inputs) == 2
        return AttrCvt("floor_divide")(inputs, attr)

    return _impl


def _floormod():
    def _impl(inputs, attr, params, mod):
        assert len(inputs) == 2
        return AttrCvt("floor_mod")(inputs, attr)

    return _impl


def _logical(name):
    def _impl(inputs, attr, params, mod):
        return AttrCvt(op_name=name)(inputs, attr)

    return _impl


def _space_to_batch_nd():
    def _impl(inputs, attr, params, mod):
        block_shape = _get_list_param(params, inputs[1], mod)

        paddings = _get_list_param(params, inputs[2], mod)
        paddings = np.squeeze(paddings)
        if len(paddings.shape) == 1:
            paddings = np.expand_dims(paddings, axis=0)
        paddings = paddings.tolist()

        attr["block_shape"] = block_shape
        attr["paddings"] = paddings
        out = AttrCvt("space_to_batch_nd", ignores=["Tblock_shape", "Tpaddings"])([inputs[0]], attr)

        return out

    return _impl


def _batch_to_space_nd():
    def _impl(inputs, attr, params, mod):
        block_shape = _get_list_param(params, inputs[1], mod)

        crops = _get_list_param(params, inputs[2], mod)
        crops = np.squeeze(crops)
        if len(crops.shape) == 1:
            crops = np.expand_dims(crops, axis=0)
        crops = crops.tolist()

        attr["block_shape"] = block_shape
        attr["crops"] = crops
        out = AttrCvt("batch_to_space_nd", ignores=["Tblock_shape", "Tcrops"])([inputs[0]], attr)

        return out

    return _impl


def _atan2():
    def _impl(inputs, attr, params, mod):
        divide = _elemwise("divide")(inputs, attr, params, mod)
        return get_relay_op("atan")(divide)

    return _impl


def _prod():
    def _impl(inputs, attr, params, mod):
        axis = _get_num_param(params, inputs[1])
        keepdims = attr["keep_dims"]
        return _op.prod(inputs[0], int(axis), keepdims=keepdims)

    return _impl


def _log1p():
    # op description: https://www.tensorflow.org/api_docs/python/tf/math/log1p
    def _impl(inputs, attr, params, mod):
        one = tvm.relay.const(1, attr["T"].name)
        add_out = get_relay_op("add")(inputs[0], one)
        return get_relay_op("log")(add_out)

    return _impl


def _one_hot():
    def _impl(inputs, attr, params, mod):
        depth = int(_get_num_param(params, inputs[1]))
        dtype = attr["T"].name

        on_value = _get_num_param(params, inputs[2])
        off_value = _get_num_param(params, inputs[3])
        new_inputs = [
            inputs[0],
            tvm.relay.const(on_value, dtype),
            tvm.relay.const(off_value, dtype),
        ]
        return AttrCvt("one_hot", ignores=["TI"], extras={"depth": depth, "dtype": dtype})(
            new_inputs, attr
        )

    return _impl


def _squared_difference():
    def _impl(inputs, attr, params, mod):
        difference = _op.subtract(inputs[0], inputs[1])
        return _op.multiply(difference, difference)

    return _impl


def _size():
    def _impl(inputs, attr, params, mod):
        new_attr = attr
        new_attr["out_type"] = attr["out_type"].name
        return AttrCvt("ndarray_size", transforms={"out_type": "dtype"})(inputs, new_attr)

    return _impl


def _add_n():
    def _impl(inputs, attr, params, mod):
        if not isinstance(inputs, tuple):
            inputs = list(inputs)
        assert len(inputs) > 0, "add_n take >=1 inputs, but 0 given."
        _res = inputs[0]
        for each in inputs[1:]:
            _res = _op.add(_res, each)
        return _res

    return _impl


def _LSTMBlockCell():
    def _impl(inputs, attr, params, mod):
        """LSTM Block cell.
        Calculations and return values are described in:
        https://github.com/tensorflow/tensorflow/blob/
        r1.8/tensorflow/contrib/rnn/python/ops/lstm_ops.py#L41-L114

        Parameters
        ----------
        inputs : relay.Expr
            Input data
        in_state_c: list of relay.Expr
            Cell state input values for all the layers
        in_state_h: list of relay.Expr
            Hidden state input values for all the layers
        attrs : dict
            Dict of operator attributes
        params : dict
            List of pretrained weights and bias

        Returns
        -------
        relay.Expr.TupleWapper
            [i, cs, f, o, ci, co, h]
        """
        in_data = inputs[0]
        in_state_c = inputs[1]
        in_state_h = inputs[2]
        in_weight = inputs[3]
        in_bias = inputs[7]
        forget_bias = attr.pop("forget_bias")
        input_shape = _infer_shape(inputs[0], mod)
        weight_shape = _infer_shape(inputs[3], mod)
        batch_size, input_size = input_shape[0], input_shape[1]
        num_hidden_layers = weight_shape[1]

        in_data = _op.reshape(in_data, newshape=(batch_size, input_size))
        ixh = _op.concatenate([in_data, in_state_h], axis=1)
        in_weight = _op.transpose(in_weight, axes=None)
        gates = _op.nn.dense(ixh, in_weight, units=num_hidden_layers)
        gates_bias = _op.add(gates, in_bias)
        gate_list = _op.split(gates_bias, indices_or_sections=4, axis=1)
        in_gate = _op.sigmoid(gate_list[0])
        in_transform = _op.tanh(gate_list[1])
        forget_gate = _op.add(gate_list[2], tvm.relay.const(forget_bias, attr["T"].name))
        forget_gate = _op.sigmoid(forget_gate)
        out_gate = _op.sigmoid(gate_list[3])
        next_c = _op.add(_op.multiply(forget_gate, in_state_c), _op.multiply(in_gate, in_transform))
        co = _op.tanh(next_c)
        next_h = out_gate * co

        return tvm.relay.TupleWrapper(
            tvm.relay.Tuple([in_gate, next_c, forget_gate, out_gate, in_transform, co, next_h]), 7
        )

    return _impl


def _unique(return_counts=True):
    def _impl(inputs, attr, params, mod):
        assert len(inputs) == 1
        data = inputs[0]
        if return_counts:
            [unique, _, inverse_indices, num_uniq, counts] = _op.unique(
                data, is_sorted=False, return_counts=True
            )
            unique_sliced = _op.strided_slice(unique, begin=[0], end=num_uniq, slice_mode="size")
            counts_sliced = _op.strided_slice(counts, begin=[0], end=num_uniq, slice_mode="size")
            return _expr.TupleWrapper(
                _expr.Tuple([unique_sliced, inverse_indices, counts_sliced]),
                3,
            )
        [unique, _, inverse_indices, num_uniq] = _op.unique(
            data, is_sorted=False, return_counts=False
        )
        unique_sliced = _op.strided_slice(unique, begin=[0], end=num_uniq, slice_mode="size")
        return _expr.TupleWrapper(
            _expr.Tuple([unique_sliced, inverse_indices]),
            2,
        )

    return _impl


# compatible operators that do NOT require any conversion.
_identity_list = []

# Operators that get pruned away when the complete graph is frozen.
# These operators are not needed for inference.
_freezed_graph_pruned_op_list = [
    "ReadVariableOp",
    "ResourceGather",
    "Variable",
    "VariableV2",
    "VarHandleOp",
    "Assign",
    "AssignVariableOp",
]


# _convert_map defines maps of name to converter functor(callable)
# for 1 to 1 mapping, use Renamer if nothing but name is different
# use AttrCvt if attributes need to be converted
# for 1 to N mapping(composed), use custom callable functions
# for N to 1 mapping, currently not supported(?)
_convert_map = {
    "Abs": AttrCvt("abs"),
    "Acos": AttrCvt("acos"),
    "Acosh": AttrCvt("acosh"),
    "Add": _elemwise("add"),
    "AddN": _add_n(),
    "AddV2": _elemwise("add"),
    "All": _reduce("all"),
    "Any": _reduce("any"),
    "ArgMax": _argx(_op.argmax, "argmax"),
    "ArgMin": _argx(_op.argmin, "argmin"),
    "Asin": AttrCvt("asin"),
    "Asinh": AttrCvt("asinh"),
    "Assert": _assert(),
    "Atan": AttrCvt("atan"),
    "Atanh": AttrCvt("atanh"),
    "Atan2": _atan2(),
    "AvgPool": _pooling("avg_pool"),
    "AvgPool3D": _pool3d("avg_pool3d"),
    "BatchMatMul": _batch_matmul(),
    "BatchMatMulV2": _batch_matmul(),
    "BatchNormWithGlobalNormalization": _batch_norm(),
    "BatchToSpaceND": _batch_to_space_nd(),
    "BiasAdd": _bias_add(),
    "BroadcastTo": _broadcast_to(),
    "BroadcastArgs": _broadcast_args(),
    "Cast": _cast(),
    "Ceil": AttrCvt("ceil"),
    "CheckNumerics": _check_numerics(),
    "ClipByValue": _clip_by_value(),
    "Concat": _concat(),
    "ConcatV2": _concatV2(),
    "Conv2D": _conv("conv"),
    "Conv2DBackpropInput": _conv("conv_transpose"),
    "Conv3D": _conv3d("conv"),
    "Conv3DBackpropInputV2": _conv3d("conv_transpose"),
    "Cos": AttrCvt("cos"),
    "Cosh": AttrCvt("cosh"),
    "CropAndResize": _crop_and_resize(),
    "DecodeJpeg": _decode_image(),
    "DepthToSpace": _depth_to_space(),
    "DepthwiseConv2dNative": _conv("depthwise"),
    "Dilation2D": _dilation2d(),
    "Elu": _elu(),
    "Equal": _broadcast("equal"),
    "Erf": AttrCvt("erf"),
    "EuclideanNorm": _euclidean_norm(),
    "Exp": AttrCvt("exp"),
    "ExpandDims": _expand_dims(),
    "Expm1": _expm1(),
    "Fill": _fill(),
    "Floor": AttrCvt("floor"),
    "FloorDiv": _floordiv(),
    "FloorMod": _floormod(),
    "FusedBatchNorm": _fused_batch_norm(),
    "FusedBatchNormV2": _fused_batch_norm(),
    "FusedBatchNormV3": _fused_batch_norm(),
    "Gather": _gather(),
    "GatherNd": _gather_nd(),
    "GatherV2": _gather(),
    "Greater": _broadcast("greater"),
    "GreaterEqual": _broadcast("greater_equal"),
    "Identity": _identity(),
    "IdentityN": _identityn(),
    "IsFinite": AttrCvt("isfinite"),
    "IsInf": AttrCvt("isinf"),
    "IsNan": AttrCvt("isnan"),
    "LeakyRelu": AttrCvt("leaky_relu"),
    "LeftShift": AttrCvt("left_shift"),
    "Less": _broadcast("less"),
    "LessEqual": _broadcast("less_equal"),
    "Log": AttrCvt("log"),
    "Log1p": _log1p(),
    "LogicalAnd": _logical("logical_and"),
    "LogicalNot": _logical("logical_not"),
    "LogicalOr": _logical("logical_or"),
    "LogSoftmax": AttrCvt("log_softmax"),
    "LRN": _lrn(),
    "LSTMBlockCell": _LSTMBlockCell(),
    "MatMul": _matmul(),
    "Max": _reduce("max"),
    "Maximum": _elemwise("maximum"),
    "MaxPool": _pooling("max_pool"),
    "MaxPool3D": _pool3d("max_pool3d"),
    "Mean": _mean(),
    "Min": _reduce("min"),
    "Minimum": _elemwise("minimum"),
    "MirrorPad": _mirror_pad(),
    "Mod": _elemwise("mod"),
    "Mul": _elemwise("multiply"),
    "Neg": AttrCvt("negative"),
    "NonMaxSuppressionV2": _nms(),
    "NonMaxSuppressionV3": _nms(),
    "NonMaxSuppressionV4": _nms(),
    "NonMaxSuppressionV5": _nms(True),
    "CombinedNonMaxSuppression": _combined_nms(),
    "NoOp": _no_op(),
    "NotEqual": _broadcast("not_equal"),
    "OneHot": _one_hot(),
    "Pack": _pack(),
    "Pad": _pad("Pad"),
    "PadV2": _pad("PadV2"),
    "Pow": _elemwise("power"),
    "Prod": _prod(),
    "Range": _range(),
    "Rank": _rank(),
    "RealDiv": _elemwise("divide"),
    "Relu": AttrCvt("relu"),
    "Relu6": _relu6(),
    "Reshape": _reshape(),
    "ResizeBicubic": _resize("bilinear"),
    "ResizeBilinear": _resize("bilinear"),
    "ResizeNearestNeighbor": _resize("nearest_neighbor"),
    "ReverseV2": _reverse_v2(),
    "RightShift": AttrCvt("right_shift"),
    "Rint": AttrCvt("round"),
    "Round": AttrCvt("round"),
    "Rsqrt": _rsqrt(),
    "Select": _where(),
    "SelectV2": _where(),
    "Selu": _selu(),
    "Shape": _shape(),
    "Sigmoid": AttrCvt("sigmoid"),
    "Sign": AttrCvt("sign"),
    "Sin": AttrCvt("sin"),
    "Sinh": AttrCvt("sinh"),
    "Size": _size(),
    "Slice": _slice(),
    "Softmax": _softmax(),
    "Softplus": _softplus(),
    "Softsign": _softsign(),
    "SpaceToBatchND": _space_to_batch_nd(),
    "SpaceToDepth": _space_to_depth(),
    "SparseToDense": _sparse_to_dense(),
    "SparseTensorDenseMatMul": _sparse_tensor_dense_matmul(),
    "SparseFillEmptyRows": _sparse_fill_empty_rows(),
    "SparseReshape": _sparse_reshape(),
    "SegmentSum": _math_segment_sum(),
    "SparseSegmentSum": _sparse_segment_sum(),
    "SparseSegmentSumWithNumSegments": _sparse_segment_sum_with_num_segments(),
    "SparseSegmentSqrtN": _sparse_segment_sum_sqrtn(),
    "SparseSegmentSqrtNWithNumSegments": _sparse_segment_sum_sqrtn_with_num_segments(),
    "SparseSegmentMean": _sparse_segment_mean(),
    "SparseSegmentMeanWithNumSegments": _sparse_segment_mean_with_num_segments(),
    "SparseTensorDenseAdd": _sparse_tensor_dense_add(),
    "Split": _split(False),
    "SplitV": _split(True),
    "Sqrt": AttrCvt("sqrt"),
    "Square": _square(),
    "SquaredDifference": _squared_difference(),
    "Squeeze": _squeeze(),
    "StopGradient": _identity(),
    "StridedSlice": _stridedSlice(),
    "Sub": _elemwise("subtract"),
    "Sum": _sum(),
    "Tan": AttrCvt("tan"),
    "Tanh": AttrCvt("tanh"),
    "TensorArrayConcatV3": _tensor_array_concat(),
    "TensorArrayGatherV3": _tensor_array_gather(),
    "TensorArrayReadV3": _tensor_array_read(),
    "TensorArrayScatterV3": _tensor_array_scatter(),
    "TensorArraySizeV3": _tensor_array_size(),
    "TensorArraySplitV3": _tensor_array_split(),
    "TensorArrayV3": _tensor_array(),
    "TensorArrayWriteV3": _tensor_array_write(),
    "Tile": _tile(),
    "TopKV2": _topk(),
    "Transpose": _transpose(),
    "TruncateMod": _elemwise("mod"),
    "Unique": _unique(False),
    "UniqueWithCounts": _unique(True),
    "Unpack": _unpack(),
    "UnravelIndex": _unravel_index(),
    "Where": _where(),
    "ZerosLike": AttrCvt("zeros_like"),
}

# An internal list to contain all the control flow primitives used in Tensorflow
# 1.x.
_control_flow_nodes = ["Merge", "Switch", "NextIteration", "Exit", "Enter", "LoopCond"]

# A map to record tensor array write ops and input ta/tensor indices
# Value is (index of tensor array, index of written node)
_tensor_array_write_ops = {
    "TensorArrayWrite": (3, 2),
    "TensorArrayScatter": (0, 2),
    "TensorArraySplit": (0, 1),
}


def is_tensor_array_constuctor(tf_node):
    """Check whether is tensor array constructor node."""
    is_ta = False
    ta_start = "TensorArrayV"
    if tf_node.op.startswith(ta_start):
        is_ta = tf_node.op[len(ta_start)].isnumeric()
    return is_ta


def find_parent_loop_name(node_name, while_loop_name_set):
    """Find name of direct parent while loop."""
    ploop_name = ""
    name_prefix = node_name.rsplit("/", 1)[0]
    if name_prefix.startswith("^"):
        name_prefix = name_prefix[1:]
    for lname in while_loop_name_set:
        if name_prefix.startswith(lname) and len(ploop_name) < len(lname):
            ploop_name = lname

    if len(ploop_name) == 0:
        ploop_name = name_prefix

    return ploop_name


def _in_while_loop(control_flow_node_map, op_name):
    """
    Check if a given control flow operator is part of a while loop execution
    frame. This is based on the fact that there is only one occurrence of
    `LoopCond` for a loop execution frame and it is only presented in the loop
    construct.

    Parameters
    ----------
    control_flow_node_map : Dict[str, Set[str]]
        A dictionay contains the unique control flow execution frame name to
        a set of primitive operators mapping.

    op_name : str
        The name of a control flow primitive.

    Returns
    -------
    ret : bool
        Return true if the operator is in a while loop execution frame,
    otherwise, return false.
    """
    return op_name in control_flow_node_map and "LoopCond" in control_flow_node_map[op_name]


class RewriteSubgraph(ExprMutator):
    """
    A helper class to rewrite expr in while loop function to variable.

    Parameters
    ----------
    rewrite_map : Dict[expr, expr]
        A dictionay contains a set of expr to var mapping.
    """

    def __init__(self, rewrite_map):
        ExprMutator.__init__(self)
        self.rewrite_map = rewrite_map

    def visit(self, expr):
        if expr in self.rewrite_map:
            return self.rewrite_map[expr]
        return super().visit(expr)


def rewrite_subgraph(expr, rewrites):
    """Rewrite loop body."""
    return RewriteSubgraph(rewrites).visit(expr)


class Branch:
    """A class contains the components that are used to build up a Relay if
    node.

    Parameters
    ----------
    cond : tvm.relay.Expr
        The condition of a if node.

    true_branch : tvm.relay.Expr
        The body of the true branch of a if expression.

    false_branch: tvm.relay.Expr
        The body of the false branch of a if expression.

    _if : tvm.relay.Expr
        An internal variable indicates where an if expression is already created
        for a matched TF condition construct.

    Examples
    --------
    The following is a cond statement written in TensorFlow:

    .. code-block:: python

        def vanilla_cond():
            i = tf.constant(1)
            j = tf.constant(4)

             def f1():
                return tf.multiply(1, 17)

             def f2():
                return tf.add(4, 23)
            r = tf.cond(tf.less(i, j), f1, f2)

    This condition statement should be converted into Relay in the following
    form:

    .. code-block:: python

        fn (%Const: Tensor[(1,), int32],
            %Const_1: Tensor[(1,), int32],
            %cond/Mul/x: Tensor[(1,), int32],
            %cond/Mul/y: Tensor[(1,), int32],
            %cond/Add/x: Tensor[(1,), int32],
            %cond/Add/y: Tensor[(1,), int32]) {
          %0 = less(%Const, %Const_1) # ty=Tensor[(1,), bool]
          %1 = min(%0)
          if (%1) {
            %2 = multiply(%cond/Mul/x, %cond/Mul/y)
            %2
          }  else {
            %3 = add(%cond/Add/x, %cond/Add/y)
            %3
          }
        }
    """

    def __init__(self):
        self._if = None
        self.cond = None
        self.true_branch = None
        self.false_branch = None

    def _if_node(self):
        """An internal API to create a relay if node from the matched TF
        condition construct.
        """
        # `cond`  returns a tensor that contains boolean values. We add a `min`
        # operator to checks if there is any false value. If so, this condition
        # doesn't not hold.
        cond = tvm.relay.op.min(self.cond)
        return tvm.relay.If(cond, self.true_branch, self.false_branch)

    def if_node(self):
        """Create an tvm.relay.If node if it hasn't been created yet."""
        if self._if is None:
            self._if = self._if_node()
        return self._if


class VarChecker(ExprVisitor):
    """Check whether a Variable is used in loop body.

    Parameters
    ----------
    var : relay.expr.Var
        Relay Variable to be checked.
    """

    def __init__(self, var):
        ExprVisitor.__init__(self)
        self._var = var
        self.used = False

    def visit(self, expr):
        if self._var == expr:
            self.used = True
        super().visit(expr)


class Loop:
    """
    A class contains the components that are used to build up a Relay
    recursive call.
    Parameters
    ----------
    mod : tvm.IRModule
        Module for current parsed IR.

    loop_name : str
        Name prefix of while loop in TensorFlow graph.

    lvar2expr : dict from str to dict from Relay.expr.Var to Relay.expr
        A dictionary recording all loop vars and corresponding
        relay expression.

    Examples
    --------
    The following is a vanilla loop from TensorFlow:
    .. code-block:: python
        i = tf.constant(0)
        c = lambda i: tf.less(i, 10)
        b = lambda i: tf.add(i, 1)
        r = tf.while_loop(c, b, [i])
    It will be converted to the following recursive call in Relay:
    .. code-block:: python
        fn (%while/Less/y: Tensor[(1,), int32],
            %while/Add/y: Tensor[(1,), int32],
            %Const: Tensor[(1,), int32]) {
          %0 = fn(%loop_var0: Tensor[(1,), int32]) {
            %1 = less(%loop_var0, %while/Less/y)
            %2 = min(%1)
            if (%2) {
              %3 = add(%loop_var0, %while/Add/y)
              free_var %while_loop
              %4 = %while_loop(%3)
              %4
            }    else {
              %5 = (%loop_var0,)
              %5
            }
          }
          let %while_loop1 = %0
          %6 = %while_loop1(%Const)
          %6
        }
    """

    def __init__(self, mod, loop_name, lvar2expr):
        self.cond = None
        self.body = []
        self._loop = None
        self._mod = mod
        self._loop_name = loop_name
        self._lvar2expr = lvar2expr
        self.loop_vars = []

        self.aligned = False

    def _while_loop(self):
        """An internal API to create a Relay recursive call for a matched TF
        `while_loop` construct.
        """
        bind_map = {}
        wl = tvm.relay.var("while_loop")
        sb = tvm.relay.scope_builder.ScopeBuilder()

        lv_list = []
        expr_list = []
        extra_vars = []

        for i, lv in enumerate(self.loop_vars):
            if self._loop_name not in self._lvar2expr:
                self._lvar2expr[self._loop_name] = {}

            # Handle the case when loop var is not properly lifted.
            # This can happen when loop var node name is set accidentally
            # beginning with loop name.
            if lv not in self._lvar2expr[self._loop_name]:
                var_name = "{}_loop_var_{}".format(self._loop_name, i)
                var_type = _infer_type(lv, self._mod).checked_type
                loop_var = tvm.relay.var(var_name, type_annotation=var_type)
                self._lvar2expr[self._loop_name][loop_var] = lv
                bind_map[lv] = loop_var
                self.loop_vars[i] = loop_var
                lv = loop_var

            lv_list.append(lv)
            expr_list.append(self._lvar2expr[self._loop_name][lv])

        if bind_map:
            self.cond = rewrite_subgraph(self.cond, bind_map)
            self.body = [rewrite_subgraph(b, bind_map) for b in self.body]

        cond = tvm.relay.op.min(self.cond)

        for lv, exp in self._lvar2expr[self._loop_name].items():
            if lv not in self.loop_vars:
                var_checker = VarChecker(lv)
                for bd in self.body + [cond]:
                    var_checker.visit(bd)
                    if var_checker.used:
                        lv_list.append(lv)
                        expr_list.append(exp)
                        extra_vars.append(lv)
                        break

        with sb.if_scope(cond):
            sb.ret(wl(*list(self.body + extra_vars)))
        with sb.else_scope():
            sb.ret(tvm.relay.Tuple(lv_list))

        loop_fn = tvm.relay.Function(lv_list, sb.get())
        sb = tvm.relay.scope_builder.ScopeBuilder()
        sb.let(wl, loop_fn)
        loop_ret = wl(*expr_list)

        sb.ret(loop_ret)
        ret = sb.get()
        return ret

    def while_loop(self):
        """Instantiate a while loop if it has not been created yet."""
        if self._loop is None:
            self._loop = self._while_loop()
            return self._loop
        return self._loop


class GraphProto(object):
    """A helper class for handling relay graph copying from Tensorflow GraphDef.
    Definition:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto
    """

    def __init__(self):
        self._nodes = {}
        self._tf_node_map = {}
        self._params = {}
        self._input_shapes = {}
        self._output_shapes = {}
        self._num_rnn_layer = False
        self._input_shapes = {}
        self._loops = {}
        self._branches = {}
        self._mod = IRModule({})
        self._prelude = Prelude(self._mod)
        self._control_flow_node_map = defaultdict(set)
        self._loop_body_order = {}
        self._loop_var_order = {}
        self._lvar2expr = {}
        self._lname_map = {}
        self._sorted_cf_node_names = []
        self._while_loop_name_set = set()
        self._main_graph_proto = self
        self._tensor_array_shapes = {}
        self._tensor_array_shape_nodes = {}

    def _get_relay_func(self, graph, layout="NHWC", shape=None, outputs=None):
        """Construct relay nodes from tensorflow graph definition - GraphDef.

        Follow the tensorflow graph definition to parse and convert it to Relay.
        Some of the assumptions listed below.

            -> All Placeholders are considered as graph input.
            -> All Const nodes are params.
            -> Last node is assumed as graph output.
            -> _output_shapes : Graph should be frozen with add_shapes=True.
                                Or user can pass input shape dictionary optionally.
            -> DecodeJpeg, ResizeBilinear: These are dummy operators.
                                           Hence user should handle preprocessing outside.
            -> CheckNumerics: No implementation as of now for this.
                              Just copies input to output.

        Parameters
        ----------
        graph : tensorflow graph definition object
            The loaded tensorflow GraphDef

        layout : target layout to be used (Optional)
            NCHW only supported now to enable NHWC models on GPU.

        shape : Dictionary of input dimensions (Optional)
            Graph level input shape dictionary.

        outputs : List of output tensor names (Optional)
            if not specified then the last node is assumed as graph output.

        Returns
        -------
        mod : tvm.IRModule
            The module that optimizations will be performed on.

        params : dict
            A dict of name: tvm.nd.array pairs, used as pretrained weights
        """
        try:
            from tensorflow.python.framework import tensor_util
        except ImportError as e:
            raise ImportError("Unable to import tensorflow which is required {}".format(e))

        missing_operators = self._parse_import_prerequisites(graph)
        control_flow_nodes = []
        ta_write_nodes = []
        ta_gather_nodes = []
        ta_construct_nodes = []
        self._in_shape = shape
        self._layout = layout
        self._graph = graph

        if missing_operators:
            freezed_ops = [op for op in missing_operators if op in _freezed_graph_pruned_op_list]
            if freezed_ops:
                raise Exception(
                    "Graph is not frozen. Provide a frozen graph. "
                    "Found operators {}".format(freezed_ops)
                )

            raise NotImplementedError(
                "The following operators are not implemented: {}".format(missing_operators)
            )

        for node in graph.node:
            node_name_prefix = node.name.rsplit("/", 1)[0]
            self._control_flow_node_map[node_name_prefix].add(node.op)
            self._tf_node_map[node.name] = node

            # Parse output_shapes attribute
            parsed_attr = self._parse_attr(node.attr)
            if "_output_shapes" in parsed_attr:
                self._output_shapes[node.name] = [
                    tensor_util.TensorShapeProtoToList(tshape)
                    for tshape in parsed_attr["_output_shapes"]
                ]
            else:
                self._output_shapes[node.name] = [None]

            # Parse placeholder and const here since input shape info is required.
            if node.op == "Placeholder" or node.op == "PlaceholderWithDefault":
                # Give priority to user argument.
                if shape and node.name in shape:
                    self._input_shapes[node.name] = list(shape[node.name])
                else:
                    self._input_shapes[node.name] = tensor_util.TensorShapeProtoToList(
                        node.attr["shape"].shape
                    )
                    for idx, dim in enumerate(self._input_shapes[node.name]):
                        if dim < 0:
                            self._input_shapes[node.name][idx] = Any()

                self._output_shapes[node.name] = [self._input_shapes[node.name]]
                attr = self._parse_attr(node.attr)
                self._nodes[node.name] = [
                    _expr.var(
                        node.name, shape=self._input_shapes[node.name], dtype=attr["dtype"].name
                    )
                ]

                # Ignore user's input shape for Non placeholder
            elif node.op == "Const":
                tensor_value = node.attr["value"].tensor
                self._input_shapes[node.name] = tensor_util.TensorShapeProtoToList(
                    tensor_value.tensor_shape
                )
                self._output_shapes[node.name] = [self._input_shapes[node.name]]
                if shape and node.name in shape:
                    warnings.warn(
                        "Ignore the passed shape. Shape in graphdef "
                        "will be used for operator %s." % node.name
                    )
                for key, value in node.attr.items():
                    self._parse_param(key, value, node.name, self._in_shape)
            elif node.op in _control_flow_nodes:
                # We assume that the direct parent node of Exit is a while loop block
                if node.op == "Exit":
                    self._while_loop_name_set.add(node_name_prefix)
                control_flow_nodes.append(node)
            elif node.op.startswith("TensorArray"):
                if is_tensor_array_constuctor(node):
                    ta_construct_nodes.append(node)
                else:
                    for ta_write_name, idx in _tensor_array_write_ops.items():
                        if node.op.startswith(ta_write_name):
                            ta_write_nodes.append((node, idx))
                            break
                    if node.op.startswith("TensorArrayGather"):
                        ta_gather_nodes.append(node)

        # Use tensor array gather to infer static tensor array shape
        for gather_node in ta_gather_nodes:
            input_ta_name = gather_node.input[0]
            input_ta_node = self._tf_node_map[input_ta_name]
            if is_tensor_array_constuctor(input_ta_node):
                gather_attr = self._parse_attr(gather_node.attr)
                if "element_shape" not in gather_attr:
                    continue
                raw_elem_shape = tensor_util.TensorShapeProtoToList(gather_attr["element_shape"])
                elem_shape = []
                for dim in raw_elem_shape:
                    if dim < 0:
                        elem_shape.append(Any())
                    else:
                        elem_shape.append(int(dim))
                self._tensor_array_shapes[input_ta_node.name] = elem_shape

        # Fetch node contains static tensor array shape
        for item in ta_write_nodes:
            wnode = item[0]
            ta_idx, inode_idx = item[1]

            stack = [self._tf_node_map[wnode.input[ta_idx].split(":")[0]]]
            while stack:
                cnode = stack.pop(0)
                if not cnode.op.startswith("TensorArray"):
                    for iname in cnode.input:
                        stack.append(self._tf_node_map[iname.split(":")[0]])
                elif cnode.name != wnode.name:
                    if is_tensor_array_constuctor(cnode):
                        inode = self._tf_node_map[wnode.input[inode_idx].split(":")[0]]
                        tn = wnode.input[inode_idx].split(":")
                        output_index = int(tn[1]) if len(tn) > 1 else 0
                        self._tensor_array_shape_nodes[cnode.name] = (inode, wnode.op, output_index)
                    break

        # First, parse all control flow nodes.
        # Convert tf.cond to Branch and tf.while_loop to Loop.
        sorted_cf_nodes = []
        exit_pos_map = {}
        ordered_prefix = []
        # Sort control flow nodes to move all Exit nodes to the end
        # of corresponding while_loop block.
        for node in control_flow_nodes:
            loop_name = find_parent_loop_name(node.name, self._while_loop_name_set)
            if node.op == "Exit":
                if loop_name not in exit_pos_map:
                    ordered_prefix.append(loop_name)
                    exit_pos_map[loop_name] = len(sorted_cf_nodes)
                sorted_cf_nodes.append(node)
            elif loop_name in self._while_loop_name_set:
                if loop_name not in exit_pos_map:
                    sorted_cf_nodes.append(node)
                else:
                    sorted_cf_nodes.insert(exit_pos_map[loop_name], node)
                    for j in range(ordered_prefix.index(loop_name), len(ordered_prefix)):
                        exit_pos_map[ordered_prefix[j]] += 1
            else:
                sorted_cf_nodes.append(node)

        for node in sorted_cf_nodes:
            self._sorted_cf_node_names.append(node.name)

        for node in sorted_cf_nodes:
            self._backtrack_construct(node.name)

        # Second, parse other nodes to re-create TF graph using Relay operators.
        for node in graph.node:
            self._backtrack_construct(node.name)

        out = []
        if outputs is None:
            last_node = graph.node[-1]
            op = self._nodes[last_node.name.split(":")[0]]
            if last_node.op == "Exit":
                out = [op[0].tuple_value]
            else:
                out = op
        else:
            for out_name in outputs:
                if ":" in out_name:
                    out_name, out_num = out_name.split(":")
                    out_num = int(out_num)
                    out.append(self._nodes[out_name][out_num])
                else:
                    out.append(self._nodes[out_name][0])

        if isinstance(out, _expr.TupleWrapper):
            out = out.tuple_value
        else:
            out = out[0] if len(out) == 1 else _expr.Tuple(out)
        fvars = analysis.free_vars(out)
        func = _function.Function(fvars, out)
        final_params = {}
        for fv in fvars:
            if fv.name_hint in self._params:
                final_params[fv.name_hint] = self._params[fv.name_hint]
        self._params = final_params
        return func

    def from_tensorflow(self, graph, layout="NHWC", shape=None, outputs=None):
        """Wrapper to _get_relay_func which converts Tensorflow graph to Relay function
        which is used as main function for the Relay module
        """
        func = self._get_relay_func(graph, layout=layout, shape=shape, outputs=outputs)
        self._mod["main"] = func
        return self._mod, self._params

    def _parse_import_prerequisites(self, graph):
        """Calculate the named preconditions from TensorFlow `graph`.
        Return prerequisites for parsing:
        a. Set of operator names which don't have their mapping in TVM, i.e.
            which are not supported
        """
        missing_operators = set()
        from tensorflow.python.framework import op_def_registry

        for node in graph.node:
            getOpDef = (
                op_def_registry._registered_ops.get
                if hasattr(op_def_registry, "_registered_ops")
                else op_def_registry.get
            )
            op_def = getOpDef(node.op)
            if node.op == "Placeholder" or node.op == "PlaceholderWithDefault":
                pass
            elif node.op == "Const":
                pass
            elif node.op in ["PartitionedCall", "StatefulPartitionedCall"]:
                pass
            else:
                if any([node.op in t for t in [_identity_list, _convert_map, _control_flow_nodes]]):
                    pass
                elif op_def is not None and op_def.is_stateful:
                    missing_operators.add(node.op)
                else:
                    missing_operators.add(node.op)

        return missing_operators

    def _parse_param(self, key, value, name, shape):
        try:
            from tensorflow.python.framework import tensor_util
        except ImportError as e:
            raise ImportError("Unable to import tensorflow which is required {}".format(e))

        if key == "value":
            np_array = tensor_util.MakeNdarray(value.tensor)

            if np_array.dtype == np.dtype(object):
                # Object types are generally tensorflow DT_STRING (DecodeJpeg op).
                # Just leave it as placeholder.
                if shape and name in shape:
                    var_shape = shape[name]
                else:
                    var_shape = tensor_util.TensorShapeProtoToList(value.tensor.tensor_shape)
                self._nodes[name] = [_expr.var(name, shape=var_shape, dtype="uint8")]
                return

            array_ndim = len(np_array.shape)
            if array_ndim == 0:
                self._nodes[name] = [tvm.relay.const(np_array, np_array.dtype)]
            else:
                self._params[name] = tvm.nd.array(np_array)
                self._nodes[name] = [
                    _expr.var(name, shape=self._params[name].shape, dtype=self._params[name].dtype)
                ]
        else:
            if key not in ("dtype", "_output_shapes", "_class"):
                raise NotImplementedError(
                    "Other attributes for a Const(param) Node {} ? .".format(key)
                )

    def _get_attr(self, buf):
        """Returns the value of the attr of this buf with the given `name`.

        Args:
          buf: attrvalue protobuf.

        Returns:
          The value of the attr, as a Python object.

        Raises:
          ValueError: If this op does not have an attr with the given `name`.
        """
        fields = ["s", "i", "f", "b", "type", "shape", "tensor", "func"]

        x = buf

        ret = []

        try:
            from tensorflow.python.framework import dtypes
        except ImportError as e:
            raise ImportError("Unable to import tensorflow which is required {}".format(e))

        # Treat an empty oneof value as an empty list.
        if not x.WhichOneof("value"):
            return ret
        if x.HasField("list"):
            for f in fields:
                if getattr(x.list, f):
                    if f == "type":
                        ret += [dtypes.as_dtype(x) for x in list(getattr(x.list, f))]
                    else:
                        ret += list(getattr(x.list, f))
        else:
            for f in fields:
                if x.HasField(f):
                    if f == "type":
                        ret = dtypes.as_dtype(getattr(x, f))
                    else:
                        ret = getattr(x, f)
        return ret

    def _parse_attr(self, attr_proto):
        """Convert a list of AttributeProto to a dict, with names as keys."""
        attrs = {}
        for key, value in attr_proto.items():
            attrs[key] = self._get_attr(value)

        return attrs

    def _convert_control_flow_operator(self, node, inputs, attrs, control_flow_node_map):
        """
        Convert the Relay control flow primitive into corresponding component
        of a Relay control flow construct, i.e. `tf.cond` and `tf.while_loop`
        are converted in Relay `If` and recusrive call, respectively.

        Parameters
        ----------
        node: TensorFlow graph node object.
            A TensorFlow graph node object.

        inputs : List[tvm.relay.Expr]
            List of input symbols.

        attrs : Dict[tvm.Attrs]
            Dict of operator attributes.

        control_flow_node_map : Dict[str, Set[str]]
            A dictionary contains the execution frame name to primitives
            mapping.

        Returns
        -------
        op : tvm.relay.Expr
            Converted relay expression.
        """
        node_name_prefix = node.name.rsplit("/", 1)[0]
        plname = find_parent_loop_name(node.name, self._while_loop_name_set)
        if node.op == "Merge":
            if _in_while_loop(self._control_flow_node_map, node_name_prefix):
                op = self._licm_construct(plname, node.input[0])
                if node_name_prefix not in self._loops:
                    self._loops[node_name_prefix] = Loop(self._mod, plname, self._lvar2expr)
            else:
                if node_name_prefix not in self._branches:
                    switch_prefix = node_name_prefix + "/Switch"
                    merge_idx = self._sorted_cf_node_names.index(node.name)
                    for i in range(merge_idx - 1, -1, -1):
                        cf_name = self._sorted_cf_node_names[i]
                        if cf_name.startswith(switch_prefix):
                            self._backtrack_construct(cf_name)
                            break

                branch = self._branches[node_name_prefix]
                false_br = self._licm_construct(plname, node.input[0])
                true_br = self._licm_construct(plname, node.input[1])
                branch.true_branch = true_br
                branch.false_branch = false_br
                op = branch.if_node()
                if node_name_prefix not in self._while_loop_name_set:
                    try:
                        cond_val = np.all(
                            _infer_value(branch.cond, self._params, self._mod).numpy()
                        )
                        if cond_val:
                            op = branch.true_branch
                        else:
                            op = branch.false_branch
                    except Exception:
                        op = branch.if_node()
        elif node.op == "Exit":
            loop = self._loops[node_name_prefix]

            # Check whether the order of loop variables aligns
            # with loop body. If not, create new loop variable list
            # with correct order.
            if not loop.aligned:
                loop_vars = []
                for i in self._loop_body_order[node_name_prefix]:
                    for j, k in enumerate(self._loop_var_order[node_name_prefix]):
                        if k == i:
                            loop_vars.append(loop.loop_vars[j])
                loop.loop_vars = loop_vars
                loop.aligned = True
            exit_name = node.name.split("/")[-1]
            if "_" in exit_name:
                exit_number = int(exit_name[5:])
            else:
                exit_number = 0
            expr = loop.while_loop()
            body_pos = exit_number
            for i, j in enumerate(self._loop_body_order[node_name_prefix]):
                if exit_number == j:
                    body_pos = i
                    break
            op = _expr.TupleGetItem(expr, body_pos)
        elif node.op == "Enter":
            op = self._licm_construct(plname, node.input[0])
        elif node.op == "LoopCond":
            op = self._licm_construct(plname, node.input[0])
            self._loops[node_name_prefix].cond = op
        elif node.op == "Switch":
            op = self._licm_construct(plname, node.input[0])
            cond = self._licm_construct(plname, node.input[1])
            if _in_while_loop(self._control_flow_node_map, node_name_prefix):
                if node_name_prefix not in self._loop_var_order:
                    self._loop_var_order[node_name_prefix] = []
                if node.name.endswith("Switch"):
                    self._loop_var_order[node_name_prefix].append(0)
                else:
                    self._loop_var_order[node_name_prefix].append(
                        int(node.name.split("Switch_")[-1])
                    )
                self._loops[node_name_prefix].loop_vars.append(op)
            else:
                if node_name_prefix not in self._branches:
                    self._branches[node_name_prefix] = Branch()
                self._branches[node_name_prefix].cond = cond
        elif node.op == "NextIteration":
            if node_name_prefix not in self._loop_body_order:
                self._loop_body_order[node_name_prefix] = []
            if node.name.endswith("NextIteration"):
                self._loop_body_order[node_name_prefix].append(0)
            else:
                self._loop_body_order[node_name_prefix].append(
                    int(node.name.split("NextIteration_")[-1])
                )
            op = self._licm_construct(plname, node.input[0])
            self._loops[node_name_prefix].body.append(op)
        else:
            raise Exception("Cannot identify control flow operator: " + "{}".format(node.op))

        return op

    def _partition_call_operator(self, inputs, attr):
        """
        Convert the Relay Partition call ops into Relay Function calls and
        function definitions from Tensorflow graph library attribute to Relay global
        functions

        Parameters
        ----------
        node: TensorFlow graph node object.
            A TensorFlow graph node object.

        inputs : List[tvm.relay.Expr]
            List of input symbols.

        attrs : Dict[tvm.Attrs]
            Dict of operator attributes.

        Returns
        -------
        op : tvm.relay.Expr
            Converted relay expression.
        """

        try:
            from tensorflow.python.framework import function_def_to_graph
        except ImportError as e:
            raise ImportError("Unable to import tensorflow which is required {}".format(e))

        main_graph_proto = self._main_graph_proto
        outer_graph_def = main_graph_proto._graph

        node_func_name = attr.get("f").name
        func = next(
            (f for f in outer_graph_def.library.function if f.signature.name == node_func_name),
            None,
        )
        if func:
            devices = set(node.device for node in func.node_def)
            if len(devices) > 1:
                raise Exception(
                    "Found inconsistent Device assignment in the "
                    "Stateful Partitioned SubGraph. Rejecting "
                    "the subgraph "
                )
            # Convert function definition to graph
            func_input_shapes = func.attr["_input_shapes"].list.shape
            subgraph, _ = function_def_to_graph.function_def_to_graph_def(func, func_input_shapes)

            # Computing subgraph's input shape dictionary
            subgraph_shape_dict, input_expr_dict = {}, {}
            for f_arg, input in zip(func.signature.input_arg, inputs):
                input_expr_dict[f_arg.name] = input
                subgraph_shape_dict[f_arg.name] = _infer_shape(input, main_graph_proto._mod)

            func_name = "func_{}".format(func.signature.name)
            try:
                global_func = main_graph_proto._mod[func_name]
                sub_func = global_func
                sub_params = main_graph_proto._params
            except ValueError:
                # Construct relay nodes from the subgraph
                g1 = SubGraphProto(main_graph_proto)
                sub_func, sub_params = g1.from_tensorflow(subgraph, shape=subgraph_shape_dict)
                main_graph_proto._params.update(sub_params)
                func_expr = _function.Function(sub_func.params, sub_func.body)
                global_func = tvm.relay.GlobalVar(func_name)
                main_graph_proto._mod[global_func] = func_expr
                main_graph_proto._mod = InferType()(main_graph_proto._mod)

            param_exprs = []
            for param_expr in sub_func.params:
                # sub_params is subset of sub_func.params
                param_name = param_expr.vid.name_hint
                if param_name in input_expr_dict.keys():
                    param_exprs.append(input_expr_dict[param_name])
                elif param_name in sub_params.keys():
                    param_exprs.append(param_expr)
                else:
                    raise Exception("Input parameter {} not found".format(param_name))

            sb = tvm.relay.scope_builder.ScopeBuilder()
            loop_ret = global_func(*param_exprs)
            sb.ret(loop_ret)
            ret = sb.get()
        else:
            raise Exception("Function not found - {}".format(node_func_name))
        return ret

    def _convert_operator(
        self, op_name, node_name, inputs, attrs, identity_list=None, convert_map=None
    ):
        """Convert from Tensorflow operator to relay operator.
        The converter must specify conversions explicitly for incompatible name, and
        apply handlers to operator attributes.

        Parameters
        ----------
        op_name : str
            Operator name, such as Conv2D, AvgPool
        inputs : list of relay.op
            List of input symbols.
        attrs : dict
            Dict of operator attributes
        identity_list : list
            List of operators that don't require conversion
        convert_map : dict
            Dict of name : callable, where name is the op's name that
            require conversion to relay, callable are functions which
            take attrs and return (new_op_name, new_attrs)

        Returns
        -------
        sym : relay.op
            Converted relay operator
        """
        identity_list = identity_list if identity_list else _identity_list
        convert_map = convert_map if convert_map else _convert_map
        if op_name in identity_list:
            sym = get_relay_op(op_name)(*inputs, **attrs)
        elif op_name in convert_map:
            if _need_prelude_for_shape_inference(op_name):
                sym = convert_map[op_name](inputs, attrs, self._params, self._prelude)
            else:
                sym = convert_map[op_name](inputs, attrs, self._params, self._mod)
        elif op_name in ["PartitionedCall", "StatefulPartitionedCall"]:
            sym = self._partition_call_operator(inputs, attrs)
        else:
            raise NotImplementedError("Operator {} not implemented.".format(op_name))

        sym = self._set_span(sym, node_name)

        return sym

    @staticmethod
    def _set_span(sym, node_name):
        span = tvm.relay.Span(tvm.relay.SourceName(node_name), 0, 0, 0, 0)
        if isinstance(sym, _expr.Call) and sym.span is None:
            sym = _expr.Call(sym.op, sym.args, sym.attrs, sym.type_args, span)
        elif isinstance(sym, _expr.TupleWrapper):
            tuple_value = sym.tuple_value
            if isinstance(tuple_value, _expr.Call) and tuple_value.span is None:
                tuple_value = _expr.Call(
                    tuple_value.op, tuple_value.args, tuple_value.attrs, tuple_value.type_args, span
                )
                sym = _expr.TupleWrapper(tuple_value, sym.size)
        return sym

    def _licm_construct(self, loop_name, node_name):
        """Construct a node by considering whether it is
        loop invariant with the given while loop. If yes, we
        generate a loop Variable. Otherwise, return regular
        converted relay expression.

        Parameters
        ----------
        loop_name : str
            TensorFlow while loop name to be checked.

        node_name : str
            TensorFlow node name.

        Returns
        -------
        out : relay.Expr or relay.Var
            Converted relay expression or loop var.
        """
        actual_expr = self._backtrack_construct(node_name)
        tn = node_name.split(":")
        node_name = tn[0].split("^")[-1]
        cloop_name = find_parent_loop_name(node_name, self._while_loop_name_set)

        if loop_name in self._while_loop_name_set and not cloop_name.startswith(loop_name):
            if loop_name not in self._lvar2expr:
                self._lvar2expr[loop_name] = {}
            if loop_name not in self._lname_map:
                self._lname_map[loop_name] = {}

            if node_name not in self._lname_map[loop_name]:
                var_name = "{}_loop_var".format(node_name)
                var_type = _infer_type(actual_expr, self._mod).checked_type
                loop_var = tvm.relay.var(var_name, type_annotation=var_type)
                try:
                    extra_param = _infer_value(actual_expr, self._params, self._mod)
                    self._params[var_name] = extra_param
                except Exception:
                    pass
                self._lvar2expr[loop_name][loop_var] = actual_expr
                self._lname_map[loop_name][node_name] = loop_var
                ret = loop_var
            else:
                ret = self._lname_map[loop_name][node_name]
        else:
            ret = actual_expr

        return ret

    def _backtrack_construct(self, node_name):
        """Convert a specific tensorflow node to relay expression.

        If any of its ancestor node is not converted yet, backtrack as
        far as input node and covert all nodes on the path.

        This is required when parsing control flow nodes, since the parsing
        order may not follow the original graph def.

        Parameters
        ----------
        node_name : str
            TensorFlow node name.

        Returns
        -------
        op : relay.Expr
            Converted relay expression
        """
        try:
            from tensorflow.python.framework import tensor_util
        except ImportError as e:
            raise ImportError("Unable to import tensorflow which is required {}".format(e))

        input_op_name = node_name.split(":")[0].split("^")[-1]
        if input_op_name not in self._nodes:
            node = self._tf_node_map[input_op_name]
            attr = self._parse_attr(node.attr)

            if node.op in _control_flow_nodes:
                attr = self._parse_attr(node.attr)
                op = self._convert_control_flow_operator(
                    node, [], attr, self._control_flow_node_map
                )
            else:
                attr["_output_shapes"] = self._output_shapes[input_op_name]
                attr["_node_name"] = node.name
                attr["_target_layout"] = self._layout

                inputs = [self._backtrack_construct(iname) for iname in node.input]

                plname = find_parent_loop_name(node_name, self._while_loop_name_set)

                # For TensorArrayV3 op, we need to infer shape first
                if is_tensor_array_constuctor(node):
                    raw_elem_shape = tensor_util.TensorShapeProtoToList(attr["element_shape"])
                    elem_shape = []
                    for dim in raw_elem_shape:
                        if dim < 0:
                            elem_shape.append(Any())
                        else:
                            elem_shape.append(dim)

                    if elem_shape:
                        attr["shape"] = elem_shape
                    if attr["identical_element_shapes"] or elem_shape:
                        shape_node, wnode_op, output_index = self._tensor_array_shape_nodes[
                            node.name
                        ]
                        name = shape_node.name
                        if output_index > 0:
                            name += ":" + str(output_index)
                        converted = self._backtrack_construct(name)
                        shape = _infer_shape(converted, self._mod)
                        if wnode_op.startswith("TensorArraySplit"):
                            shape = (Any(),) + shape[1:]
                        elif wnode_op.startswith("TensorArrayScatter"):
                            shape = shape[1:]

                        if node.name in self._tensor_array_shapes:
                            preset_shape = self._tensor_array_shapes[node.name]
                            shape = _get_more_static_shape(shape, preset_shape)

                        if "shape" in attr:
                            attr["shape"] = _get_more_static_shape(shape, attr["shape"])
                        else:
                            attr["shape"] = shape

                # LICM
                if plname in self._while_loop_name_set:
                    for i, iname in enumerate(node.input):
                        actual_input = self._licm_construct(plname, iname)
                        inputs[i] = actual_input

                op = self._convert_operator(node.op, node.name, inputs, attr)
            if isinstance(op, np.ndarray):
                self._params[node.name] = tvm.nd.array(op)
                op = [
                    _expr.var(
                        node.name,
                        shape=self._params[node.name].shape,
                        dtype=self._params[node.name].dtype,
                    )
                ]

            elif isinstance(op, (_expr.Expr, _expr.TupleGetItem)):
                op = [op]

            self._nodes[input_op_name] = op

        out = self._nodes[input_op_name]

        if isinstance(out, _expr.TupleWrapper):
            tn = node_name.split(":")
            tensor_slot = int(tn[1]) if len(tn) > 1 else 0
            return out[tensor_slot]
        return out[0]


class SubGraphProto(GraphProto):
    """A helper class for handling relay subgraph copying from Tensorflow GraphDef."""

    def __init__(self, main_graph_proto):
        super().__init__()
        self._main_graph_proto = main_graph_proto  # holds main graph proto object

    def from_tensorflow(self, graph, layout="NHWC", shape=None, outputs=None):
        """Wrapper to _get_relay_func which converts Tensorflow graph to Relay function.
        Return Relay function and params
        """
        func = self._get_relay_func(graph, layout=layout, shape=shape, outputs=outputs)
        return func, self._params


def from_tensorflow(graph, layout="NHWC", shape=None, outputs=None):
    """Load tensorflow graph which is a python tensorflow graph object into relay.
    The companion parameters will be handled automatically.

    Parameters
    ----------
    graph : GraphDef object
        Tensorflow GraphDef

    layout : target layout to be used (Optional)
        NCHW only supported now to enable NHWC models on GPU.

    shape : Dictionary of input dimensions (Optional)
        Graph level input shape dictionary.

    outputs : List of output tensor names (Optional)
        if not specified then the last node is assumed as graph output.

    Returns
    -------
    mod : tvm.IRModule
        The module that optimizations will be performed on.

    params : dict of str to tvm.nd.NDArray
        Dict of converted parameters stored in tvm.nd.NDArray format
    """

    g = GraphProto()
    mod, params = g.from_tensorflow(graph, layout, shape, outputs)
    return mod, params
