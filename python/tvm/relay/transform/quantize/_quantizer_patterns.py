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

"""Patterns to quantize and how to quantize them."""

import tvm
from tvm import relay

from tvm.relay.transform.quantize import CalibrationCallback
from tvm.relay.dataflow_pattern import (
    is_op,
    wildcard,
    is_constant,
    DFPatternCallback,
    _DFPatternCallback,
)
from tvm.relay.dataflow_pattern import ffi as pattern_ffi
from tvm.relay.frontend.common import infer_type
from tvm.relay.op.nn.utils import get_pad_tuple2d
from . import _ffi as ffi


class QuantizerPattern(DFPatternCallback):
    """DFPatternCallback to rewrite patterns as quantized. Also contains extra information
    used for quantization and calibration.

    Parameters
    ----------
    calibration_callback : CalibrationCallback
        The method we will use to calibrate the nn.conv2d pattern.
    """

    # Counts the number of times we've added a scale and zp for variable naming
    # This needs to be a global variable and not initialized in __init__ because
    # each scale and zero point must be unique, even if they are created by different
    # instances.
    scales_count = 0
    zp_count = 0

    def __init__(self, calibration_callback: CalibrationCallback = None):
        super().__init__()
        self.calibration_callback = calibration_callback

    def calibrate_pattern(self, calibration_info):
        """Calculates the scale and zero points for quantizing parts of a generic pattern. By
        default, we call the calibrate_pattern method of the CalibrationCallback object that is
        passed into QuantizerPattern during initialization. However, if you want a pattern specific
        quantization method or a per-channel quantization method, you should overwrite the
        QuantizerPattern's calibrate_pattern method.

        Parameters
        ----------
        calibration_info : CalibrationInfo
            The class containing relevant information and utility functions to calibrate one
            instance of a pattern.

        Returns
        -------
        scale_zp_map : Dictionary
            A map from the names of scales and zero point variables in this pattern to their
            values.
        """
        return self.calibration_callback.calibrate_pattern(calibration_info)

    def callback(self, pre, post, node_map):
        raise NotImplementedError

    def scale(self, name, is_weight=False):
        """Helper to create the scale variable for qnn.quantize when rewriting our pattern.

        Parameters
        ----------
        name : str
            Identifier at the beginning of the scale variable.

        is_weight : bool
            Whether this scale is a weight scale or a data scale. If it is a weight scale, we
            the returned variable has shape (channels,). Only used for per-channel quantization.

        Returns
        -------
        var : relay.Var
            Relay variable for scale. If the input name is 'conv2d_data', then the name of the
            relay variable might be 'conv2d_data_scale_0'.
        """

        var = relay.var(
            str(name) + "_scale_" + str(QuantizerPattern.scales_count), shape=(), dtype="float32"
        )
        QuantizerPattern.scales_count += 1
        return var

    def zero_point(self, name):
        """Helper to create the zero point variable for qnn.quantize when rewriting our
        our pattern.

        Parameters
        ----------
        name : str
            Identifier at the beginning of the variable.

        Returns
        -------
        var : relay.Var
            Relay variable for scale. If the input name is 'conv2d_data', then the name of the
            relay variable might be 'conv2d_data_zero_pt_0'.
        """
        var = relay.var(
            str(name) + "_zero_pt_" + str(QuantizerPattern.zp_count), shape=(), dtype="int32"
        )
        QuantizerPattern.zp_count += 1
        return var

    def create_scale_zps(self, left_name, right_name):
        """Helper to create scales and zero points for binops.

        Parameters
        ----------
        left_name : str
            Identifier of the left hand side scale and zero point.

        right_name : str
            Identifier of the right hand side scale and zero point.
        """
        data_scale = self.scale(left_name)
        data_zp = self.zero_point(left_name)
        weight_scale = self.scale(right_name)
        weight_zp = self.zero_point(right_name)
        self.scale_zps = [data_scale, data_zp, weight_scale, weight_zp]


class Conv2DPattern(QuantizerPattern):
    def __init__(self, calibration_callback: CalibrationCallback = None):
        """Pattern to rewrite nn.conv2d ops as qnn.conv2d ops.

        Parameters
        ----------
        calibration_callback : CalibrationCallback
            The method we will use to calibrate this pattern.
        """
        super().__init__(calibration_callback)
        self.input = wildcard()
        self.conv_weight = wildcard()
        self.inputs = [self.input, self.conv_weight]
        self.conv2d = is_op("nn.conv2d")(self.input, self.conv_weight)
        self.pattern = self.conv2d
        self.attrs = None
        self.weight_channel_axis = None
        self.data_channel_axis = None
        self.channels = None

    def get_kernel_size(self, kernel_shape, kernel_layout):
        """Gets the size of the kernel.

        Parameters
        ----------
        kernel_shape : NDArray
            Shape of the kernel

        kernel_layout : str
            Layout of the kernel

        Returns
        -------
            kernel_size : NDArray
                Size of the kernel
        """
        if kernel_layout == "OIHW":
            kernel_size = tuple(kernel_shape[2:4])
        elif kernel_layout == "HWIO":
            kernel_size = tuple(kernel_shape[0:2])
        else:
            raise ValueError(
                "Quantizting kernel layout %s for conv2d is not yet supported."
                + "Please use OIHW or HWIO",
                kernel_layout,
            )
        return kernel_size

    def get_attrs(self, attrs, kernel_shape):
        """Constructs the attributes for qnn.conv2d.

        Parameters
        ----------
        attrs : dict
            Attributes of the original nn.conv2d

        kernel_shape : NDArray
            Shape of the kernel

        Returns
        -------
            quantized_attrs : dict
                Attributes for the qnn.conv2d
        """
        new_attr_dict = {}
        self.kernel_layout = attrs["kernel_layout"]
        data_layout = attrs["data_layout"]

        if self.kernel_layout == "OIHW":
            self.weight_channel_axis = 0
        elif self.kernel_layout == "HWIO":
            self.weight_channel_axis = 3
        else:
            raise ValueError(
                "Quantizing kernel layout %s for conv2d is not yet supported."
                + "Please use OIHW or HWIO",
                self.kernel_layout,
            )

        if data_layout == "NCHW":
            self.data_channel_axis = 1
        elif data_layout == "NHWC":
            self.data_channel_axis = 3
        else:
            raise ValueError(
                "Quantizing data layout %s for conv2d is not yet supported."
                + "Please use NCHW or NHWC",
                data_layout,
            )

        for attr in attrs.keys():
            attr_value = attrs[attr]
            if isinstance(attr_value, tvm.ir.container.Array):
                attr_value = tuple(attr_value)
            if attr == "kernel_size":
                kernel_size = attrs[attr]
                if kernel_size is None:
                    kernel_size = self.get_kernel_size(self.kernel_layout, kernel_shape)
                else:
                    kernel_size = tuple([k.value for k in attrs[attr]])
                new_attr_dict[attr] = kernel_size
            elif attr == "channels":
                self.channels = attrs[attr]
                if self.channels is None:
                    self.channels = kernel_shape[self.weight_channel_axis]
                if isinstance(self.channels, tvm.tir.expr.IntImm):
                    self.channels = self.channels.value
                new_attr_dict[attr] = self.channels
            elif attr == "padding":
                # We don't need to put padding in attr dict because we explicitly construct padding
                self.padding = attrs[attr]
            else:
                new_attr_dict[attr] = attr_value

        new_attr_dict["out_dtype"] = "int32"
        self.attrs = new_attr_dict

    def quantize_args(self):
        """Helper to quantize the arguments to the qnn.conv2d."""
        quantized_data = relay.qnn.op.quantize(
            self.args[0], self.scale_zps[0], self.scale_zps[1], axis=self.data_channel_axis
        )
        quantized_weight = relay.qnn.op.quantize(
            self.args[1], self.scale_zps[2], self.scale_zps[3], axis=self.weight_channel_axis
        )
        self.quantized_args = [quantized_data, quantized_weight]

    def create_conv(self, args, node_map):
        """Creates the qnn.conv2d.

        Parameters
        ----------
        args : List[relay.Expr]
            Quantized arguments for the qnn.conv2d.

        node_map : tvm.ir.container.Map
            Node map from DFPatternCallback's callback

        Returns
        -------
        q_conv2d : relay.Expr
            Quantized version of the pattern.
        """
        return relay.qnn.op.conv2d(*args, **self.attrs)

    def callback(self, pre, post, node_map):
        self.args = [node_map[i][0] for i in self.inputs]
        conv2d = node_map[self.conv2d][0]

        self.out_dtype = conv2d.checked_type.dtype

        self.get_attrs(conv2d.attrs, infer_type(self.args[1]).checked_type.shape)

        self.create_scale_zps("conv2d_data", "conv2d_weight")
        self.quantize_args()

        conv_scale = self.scale_zps[0] * self.scale_zps[2]  # data_scale * weight_scale

        # Conv zp is zero since QNN deals with input zps for us
        conv_zp = relay.const(0, dtype="int32")
        # args = [quantized_data, quantized_weight, data_zp, weight_zp, data_scale, weight_scale]
        args = self.quantized_args[0:2] + [self.scale_zps[i] for i in [1, 3, 0, 2]]

        if self.padding is not None:

            top, left, bottom, right = [p.value for p in get_pad_tuple2d(self.padding)]
            if self.kernel_layout == "OIHW":
                pad_width = ((0, 0), (0, 0), (top, bottom), (left, right))
            elif self.kernel_layout == "HWIO":
                pad_width = (
                    (top, bottom),
                    (left, right),
                    (0, 0),
                    (0, 0),
                )
            pad_val = 0
            args[0] = relay.op.nn.pad(args[0], pad_width, pad_val)

        # Construct quantized qnn.conv2d and dequantize
        qnn_call = self.create_conv(args, node_map)
        dequantized_call = relay.qnn.op.dequantize(
            qnn_call, conv_scale, conv_zp, out_dtype=self.out_dtype, axis=self.data_channel_axis
        )

        return dequantized_call


class Conv2DBiasAddPattern(Conv2DPattern):
    """Pattern to rewrite nn.conv2d -> nn.bias_add pattern as qnn.conv2d -> nn.bias_add.

    Parameters
    ----------
    calibration_callback : CalibrationCallback
        The method we will use to calibrate this pattern.
    """

    def __init__(self, calibration_callback: CalibrationCallback = None):
        super().__init__(calibration_callback)
        self.bias_weight = is_constant()
        self.inputs.append(self.bias_weight)
        self.add = is_op("add")(self.conv2d, self.bias_weight)
        self.bias_add = is_op("nn.bias_add")(self.conv2d, self.bias_weight)
        self.pattern = self.bias_add | self.add

    def quantize_args(self):
        """Quantizes the arguments to the nn.conv2d -> nn.bias_add pattern."""
        super().quantize_args()
        quantized_bias = relay.qnn.op.quantize(
            self.args[2], self.scale_zps[0], self.scale_zps[1], axis=0, out_dtype="int32"
        )
        self.quantized_args.append(quantized_bias)

    def create_conv(self, args, node_map):
        """Creates the qnn.dense -> nn.bias_add.

        Parameters
        ----------
        args : List[relay.Expr]
            Quantized arguments for the qnn.conv2d and bias_add.

        node_map : tvm.ir.container.Map
            Node map from DFPatternCallback's callback

        Returns
        -------
        q_conv2d : relay.Expr
            Quantized version of the pattern.
        """
        qnn_call = relay.qnn.op.conv2d(*args, **self.attrs)
        if node_map.get(self.add) is not None:
            bias_add = relay.op.add(qnn_call, self.quantized_args[2])
        else:  # self.bias_add in node_map
            bias_add = relay.op.nn.bias_add(
                qnn_call, self.quantized_args[2], axis=self.data_channel_axis
            )
        return bias_add


class DensePattern(QuantizerPattern):
    """Pattern to rewrite nn.dense pattern as qnn.dense.
    Parameters
    ----------
    calibration_callback : CalibrationCallback
        The method we will use to calibrate this pattern.
    """

    def __init__(self, calibration_callback: CalibrationCallback = None):
        super().__init__(calibration_callback)
        self.data = wildcard()
        self.weight = wildcard()
        self.inputs = [self.data, self.weight]

        self.dense = is_op("nn.dense")(self.data, self.weight)

        self.pattern = self.dense
        self.attrs = None
        self.units = None

    def get_attrs(self, attrs, weight_shape):
        """Constructs the attributes for qnn.conv2d.

        Parameters
        ----------
        attrs : dict
            Attributes of the original nn.dense

        weight_shape : NDArray
            Shape of the dense weights

        Returns
        -------
            quantized_attrs : dict
                Attributes for the qnn.conv2d
        """
        self.attrs = {}
        units = attrs["units"]
        if units is None:
            units = weight_shape[0]
        self.units = units.value
        self.attrs["units"] = self.units

    def quantize_args(self):
        """Quantizes the arguments to the nn.dense pattern."""
        # Quantize data and construct args for qnn.dense
        quantized_data = relay.qnn.op.quantize(self.args[0], self.scale_zps[0], self.scale_zps[1])
        quantized_weight = relay.qnn.op.quantize(
            self.args[1], self.scale_zps[2], self.scale_zps[3], axis=0
        )  # Axis = 0 for per channel quantization
        self.quantized_args = [quantized_data, quantized_weight]

    def create_dense(self, args, node_map):
        """Creates the qnn.dense.

        Parameters
        ----------
        args : List[relay.Expr]
            Quantized arguments for the qnn.dense.

        node_map : tvm.ir.container.Map
            Node map from DFPatternCallback's callback

        Returns
        -------
        q_dense : relay.Expr
            Quantized version of the pattern.
        """
        qnn_call = relay.qnn.op.dense(*args, **self.attrs)
        return qnn_call

    def callback(self, pre, post, node_map):
        self.args = [node_map[i][0] for i in self.inputs]
        weight = node_map[self.weight][0]

        dense = node_map[self.dense][0]
        out_dtype = dense.checked_type.dtype
        self.get_attrs(dense.attrs, infer_type(weight).checked_type.shape)
        self.create_scale_zps("dense_data", "dense_weight")
        self.quantize_args()

        # args = [quantized_data, quantized_weight, data_zp, weight_zp, data_scale, weight_scale]
        args = self.quantized_args[0:2] + [self.scale_zps[i] for i in [1, 3, 0, 2]]
        qnn_call = self.create_dense(args, node_map)

        deq_call = relay.qnn.op.dequantize(
            qnn_call,
            self.scale_zps[0] * self.scale_zps[2],
            relay.const(0, dtype="int32"),
            out_dtype=out_dtype,
            axis=1,
        )

        return deq_call


class DenseBiasAddPattern(DensePattern):
    """Pattern to rewrite nn.dense -> add and nn.dense -> nn.bias_add pattern as qnn.dense -> nn.bias_add.

    Parameters
    ----------
    calibration_callback : CalibrationCallback
        The method we will use to calibrate this pattern.
    """

    def __init__(self, calibration_callback: CalibrationCallback = None):
        super().__init__(calibration_callback)
        self.bias_weight = is_constant()
        self.inputs.append(self.bias_weight)
        self.bias_add = is_op("nn.bias_add")(self.dense, self.bias_weight)
        self.add = is_op("add")(self.dense, self.bias_weight)
        self.pattern = self.bias_add | self.add

    def quantize_args(self):
        super().quantize_args()
        quantized_bias = relay.qnn.op.quantize(
            self.args[2], self.scale_zps[0], self.scale_zps[1], axis=0, out_dtype="int32"
        )
        self.quantized_args.append(quantized_bias)

    def create_dense(self, args, node_map):
        qnn_call = relay.qnn.op.dense(*args, **self.attrs)
        if node_map.get(self.add) is not None:
            bias_add = relay.op.add(qnn_call, self.quantized_args[2])
        else:  # self.bias_add in node_map
            bias_add = relay.op.nn.bias_add(
                qnn_call, self.quantized_args[2], axis=1  # Axis is always 1 for dense
            )
        return bias_add


class AddPattern(QuantizerPattern):
    """Pattern to rewrite add as quantized.

    Parameters
    ----------
    calibration_callback : CalibrationCallback
        The method we will use to calibrate this pattern.
    """

    def __init__(self, calibration_callback: CalibrationCallback = None):
        super().__init__(calibration_callback)
        self.lhs = wildcard()
        self.rhs = wildcard()
        self.add = is_op("add")(self.lhs, self.rhs)
        self.pattern = self.add

    def callback(self, pre, post, node_map):
        lhs = node_map[self.lhs][0]
        rhs = node_map[self.rhs][0]

        add = node_map[self.add][0]

        out_dtype = infer_type(add).checked_type.dtype

        # Create quantization parameters for arguments to this addition
        self.create_scale_zps("add_lhs", "add_rhs")

        # Quantize, dequantize, and requantize inputs to have scale lhs_scale + rhs_scale
        # (Scale represents the lowest possible value representable in the quantized type,
        # so the smallest representable output is lhs_scale + rhs_scale)

        # We do this to avoid the requantize op in qnn's add, which causes issues with compilation
        # Requantize will be inserted in a future pass
        lhs_scale, lhs_zp, rhs_scale, rhs_zp = self.scale_zps
        quantized_lhs = relay.qnn.op.quantize(lhs, lhs_scale, lhs_zp)
        quantized_rhs = relay.qnn.op.quantize(rhs, rhs_scale, rhs_zp)

        dequantized_lhs = relay.qnn.op.dequantize(
            quantized_lhs, lhs_scale, relay.const(0, dtype="int32"), out_dtype=out_dtype
        )
        dequantized_rhs = relay.qnn.op.dequantize(
            quantized_rhs, rhs_scale, relay.const(0, dtype="int32"), out_dtype=out_dtype
        )

        add_scale = relay.op.add(lhs_scale, rhs_scale)

        requantized_lhs = relay.qnn.op.quantize(
            dequantized_lhs, add_scale, relay.const(0, dtype="int32")
        )
        requantized_rhs = relay.qnn.op.quantize(
            dequantized_rhs, add_scale, relay.const(0, dtype="int32")
        )

        add = relay.op.add(requantized_lhs, requantized_rhs)
        dequantized_call = relay.qnn.op.dequantize(
            add, add_scale, relay.const(0, dtype="int32"), out_dtype=out_dtype
        )

        return dequantized_call


class MultiplyPattern(QuantizerPattern):
    """Pattern to rewrite multiply as quantized.

    Parameters
    ----------
    calibration_callback : CalibrationCallback
        The method we will use to calibrate this pattern.
    """

    def __init__(self, calibration_callback: CalibrationCallback = None):
        super().__init__(calibration_callback)
        self.lhs = wildcard()
        self.rhs = wildcard()

        self.multiply = is_op("multiply")(self.lhs, self.rhs)
        self.pattern = self.multiply

    def callback(self, pre, post, node_map):
        lhs = node_map[self.lhs][0]
        rhs = node_map[self.rhs][0]

        multiply = node_map[self.multiply][0]

        out_dtype = infer_type(multiply).checked_type.dtype

        # Create quantization parameters for arguments to this multiplication.
        self.create_scale_zps("mul_lhs", "mul_rhs")
        lhs_scale, lhs_zp, rhs_scale, rhs_zp = self.scale_zps

        # Quantize inputs and construct args for multiply
        quantized_lhs = tvm.relay.cast(relay.qnn.op.quantize(lhs, lhs_scale, lhs_zp), "int32")
        quantized_rhs = tvm.relay.cast(relay.qnn.op.quantize(rhs, rhs_scale, rhs_zp), "int32")

        # Use normal relay multiply instead of qnn multiply to avoid requantize in qnn.mul
        # Subtract zero points to center on zero so that we can multiply lhs, rhs directly
        zeroed_quantized_lhs = relay.op.subtract(quantized_lhs, lhs_zp)
        zeroed_quantized_rhs = relay.op.subtract(quantized_rhs, rhs_zp)

        multiply = relay.op.multiply(zeroed_quantized_lhs, zeroed_quantized_rhs)
        dequantized_call = relay.qnn.op.dequantize(
            multiply, lhs_scale * rhs_scale, relay.const(0, dtype="int32"), out_dtype=out_dtype
        )

        return dequantized_call


class PerChannelPattern:
    """A parent class for patterns that will be per-channel quantized. PerChannelPattern should
    only be inherited by a class that also inherits QuantizerPattern or a subclass of it.
    """

    def extract_attrs(self, pre, post, node_map):
        """A callback to get the quantized attributes of this pattern. Usually, we just call
        self.get_attrs on the attributes of the original, unquantized node to construct the
        quantized attributes. Since this callback is used by the pattern rewriter, we must return
        a relay.Expr from it.

        Parameters
        ----------
        pre : relay.Expr
            Expression before transformation

        post : relay.Expr
            Expression after transformation

        node_map : Map of pattern to relay.Expr
            Contains expressions matching parts of the pattern.

        Returns
        -------
        post : relay.Expr
            Expression to rewrite the input expression as. We don't actually want to rewrite
            anything in this pass, so you should just return post.
        """
        raise NotImplementedError()

    def get_scale_size(self):
        """Returns the size of the per-channel scale variable

        Returns
        -------
        scale_size : tuple
            The size of the scale variable
        """
        raise NotImplementedError

    def weight_scale(self, name):
        """Helper to create a variable for a per-channel scale.
        Parameters
        ----------
        name : str
            Name of the variable
        """
        var = relay.var(
            str(name) + "_scale_" + str(QuantizerPattern.scales_count),
            shape=self.get_scale_size(),
            dtype="float32",
        )
        QuantizerPattern.scales_count += 1
        return var

    def create_scale_zps(self, left_name, right_name):
        """Helper to create scales and zero points for binops, with the per channel weight scale quantized.

        Parameters
        ----------
        left_name : str
            Identifier of the left hand side scale and zero point.

        right_name : str
            Identifier of the right hand side scale and zero point.
        """
        # Create quantization parameters for arguments with per channel on the right
        data_scale = self.scale(left_name)
        data_zp = self.zero_point(left_name)

        weight_scale = self.weight_scale(right_name)
        weight_zp = self.zero_point(right_name)
        self.scale_zps = [data_scale, data_zp, weight_scale, weight_zp]

    def attr_callback(self, expr):
        """A function to get the attributes of the quantized version of the current
        pattern. Meant to be called from inside calibrate_pattern.

        Parameters
        ----------
        expr : relay.Expr
            Expression that we want the attributes from. This will be the unquantized
            version of the expression.
        """
        pattern_ffi.rewrite(
            [_DFPatternCallback(self.pattern, self.extract_attrs, self.require_type)],
            infer_type(expr),
            tvm.ir.IRModule(),
            False,
        )
