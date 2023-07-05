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
"""Definition of generic operator strategy."""

from tvm.target import override_native_generic_func


def wrap_topi_schedule(topi_schedule):
    """Wrap TOPI schedule which doesn't use attrs"""

    def wrapper(_attrs, outs, target):
        with target:
            return topi_schedule(outs)

    return wrapper


def wrap_topi_compute(topi_compute):
    """Wrap TOPI compute which doesn't use attrs"""

    def wrapper(_attrs, inputs, _out_type):
        return [topi_compute(*inputs)]

    return wrapper


def wrap_compute_quantize(topi_compute):
    """Wrap TOPI compute which use axis and out data type from attrs"""

    def wrapper(attrs, inputs, _out_type):
        axis = attrs.axis
        out_dtype = attrs.out_dtype
        args = [*inputs, axis, out_dtype]
        return [topi_compute(*args)]

    return wrapper


def wrap_compute_dequantize(topi_compute):
    """Wrap TOPI compute which use axis from attrs"""

    def wrapper(attrs, inputs, _out_type):
        args = [*inputs, attrs.axis]
        return [topi_compute(*args)]

    return wrapper


def wrap_topi_qnn_conv2d(topi_compute):
    """Wrap TOPI compute which use conv2d attrs and output data type"""

    def wrapper(attrs, inputs, out_type):
        out_dtype = out_type.dtype
        oshape = out_type.shape
        strides = attrs.strides
        padding = attrs.padding
        dilation = attrs.dilation
        if len([*inputs]) == 11:
            args = [*inputs, strides, padding, dilation, oshape, out_dtype]
        elif len([*inputs]) == 10:
            args = [  # QNN Conv2d params:
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                # Bias argument
                None,
                # Requantization params:
                inputs[6],
                inputs[7],
                inputs[8],
                inputs[9],
                # Conv2d attrs:
                strides,
                padding,
                dilation,
                oshape,
                out_dtype,
            ]
        else:
            assert len([*inputs]) == 6
            args = [  # QNN Conv2d params:
                *inputs,
                # Bias argument:
                None,
                # Requantization params:
                None,
                None,
                None,
                None,
                strides,
                padding,
                dilation,
                oshape,
                out_dtype,
            ]
        return [topi_compute(*args)]

    return wrapper


def wrap_topi_qnn_dense(topi_compute):
    """Wrap TOPI compute which use qnn.dense attrs"""

    def wrapper(_attrs, inputs, out_type):
        out_dtype = out_type.dtype
        if len([*inputs]) == 11:
            args = [*inputs, out_dtype]
        elif len([*inputs]) == 10:
            args = [  # QNN Dense params:
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                # Bias argument
                None,
                # Requantization params:
                inputs[6],
                inputs[7],
                inputs[8],
                inputs[9],
                out_dtype,
            ]
        else:
            assert len([*inputs]) == 6
            args = [  # QNN Dense params:
                *inputs,
                # Bias argument:
                None,
                # Requantization params:
                None,
                None,
                None,
                None,
                out_dtype,
            ]
        return [topi_compute(*args)]

    return wrapper


def wrap_compute_qnn_avg_pool2d(topi_compute):
    """Wrap qnn.avg_pool2d topi compute"""

    def wrapper(attrs, inputs, out_type):
        kernel = attrs.pool_size
        strides = attrs.strides
        padding = attrs.padding
        dilation = attrs.dilation
        count_include_pad = attrs.count_include_pad
        oshape = out_type.shape
        odtype = out_type.dtype
        args = [
            inputs[0],
            kernel,
            strides,
            padding,
            dilation,
            count_include_pad,
            oshape,
            odtype,
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
        ]
        return [topi_compute(*args)]

    return wrapper


def wrap_topi_concatenate(topi_compute):
    """Wrap TOPI compute which use qnn.concatenate attrs"""

    def wrapper(attrs, inputs, out_type):
        return [topi_compute(inputs, attrs.axis, out_type.dtype)]

    return wrapper


def wrap_topi_qnn_batch_matmul(topi_compute):
    """Wrap TOPI compute which use qnn.batch_matmul attrs"""

    def wrapper(attrs, inputs, _out_type):
        assert len([*inputs]) == 6
        args = [*inputs, attrs.transpose_a, attrs.transpose_b, attrs.out_dtype]
        return [topi_compute(*args)]

    return wrapper


@override_native_generic_func("qnn_quantize_strategy")
def qnn_quantize_strategy(attrs, inputs, out_type, target):
    """qnn.quantize generic strategy"""
    raise RuntimeError(
        "qnn.quantize is currently only supported with Hexagon. "
        "Please run QNN Canonicalize pass to decompose this op into supported ops."
    )


@override_native_generic_func("qnn_dequantize_strategy")
def qnn_dequantize_strategy(attrs, inputs, out_type, target):
    """qnn.dequantize generic strategy"""
    raise RuntimeError(
        "qnn.dequantize is currently only supported with Hexagon. "
        "Please run QNN Canonicalize pass to decompose this op into supported ops."
    )


@override_native_generic_func("qnn_requantize_strategy")
def qnn_requantize_strategy(attrs, inputs, out_type, target):
    """qnn.requantize generic strategy"""
    raise RuntimeError(
        "qnn.requantize is currently only supported with Hexagon. "
        "Please run QNN Canonicalize pass to decompose this op into supported ops."
    )


@override_native_generic_func("qnn_add_strategy")
def qnn_add_strategy(attrs, inputs, out_type, target):
    """qnn.add generic strategy"""
    raise RuntimeError(
        "qnn.add is currently only supported with Hexagon. "
        "Please run QNN Canonicalize pass to decompose this op into supported ops."
    )


@override_native_generic_func("qnn_subtract_strategy")
def qnn_subtract_strategy(attrs, inputs, out_type, target):
    """qnn.subtract generic strategy"""
    raise RuntimeError(
        "qnn.subtract is currently only supported with Hexagon. "
        "Please run QNN Canonicalize pass to decompose this op into supported ops."
    )


@override_native_generic_func("qnn_mul_strategy")
def qnn_mul_strategy(attrs, inputs, out_type, target):
    """qnn.mul generic strategy"""
    raise RuntimeError(
        "qnn.mul is currently only supported with Hexagon. "
        "Please run QNN Canonicalize pass to decompose this op into supported ops."
    )


@override_native_generic_func("qnn_tanh_strategy")
def qnn_tanh_strategy(attrs, inputs, out_type, target):
    """qnn.tanh generic strategy"""
    raise RuntimeError(
        "qnn.tanh is currently only supported with Hexagon. "
        "Please run QNN Canonicalize pass to decompose this op into supported ops."
    )


@override_native_generic_func("qnn_concatenate_strategy")
def qnn_concatenate_strategy(attrs, inputs, out_type, target):
    """qnn.concatenate generic strategy"""
    raise RuntimeError(
        "qnn.concatenate is currently only supported with Hexagon. "
        "Please run QNN Canonicalize pass to decompose this op into supported ops."
    )


@override_native_generic_func("qnn_conv2d_strategy")
def qnn_conv2d_strategy(attrs, inputs, out_type, target):
    """qnn.conv2d generic strategy"""
    raise RuntimeError(
        "qnn.conv2d is currently only supported with Hexagon. "
        "Please run QNN Canonicalize pass to decompose this op into supported ops."
    )


@override_native_generic_func("qnn_dense_strategy")
def qnn_dense_strategy(attrs, inputs, out_type, target):
    """qnn.dense generic strategy"""
    raise RuntimeError(
        "qnn.dense is currently only supported with Hexagon. "
        "Please run QNN Canonicalize pass to decompose this op into supported ops."
    )


@override_native_generic_func("qnn_dense_pack_strategy")
def qnn_dense_pack_strategy(attrs, inputs, out_type, target):
    """qnn.contrib_dense_pack generic strategy"""
    raise RuntimeError("qnn.contrib_dense_pack is currently only supported with Hexagon. ")


@override_native_generic_func("qnn_batch_matmul_strategy")
def qnn_batch_matmul_strategy(attrs, inputs, out_type, target):
    """qnn.batch_matmul generic strategy"""
    raise RuntimeError(
        "qnn.batch_matmul is currently only supported with Hexagon. "
        "Please run QNN Canonicalize pass to decompose this op into supported ops."
    )


@override_native_generic_func("qnn_avg_pool2d_strategy")
def qnn_avg_pool2d_strategy(attrs, inputs, out_type, target):
    """qnn.avg_pool2d generic strategy"""
    raise RuntimeError(
        "qnn.avg_pool2d is currently only supported with Hexagon. "
        "Please run QNN Canonicalize pass to decompose this op into supported ops."
    )
