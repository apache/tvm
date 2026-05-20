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

import json
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.backend.pattern_registry import get_patterns_with_prefix
from tvm.script import relax as R
from tvm.script import tirx as T


@tvm.script.ir_module
class ReluModule:
    @R.function
    def main(x: R.Tensor((2, 3), "float32")):
        with R.dataflow():
            z = relax.op.nn.relu(x)
            R.output(z)
        return z


@tvm.script.ir_module
class ReluFloat16Module:
    @R.function
    def main(x: R.Tensor((2, 3), "float16")):
        with R.dataflow():
            z = relax.op.nn.relu(x)
            R.output(z)
        return z


@tvm.script.ir_module
class ReluSymbolicModule:
    @R.function
    def main(x: R.Tensor(("n", 3), "float32")):
        with R.dataflow():
            z = relax.op.nn.relu(x)
            R.output(z)
        return z


@tvm.script.ir_module
class AddModule:
    @R.function
    def main(x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")):
        with R.dataflow():
            z = relax.op.add(x, y)
            R.output(z)
        return z


@tvm.script.ir_module
class MultiplyModule:
    @R.function
    def main(x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")):
        with R.dataflow():
            z = relax.op.multiply(x, y)
            R.output(z)
        return z


@tvm.script.ir_module
class FullyConnectedBiasGeluParamModule:
    @R.function
    def main(
        x: R.Tensor((4, 8), "float32"),
        w: R.Tensor((8, 16), "float32"),
        b: R.Tensor((16,), "float32"),
    ):
        with R.dataflow():
            y = R.matmul(x, w)
            z = R.add(y, b)
            out = R.nn.gelu(z)
            R.output(out)
        return out


@tvm.script.ir_module
class FullyConnectedBiasApproxGeluParamModule:
    @R.function
    def main(
        x: R.Tensor((4, 8), "float32"),
        w: R.Tensor((8, 16), "float32"),
        b: R.Tensor((16,), "float32"),
    ):
        with R.dataflow():
            y = R.matmul(x, w)
            z = R.add(y, b)
            out = R.nn.gelu_tanh(z)
            R.output(out)
        return out


@tvm.script.ir_module
class MLPResidualModule:
    @R.function
    def main(
        x: R.Tensor((4, 8), "float32"),
        residual: R.Tensor((4, 16), "float32"),
        w: R.Tensor((8, 16), "float32"),
        b: R.Tensor((16,), "float32"),
    ):
        with R.dataflow():
            y = R.matmul(x, w)
            z = R.add(y, b)
            gelu = R.nn.gelu(z)
            out = R.add(gelu, residual)
            R.output(out)
        return out


@tvm.script.ir_module
class GeluModule:
    @R.function
    def main(x: R.Tensor((4, 16), "float32")):
        with R.dataflow():
            z = R.nn.gelu(x)
            R.output(z)
        return z


@tvm.script.ir_module
class GeluFloat16Module:
    @R.function
    def main(x: R.Tensor((4, 16), "float16")):
        with R.dataflow():
            z = R.nn.gelu(x)
            R.output(z)
        return z


@tvm.script.ir_module
class GeluSymbolicModule:
    @R.function
    def main(x: R.Tensor(("n", 16), "float32")):
        with R.dataflow():
            z = R.nn.gelu(x)
            R.output(z)
        return z


@tvm.script.ir_module
class SoftmaxLastAxisModule:
    @R.function
    def main(x: R.Tensor((2, 3, 4), "float32")):
        with R.dataflow():
            z = R.nn.softmax(x, axis=-1)
            R.output(z)
        return z


@tvm.script.ir_module
class SoftmaxAxis0Module:
    @R.function
    def main(x: R.Tensor((2, 3, 4), "float32")):
        with R.dataflow():
            z = R.nn.softmax(x, axis=0)
            R.output(z)
        return z


@tvm.script.ir_module
class AddBroadcastModule:
    @R.function
    def main(x: R.Tensor((2, 3), "float32"), y: R.Tensor((3,), "float32")):
        with R.dataflow():
            z = relax.op.add(x, y)
            R.output(z)
        return z


@tvm.script.ir_module
class QuantizeModule:
    @R.function
    def main(x: R.Tensor((2, 4), "float32")) -> R.Tensor((2, 4), "int8"):
        with R.dataflow():
            z = R.quantize(
                x,
                R.const(0.5, "float32"),
                R.const(0, "int8"),
                axis=-1,
                out_dtype="int8",
            )
            R.output(z)
        return z


@tvm.script.ir_module
class DequantizeModule:
    @R.function
    def main(x: R.Tensor((2, 4), "int8")) -> R.Tensor((2, 4), "float32"):
        with R.dataflow():
            z = R.dequantize(
                x,
                R.const(0.5, "float32"),
                R.const(0, "int8"),
                axis=-1,
                out_dtype="float32",
            )
            R.output(z)
        return z


@tvm.script.ir_module
class QS8FullyConnectedModule:
    @R.function
    def main(x: R.Tensor((2, 3), "int8")) -> R.Tensor((2, 4), "int8"):
        with R.dataflow():
            x_f = R.dequantize(
                x, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="float32"
            )
            w = R.const(
                np.array([[1, -2, 3, -4], [2, 1, -1, 3], [-3, 2, 1, -2]], dtype="int8")
            )
            w_f = R.dequantize(
                w,
                R.const(np.array([0.5, 0.25, 0.125, 0.375], dtype="float32")),
                R.const(0, "int8"),
                axis=1,
                out_dtype="float32",
            )
            y = R.matmul(x_f, w_f)
            z = R.quantize(
                y, R.const(0.5, "float32"), R.const(0, "int8"), axis=-1, out_dtype="int8"
            )
            R.output(z)
        return z


@tvm.script.ir_module
class QS8FullyConnectedBiasRelu6Module:
    @R.function
    def main(x: R.Tensor((2, 3), "int8")) -> R.Tensor((2, 4), "int8"):
        with R.dataflow():
            x_f = R.dequantize(
                x, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="float32"
            )
            w = R.const(
                np.array([[1, -2, 3, -4], [2, 1, -1, 3], [-3, 2, 1, -2]], dtype="int8")
            )
            w_f = R.dequantize(
                w,
                R.const(np.array([0.5, 0.25, 0.125, 0.375], dtype="float32")),
                R.const(0, "int8"),
                axis=1,
                out_dtype="float32",
            )
            b = R.const(np.array([1, -2, 3, -4], dtype="int32"))
            b_f = R.dequantize(
                b,
                R.const(np.array([0.125, 0.0625, 0.03125, 0.09375], dtype="float32")),
                R.const(0, "int32"),
                axis=0,
                out_dtype="float32",
            )
            y = R.matmul(x_f, w_f)
            biased = relax.op.add(y, b_f)
            clipped = relax.op.clip(biased, 0, 6)
            z = R.quantize(
                clipped, R.const(0.5, "float32"), R.const(0, "int8"), axis=-1, out_dtype="int8"
            )
            R.output(z)
        return z


@tvm.script.ir_module
class QS8Conv2DBiasReluModule:
    @R.function
    def main(x: R.Tensor((1, 4, 4, 2), "int8")) -> R.Tensor((1, 2, 2, 3), "int8"):
        with R.dataflow():
            x_f = R.dequantize(
                x, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="float32"
            )
            w = R.const(np.arange(-27, 27, dtype="int8").reshape(3, 3, 3, 2))
            w_f = R.dequantize(
                w,
                R.const(np.array([0.5, 0.25, 0.125], dtype="float32")),
                R.const(0, "int8"),
                axis=0,
                out_dtype="float32",
            )
            y = relax.op.nn.conv2d(
                x_f,
                w_f,
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                groups=1,
                data_layout="NHWC",
                kernel_layout="OHWI",
                out_layout="NHWC",
            )
            b = R.const(np.array([1, -2, 3], dtype="int32"))
            b_f = R.dequantize(
                b,
                R.const(np.array([0.125, 0.0625, 0.03125], dtype="float32")),
                R.const(0, "int32"),
                axis=0,
                out_dtype="float32",
            )
            biased = relax.op.add(y, b_f)
            relu = relax.op.nn.relu(biased)
            z = R.quantize(
                relu, R.const(0.5, "float32"), R.const(0, "int8"), axis=-1, out_dtype="int8"
            )
            R.output(z)
        return z


@tvm.script.ir_module
class QS8DepthwiseConv2DBiasRelu6Module:
    @R.function
    def main(x: R.Tensor((1, 4, 4, 2), "int8")) -> R.Tensor((1, 2, 2, 2), "int8"):
        with R.dataflow():
            x_f = R.dequantize(
                x, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="float32"
            )
            w = R.const(np.arange(-9, 9, dtype="int8").reshape(3, 3, 2, 1))
            w_f = R.dequantize(
                w,
                R.const(np.array([0.5, 0.25], dtype="float32")),
                R.const(0, "int8"),
                axis=2,
                out_dtype="float32",
            )
            y = relax.op.nn.conv2d(
                x_f,
                w_f,
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                groups=2,
                data_layout="NHWC",
                kernel_layout="HWOI",
                out_layout="NHWC",
            )
            b = R.const(np.array([1, -2], dtype="int32"))
            b_f = R.dequantize(
                b,
                R.const(np.array([0.125, 0.0625], dtype="float32")),
                R.const(0, "int32"),
                axis=0,
                out_dtype="float32",
            )
            biased = relax.op.add(y, b_f)
            clipped = relax.op.clip(biased, 0, 6)
            z = R.quantize(
                clipped, R.const(0.5, "float32"), R.const(0, "int8"), axis=-1, out_dtype="int8"
            )
            R.output(z)
        return z


@tvm.script.ir_module
class DynamicRangeFullyConnectedModule:
    @R.function
    def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 4), "float32"):
        with R.dataflow():
            w = R.const(
                np.array([[1, -2, 3, -4], [2, 1, -1, 3], [-3, 2, 1, -2]], dtype="int8")
            )
            w_f = R.dequantize(
                w,
                R.const(np.array([0.5, 0.25, 0.125, 0.375], dtype="float32")),
                R.const(0, "int8"),
                axis=1,
                out_dtype="float32",
            )
            z = R.matmul(x, w_f)
            R.output(z)
        return z


@tvm.script.ir_module
class DynamicRangeFullyConnectedBiasRelu6Module:
    @R.function
    def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 4), "float32"):
        with R.dataflow():
            w = R.const(
                np.array([[1, -2, 3, -4], [2, 1, -1, 3], [-3, 2, 1, -2]], dtype="int8")
            )
            w_f = R.dequantize(
                w,
                R.const(np.array([0.5, 0.25, 0.125, 0.375], dtype="float32")),
                R.const(0, "int8"),
                axis=1,
                out_dtype="float32",
            )
            b = R.const(np.array([0.125, -0.25, 0.375, -0.5], dtype="float32"))
            y = R.matmul(x, w_f)
            biased = relax.op.add(y, b)
            z = relax.op.clip(biased, 0, 6)
            R.output(z)
        return z


@tvm.script.ir_module
class DynamicRangeTinyFullyConnectedModule:
    @R.function
    def main(x: R.Tensor((1, 2), "float32")) -> R.Tensor((1, 2), "float32"):
        with R.dataflow():
            w = R.const(np.array([[1, -2], [2, 1]], dtype="int8"))
            w_f = R.dequantize(
                w,
                R.const(np.array([0.5, 0.25], dtype="float32")),
                R.const(0, "int8"),
                axis=1,
                out_dtype="float32",
            )
            z = R.matmul(x, w_f)
            R.output(z)
        return z


@tvm.script.ir_module
class DynamicRangeFullyConnectedPerTensorWeightModule:
    @R.function
    def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 4), "float32"):
        with R.dataflow():
            w = R.const(np.ones((3, 4), dtype="int8"))
            w_f = R.dequantize(
                w, R.const(0.5, "float32"), R.const(0, "int8"), axis=1, out_dtype="float32"
            )
            z = R.matmul(x, w_f)
            R.output(z)
        return z


@tvm.script.ir_module
class DynamicRangeFullyConnectedBadWeightZeroPointModule:
    @R.function
    def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 4), "float32"):
        with R.dataflow():
            w = R.const(np.ones((3, 4), dtype="int8"))
            w_f = R.dequantize(
                w,
                R.const(np.array([0.5, 0.25, 0.125, 0.375], dtype="float32")),
                R.const(1, "int8"),
                axis=1,
                out_dtype="float32",
            )
            z = R.matmul(x, w_f)
            R.output(z)
        return z


@tvm.script.ir_module
class DynamicRangeFullyConnectedQU8WeightModule:
    @R.function
    def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 4), "float32"):
        with R.dataflow():
            w = R.const(np.ones((3, 4), dtype="uint8"))
            w_f = R.dequantize(
                w,
                R.const(np.array([0.5, 0.25, 0.125, 0.375], dtype="float32")),
                R.const(0, "uint8"),
                axis=1,
                out_dtype="float32",
            )
            z = R.matmul(x, w_f)
            R.output(z)
        return z


@tvm.script.ir_module
class DynamicBatchFullyConnectedModule:
    @R.function
    def main(x: R.Tensor(("n", 3), "float32")) -> R.Tensor(("n", 4), "float32"):
        with R.dataflow():
            w = R.const(
                np.array([[1.0, -2.0, 3.0, -4.0], [2.0, 1.0, -1.0, 3.0], [-3.0, 2.0, 1.0, -2.0]], dtype="float32")
            )
            z = R.matmul(x, w)
            R.output(z)
        return z


@tvm.script.ir_module
class DynamicBatchFullyConnectedWithAttrsModule:
    @R.function
    def main(
        x: R.Tensor(("n", 3), "float32"),
        w: R.Tensor((3, 4), "float32"),
    ) -> R.Tensor(("n", 4), "float32"):
        R.func_attr({"tir_var_upper_bound": {"n": T.int64(4)}, "tir_var_lower_bound": {"n": T.int64(1)}})
        with R.dataflow():
            z = R.matmul(x, w)
            R.output(z)
        return z


@tvm.script.ir_module
class DynamicBatchFullyConnectedParamModule:
    @R.function
    def main(
        x: R.Tensor(("n", 3), "float32"),
        w: R.Tensor((3, 4), "float32"),
    ) -> R.Tensor(("n", 4), "float32"):
        with R.dataflow():
            z = R.matmul(x, w)
            R.output(z)
        return z


@tvm.script.ir_module
class DynamicBatchConv2DModule:
    @R.function
    def main(x: R.Tensor(("n", 5, 5, 3), "float32")) -> R.Tensor(("n", 3, 3, 4), "float32"):
        with R.dataflow():
            w = R.const(
                np.linspace(-0.2, 0.2, num=4 * 3 * 3 * 3, dtype="float32").reshape(4, 3, 3, 3)
            )
            z = relax.op.nn.conv2d(
                x,
                w,
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                groups=1,
                data_layout="NHWC",
                kernel_layout="OHWI",
                out_layout="NHWC",
            )
            R.output(z)
        return z


@tvm.script.ir_module
class DynamicBatchConv2DParamModule:
    @R.function
    def main(
        x: R.Tensor(("n", 5, 5, 3), "float32"),
        w: R.Tensor((4, 3, 3, 3), "float32"),
    ) -> R.Tensor(("n", 3, 3, 4), "float32"):
        with R.dataflow():
            z = relax.op.nn.conv2d(
                x,
                w,
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                groups=1,
                data_layout="NHWC",
                kernel_layout="OHWI",
                out_layout="NHWC",
            )
            R.output(z)
        return z


@tvm.script.ir_module
class DynamicHWConv2DModule:
    @R.function
    def main(x: R.Tensor(("n", "h", 5, 3), "float32")):
        with R.dataflow():
            w = R.const(np.zeros((4, 3, 3, 3), dtype="float32"))
            z = relax.op.nn.conv2d(
                x,
                w,
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                groups=1,
                data_layout="NHWC",
                kernel_layout="OHWI",
                out_layout="NHWC",
            )
            R.output(z)
        return z


@tvm.script.ir_module
class DynamicChannelConv2DModule:
    @R.function
    def main(x: R.Tensor(("n", 5, 5, "c"), "float32")):
        with R.dataflow():
            w = R.const(np.zeros((4, 3, 3, 3), dtype="float32"))
            z = relax.op.nn.conv2d(
                x,
                w,
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                groups=1,
                data_layout="NHWC",
                kernel_layout="OHWI",
                out_layout="NHWC",
            )
            R.output(z)
        return z


@tvm.script.ir_module
class DynamicBatchQS8FullyConnectedModule:
    @R.function
    def main(x: R.Tensor(("n", 3), "int8")) -> R.Tensor(("n", 4), "int8"):
        with R.dataflow():
            x_f = R.dequantize(
                x, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="float32"
            )
            w = R.const(
                np.array([[1, -2, 3, -4], [2, 1, -1, 3], [-3, 2, 1, -2]], dtype="int8")
            )
            w_f = R.dequantize(
                w,
                R.const(np.array([0.5, 0.25, 0.125, 0.375], dtype="float32")),
                R.const(0, "int8"),
                axis=1,
                out_dtype="float32",
            )
            y = R.matmul(x_f, w_f)
            z = R.quantize(
                y, R.const(0.5, "float32"), R.const(0, "int8"), axis=-1, out_dtype="int8"
            )
            R.output(z)
        return z


@tvm.script.ir_module
class DynamicBatchDynamicRangeFullyConnectedModule:
    @R.function
    def main(x: R.Tensor(("n", 3), "float32")) -> R.Tensor(("n", 4), "float32"):
        with R.dataflow():
            w = R.const(
                np.array([[1, -2, 3, -4], [2, 1, -1, 3], [-3, 2, 1, -2]], dtype="int8")
            )
            w_f = R.dequantize(
                w,
                R.const(np.array([0.5, 0.25, 0.125, 0.375], dtype="float32")),
                R.const(0, "int8"),
                axis=1,
                out_dtype="float32",
            )
            z = R.matmul(x, w_f)
            R.output(z)
        return z


@tvm.script.ir_module
class QS8ReshapeModule:
    @R.function
    def main(x: R.Tensor((2, 3), "int8")) -> R.Tensor((1, 6), "int8"):
        with R.dataflow():
            x_f = R.dequantize(
                x, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="float32"
            )
            y = relax.op.reshape(x_f, (1, 6))
            z = R.quantize(
                y, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="int8"
            )
            R.output(z)
        return z


@tvm.script.ir_module
class QS8FlattenModule:
    @R.function
    def main(x: R.Tensor((2, 3, 4), "int8")) -> R.Tensor((24,), "int8"):
        with R.dataflow():
            x_f = R.dequantize(
                x, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="float32"
            )
            y = relax.op.flatten(x_f)
            z = R.quantize(
                y, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="int8"
            )
            R.output(z)
        return z


@tvm.script.ir_module
class QS8CopyModule:
    @R.function
    def main(x: R.Tensor((2, 3), "int8")) -> R.Tensor((2, 3), "int8"):
        with R.dataflow():
            x_f = R.dequantize(
                x, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="float32"
            )
            z = R.quantize(
                x_f, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="int8"
            )
            R.output(z)
        return z


@tvm.script.ir_module
class QS8MaxPool2DModule:
    @R.function
    def main(x: R.Tensor((1, 4, 4, 2), "int8")) -> R.Tensor((1, 2, 2, 2), "int8"):
        with R.dataflow():
            x_f = R.dequantize(
                x, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="float32"
            )
            y = relax.op.nn.max_pool2d(
                x_f,
                pool_size=[2, 2],
                strides=[2, 2],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                ceil_mode=False,
                layout="NHWC",
                out_layout="NHWC",
            )
            z = R.quantize(
                y, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="int8"
            )
            R.output(z)
        return z


@tvm.script.ir_module
class QS8AvgPool2DModule:
    @R.function
    def main(x: R.Tensor((1, 4, 4, 2), "int8")) -> R.Tensor((1, 2, 2, 2), "int8"):
        with R.dataflow():
            x_f = R.dequantize(
                x, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="float32"
            )
            y = relax.op.nn.avg_pool2d(
                x_f,
                pool_size=[2, 2],
                strides=[2, 2],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                ceil_mode=False,
                count_include_pad=False,
                layout="NHWC",
                out_layout="NHWC",
            )
            z = R.quantize(
                y, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="int8"
            )
            R.output(z)
        return z


@tvm.script.ir_module
class QS8GlobalAvgPoolAsAvgPool2DModule:
    @R.function
    def main(x: R.Tensor((1, 4, 4, 2), "int8")) -> R.Tensor((1, 1, 1, 2), "int8"):
        with R.dataflow():
            x_f = R.dequantize(
                x, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="float32"
            )
            y = relax.op.nn.avg_pool2d(
                x_f,
                pool_size=[4, 4],
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                ceil_mode=False,
                count_include_pad=False,
                layout="NHWC",
                out_layout="NHWC",
            )
            z = R.quantize(
                y, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="int8"
            )
            R.output(z)
        return z


@tvm.script.ir_module
class QS8AddModule:
    @R.function
    def main(x: R.Tensor((1, 4, 4, 2), "int8"), y: R.Tensor((1, 4, 4, 2), "int8")) -> R.Tensor(
        (1, 4, 4, 2), "int8"
    ):
        with R.dataflow():
            x_f = R.dequantize(
                x, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="float32"
            )
            y_f = R.dequantize(
                y, R.const(0.5, "float32"), R.const(0, "int8"), axis=-1, out_dtype="float32"
            )
            added = relax.op.add(x_f, y_f)
            z = R.quantize(
                added, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="int8"
            )
            R.output(z)
        return z


@tvm.script.ir_module
class QS8AddRelu6Module:
    @R.function
    def main(x: R.Tensor((1, 4, 4, 2), "int8"), y: R.Tensor((1, 4, 4, 2), "int8")) -> R.Tensor(
        (1, 4, 4, 2), "int8"
    ):
        with R.dataflow():
            x_f = R.dequantize(
                x, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="float32"
            )
            y_f = R.dequantize(
                y, R.const(0.5, "float32"), R.const(0, "int8"), axis=-1, out_dtype="float32"
            )
            added = relax.op.add(x_f, y_f)
            clipped = relax.op.clip(added, 0, 6)
            z = R.quantize(
                clipped, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="int8"
            )
            R.output(z)
        return z


@tvm.script.ir_module
class QS8ReshapeMismatchedQParamsModule:
    @R.function
    def main(x: R.Tensor((2, 3), "int8")) -> R.Tensor((1, 6), "int8"):
        with R.dataflow():
            x_f = R.dequantize(
                x, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="float32"
            )
            y = relax.op.reshape(x_f, (1, 6))
            z = R.quantize(
                y, R.const(0.5, "float32"), R.const(0, "int8"), axis=-1, out_dtype="int8"
            )
            R.output(z)
        return z


@tvm.script.ir_module
class QS8MaxPoolNCHWModule:
    @R.function
    def main(x: R.Tensor((1, 2, 4, 4), "int8")) -> R.Tensor((1, 2, 2, 2), "int8"):
        with R.dataflow():
            x_f = R.dequantize(
                x, R.const(0.25, "float32"), R.const(0, "int8"), axis=1, out_dtype="float32"
            )
            y = relax.op.nn.max_pool2d(
                x_f,
                pool_size=[2, 2],
                strides=[2, 2],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                ceil_mode=False,
                layout="NCHW",
                out_layout="NCHW",
            )
            z = R.quantize(
                y, R.const(0.25, "float32"), R.const(0, "int8"), axis=1, out_dtype="int8"
            )
            R.output(z)
        return z


@tvm.script.ir_module
class QS8AddBroadcastModule:
    @R.function
    def main(x: R.Tensor((1, 4, 4, 2), "int8"), y: R.Tensor((2,), "int8")) -> R.Tensor(
        (1, 4, 4, 2), "int8"
    ):
        with R.dataflow():
            x_f = R.dequantize(
                x, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="float32"
            )
            y_f = R.dequantize(
                y, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="float32"
            )
            added = relax.op.add(x_f, y_f)
            z = R.quantize(
                added, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="int8"
            )
            R.output(z)
        return z


@tvm.script.ir_module
class ClipModule:
    @R.function
    def main(x: R.Tensor((2, 3), "float32")):
        with R.dataflow():
            z = relax.op.clip(x, 0, 6)
            R.output(z)
        return z


@tvm.script.ir_module
class SigmoidModule:
    @R.function
    def main(x: R.Tensor((2, 3), "float32")):
        with R.dataflow():
            z = relax.op.sigmoid(x)
            R.output(z)
        return z


@tvm.script.ir_module
class TanhModule:
    @R.function
    def main(x: R.Tensor((2, 3), "float32")):
        with R.dataflow():
            z = relax.op.tanh(x)
            R.output(z)
        return z


@tvm.script.ir_module
class ConvBiasReluPoolModule:
    @R.function
    def main(
        x: R.Tensor((1, 5, 5, 3), "float32"),
        w: R.Tensor((4, 3, 3, 3), "float32"),
        b: R.Tensor((4,), "float32"),
    ):
        with R.dataflow():
            conv = relax.op.nn.conv2d(
                x,
                w,
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                groups=1,
                data_layout="NHWC",
                kernel_layout="OHWI",
                out_layout="NHWC",
            )
            biased = relax.op.add(conv, b)
            relu = relax.op.nn.relu(biased)
            z = relax.op.nn.max_pool2d(
                relu,
                pool_size=[2, 2],
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                ceil_mode=False,
                layout="NHWC",
                out_layout="NHWC",
            )
            R.output(z)
        return z


@tvm.script.ir_module
class TinyCNNModule:
    @R.function
    def main(
        x: R.Tensor((1, 8, 8, 3), "float32"),
        residual: R.Tensor((1, 3, 3, 4), "float32"),
        w: R.Tensor((4, 3, 3, 3), "float32"),
        b: R.Tensor((4,), "float32"),
    ):
        with R.dataflow():
            conv = relax.op.nn.conv2d(
                x,
                w,
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                groups=1,
                data_layout="NHWC",
                kernel_layout="OHWI",
                out_layout="NHWC",
            )
            biased = relax.op.add(conv, b)
            relu = relax.op.nn.relu(biased)
            pooled = relax.op.nn.max_pool2d(
                relu,
                pool_size=[2, 2],
                strides=[2, 2],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                ceil_mode=False,
                layout="NHWC",
                out_layout="NHWC",
            )
            added = relax.op.add(pooled, residual)
            z = relax.op.tanh(added)
            R.output(z)
        return z


@tvm.script.ir_module
class ConvNCHWModule:
    @R.function
    def main(x: R.Tensor((1, 3, 5, 5), "float32")):
        with R.dataflow():
            w = R.const(np.zeros((4, 3, 3, 3), dtype="float32"))
            z = relax.op.nn.conv2d(
                x,
                w,
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                groups=1,
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_layout="NCHW",
            )
            R.output(z)
        return z


@tvm.script.ir_module
class ConvDynamicWeightModule:
    @R.function
    def main(
        x: R.Tensor((1, 5, 5, 3), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
    ):
        with R.dataflow():
            z = relax.op.nn.conv2d(
                x,
                w,
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                groups=1,
                data_layout="NHWC",
                kernel_layout="OHWI",
                out_layout="NHWC",
            )
            R.output(z)
        return z


@tvm.script.ir_module
class AvgPoolPaddedModule:
    @R.function
    def main(x: R.Tensor((1, 5, 5, 3), "float32")):
        with R.dataflow():
            z = relax.op.nn.avg_pool2d(
                x,
                pool_size=[2, 2],
                strides=[1, 1],
                padding=[1, 1, 1, 1],
                dilation=[1, 1],
                ceil_mode=False,
                count_include_pad=False,
                layout="NHWC",
                out_layout="NHWC",
            )
            R.output(z)
        return z


def _has_xnnpack_codegen():
    return tvm.get_global_func("relax.ext.xnnpack", allow_missing=True) is not None


def _has_xnnpack_runtime():
    return tvm.get_global_func("runtime.XNNPACKJSONRuntimeCreate", allow_missing=True) is not None


def _xnnpack_capability(name):
    func = tvm.get_global_func("runtime.XNNPACKJSONRuntimeGetCapabilities", allow_missing=True)
    if func is None:
        return False
    return bool(int(func()[name]))


def _xnnpack_capabilities():
    func = tvm.get_global_func("runtime.XNNPACKJSONRuntimeGetCapabilities", allow_missing=True)
    if func is None:
        return {}
    return {str(key): int(value) for key, value in func().items()}


def _load_xnnpack_benchmark_module():
    path = pathlib.Path(__file__).with_name("benchmark_xnnpack.py")
    spec = importlib.util.spec_from_file_location("benchmark_xnnpack", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _quant_metadata_validator():
    return tvm.get_global_func(
        "runtime.XNNPACKJSONRuntimeValidateQuantizationMetadata", allow_missing=True
    )


def _quant_tensor_smoke():
    return tvm.get_global_func(
        "runtime.XNNPACKJSONRuntimeQuantizedTensorDefinitionSmoke", allow_missing=True
    )


def _xnnpack_runtime_create():
    return tvm.get_global_func("runtime.XNNPACKJSONRuntimeCreate", allow_missing=True)


def _has_codegen_attr(mod):
    found = False

    def visit(expr):
        nonlocal found
        if (
            isinstance(expr, relax.Function)
            and expr.attrs
            and expr.attrs.get("Codegen") == "xnnpack"
        ):
            found = True

    for func in mod.functions.values():
        if isinstance(func, relax.Function):
            visit(func)
            relax.analysis.post_order_visit(func, visit)

    return found


def _has_external_mods(mod):
    return (
        mod.attrs is not None
        and "external_mods" in mod.attrs
        and len(mod.attrs["external_mods"]) > 0
    )


def _count_xnnpack_partitions(mod):
    count = 0

    for func in mod.functions.values():
        if (
            isinstance(func, relax.Function)
            and func.attrs
            and func.attrs.get("Codegen") == "xnnpack"
        ):
            count += 1

    return count


def _partition(mod, precision="fp32", **kwargs):
    from tvm.relax.backend.xnnpack import (
        XNNPACKCostConfig,
        XNNPACKPartitionConfig,
        XNNPACKRuntimeConfig,
        partition_for_xnnpack,
    )

    cost_keys = {
        "partition_policy",
        "layout",
        "min_subgraph_size",
        "min_compute_to_copy_ratio",
        "allow_isolated_elementwise",
        "allow_layout_rewrite",
        "allow_cast_boundary",
        "report_partition_decisions",
    }
    cost_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in cost_keys}
    dynamic_shape_policy = kwargs.pop("dynamic_shape_policy", "none")
    dynamic_batch_bounds = kwargs.pop("dynamic_batch_bounds", None)
    if kwargs:
        raise TypeError(f"Unsupported _partition test options: {sorted(kwargs)}")

    return partition_for_xnnpack(
        mod,
        config=XNNPACKPartitionConfig(
            runtime=XNNPACKRuntimeConfig(precision=precision),
            cost=XNNPACKCostConfig(**cost_kwargs),
            dynamic_shape_policy=dynamic_shape_policy,
            dynamic_batch_bounds=dynamic_batch_bounds,
        ),
    )


def _bind_cnn_params(mod=ConvBiasReluPoolModule):
    weight = np.arange(4 * 3 * 3 * 3).reshape(4, 3, 3, 3).astype("float32") / 100.0
    bias = np.array([0.1, -0.2, 0.3, -0.4], dtype="float32")
    return relax.transform.BindParams("main", {"w": weight, "b": bias})(mod)


def _bind_tiny_cnn_params():
    weight = np.linspace(-0.2, 0.2, num=4 * 3 * 3 * 3, dtype="float32").reshape(4, 3, 3, 3)
    bias = np.array([0.15, -0.05, 0.25, -0.10], dtype="float32")
    return relax.transform.BindParams("main", {"w": weight, "b": bias})(TinyCNNModule)


def _mlp_weight_bias():
    weight = np.linspace(-0.3, 0.3, num=8 * 16, dtype="float32").reshape(8, 16)
    bias = np.linspace(-0.2, 0.2, num=16, dtype="float32")
    return weight, bias


def _bind_mlp_params(mod):
    weight, bias = _mlp_weight_bias()
    return relax.transform.BindParams("main", {"w": weight, "b": bias})(mod)


def _dynamic_batch_fc_weight():
    return np.array(
        [[1.0, -2.0, 3.0, -4.0], [2.0, 1.0, -1.0, 3.0], [-3.0, 2.0, 1.0, -2.0]],
        dtype="float32",
    )


def _bind_dynamic_batch_fc_params():
    return relax.transform.BindParams("main", {"w": _dynamic_batch_fc_weight()})(
        DynamicBatchFullyConnectedParamModule
    )


def _bind_dynamic_batch_fc_attrs_params():
    return relax.transform.BindParams("main", {"w": _dynamic_batch_fc_weight()})(
        DynamicBatchFullyConnectedWithAttrsModule
    )


def _dynamic_batch_conv_weight():
    return np.linspace(-0.2, 0.2, num=4 * 3 * 3 * 3, dtype="float32").reshape(4, 3, 3, 3)


def _bind_dynamic_batch_conv_params():
    return relax.transform.BindParams("main", {"w": _dynamic_batch_conv_weight()})(
        DynamicBatchConv2DParamModule
    )


def _tiny_cnn_inputs():
    rng = np.random.default_rng(0)
    x_np = rng.uniform(-1.0, 1.0, size=(1, 8, 8, 3)).astype("float32")
    residual_np = rng.uniform(-0.5, 0.5, size=(1, 3, 3, 4)).astype("float32")
    return x_np, residual_np


def _run_tiny_cnn_with_options(options=None, precision="fp32", rtol=1e-5, atol=1e-5):
    bound_mod = _bind_tiny_cnn_params()
    partitioned = _partition(bound_mod, precision=precision)
    assert _count_xnnpack_partitions(partitioned) == 4
    partitioned = relax.transform.RunCodegen({"xnnpack": options or {}})(partitioned)
    assert _has_external_mods(partitioned)

    x_np, residual_np = _tiny_cnn_inputs()
    ref_ex = tvm.compile(bound_mod, target="llvm")
    ref_vm = relax.VirtualMachine(ref_ex, tvm.cpu())
    expected = ref_vm["main"](
        tvm.runtime.tensor(x_np), tvm.runtime.tensor(residual_np)
    ).numpy()

    xnn_ex = tvm.compile(partitioned, target="llvm")
    xnn_vm = relax.VirtualMachine(xnn_ex, tvm.cpu())
    result = xnn_vm["main"](
        tvm.runtime.tensor(x_np), tvm.runtime.tensor(residual_np)
    ).numpy()
    tvm.testing.assert_allclose(result, expected, rtol=rtol, atol=atol)
    return partitioned, expected, (x_np, residual_np)


def _run_first_external_module(mod, inputs, output_shape, output_dtype="float32"):
    ext_mod = mod.attrs["external_mods"][0]
    symbol = ext_mod["get_symbol"]()
    const_names = list(ext_mod["get_const_vars"]())
    const_map = mod.attrs.get("const_name_to_constant", {})
    consts = [const_map[name] for name in const_names]
    ext_mod["__init_" + symbol](consts)

    output_np = np.empty(output_shape, dtype=output_dtype)
    output = tvm.runtime.tensor(output_np)
    ext_mod[symbol](*[tvm.runtime.tensor(input_np) for input_np in inputs], output)
    return ext_mod, output.numpy()


def _init_first_external_module(mod):
    ext_mod = mod.attrs["external_mods"][0]
    symbol = ext_mod["get_symbol"]()
    const_names = list(ext_mod["get_const_vars"]())
    const_map = mod.attrs.get("const_name_to_constant", {})
    consts = [const_map[name] for name in const_names]
    ext_mod["__init_" + symbol](consts)
    return ext_mod, symbol


def _first_external_runtime_options(mod):
    ext_mod = mod.attrs["external_mods"][0]
    return ext_mod["get_runtime_options"]()


def _first_external_graph_json(mod):
    return str(mod.attrs["external_mods"][0].inspect_source("json"))


def _assert_report_fields(report):
    assert report
    expected_fields = {
        "candidate_id",
        "accepted",
        "reason",
        "op_list",
        "dtype",
        "layout",
        "estimated_flops",
        "copy_bytes",
        "padded_copy_bytes",
        "layout_transform_bytes",
        "cast_bytes",
        "external_input_count",
        "external_output_count",
        "boundary_count",
        "compute_to_copy_ratio",
        "policy",
        "quantized",
        "qscheme",
        "qdq_boundary_count",
        "qparam_source",
        "qparam_validation_result",
        "quantized_op_type",
        "qparams_summary",
        "qparam_equality_required",
        "qparam_rejection_reason",
        "dynamic_batch",
        "dynamic_batch_symbol",
        "dynamic_batch_lower",
        "dynamic_batch_upper",
        "estimated_min_flops",
        "estimated_max_flops",
        "estimated_min_copy_bytes",
        "estimated_max_copy_bytes",
    }
    assert expected_fields.issubset(report[0].keys())


def test_xnnpack_python_module_importable():
    from tvm.relax.backend.xnnpack import (
        XNNPACKCostConfig,
        XNNPACKPartitionConfig,
        XNNPACKRuntimeConfig,
        partition_for_xnnpack,
    )

    assert callable(partition_for_xnnpack)
    assert XNNPACKPartitionConfig().runtime == XNNPACKRuntimeConfig()
    assert XNNPACKPartitionConfig().cost == XNNPACKCostConfig()


def test_partition_for_xnnpack_rejects_invalid_precision():
    from tvm.relax.backend.xnnpack import (
        XNNPACKPartitionConfig,
        XNNPACKRuntimeConfig,
        partition_for_xnnpack,
    )

    with pytest.raises(ValueError, match="Unsupported XNNPACK precision"):
        partition_for_xnnpack(
            ReluModule,
            config=XNNPACKPartitionConfig(runtime=XNNPACKRuntimeConfig(precision="explicit_fp16")),
        )


def test_partition_for_xnnpack_rejects_old_keyword_options():
    from tvm.relax.backend.xnnpack import partition_for_xnnpack

    with pytest.raises(TypeError):
        partition_for_xnnpack(ReluModule, quantization="weight_only")


def test_partition_for_xnnpack_rejects_invalid_dynamic_shape_policy():
    from tvm.relax.backend.xnnpack import XNNPACKPartitionConfig, partition_for_xnnpack

    with pytest.raises(ValueError, match="Unsupported XNNPACK dynamic_shape_policy"):
        partition_for_xnnpack(
            ReluModule, config=XNNPACKPartitionConfig(dynamic_shape_policy="full")
        )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"partition_policy": "fast"}, "partition_policy"),
        ({"layout": "NCHW"}, "layout policy"),
        ({"min_subgraph_size": 0}, "min_subgraph_size"),
        ({"min_compute_to_copy_ratio": -1.0}, "min_compute_to_copy_ratio"),
    ],
)
def test_partition_for_xnnpack_rejects_invalid_policy_options(kwargs, match):
    from tvm.relax.backend.xnnpack import XNNPACKCostConfig, XNNPACKPartitionConfig, partition_for_xnnpack

    with pytest.raises(ValueError, match=match):
        partition_for_xnnpack(ReluModule, config=XNNPACKPartitionConfig(cost=XNNPACKCostConfig(**kwargs)))


def test_partition_for_xnnpack_dynamic_batch_requires_bounds():
    from tvm.relax.backend.xnnpack import XNNPACKPartitionConfig, partition_for_xnnpack

    with pytest.raises(ValueError, match="dynamic_shape_policy='batch_only' requires"):
        partition_for_xnnpack(
            DynamicBatchFullyConnectedModule,
            config=XNNPACKPartitionConfig(dynamic_shape_policy="batch_only"),
        )


@pytest.mark.parametrize("bounds", [{"n": 4}, {"n": (1, 4)}, {"n": [1, 4]}])
def test_partition_for_xnnpack_dynamic_batch_partitions_fully_connected_with_api_bounds(bounds):
    mod = _partition(
        _bind_dynamic_batch_fc_params(),
        dynamic_shape_policy="batch_only",
        dynamic_batch_bounds=bounds,
    )
    assert _has_codegen_attr(mod)
    assert "dynamic_batch_fully_connected" in mod.script()


def test_partition_for_xnnpack_dynamic_batch_infers_function_attrs():
    mod = _partition(_bind_dynamic_batch_fc_attrs_params(), dynamic_shape_policy="batch_only")
    assert _has_codegen_attr(mod)
    assert "dynamic_batch_fully_connected" in mod.script()


def test_partition_for_xnnpack_dynamic_batch_default_policy_rejects_symbolic_batch():
    mod = _partition(DynamicBatchFullyConnectedModule)
    assert not _has_codegen_attr(mod)


def test_partition_for_xnnpack_dynamic_batch_partitions_conv2d():
    mod = _partition(
        _bind_dynamic_batch_conv_params(),
        dynamic_shape_policy="batch_only",
        dynamic_batch_bounds={"n": 4},
    )
    assert _has_codegen_attr(mod)
    assert "dynamic_batch_conv2d" in mod.script()


@pytest.mark.parametrize(
    "mod, kwargs",
    [
        (DynamicHWConv2DModule, {}),
        (DynamicChannelConv2DModule, {}),
        (DynamicBatchQS8FullyConnectedModule, {}),
        (DynamicBatchDynamicRangeFullyConnectedModule, {}),
    ],
)
def test_partition_for_xnnpack_dynamic_batch_rejects_unsupported_dynamic_cases(mod, kwargs):
    mod = _partition(
        mod,
        dynamic_shape_policy="batch_only",
        dynamic_batch_bounds={"n": 4, "h": 5, "h_out": 3, "c": 3},
        **kwargs,
    )
    assert not _has_codegen_attr(mod)


def test_xnnpack_dynamic_batch_partition_report_fields():
    mod, report = _partition(
        _bind_dynamic_batch_conv_params(),
        dynamic_shape_policy="batch_only",
        dynamic_batch_bounds={"n": (1, 4)},
        report_partition_decisions=True,
    )
    assert _has_codegen_attr(mod)
    _assert_report_fields(report)
    accepted = [entry for entry in report if entry["accepted"] and entry["dynamic_batch"]]
    assert accepted
    assert accepted[0]["dynamic_batch_symbol"] == "n"
    assert accepted[0]["dynamic_batch_lower"] == 1
    assert accepted[0]["dynamic_batch_upper"] == 4
    assert accepted[0]["estimated_min_flops"] <= accepted[0]["estimated_max_flops"]


def test_xnnpack_registers_relu_pattern():
    import tvm.relax.backend.xnnpack  # noqa: F401

    pattern_names = {pattern.name for pattern in get_patterns_with_prefix("xnnpack")}
    assert {
        "xnnpack.qs8_reshape",
        "xnnpack.qs8_flatten",
        "xnnpack.qs8_copy",
        "xnnpack.qs8_max_pool2d",
        "xnnpack.qs8_add",
        "xnnpack.conv2d_bias_relu",
        "xnnpack.max_pool2d",
        "xnnpack.add",
        "xnnpack.fully_connected_bias_gelu",
        "xnnpack.fully_connected_bias_approx_gelu",
        "xnnpack.softmax",
        "xnnpack.clip",
        "xnnpack.relu",
        "xnnpack.gelu",
        "xnnpack.approx_gelu",
        "xnnpack.sigmoid",
        "xnnpack.tanh",
    }.issubset(pattern_names)


def test_partition_for_xnnpack_partitions_static_float32_relu():
    mod = _partition(ReluModule)
    assert _has_codegen_attr(mod)


@pytest.mark.parametrize(
    "mod, composite",
    [
        (FullyConnectedBiasGeluParamModule, "fully_connected_bias_gelu"),
        (FullyConnectedBiasApproxGeluParamModule, "fully_connected_bias_approx_gelu"),
    ],
)
def test_partition_for_xnnpack_partitions_mlp_fully_connected_gelu(mod, composite):
    mod = _partition(_bind_mlp_params(mod))
    assert _has_codegen_attr(mod)
    assert composite in mod.script()


def test_partition_for_xnnpack_partitions_last_axis_softmax():
    mod = _partition(SoftmaxLastAxisModule)
    assert _has_codegen_attr(mod)
    assert "xnnpack.softmax" in mod.script()


@pytest.mark.parametrize(
    "mod",
    [SoftmaxAxis0Module, GeluFloat16Module, GeluSymbolicModule],
)
def test_partition_for_xnnpack_rejects_unsupported_mlp_patterns(mod):
    mod = _partition(mod)
    assert not _has_codegen_attr(mod)


@pytest.mark.parametrize("mod", [GeluModule, SoftmaxLastAxisModule])
def test_xnnpack_cost_policy_rejects_isolated_mlp_unary(mod):
    mod, report = _partition(
        mod,
        partition_policy="cost",
        report_partition_decisions=True,
    )
    assert not _has_codegen_attr(mod)
    _assert_report_fields(report)
    assert any(entry["reason"] == "rejected_isolated_elementwise" for entry in report)


def test_partition_for_xnnpack_partitions_mlp_residual_block():
    mod = _partition(_bind_mlp_params(MLPResidualModule))
    assert _has_codegen_attr(mod)
    assert _count_xnnpack_partitions(mod) >= 2


def test_partition_for_xnnpack_records_precision_attr():
    mod = _partition(ReluModule, precision="fp16_hint")
    precisions = [
        func.attrs.get("xnnpack_precision")
        for func in mod.functions.values()
        if isinstance(func, relax.Function)
        and func.attrs
        and func.attrs.get("Codegen") == "xnnpack"
    ]
    assert precisions
    assert set(precisions) == {"fp16_hint"}


@pytest.mark.parametrize(
    "mod",
    [
        MultiplyModule,
        AddBroadcastModule,
        ReluFloat16Module,
        ReluSymbolicModule,
        ConvNCHWModule,
        ConvDynamicWeightModule,
        AvgPoolPaddedModule,
    ],
)
def test_partition_for_xnnpack_rejects_unsupported_patterns(mod):
    mod = _partition(mod)
    assert not _has_codegen_attr(mod)

    mod = relax.transform.RunCodegen()(mod)
    assert not _has_external_mods(mod)


@pytest.mark.parametrize("policy", ["greedy", "cost", "debug_all_supported"])
@pytest.mark.parametrize("mod", [QuantizeModule, DequantizeModule])
def test_partition_for_xnnpack_does_not_partition_qdq(policy, mod):
    mod = _partition(mod, partition_policy=policy)
    assert not _has_codegen_attr(mod)

    mod = relax.transform.RunCodegen()(mod)
    assert not _has_external_mods(mod)


@pytest.mark.parametrize(
    "mod",
    [
        DynamicRangeFullyConnectedModule,
        DynamicRangeFullyConnectedBiasRelu6Module,
        DynamicRangeFullyConnectedPerTensorWeightModule,
        DynamicRangeFullyConnectedBadWeightZeroPointModule,
        DynamicRangeFullyConnectedQU8WeightModule,
        QS8Conv2DBiasReluModule,
        QS8DepthwiseConv2DBiasRelu6Module,
        QS8FullyConnectedModule,
        QS8FullyConnectedBiasRelu6Module,
    ],
)
def test_partition_for_xnnpack_rejects_pruned_quantized_patterns(mod):
    mod = _partition(mod)
    assert not _has_codegen_attr(mod)


@tvm.script.ir_module
class QS8FullyConnectedBadWeightZeroPointModule:
    @R.function
    def main(x: R.Tensor((2, 3), "uint8")) -> R.Tensor((2, 4), "int8"):
        with R.dataflow():
            x_f = R.dequantize(
                x, R.const(0.25, "float32"), R.const(0, "uint8"), axis=-1, out_dtype="float32"
            )
            w = R.const(np.ones((3, 4), dtype="int8"))
            w_f = R.dequantize(
                w, R.const(0.5, "float32"), R.const(1, "int8"), axis=1, out_dtype="float32"
            )
            y = R.matmul(x_f, w_f)
            z = R.quantize(
                y, R.const(0.5, "float32"), R.const(0, "int8"), axis=-1, out_dtype="int8"
            )
            R.output(z)
        return z


@pytest.mark.parametrize("mod", [QS8FullyConnectedBadWeightZeroPointModule])
def test_partition_for_xnnpack_rejects_invalid_qs8_qparams(mod):
    mod = _partition(mod)
    assert not _has_codegen_attr(mod)


@pytest.mark.parametrize(
    "mod",
    [
        QS8ReshapeModule,
        QS8FlattenModule,
        QS8CopyModule,
        QS8MaxPool2DModule,
        QS8AddModule,
        QS8AddRelu6Module,
    ],
)
def test_partition_for_xnnpack_partitions_static_qs8_island_ops(mod):
    mod = _partition(mod)
    assert _has_codegen_attr(mod)


@pytest.mark.parametrize(
    "mod",
    [
        QS8ReshapeMismatchedQParamsModule,
        QS8MaxPoolNCHWModule,
        QS8AvgPool2DModule,
        QS8GlobalAvgPoolAsAvgPool2DModule,
        QS8AddBroadcastModule,
    ],
)
def test_partition_for_xnnpack_rejects_unsupported_qs8_island_ops(mod):
    mod = _partition(mod)
    assert not _has_codegen_attr(mod)


def test_xnnpack_cost_policy_reports_qs8_island_rejections():
    reshape_mod, reshape_report = _partition(
        QS8ReshapeModule,
        partition_policy="cost",
        report_partition_decisions=True,
    )
    add_mod, add_report = _partition(
        QS8AddModule,
        partition_policy="cost",
        report_partition_decisions=True,
    )
    assert not _has_codegen_attr(reshape_mod)
    assert not _has_codegen_attr(add_mod)
    _assert_report_fields(reshape_report)
    assert any(entry["reason"] == "rejected_low_compute_to_copy_ratio" for entry in reshape_report)
    assert any(entry["reason"] == "rejected_isolated_elementwise" for entry in add_report)
    accepted_debug, debug_report = _partition(
        QS8AddModule,
        partition_policy="debug_all_supported",
        report_partition_decisions=True,
    )
    assert _has_codegen_attr(accepted_debug)
    assert any(entry["quantized_op_type"] == "qs8_add" for entry in debug_report)


def test_partition_for_xnnpack_rejects_float16_even_with_fp16_policy():
    mod = _partition(ReluFloat16Module, precision="fp16_hint")
    assert not _has_codegen_attr(mod)


@pytest.mark.parametrize("mod", [AddModule, ClipModule, SigmoidModule, TanhModule])
def test_partition_for_xnnpack_partitions_supported_static_fp32_patterns(mod):
    mod = _partition(mod)
    assert _has_codegen_attr(mod)


def test_partition_for_xnnpack_partitions_bound_cnn_pattern():
    mod = _partition(_bind_cnn_params())
    assert _has_codegen_attr(mod)


def test_partition_for_xnnpack_tiny_cnn_partition_count():
    mod = _partition(_bind_tiny_cnn_params())
    assert _count_xnnpack_partitions(mod) == 4


def test_xnnpack_greedy_policy_preserves_partition_count():
    mod = _partition(_bind_tiny_cnn_params(), partition_policy="greedy")
    assert _count_xnnpack_partitions(mod) == 4


def test_xnnpack_debug_policy_partitions_supported_patterns():
    mod = _partition(ReluModule, partition_policy="debug_all_supported")
    assert _has_codegen_attr(mod)


def test_xnnpack_cost_policy_rejects_isolated_unary_and_small_binary():
    relu_mod, relu_report = _partition(
        ReluModule,
        partition_policy="cost",
        report_partition_decisions=True,
    )
    add_mod, add_report = _partition(
        AddModule,
        partition_policy="cost",
        report_partition_decisions=True,
    )
    assert not _has_codegen_attr(relu_mod)
    assert not _has_codegen_attr(add_mod)
    _assert_report_fields(relu_report)
    assert any(entry["reason"] == "rejected_isolated_elementwise" for entry in relu_report)
    assert any(entry["reason"] == "rejected_isolated_elementwise" for entry in add_report)


def test_xnnpack_cost_policy_accepts_conv_and_tiny_cnn_island():
    conv_mod, conv_report = _partition(
        _bind_cnn_params(),
        partition_policy="cost",
        report_partition_decisions=True,
    )
    tiny_mod, tiny_report = _partition(
        _bind_tiny_cnn_params(),
        partition_policy="cost",
        report_partition_decisions=True,
    )
    assert _count_xnnpack_partitions(conv_mod) >= 1
    assert _count_xnnpack_partitions(tiny_mod) >= 1
    assert any(entry["reason"] == "accepted_compute_heavy" for entry in conv_report)
    assert any(entry["reason"] == "accepted_compute_heavy" for entry in tiny_report)


def test_xnnpack_cost_policy_reports_float16_rejection():
    mod, report = _partition(
        ReluFloat16Module,
        precision="fp16_hint",
        partition_policy="cost",
        report_partition_decisions=True,
    )
    assert not _has_codegen_attr(mod)
    _assert_report_fields(report)
    assert any(entry["reason"] == "rejected_unsupported_dtype" for entry in report)


def test_xnnpack_cost_policy_reports_layout_rewrite_rejection():
    mod, report = _partition(
        ConvNCHWModule,
        partition_policy="cost",
        layout="NHWC",
        report_partition_decisions=True,
    )
    assert not _has_codegen_attr(mod)
    _assert_report_fields(report)
    assert any(entry["reason"] == "rejected_layout_rewrite_overhead" for entry in report)


def test_xnnpack_partition_report_has_stable_fields_and_reasons():
    _, report = _partition(
        _bind_cnn_params(),
        partition_policy="cost",
        report_partition_decisions=True,
    )
    _assert_report_fields(report)
    assert report[0]["candidate_id"] == 0
    assert report[0]["policy"] == "cost"
    assert isinstance(report[0]["op_list"], list)


def test_xnnpack_benchmark_report_helpers_are_stable():
    bench = _load_xnnpack_benchmark_module()

    class FakeResult:
        results = [0.001, 0.002, 0.004]

    formatted = bench.format_result(FakeResult())
    assert "p50_ms" in formatted
    assert "p90_ms" in formatted
    assert "p99_ms" in formatted
    assert "steady_state_mean_ms" in formatted

    summary = bench.summarize_partition_report(
        [
            {
                "accepted": True,
                "reason": "accepted_compute_heavy",
                "copy_bytes": 16,
                "padded_copy_bytes": 64,
                "layout_transform_bytes": 0,
                "cast_bytes": 0,
                "estimated_flops": 128,
            },
            {
                "accepted": False,
                "reason": "rejected_low_compute_to_copy_ratio",
                "copy_bytes": 4,
                "padded_copy_bytes": 32,
                "layout_transform_bytes": 0,
                "cast_bytes": 0,
                "estimated_flops": 2,
            },
        ]
    )
    assert summary["accepted"] == 1
    assert summary["rejected"] == 1
    assert summary["totals"]["copy_bytes"] == 20
    assert summary["reasons"]["rejected_low_compute_to_copy_ratio"] == 1
    assert summary["accepted_candidate_ratio"] == 0.5
    assert summary["accepted_flop_ratio"] > 0.0


def test_xnnpack_benchmark_model_listing_and_args():
    bench = _load_xnnpack_benchmark_module()
    models = set(bench.available_models())
    assert "xnnpack_large_cnn_fp32" in models
    assert "xnnpack_large_mlp_fp32" in models
    assert "xnnpack_large_qs8_island" in models
    assert "torchvision:mobilenet_v2" in models

    args = bench.parse_args(
        [
            "--model",
            "xnnpack_cnn_fp32",
            "--model-size",
            "large",
            "--compare-models",
            "xnnpack_large_cnn_fp32,xnnpack_large_mlp_fp32",
            "--dump-partition-report-json",
            "/tmp/xnnpack-report.json",
        ]
    )
    assert args.model_size == "large"
    assert args.compare_models == "xnnpack_large_cnn_fp32,xnnpack_large_mlp_fp32"
    assert bench.resolve_model_name(args.model, args.quantization_mode, args.model_size) == (
        "xnnpack_large_cnn_fp32"
    )


@pytest.mark.parametrize(
    "loader",
    ["load_large_cnn", "load_large_mlp", "load_large_static_qs8_island"],
)
def test_xnnpack_benchmark_large_fixtures_construct_without_torch(loader):
    bench = _load_xnnpack_benchmark_module()
    mod, inputs, model_name = getattr(bench, loader)(seed=0)
    metadata = bench.model_metadata(mod, inputs, model_name)
    assert metadata["fixture_size"] == "large"
    assert metadata["input_shapes"]
    assert metadata["op_count_estimate"] > 0


@pytest.mark.parametrize(
    "loader",
    ["load_large_cnn", "load_large_mlp", "load_large_static_qs8_island"],
)
def test_xnnpack_benchmark_large_fixtures_partition_report(loader):
    bench = _load_xnnpack_benchmark_module()
    mod, _, _ = getattr(bench, loader)(seed=0)
    mod, report = _partition(mod, report_partition_decisions=True)
    assert _has_codegen_attr(mod)
    _assert_report_fields(report)
    summary = bench.summarize_partition_report(report)
    assert summary["candidates"] >= summary["accepted"] >= 1


def test_xnnpack_benchmark_torchvision_missing_dependency_reports_cleanly(monkeypatch):
    bench = _load_xnnpack_benchmark_module()
    original_find_spec = bench.importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in ("torch", "torchvision"):
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(bench.importlib.util, "find_spec", fake_find_spec)
    args = bench.parse_args(
        ["--model", "torchvision:resnet18", "--number", "1", "--repeat", "1"]
    )
    result = bench.run_benchmark(args)
    assert result["baseline_status"] == "not run"
    assert "torch and torchvision are required" in result["load_error"]


def test_xnnpack_benchmark_static_qs8_fixture_partitions():
    bench = _load_xnnpack_benchmark_module()
    mod, _, _ = bench.load_static_qs8_island(seed=0)
    mod, report = _partition(mod, report_partition_decisions=True)
    assert _has_codegen_attr(mod)
    _assert_report_fields(report)
    assert any(entry["accepted"] and entry["quantized"] for entry in report)


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
def test_xnnpack_relu_vm_execution():
    mod = _partition(ReluModule)
    assert _has_codegen_attr(mod)
    mod = relax.transform.RunCodegen()(mod)
    assert _has_external_mods(mod)

    ex = tvm.compile(mod, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())

    x_np = np.array([[-1.0, 0.0, 1.5], [2.0, -3.0, 4.0]], dtype="float32")
    result = vm["main"](tvm.runtime.tensor(x_np)).numpy()
    tvm.testing.assert_allclose(result, np.maximum(x_np, 0.0), rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
@pytest.mark.parametrize(
    "mod, capability",
    [
        (FullyConnectedBiasGeluParamModule, "unary_gelu"),
        (FullyConnectedBiasApproxGeluParamModule, "unary_approxgelu"),
    ],
)
def test_xnnpack_mlp_fully_connected_gelu_vm_execution(mod, capability):
    capabilities = _xnnpack_capabilities()
    if not capabilities.get("fully_connected") or not capabilities.get("transpose_weights"):
        pytest.skip("XNNPACK fully_connected subgraph API is unavailable")
    if not capabilities.get(capability):
        pytest.skip(f"XNNPACK {capability} API is unavailable")
    bound_mod = _bind_mlp_params(mod)
    partitioned = _partition(bound_mod)
    assert _has_codegen_attr(partitioned)
    partitioned = relax.transform.RunCodegen()(partitioned)
    assert _has_external_mods(partitioned)

    x_np = np.linspace(-1.0, 1.0, num=4 * 8, dtype="float32").reshape(4, 8)
    ref_ex = tvm.compile(bound_mod, target="llvm")
    ref_vm = relax.VirtualMachine(ref_ex, tvm.cpu())
    expected = ref_vm["main"](tvm.runtime.tensor(x_np)).numpy()

    xnn_ex = tvm.compile(partitioned, target="llvm")
    xnn_vm = relax.VirtualMachine(xnn_ex, tvm.cpu())
    result = xnn_vm["main"](tvm.runtime.tensor(x_np)).numpy()
    tvm.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
def test_xnnpack_softmax_vm_execution():
    if not _xnnpack_capabilities().get("softmax"):
        pytest.skip("XNNPACK softmax subgraph API is unavailable")
    partitioned = _partition(SoftmaxLastAxisModule)
    assert _has_codegen_attr(partitioned)
    partitioned = relax.transform.RunCodegen()(partitioned)
    assert _has_external_mods(partitioned)

    x_np = np.linspace(-2.0, 2.0, num=2 * 3 * 4, dtype="float32").reshape(2, 3, 4)
    x_shifted = x_np - np.max(x_np, axis=-1, keepdims=True)
    expected = np.exp(x_shifted) / np.sum(np.exp(x_shifted), axis=-1, keepdims=True)

    xnn_ex = tvm.compile(partitioned, target="llvm")
    xnn_vm = relax.VirtualMachine(xnn_ex, tvm.cpu())
    result = xnn_vm["main"](tvm.runtime.tensor(x_np)).numpy()
    tvm.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
def test_xnnpack_mlp_residual_vm_execution():
    capabilities = _xnnpack_capabilities()
    if not (
        capabilities.get("fully_connected")
        and capabilities.get("transpose_weights")
        and capabilities.get("unary_gelu")
    ):
        pytest.skip("XNNPACK fully_connected/GELU APIs are unavailable")
    bound_mod = _bind_mlp_params(MLPResidualModule)
    partitioned = _partition(bound_mod)
    assert _has_codegen_attr(partitioned)
    partitioned = relax.transform.RunCodegen()(partitioned)
    assert _has_external_mods(partitioned)

    x_np = np.linspace(-1.0, 1.0, num=4 * 8, dtype="float32").reshape(4, 8)
    residual_np = np.linspace(0.25, -0.25, num=4 * 16, dtype="float32").reshape(4, 16)
    ref_ex = tvm.compile(bound_mod, target="llvm")
    ref_vm = relax.VirtualMachine(ref_ex, tvm.cpu())
    expected = ref_vm["main"](
        tvm.runtime.tensor(x_np), tvm.runtime.tensor(residual_np)
    ).numpy()

    xnn_ex = tvm.compile(partitioned, target="llvm")
    xnn_vm = relax.VirtualMachine(xnn_ex, tvm.cpu())
    result = xnn_vm["main"](
        tvm.runtime.tensor(x_np), tvm.runtime.tensor(residual_np)
    ).numpy()
    tvm.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
def test_xnnpack_cnn_vm_execution():
    bound_mod = _bind_cnn_params()
    partitioned = _partition(bound_mod)
    assert _has_codegen_attr(partitioned)
    partitioned = relax.transform.RunCodegen()(partitioned)
    assert _has_external_mods(partitioned)

    x_np = np.linspace(-1.0, 1.0, num=1 * 5 * 5 * 3, dtype="float32").reshape(1, 5, 5, 3)

    ref_ex = tvm.compile(bound_mod, target="llvm")
    ref_vm = relax.VirtualMachine(ref_ex, tvm.cpu())
    expected = ref_vm["main"](tvm.runtime.tensor(x_np)).numpy()

    xnn_ex = tvm.compile(partitioned, target="llvm")
    xnn_vm = relax.VirtualMachine(xnn_ex, tvm.cpu())
    result = xnn_vm["main"](tvm.runtime.tensor(x_np)).numpy()
    tvm.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
def test_xnnpack_tiny_cnn_vm_execution():
    _run_tiny_cnn_with_options()


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
def test_xnnpack_cost_policy_tiny_cnn_vm_execution():
    bound_mod = _bind_tiny_cnn_params()
    partitioned = _partition(bound_mod, partition_policy="cost")
    assert _count_xnnpack_partitions(partitioned) >= 1
    partitioned = relax.transform.RunCodegen()(partitioned)
    assert _has_external_mods(partitioned)

    x_np, residual_np = _tiny_cnn_inputs()
    ref_ex = tvm.compile(bound_mod, target="llvm")
    ref_vm = relax.VirtualMachine(ref_ex, tvm.cpu())
    expected = ref_vm["main"](
        tvm.runtime.tensor(x_np), tvm.runtime.tensor(residual_np)
    ).numpy()

    xnn_ex = tvm.compile(partitioned, target="llvm")
    xnn_vm = relax.VirtualMachine(xnn_ex, tvm.cpu())
    result = xnn_vm["main"](
        tvm.runtime.tensor(x_np), tvm.runtime.tensor(residual_np)
    ).numpy()
    tvm.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
def test_xnnpack_cost_policy_rejected_relu_has_no_external_modules():
    mod = _partition(ReluModule, partition_policy="cost")
    assert not _has_codegen_attr(mod)
    mod = relax.transform.RunCodegen()(mod)
    assert not _has_external_mods(mod)


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
def test_xnnpack_cost_policy_composes_with_runtime_options():
    if not _xnnpack_capability("fp16_hint"):
        pytest.skip("XNNPACK FP16 hint flag is unavailable")
    mod = _partition(_bind_cnn_params(), partition_policy="cost", precision="fp16_hint")
    assert _has_codegen_attr(mod)
    options = {"num_threads": 1, "precision": "fp16_hint"}
    mod = relax.transform.RunCodegen({"xnnpack": options})(mod)
    assert "precision=fp16_hint" in _first_external_runtime_options(mod)


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
def test_xnnpack_dynamic_batch_fully_connected_external_runtime():
    if not _xnnpack_capability("dynamic_batch_runtime"):
        pytest.skip("XNNPACK runtime reshape APIs are unavailable")
    partitioned = _partition(
        _bind_dynamic_batch_fc_params(),
        dynamic_shape_policy="batch_only",
        dynamic_batch_bounds={"n": 4},
    )
    codegen_mod = relax.transform.RunCodegen()(partitioned)
    assert _has_external_mods(codegen_mod)
    ext_mod, symbol = _init_first_external_module(codegen_mod)
    weight = _dynamic_batch_fc_weight()

    counters = []
    for n in [1, 1, 2, 4]:
        x_np = np.arange(n * 3, dtype="float32").reshape(n, 3) / 4.0
        output = tvm.runtime.tensor(np.empty((n, 4), dtype="float32"))
        ext_mod[symbol](tvm.runtime.tensor(x_np), output)
        tvm.testing.assert_allclose(output.numpy(), x_np @ weight, rtol=1e-5, atol=1e-5)
        counters.append(json.loads(ext_mod["get_runtime_counters"]()))
    assert counters[0]["reshape_count"] == 1
    assert counters[1]["reshape_count"] == counters[0]["reshape_count"]
    assert counters[2]["reshape_count"] == counters[1]["reshape_count"] + 1
    assert counters[3]["last_batch_size"] == 4


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
def test_xnnpack_dynamic_batch_conv2d_external_runtime():
    if not _xnnpack_capability("dynamic_batch_runtime"):
        pytest.skip("XNNPACK runtime reshape APIs are unavailable")
    bound_mod = _bind_dynamic_batch_conv_params()
    partitioned = _partition(
        bound_mod,
        dynamic_shape_policy="batch_only",
        dynamic_batch_bounds={"n": 4},
    )
    codegen_mod = relax.transform.RunCodegen()(partitioned)
    assert _has_external_mods(codegen_mod)
    ext_mod, symbol = _init_first_external_module(codegen_mod)

    ref_ex = tvm.compile(bound_mod, target="llvm")
    ref_vm = relax.VirtualMachine(ref_ex, tvm.cpu())
    counters = []
    for n in [1, 2, 4]:
        x_np = np.linspace(-1.0, 1.0, num=n * 5 * 5 * 3, dtype="float32").reshape(n, 5, 5, 3)
        expected = ref_vm["main"](tvm.runtime.tensor(x_np)).numpy()
        output = tvm.runtime.tensor(np.empty((n, 3, 3, 4), dtype="float32"))
        ext_mod[symbol](tvm.runtime.tensor(x_np), output)
        tvm.testing.assert_allclose(output.numpy(), expected, rtol=1e-5, atol=1e-5)
        counters.append(json.loads(ext_mod["get_runtime_counters"]()))
    assert counters[0]["reshape_count"] == 1
    assert counters[-1]["last_batch_size"] == 4


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
def test_xnnpack_dynamic_batch_out_of_bounds_fails_clearly():
    if not _xnnpack_capability("dynamic_batch_runtime"):
        pytest.skip("XNNPACK runtime reshape APIs are unavailable")
    partitioned = _partition(
        _bind_dynamic_batch_fc_params(),
        dynamic_shape_policy="batch_only",
        dynamic_batch_bounds={"n": 2},
    )
    codegen_mod = relax.transform.RunCodegen()(partitioned)
    ext_mod, symbol = _init_first_external_module(codegen_mod)
    x_np = np.zeros((3, 3), dtype="float32")
    output = tvm.runtime.tensor(np.empty((3, 4), dtype="float32"))
    with pytest.raises(tvm.error.TVMError, match="upper bound"):
        ext_mod[symbol](tvm.runtime.tensor(x_np), output)


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
@pytest.mark.parametrize(
    "input_np, output_np, match",
    [
        (np.ones((2, 3), dtype="int32"), np.empty((2, 3), dtype="float32"), "dtype mismatch"),
        (np.ones((6,), dtype="float32"), np.empty((2, 3), dtype="float32"), "rank mismatch"),
        (
            np.ones((3, 2), dtype="float32"),
            np.empty((3, 2), dtype="float32"),
            "shape mismatch",
        ),
        (np.ones((2, 3), dtype="float32"), np.empty((2, 3), dtype="int32"), "dtype mismatch"),
    ],
)
def test_xnnpack_runtime_rejects_invalid_external_tensors(input_np, output_np, match):
    mod = relax.transform.RunCodegen()(_partition(ReluModule))
    ext_mod, symbol = _init_first_external_module(mod)
    with pytest.raises(tvm.error.TVMError, match=match):
        ext_mod[symbol](tvm.runtime.tensor(input_np), tvm.runtime.tensor(output_np))


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
def test_xnnpack_dynamic_batch_lower_bound_fails_clearly():
    if not _xnnpack_capability("dynamic_batch_runtime"):
        pytest.skip("XNNPACK runtime reshape APIs are unavailable")
    partitioned = _partition(
        _bind_dynamic_batch_fc_params(),
        dynamic_shape_policy="batch_only",
        dynamic_batch_bounds={"n": (2, 4)},
    )
    codegen_mod = relax.transform.RunCodegen()(partitioned)
    ext_mod, symbol = _init_first_external_module(codegen_mod)
    x_np = np.zeros((1, 3), dtype="float32")
    output = tvm.runtime.tensor(np.empty((1, 4), dtype="float32"))
    with pytest.raises(tvm.error.TVMError, match="lower bound"):
        ext_mod[symbol](tvm.runtime.tensor(x_np), output)


@pytest.mark.skipif(not _has_xnnpack_runtime(), reason="XNNPACK runtime is not enabled")
def test_xnnpack_runtime_rejects_malformed_options():
    create = _xnnpack_runtime_create()
    assert create is not None
    mod = relax.transform.RunCodegen()(_partition(ReluModule))
    graph_json = _first_external_graph_json(mod)
    with pytest.raises(tvm.error.TVMError, match="must be 0 or 1"):
        create("bad_options", graph_json, [], "use_weights_cache=true;")
    with pytest.raises(tvm.error.TVMError, match="batch symbol"):
        create(
            "bad_dynamic",
            graph_json,
            [],
            "dynamic_shape_policy=batch_only;dynamic_batch_lower=1;dynamic_batch_upper=4;",
        )


@pytest.mark.skipif(not _has_xnnpack_runtime(), reason="XNNPACK runtime is not enabled")
@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda graph: graph["nodes"][1]["attrs"].pop("op_kind"), "op_kind"),
        (lambda graph: graph["nodes"][1]["attrs"].__setitem__("op_kind", "bogus"), "op_kind"),
        (lambda graph: graph["heads"].__setitem__(0, [99, 0, 0]), "output"),
        (lambda graph: graph["nodes"][1]["inputs"].__setitem__(0, [99, 0, 0]), "input"),
        (lambda graph: graph["nodes"][0]["attrs"]["dtype"].append("float32"), "shape"),
    ],
)
def test_xnnpack_runtime_rejects_malformed_json_metadata(mutate, match):
    create = _xnnpack_runtime_create()
    assert create is not None
    mod = relax.transform.RunCodegen()(_partition(ReluModule))
    graph = json.loads(_first_external_graph_json(mod))
    mutate(graph)
    with pytest.raises(tvm.error.TVMError, match=match):
        create("bad_json", json.dumps(graph), [], _first_external_runtime_options(mod))


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
def test_xnnpack_runtime_options_persist_precision():
    mod = _partition(ReluModule, precision="fp16_hint")
    mod = relax.transform.RunCodegen()(mod)
    assert "precision=fp16_hint" in _first_external_runtime_options(mod)


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
def test_xnnpack_runcodegen_precision_conflict_rejected():
    mod = _partition(ReluModule, precision="fp16_hint")
    with pytest.raises(tvm.error.TVMError, match="must match"):
        relax.transform.RunCodegen({"xnnpack": {"precision": "fp32"}})(mod)


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
def test_xnnpack_tiny_cnn_fp16_hint_precision():
    if not _xnnpack_capability("fp16_hint"):
        pytest.skip("XNNPACK FP16 hint flag is unavailable")
    mod, _, _ = _run_tiny_cnn_with_options(precision="fp16_hint", rtol=5e-2, atol=5e-2)
    assert "precision=fp16_hint" in _first_external_runtime_options(mod)


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
def test_xnnpack_tiny_cnn_fp16_force_precision():
    if not _xnnpack_capability("fp16_force"):
        pytest.skip("XNNPACK FP16 force flag is unavailable")
    try:
        mod, _, _ = _run_tiny_cnn_with_options(precision="fp16_force", rtol=5e-2, atol=5e-2)
    except tvm.error.TVMError as err:
        assert "fp16_force" in str(err) or "FP16 runtime" in str(err)
    else:
        assert "precision=fp16_force" in _first_external_runtime_options(mod)


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
def test_xnnpack_fp16_hint_composes_with_runtime_options():
    if not _xnnpack_capability("fp16_hint"):
        pytest.skip("XNNPACK FP16 hint flag is unavailable")
    options = {
        "use_weights_cache": _xnnpack_capability("weights_cache"),
        "use_workspace": _xnnpack_capability("runtime_v4") and _xnnpack_capability("workspace"),
        "profile": _xnnpack_capability("profiling"),
        "dont_spin_workers": _xnnpack_capability("dont_spin_workers"),
        "transient_indirection_buffer": _xnnpack_capability("transient_indirection_buffer"),
        "num_threads": 1,
        "precision": "fp16_hint",
    }
    mod, _, _ = _run_tiny_cnn_with_options(options, precision="fp16_hint", rtol=5e-2, atol=5e-2)
    runtime_options = _first_external_runtime_options(mod)
    assert "precision=fp16_hint" in runtime_options
    assert "num_threads=1" in runtime_options


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
@pytest.mark.parametrize("use_weights_cache", [False, True])
def test_xnnpack_tiny_cnn_weights_cache_option(use_weights_cache):
    if use_weights_cache and not _xnnpack_capability("weights_cache"):
        pytest.skip("XNNPACK weights cache is unavailable")
    _run_tiny_cnn_with_options({"use_weights_cache": use_weights_cache})


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
@pytest.mark.parametrize("use_workspace", [False, True])
def test_xnnpack_tiny_cnn_workspace_option(use_workspace):
    if use_workspace and not (
        _xnnpack_capability("runtime_v4") and _xnnpack_capability("workspace")
    ):
        pytest.skip("XNNPACK workspace runtime is unavailable")
    _run_tiny_cnn_with_options({"use_workspace": use_workspace})


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
def test_xnnpack_tiny_cnn_threading_and_runtime_flags():
    options = {
        "dont_spin_workers": _xnnpack_capability("dont_spin_workers"),
        "transient_indirection_buffer": _xnnpack_capability("transient_indirection_buffer"),
        "num_threads": 1,
    }
    _run_tiny_cnn_with_options(options)


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
def test_xnnpack_tiny_cnn_num_threads_two():
    if not _xnnpack_capability("pthreadpool"):
        pytest.skip("XNNPACK pthreadpool is unavailable")
    _run_tiny_cnn_with_options({"num_threads": 2})


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
def test_xnnpack_multiple_modules_with_weights_cache():
    if not _xnnpack_capability("weights_cache"):
        pytest.skip("XNNPACK weights cache is unavailable")
    _run_tiny_cnn_with_options({"use_weights_cache": True})
    _run_tiny_cnn_with_options({"use_weights_cache": True})


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
def test_xnnpack_profile_json():
    if not _xnnpack_capability("profiling"):
        pytest.skip("XNNPACK profiling is unavailable")
    mod = _partition(ReluModule)
    mod = relax.transform.RunCodegen({"xnnpack": {"profile": True}})(mod)
    x_np = np.array([[-1.0, 0.0, 1.5], [2.0, -3.0, 4.0]], dtype="float32")
    expected = np.maximum(x_np, 0.0)
    ext_mod, output = _run_first_external_module(mod, [x_np], expected.shape)
    tvm.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)
    profile_json = ext_mod["get_profile_json"]()
    assert "time_ns" in profile_json


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
def test_xnnpack_runtime_quantization_metadata_debug_dump_empty_for_fp32_graph():
    mod = _partition(ReluModule)
    mod = relax.transform.RunCodegen()(mod)
    x_np = np.array([[-1.0, 0.0, 1.5], [2.0, -3.0, 4.0]], dtype="float32")
    ext_mod, _ = _run_first_external_module(mod, [x_np], x_np.shape)
    assert json.loads(ext_mod["get_quantization_metadata_json"]()) == []


@pytest.mark.skipif(
    not (_has_xnnpack_codegen() and _has_xnnpack_runtime()),
    reason="XNNPACK codegen/runtime is not enabled",
)
@pytest.mark.parametrize(
    "mod, inputs, output_shape",
    [
        (QS8ReshapeModule, [np.array([[-3, -1, 2], [4, 1, -2]], dtype="int8")], (1, 6)),
        (QS8FlattenModule, [np.arange(-12, 12, dtype="int8").reshape(2, 3, 4)], (24,)),
        (QS8CopyModule, [np.array([[-3, -1, 2], [4, 1, -2]], dtype="int8")], (2, 3)),
        (QS8MaxPool2DModule, [np.arange(-16, 16, dtype="int8").reshape(1, 4, 4, 2)], (1, 2, 2, 2)),
        (
            QS8AddModule,
            [
                np.arange(-16, 16, dtype="int8").reshape(1, 4, 4, 2),
                np.arange(16, -16, -1, dtype="int8").reshape(1, 4, 4, 2),
            ],
            (1, 4, 4, 2),
        ),
        (
            QS8AddRelu6Module,
            [
                np.arange(-16, 16, dtype="int8").reshape(1, 4, 4, 2),
                np.arange(16, -16, -1, dtype="int8").reshape(1, 4, 4, 2),
            ],
            (1, 4, 4, 2),
        ),
    ],
)
def test_xnnpack_qs8_island_ops_external_runtime(mod, inputs, output_shape):
    capabilities = _xnnpack_capabilities()
    required = capabilities.get("datatype_qint8") and capabilities.get(
        "define_quantized_tensor_value"
    )
    if mod in (QS8ReshapeModule, QS8FlattenModule) and not capabilities.get("static_reshape"):
        pytest.skip("XNNPACK static reshape API is unavailable")
    if mod is QS8CopyModule and not capabilities.get("copy"):
        pytest.skip("XNNPACK copy API is unavailable")
    if not required:
        pytest.skip("XNNPACK QS8 tensor APIs are unavailable")
    partitioned = _partition(mod)
    assert _has_codegen_attr(partitioned)
    codegen_mod = relax.transform.RunCodegen()(partitioned)
    assert _has_external_mods(codegen_mod)

    ref_ex = tvm.compile(mod, target="llvm")
    ref_vm = relax.VirtualMachine(ref_ex, tvm.cpu())
    expected = ref_vm["main"](*[tvm.runtime.tensor(input_np) for input_np in inputs]).numpy()
    ext_mod, result = _run_first_external_module(
        codegen_mod, inputs, output_shape, output_dtype="int8"
    )
    max_diff = np.max(np.abs(result.astype("int16") - expected.astype("int16")))
    assert max_diff <= 1
    metadata = json.loads(ext_mod["get_quantization_metadata_json"]())
    assert metadata


@pytest.mark.skipif(not _has_xnnpack_codegen(), reason="XNNPACK codegen is not enabled")
def test_xnnpack_codegen_registration_accepts_empty_input():
    codegen = tvm.get_global_func("relax.ext.xnnpack")
    assert len(codegen([], {}, {})) == 0


@pytest.mark.skipif(not _has_xnnpack_runtime(), reason="XNNPACK runtime is not enabled")
def test_xnnpack_runtime_registration_available():
    assert tvm.get_global_func("runtime.XNNPACKJSONRuntimeCreate") is not None


@pytest.mark.skipif(not _has_xnnpack_runtime(), reason="XNNPACK runtime is not enabled")
def test_xnnpack_quantization_capabilities_are_reported():
    capabilities = _xnnpack_capabilities()
    assert "runtime_v2" in capabilities
    assert "baseline_subgraph" in capabilities
    assert "baseline_fp32_ops" in capabilities
    assert "fp16_flags" in capabilities
    assert "datatype_qint8" in capabilities
    assert "datatype_quint8" in capabilities
    assert "datatype_qcint8" in capabilities
    assert "qs8_datatypes" in capabilities
    assert "qs8_subgraph_ops" in capabilities
    assert "unary_gelu" in capabilities
    assert "unary_approxgelu" in capabilities
    assert "softmax" in capabilities
    assert "extra_quantization_params" in capabilities
    assert "runtime_reshape" in capabilities
    assert "reshape_external_value" in capabilities
    assert "setup_runtime_v2" in capabilities
    assert "get_external_value_shape" in capabilities
    assert "dynamic_batch_runtime" in capabilities


@pytest.mark.skipif(not _has_xnnpack_runtime(), reason="XNNPACK runtime is not enabled")
def test_xnnpack_quantization_metadata_per_tensor_roundtrip():
    validator = _quant_metadata_validator()
    assert validator is not None
    result = json.loads(
        validator(
            {
                "dtype": "int8",
                "qscheme": "per_tensor",
                "scale": 0.25,
                "zero_point": 3,
                "axis": -1,
                "channel_dim": 1,
                "signedness": "signed",
            },
            [2, 4],
        )
    )
    assert result["dtype"] == "int8"
    assert result["qscheme"] == "per_tensor"
    assert result["scale"] == pytest.approx(0.25)
    assert result["zero_point"] == 3
    assert result["xnn_datatype"] == "xnn_datatype_qint8"


@pytest.mark.skipif(not _has_xnnpack_runtime(), reason="XNNPACK runtime is not enabled")
def test_xnnpack_quantization_metadata_per_channel_roundtrip():
    validator = _quant_metadata_validator()
    assert validator is not None
    result = json.loads(
        validator(
            {
                "dtype": "int8",
                "qscheme": "per_channel",
                "scale": [0.25, 0.5, 1.0],
                "zero_point": 0,
                "axis": 0,
                "channel_dim": 0,
                "signedness": "signed",
            },
            [3, 3, 3, 4],
        )
    )
    assert result["qscheme"] == "per_channel"
    assert result["scale"] == pytest.approx([0.25, 0.5, 1.0])
    assert result["xnn_datatype"] == "xnn_datatype_qcint8"
    assert result["padded_scale_length"] >= 3


@pytest.mark.skipif(not _has_xnnpack_runtime(), reason="XNNPACK runtime is not enabled")
@pytest.mark.parametrize(
    "metadata, shape, match",
    [
        (
            {
                "dtype": "int8",
                "qscheme": "per_tensor",
                "scale": 0.0,
                "zero_point": 0,
                "axis": -1,
                "channel_dim": 1,
                "signedness": "signed",
            },
            [2, 4],
            "positive",
        ),
        (
            {
                "dtype": "int8",
                "qscheme": "per_tensor",
                "scale": float("nan"),
                "zero_point": 0,
                "axis": -1,
                "channel_dim": 1,
                "signedness": "signed",
            },
            [2, 4],
            "finite",
        ),
        (
            {
                "dtype": "int8",
                "qscheme": "per_tensor",
                "scale": float("inf"),
                "zero_point": 0,
                "axis": -1,
                "channel_dim": 1,
                "signedness": "signed",
            },
            [2, 4],
            "finite",
        ),
        (
            {
                "dtype": "int8",
                "qscheme": "per_tensor",
                "scale": 0.5,
                "zero_point": 200,
                "axis": -1,
                "channel_dim": 1,
                "signedness": "signed",
            },
            [2, 4],
            "zero_point",
        ),
        (
            {
                "dtype": "int8",
                "qscheme": "per_channel",
                "scale": [0.5, 1.0],
                "zero_point": 0,
                "axis": 0,
                "channel_dim": 0,
                "signedness": "signed",
            },
            [3, 3, 3, 4],
            "scale length",
        ),
        (
            {
                "dtype": "int8",
                "qscheme": "per_channel",
                "scale": [0.5, 1.0, 2.0],
                "zero_point": 0,
                "axis": 1,
                "channel_dim": 0,
                "signedness": "signed",
            },
            [3, 3, 3, 4],
            "axis must match",
        ),
        (
            {
                "dtype": "uint8",
                "qscheme": "per_channel",
                "scale": [0.5, 1.0, 2.0],
                "zero_point": 0,
                "axis": 0,
                "channel_dim": 0,
                "signedness": "unsigned",
            },
            [3, 3, 3, 4],
            "per-channel",
        ),
        (
            {
                "dtype": "uint8",
                "qscheme": "per_tensor",
                "scale": 0.5,
                "zero_point": 0,
                "axis": -1,
                "channel_dim": 1,
                "signedness": "signed",
            },
            [2, 4],
            "signedness",
        ),
        (
            {
                "dtype": "int8",
                "qscheme": "per_tensor",
                "scale": 0.5,
                "zero_point": 0,
                "axis": -1,
                "channel_dim": 1,
                "signedness": "signed",
            },
            [2**62, 8],
            "overflow",
        ),
    ],
)
def test_xnnpack_quantization_metadata_invalid_qparams(metadata, shape, match):
    validator = _quant_metadata_validator()
    assert validator is not None
    with pytest.raises(tvm.error.TVMError, match=match):
        validator(metadata, shape)


@pytest.mark.skipif(not _has_xnnpack_runtime(), reason="XNNPACK runtime is not enabled")
def test_xnnpack_quantized_tensor_definition_smoke():
    capabilities = _xnnpack_capabilities()
    if not (
        capabilities.get("define_quantized_tensor_value")
        and capabilities.get("define_channelwise_quantized_tensor_value")
    ):
        pytest.skip("XNNPACK quantized tensor definition APIs are unavailable")
    smoke = _quant_tensor_smoke()
    assert smoke is not None

    per_tensor = json.loads(
        smoke(
            {
                "dtype": "int8",
                "qscheme": "per_tensor",
                "scale": 0.5,
                "zero_point": 0,
                "axis": -1,
                "channel_dim": 1,
                "signedness": "signed",
            },
            [2, 4],
        )
    )
    assert per_tensor["xnn_datatype"] == "xnn_datatype_qint8"

    per_channel = json.loads(
        smoke(
            {
                "dtype": "int8",
                "qscheme": "per_channel",
                "scale": [0.25, 0.5, 1.0],
                "zero_point": 0,
                "axis": 0,
                "channel_dim": 0,
                "signedness": "signed",
            },
            [3, 3, 3, 4],
        )
    )
    assert per_channel["xnn_datatype"] == "xnn_datatype_qcint8"


if __name__ == "__main__":
    tvm.testing.main()
