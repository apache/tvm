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
# ruff: noqa: RUF005
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm.contrib.pickle_memoize import memoize
from tvm.relax.dpl import is_op, make_fused_bias_activation_pattern, wildcard
from tvm.script import relax as R


@tvm.script.ir_module
class Conv2dResidualBlock:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), "float32"),
        weight1: R.Tensor((64, 64, 3, 3), "float32"),
        weight2: R.Tensor((64, 64, 3, 3), "float32"),
    ):
        with R.dataflow():
            conv1 = relax.op.nn.relu(relax.op.nn.conv2d(data, weight1, padding=(1, 1)))
            conv2 = relax.op.nn.relu(relax.op.nn.conv2d(conv1, weight2, padding=(1, 1)))
            out = relax.op.add(conv2, data)
            R.output(out)

        return out


has_tensorrt = tvm.get_global_func("relax.ext.tensorrt", True)
env_checker_runtime = tvm.get_global_func("relax.is_tensorrt_runtime_enabled", True)

requires_tensorrt_codegen = pytest.mark.skipif(
    not has_tensorrt,
    reason="TENSORRT not enabled.",
)

requires_tensorrt_runtime = pytest.mark.skipif(
    not env_checker_runtime or not env_checker_runtime(),
    reason="TensorRT runtime not available",
)

pytestmark = [
    requires_tensorrt_codegen,
    requires_tensorrt_runtime,
] + tvm.testing.requires_cuda.marks()


def build_and_run(mod, inputs_np, target, legalize=False):
    dev = tvm.device(target, 0)
    with tvm.transform.PassContext(config={"relax.transform.apply_legalize_ops": legalize}):
        ex = tvm.compile(mod, target)
    vm = relax.VirtualMachine(ex, dev)
    f = vm["main"]
    inputs = [tvm.runtime.tensor(inp, dev) for inp in inputs_np]
    return f(*inputs).numpy()


def test_tensorrt_offload():
    @memoize("relax.tests.test_codegen_tensorrt.conv2d_residual")
    def get_ref():
        data_np = np.random.randn(1, 64, 56, 56).astype("float32")
        weight1_np = np.random.randn(64, 64, 3, 3).astype("float32")
        weight2_np = np.random.randn(64, 64, 3, 3).astype("float32")
        inputs = [data_np, weight1_np, weight2_np]
        ref = build_and_run(Conv2dResidualBlock, inputs, "llvm", legalize=True)
        return inputs, ref

    inputs, ref = get_ref()

    conv_pat = make_fused_bias_activation_pattern(
        "relax.nn.conv2d", with_bias=False, activation=None
    )
    relu_pat = is_op("relax.nn.relu")(wildcard())
    add_pat = is_op("relax.add")(wildcard(), wildcard())

    patterns = [
        ("tensorrt.nn.conv2d", conv_pat),
        ("tensorrt.nn.relu", relu_pat),
        ("tensorrt.add", add_pat),
    ]

    params_np = {"weight1": inputs[1], "weight2": inputs[2]}

    mod = tvm.transform.Sequential(
        [
            relax.transform.BindParams("main", params_np),
            relax.transform.FuseOpsByPattern(patterns),
            relax.transform.MergeCompositeFunctions(),
            relax.transform.RunCodegen(),
        ]
    )(Conv2dResidualBlock)

    out = build_and_run(mod, inputs[:1], "cuda")

    tvm.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)


def _offload_and_compare(mod, params_np, patterns, data_np, rtol=1e-2, atol=1e-2):
    """Offload a single-op module to TensorRT and compare against the LLVM reference.

    Each module here contains a single instance of the op under test, which both exercises the
    individual converter and avoids the structurally-identical-composite deduplication that would
    otherwise collapse repeated ops.
    """
    ref = build_and_run(mod, [data_np, *params_np.values()], "llvm", legalize=True)
    offloaded = tvm.transform.Sequential(
        [
            relax.transform.BindParams("main", params_np),
            relax.transform.FuseOpsByPattern(patterns),
            relax.transform.MergeCompositeFunctions(),
            relax.transform.RunCodegen(),
        ]
    )(mod)
    out = build_and_run(offloaded, [data_np], "cuda")
    tvm.testing.assert_allclose(out, ref, rtol=rtol, atol=atol)


def test_tensorrt_conv1d():
    # Regression test: explicit-batch (batch > 1) 1D convolution. The pre-TRT10 converter assumed an
    # implicit batch dimension and dropped the spatial dimension under explicit batch.
    @tvm.script.ir_module
    class Conv1d:
        @R.function
        def main(data: R.Tensor((2, 8, 16), "float32"), weight: R.Tensor((4, 8, 3), "float32")):
            with R.dataflow():
                out = relax.op.nn.conv1d(data, weight, padding=1)
                R.output(out)
            return out

    data = np.random.randn(2, 8, 16).astype("float32")
    weight = np.random.randn(4, 8, 3).astype("float32")
    patterns = [("tensorrt.nn.conv1d", is_op("relax.nn.conv1d")(wildcard(), wildcard()))]
    _offload_and_compare(Conv1d, {"weight": weight}, patterns, data)


def test_tensorrt_max_pool2d():
    @tvm.script.ir_module
    class MaxPool:
        @R.function
        def main(data: R.Tensor((2, 8, 16, 16), "float32")):
            with R.dataflow():
                out = relax.op.nn.max_pool2d(data, pool_size=(2, 2), strides=(2, 2))
                R.output(out)
            return out

    data = np.random.randn(2, 8, 16, 16).astype("float32")
    patterns = [("tensorrt.nn.max_pool2d", is_op("relax.nn.max_pool2d")(wildcard()))]
    _offload_and_compare(MaxPool, {}, patterns, data)


def test_tensorrt_avg_pool2d():
    @tvm.script.ir_module
    class AvgPool:
        @R.function
        def main(data: R.Tensor((2, 8, 16, 16), "float32")):
            with R.dataflow():
                out = relax.op.nn.avg_pool2d(data, pool_size=(2, 2), strides=(2, 2))
                R.output(out)
            return out

    data = np.random.randn(2, 8, 16, 16).astype("float32")
    patterns = [("tensorrt.nn.avg_pool2d", is_op("relax.nn.avg_pool2d")(wildcard()))]
    _offload_and_compare(AvgPool, {}, patterns, data)


def test_tensorrt_softmax():
    @tvm.script.ir_module
    class Softmax:
        @R.function
        def main(data: R.Tensor((2, 8, 16, 16), "float32")):
            with R.dataflow():
                out = relax.op.nn.softmax(data, axis=1)
                R.output(out)
            return out

    data = np.random.randn(2, 8, 16, 16).astype("float32")
    patterns = [("tensorrt.nn.softmax", is_op("relax.nn.softmax")(wildcard()))]
    _offload_and_compare(Softmax, {}, patterns, data)


def test_tensorrt_sigmoid():
    @tvm.script.ir_module
    class Sigmoid:
        @R.function
        def main(data: R.Tensor((2, 8, 16, 16), "float32")):
            with R.dataflow():
                out = relax.op.sigmoid(data)
                R.output(out)
            return out

    data = np.random.randn(2, 8, 16, 16).astype("float32")
    patterns = [("tensorrt.sigmoid", is_op("relax.sigmoid")(wildcard()))]
    _offload_and_compare(Sigmoid, {}, patterns, data)


def test_tensorrt_tanh():
    @tvm.script.ir_module
    class Tanh:
        @R.function
        def main(data: R.Tensor((2, 8, 16, 16), "float32")):
            with R.dataflow():
                out = relax.op.tanh(data)
                R.output(out)
            return out

    data = np.random.randn(2, 8, 16, 16).astype("float32")
    patterns = [("tensorrt.tanh", is_op("relax.tanh")(wildcard()))]
    _offload_and_compare(Tanh, {}, patterns, data)


def test_tensorrt_conv2d_transpose():
    # Default IOHW kernel layout ([in, out, h, w]); output channels are weight_shape[1].
    @tvm.script.ir_module
    class ConvTranspose:
        @R.function
        def main(
            data: R.Tensor((2, 8, 16, 16), "float32"), weight: R.Tensor((8, 4, 3, 3), "float32")
        ):
            with R.dataflow():
                out = relax.op.nn.conv2d_transpose(data, weight, padding=1)
                R.output(out)
            return out

    data = np.random.randn(2, 8, 16, 16).astype("float32")
    weight = np.random.randn(8, 4, 3, 3).astype("float32")
    patterns = [
        ("tensorrt.nn.conv2d_transpose", is_op("relax.nn.conv2d_transpose")(wildcard(), wildcard()))
    ]
    _offload_and_compare(ConvTranspose, {"weight": weight}, patterns, data)


def test_tensorrt_conv3d_transpose():
    # Default IODHW kernel layout ([in, out, d, h, w]); output channels are weight_shape[1].
    @tvm.script.ir_module
    class ConvTranspose3d:
        @R.function
        def main(
            data: R.Tensor((2, 4, 8, 8, 8), "float32"), weight: R.Tensor((4, 2, 3, 3, 3), "float32")
        ):
            with R.dataflow():
                out = relax.op.nn.conv3d_transpose(data, weight, padding=1)
                R.output(out)
            return out

    data = np.random.randn(2, 4, 8, 8, 8).astype("float32")
    weight = np.random.randn(4, 2, 3, 3, 3).astype("float32")
    patterns = [
        ("tensorrt.nn.conv3d_transpose", is_op("relax.nn.conv3d_transpose")(wildcard(), wildcard()))
    ]
    _offload_and_compare(ConvTranspose3d, {"weight": weight}, patterns, data)


def test_tensorrt_int8_calibration(monkeypatch):
    # INT8 calibration path: the first N runs feed calibration batches, then the INT8 engine is
    # built and run. Validates that the calibrator copies a full batch (batch_size * per-sample
    # elements) without over-reading the input or over-writing the device buffers, which previously
    # crashed for batch > 1.
    @tvm.script.ir_module
    class Conv2dInt8:
        @R.function
        def main(
            data: R.Tensor((2, 8, 16, 16), "float32"), weight: R.Tensor((4, 8, 3, 3), "float32")
        ):
            with R.dataflow():
                out = relax.op.nn.conv2d(data, weight, padding=1)
                R.output(out)
            return out

    data = np.random.randn(2, 8, 16, 16).astype("float32")
    weight = np.random.randn(4, 8, 3, 3).astype("float32")
    ref = build_and_run(Conv2dInt8, [data, weight], "llvm", legalize=True)

    patterns = [("tensorrt.nn.conv2d", is_op("relax.nn.conv2d")(wildcard(), wildcard()))]
    offloaded = tvm.transform.Sequential(
        [
            relax.transform.BindParams("main", {"weight": weight}),
            relax.transform.FuseOpsByPattern(patterns),
            relax.transform.MergeCompositeFunctions(),
            relax.transform.RunCodegen(),
        ]
    )(Conv2dInt8)

    num_calibration_batches = 2
    monkeypatch.setenv("TVM_TENSORRT_USE_INT8", "1")
    monkeypatch.setenv("TENSORRT_NUM_CALI_INT8", str(num_calibration_batches))

    dev = tvm.device("cuda", 0)
    vm = relax.VirtualMachine(tvm.compile(offloaded, "cuda"), dev)
    data_trt = tvm.runtime.tensor(data, dev)
    out = None
    for _ in range(num_calibration_batches + 1):
        out = vm["main"](data_trt).numpy()

    assert np.isfinite(out).all()
    # INT8 is lossy, so use a generous tolerance; the key assertion is that calibration completed
    # without a CUDA error.
    tvm.testing.assert_allclose(out, ref, rtol=0.2, atol=0.1 * float(np.abs(ref).max()))


if __name__ == "__main__":
    tvm.testing.main()
