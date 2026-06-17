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
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm.contrib.pickle_memoize import memoize
from tvm.relax.dpl import is_op, make_fused_bias_activation_pattern, wildcard
from tvm.script import relax as R
from tvm.testing import env


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
    pytest.mark.gpu,
    pytest.mark.skipif(not env.has_cuda(), reason="need cuda"),
]


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
    partitioned = tvm.transform.Sequential(
        [
            relax.transform.BindParams("main", params_np),
            relax.transform.FuseOpsByPattern(patterns),
            relax.transform.MergeCompositeFunctions(),
        ]
    )(mod)
    # Guard against a silent false pass: if no pattern matched, nothing is offloaded and the
    # comparison would trivially succeed via the TVM fallback without exercising the converter.
    assert any(
        isinstance(fn, relax.Function) and fn.attrs is not None and "Codegen" in fn.attrs
        for fn in partitioned.functions.values()
    ), "expected the op under test to be offloaded to TensorRT, but nothing was partitioned"
    offloaded = relax.transform.RunCodegen()(partitioned)
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


def test_tensorrt_matmul():
    # Regression test: Relax matmul has no transpose_a/transpose_b attrs (Relay's batch_matmul did).
    @tvm.script.ir_module
    class Matmul:
        @R.function
        def main(data: R.Tensor((4, 8), "float32"), weight: R.Tensor((8, 16), "float32")):
            with R.dataflow():
                out = relax.op.matmul(data, weight)
                R.output(out)
            return out

    data = np.random.randn(4, 8).astype("float32")
    weight = np.random.randn(8, 16).astype("float32")
    patterns = [("tensorrt.nn.batch_matmul", is_op("relax.matmul")(wildcard(), wildcard()))]
    _offload_and_compare(Matmul, {"weight": weight}, patterns, data)


def test_tensorrt_sum():
    # Regression test: Relax reduce ops (StatisticalAttrs) have no "exclude" attr.
    @tvm.script.ir_module
    class Sum:
        @R.function
        def main(data: R.Tensor((2, 3, 4), "float32")):
            with R.dataflow():
                out = relax.op.sum(data, axis=[1], keepdims=True)
                R.output(out)
            return out

    data = np.random.randn(2, 3, 4).astype("float32")
    patterns = [("tensorrt.sum", is_op("relax.sum")(wildcard()))]
    _offload_and_compare(Sum, {}, patterns, data)


def test_tensorrt_expand_dims():
    # Regression test: Relax expand_dims carries an `axis` list, not Relay's axis + num_newaxis.
    @tvm.script.ir_module
    class ExpandDims:
        @R.function
        def main(data: R.Tensor((2, 4), "float32")):
            with R.dataflow():
                out = relax.op.expand_dims(data, axis=[1, 3])
                R.output(out)
            return out

    data = np.random.randn(2, 4).astype("float32")
    patterns = [("tensorrt.expand_dims", is_op("relax.expand_dims")(wildcard()))]
    _offload_and_compare(ExpandDims, {}, patterns, data)


def test_tensorrt_layer_norm():
    # Regression test: Relax layer_norm normalizes over an `axes` list (Relay used `axis`).
    @tvm.script.ir_module
    class LayerNorm:
        @R.function
        def main(
            data: R.Tensor((2, 4, 8), "float32"),
            gamma: R.Tensor((8,), "float32"),
            beta: R.Tensor((8,), "float32"),
        ):
            with R.dataflow():
                out = relax.op.nn.layer_norm(data, gamma, beta, axes=[-1])
                R.output(out)
            return out

    data = np.random.randn(2, 4, 8).astype("float32")
    gamma = np.random.randn(8).astype("float32")
    beta = np.random.randn(8).astype("float32")
    patterns = [
        ("tensorrt.nn.layer_norm", is_op("relax.nn.layer_norm")(wildcard(), wildcard(), wildcard()))
    ]
    _offload_and_compare(LayerNorm, {"gamma": gamma, "beta": beta}, patterns, data)


def test_tensorrt_clip():
    # Regression test: Relax clip passes min/max as PrimValue arguments (Relay used a_min/a_max
    # attributes); the codegen serializes them under the op's argument names.
    @tvm.script.ir_module
    class Clip:
        @R.function
        def main(data: R.Tensor((2, 8, 16, 16), "float32")):
            with R.dataflow():
                out = relax.op.clip(data, 0.0, 6.0)
                R.output(out)
            return out

    data = (np.random.randn(2, 8, 16, 16) * 4).astype("float32")
    patterns = [("tensorrt.clip", is_op("relax.clip")(wildcard(), wildcard(), wildcard()))]
    _offload_and_compare(Clip, {}, patterns, data)


def test_tensorrt_reshape():
    # Regression test: Relax reshape takes the target shape as a Shape argument (Relay used a
    # "newshape" attribute); the codegen serializes it under the op's argument name.
    @tvm.script.ir_module
    class Reshape:
        @R.function
        def main(data: R.Tensor((2, 8, 4, 4), "float32")):
            with R.dataflow():
                out = relax.op.reshape(data, (2, 8, 16))
                R.output(out)
            return out

    data = np.random.randn(2, 8, 4, 4).astype("float32")
    patterns = [("tensorrt.reshape", is_op("relax.reshape")(wildcard(), wildcard()))]
    _offload_and_compare(Reshape, {}, patterns, data)


def test_tensorrt_strided_slice():
    # Regression test: Relax strided_slice passes axes/begin/end/strides as tuple arguments (Relay
    # used start/size/strides attributes); the codegen serializes them positionally.
    @tvm.script.ir_module
    class StridedSlice:
        @R.function
        def main(data: R.Tensor((4, 8, 16), "float32")):
            with R.dataflow():
                out = relax.op.strided_slice(
                    data, axes=[1, 2], begin=[2, 0], end=[6, 8], strides=[2, 1]
                )
                R.output(out)
            return out

    data = np.random.randn(4, 8, 16).astype("float32")
    patterns = [
        (
            "tensorrt.strided_slice",
            is_op("relax.strided_slice")(
                wildcard(), wildcard(), wildcard(), wildcard(), wildcard()
            ),
        )
    ]
    _offload_and_compare(StridedSlice, {}, patterns, data)


def test_tensorrt_split():
    # Regression test: Relax split has no Relay-style "mode"; it is multi-output. The converter
    # derives per-output extents from the codegen-recorded output shapes.
    @tvm.script.ir_module
    class Split:
        @R.function
        def main(data: R.Tensor((4, 8, 16), "float32")):
            with R.dataflow():
                parts = relax.op.split(data, 2, axis=1)
                out = relax.op.add(parts[0], parts[1])
                R.output(out)
            return out

    data = np.random.randn(4, 8, 16).astype("float32")
    # Offload the add too so both split outputs are consumed inside TensorRT (and nothing is left
    # for the VM to legalize).
    patterns = [
        ("tensorrt.split", is_op("relax.split")(wildcard())),
        ("tensorrt.add", is_op("relax.add")(wildcard(), wildcard())),
    ]
    _offload_and_compare(Split, {}, patterns, data)


def test_tensorrt_layout_transform():
    # Regression test: Relax layout_transform uses an IndexMap (Relay used src_layout/dst_layout
    # strings); the codegen translates a pure-permutation index map into a transpose. Built with the
    # BlockBuilder because the index_map lambda cannot be expressed in TVMScript.
    bb = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorStructInfo((1, 4, 8, 8), "float32"))
    with bb.function("main", [data]):
        with bb.dataflow():
            out = bb.emit(
                relax.op.layout_transform(data, index_map=lambda n, c, h, w: (n, h, w, c))
            )
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    LayoutTransform = bb.finalize()

    data_np = np.random.randn(1, 4, 8, 8).astype("float32")
    patterns = [("tensorrt.layout_transform", is_op("relax.layout_transform")(wildcard()))]
    _offload_and_compare(LayoutTransform, {}, patterns, data_np)


def test_tensorrt_sum_all_axes():
    # Edge case: Relax sum with no axis (StatisticalAttrs.axis = None) reduces over all axes.
    @tvm.script.ir_module
    class SumAll:
        @R.function
        def main(data: R.Tensor((2, 3, 4), "float32")):
            with R.dataflow():
                out = relax.op.sum(data, keepdims=True)
                R.output(out)
            return out

    data = np.random.randn(2, 3, 4).astype("float32")
    patterns = [("tensorrt.sum", is_op("relax.sum")(wildcard()))]
    _offload_and_compare(SumAll, {}, patterns, data)


def test_tensorrt_layer_norm_multi_axis():
    # Edge case: layer_norm normalizing over more than one axis.
    @tvm.script.ir_module
    class LayerNorm2:
        @R.function
        def main(
            data: R.Tensor((2, 3, 4, 5), "float32"),
            gamma: R.Tensor((4, 5), "float32"),
            beta: R.Tensor((4, 5), "float32"),
        ):
            with R.dataflow():
                out = relax.op.nn.layer_norm(data, gamma, beta, axes=[-2, -1])
                R.output(out)
            return out

    data = np.random.randn(2, 3, 4, 5).astype("float32")
    gamma = np.random.randn(4, 5).astype("float32")
    beta = np.random.randn(4, 5).astype("float32")
    patterns = [
        ("tensorrt.nn.layer_norm", is_op("relax.nn.layer_norm")(wildcard(), wildcard(), wildcard()))
    ]
    _offload_and_compare(LayerNorm2, {"gamma": gamma, "beta": beta}, patterns, data)


def test_tensorrt_matmul_batched():
    # Edge case: batched (3-D) matmul exercises TensorRT's leading-dim broadcasting.
    @tvm.script.ir_module
    class BatchMatmul:
        @R.function
        def main(data: R.Tensor((2, 4, 8), "float32"), weight: R.Tensor((2, 8, 16), "float32")):
            with R.dataflow():
                out = relax.op.matmul(data, weight)
                R.output(out)
            return out

    data = np.random.randn(2, 4, 8).astype("float32")
    weight = np.random.randn(2, 8, 16).astype("float32")
    patterns = [("tensorrt.nn.batch_matmul", is_op("relax.matmul")(wildcard(), wildcard()))]
    _offload_and_compare(BatchMatmul, {"weight": weight}, patterns, data)


def test_tensorrt_strided_slice_no_strides():
    # Edge case: strided_slice without an explicit strides argument (defaults to 1).
    @tvm.script.ir_module
    class StridedSliceNoStride:
        @R.function
        def main(data: R.Tensor((4, 8, 16), "float32")):
            with R.dataflow():
                out = relax.op.strided_slice(data, axes=[1], begin=[2], end=[6])
                R.output(out)
            return out

    data = np.random.randn(4, 8, 16).astype("float32")
    patterns = [
        (
            "tensorrt.strided_slice",
            is_op("relax.strided_slice")(wildcard(), wildcard(), wildcard(), wildcard()),
        )
    ]
    _offload_and_compare(StridedSliceNoStride, {}, patterns, data)


def test_tensorrt_split_indices():
    # Edge case: split by an explicit index list (the other indices_or_sections form).
    @tvm.script.ir_module
    class SplitIdx:
        @R.function
        def main(data: R.Tensor((4, 8, 16), "float32")):
            with R.dataflow():
                parts = relax.op.split(data, [4], axis=1)
                out = relax.op.add(parts[0], parts[1])
                R.output(out)
            return out

    data = np.random.randn(4, 8, 16).astype("float32")
    patterns = [
        ("tensorrt.split", is_op("relax.split")(wildcard())),
        ("tensorrt.add", is_op("relax.add")(wildcard(), wildcard())),
    ]
    _offload_and_compare(SplitIdx, {}, patterns, data)


if __name__ == "__main__":
    tvm.testing.main()
