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
from tvm.relax.backend.pattern_registry import get_patterns_with_prefix
from tvm.script import relax as R


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
class AddBroadcastModule:
    @R.function
    def main(x: R.Tensor((2, 3), "float32"), y: R.Tensor((3,), "float32")):
        with R.dataflow():
            z = relax.op.add(x, y)
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


def _partition(mod):
    from tvm.relax.backend.xnnpack import partition_for_xnnpack

    return partition_for_xnnpack(mod)


def _bind_cnn_params(mod=ConvBiasReluPoolModule):
    weight = np.arange(4 * 3 * 3 * 3).reshape(4, 3, 3, 3).astype("float32") / 100.0
    bias = np.array([0.1, -0.2, 0.3, -0.4], dtype="float32")
    return relax.transform.BindParams("main", {"w": weight, "b": bias})(mod)


def _bind_tiny_cnn_params():
    weight = np.linspace(-0.2, 0.2, num=4 * 3 * 3 * 3, dtype="float32").reshape(4, 3, 3, 3)
    bias = np.array([0.15, -0.05, 0.25, -0.10], dtype="float32")
    return relax.transform.BindParams("main", {"w": weight, "b": bias})(TinyCNNModule)


def test_xnnpack_python_module_importable():
    from tvm.relax.backend.xnnpack import partition_for_xnnpack

    assert callable(partition_for_xnnpack)


def test_xnnpack_registers_relu_pattern():
    import tvm.relax.backend.xnnpack  # noqa: F401

    pattern_names = {pattern.name for pattern in get_patterns_with_prefix("xnnpack")}
    assert {
        "xnnpack.conv2d_bias_relu",
        "xnnpack.max_pool2d",
        "xnnpack.add",
        "xnnpack.clip",
        "xnnpack.relu",
        "xnnpack.sigmoid",
        "xnnpack.tanh",
    }.issubset(pattern_names)


def test_partition_for_xnnpack_partitions_static_float32_relu():
    mod = _partition(ReluModule)
    assert _has_codegen_attr(mod)


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


@pytest.mark.parametrize("mod", [AddModule, ClipModule, SigmoidModule, TanhModule])
def test_partition_for_xnnpack_partitions_supported_phase3_patterns(mod):
    mod = _partition(mod)
    assert _has_codegen_attr(mod)


def test_partition_for_xnnpack_partitions_bound_cnn_pattern():
    mod = _partition(_bind_cnn_params())
    assert _has_codegen_attr(mod)


def test_partition_for_xnnpack_tiny_cnn_partition_count():
    mod = _partition(_bind_tiny_cnn_params())
    assert _count_xnnpack_partitions(mod) == 4


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
    bound_mod = _bind_tiny_cnn_params()
    partitioned = _partition(bound_mod)
    assert _count_xnnpack_partitions(partitioned) == 4

    partitioned = relax.transform.RunCodegen()(partitioned)
    assert _has_external_mods(partitioned)

    rng = np.random.default_rng(0)
    x_np = rng.uniform(-1.0, 1.0, size=(1, 8, 8, 3)).astype("float32")
    residual_np = rng.uniform(-0.5, 0.5, size=(1, 3, 3, 4)).astype("float32")

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


@pytest.mark.skipif(not _has_xnnpack_codegen(), reason="XNNPACK codegen is not enabled")
def test_xnnpack_codegen_registration_accepts_empty_input():
    codegen = tvm.get_global_func("relax.ext.xnnpack")
    assert len(codegen([], {}, {})) == 0


@pytest.mark.skipif(not _has_xnnpack_runtime(), reason="XNNPACK runtime is not enabled")
def test_xnnpack_runtime_registration_available():
    assert tvm.get_global_func("runtime.XNNPACKJSONRuntimeCreate") is not None


if __name__ == "__main__":
    tvm.testing.main()
