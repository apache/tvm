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


def _quant_metadata_validator():
    return tvm.get_global_func(
        "runtime.XNNPACKJSONRuntimeValidateQuantizationMetadata", allow_missing=True
    )


def _quant_tensor_smoke():
    return tvm.get_global_func(
        "runtime.XNNPACKJSONRuntimeQuantizedTensorDefinitionSmoke", allow_missing=True
    )


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
    from tvm.relax.backend.xnnpack import partition_for_xnnpack

    return partition_for_xnnpack(mod, precision=precision, **kwargs)


def _bind_cnn_params(mod=ConvBiasReluPoolModule):
    weight = np.arange(4 * 3 * 3 * 3).reshape(4, 3, 3, 3).astype("float32") / 100.0
    bias = np.array([0.1, -0.2, 0.3, -0.4], dtype="float32")
    return relax.transform.BindParams("main", {"w": weight, "b": bias})(mod)


def _bind_tiny_cnn_params():
    weight = np.linspace(-0.2, 0.2, num=4 * 3 * 3 * 3, dtype="float32").reshape(4, 3, 3, 3)
    bias = np.array([0.15, -0.05, 0.25, -0.10], dtype="float32")
    return relax.transform.BindParams("main", {"w": weight, "b": bias})(TinyCNNModule)


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


def _run_first_external_module(mod, inputs, output_shape):
    ext_mod = mod.attrs["external_mods"][0]
    symbol = ext_mod["get_symbol"]()
    const_names = list(ext_mod["get_const_vars"]())
    const_map = mod.attrs.get("const_name_to_constant", {})
    consts = [const_map[name] for name in const_names]
    ext_mod["__init_" + symbol](consts)

    output_np = np.empty(output_shape, dtype="float32")
    output = tvm.runtime.tensor(output_np)
    ext_mod[symbol](*[tvm.runtime.tensor(input_np) for input_np in inputs], output)
    return ext_mod, output.numpy()


def _first_external_runtime_options(mod):
    ext_mod = mod.attrs["external_mods"][0]
    return ext_mod["get_runtime_options"]()


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
    }
    assert expected_fields.issubset(report[0].keys())


def test_xnnpack_python_module_importable():
    from tvm.relax.backend.xnnpack import partition_for_xnnpack

    assert callable(partition_for_xnnpack)


def test_partition_for_xnnpack_rejects_invalid_precision():
    from tvm.relax.backend.xnnpack import partition_for_xnnpack

    with pytest.raises(ValueError, match="Unsupported XNNPACK precision"):
        partition_for_xnnpack(ReluModule, precision="explicit_fp16")


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
    from tvm.relax.backend.xnnpack import partition_for_xnnpack

    with pytest.raises(ValueError, match=match):
        partition_for_xnnpack(ReluModule, **kwargs)


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


def test_partition_for_xnnpack_rejects_float16_even_with_fp16_policy():
    mod = _partition(ReluFloat16Module, precision="fp16_hint")
    assert not _has_codegen_attr(mod)


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
    assert "datatype_qint8" in capabilities
    assert "datatype_quint8" in capabilities
    assert "datatype_qcint8" in capabilities
    assert "extra_quantization_params" in capabilities


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
