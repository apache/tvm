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


def _partition(mod):
    from tvm.relax.backend.xnnpack import partition_for_xnnpack

    return partition_for_xnnpack(mod)


def test_xnnpack_python_module_importable():
    from tvm.relax.backend.xnnpack import partition_for_xnnpack

    assert callable(partition_for_xnnpack)


def test_xnnpack_registers_relu_pattern():
    import tvm.relax.backend.xnnpack  # noqa: F401

    assert [pattern.name for pattern in get_patterns_with_prefix("xnnpack")] == ["xnnpack.relu"]


def test_partition_for_xnnpack_partitions_static_float32_relu():
    mod = _partition(ReluModule)
    assert _has_codegen_attr(mod)


@pytest.mark.parametrize("mod", [AddModule, ReluFloat16Module, ReluSymbolicModule])
def test_partition_for_xnnpack_rejects_unsupported_patterns(mod):
    mod = _partition(mod)
    assert not _has_codegen_attr(mod)

    mod = relax.transform.RunCodegen()(mod)
    assert not _has_external_mods(mod)


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


@pytest.mark.skipif(not _has_xnnpack_codegen(), reason="XNNPACK codegen is not enabled")
def test_xnnpack_codegen_registration_accepts_empty_input():
    codegen = tvm.get_global_func("relax.ext.xnnpack")
    assert len(codegen([], {}, {})) == 0


@pytest.mark.skipif(not _has_xnnpack_runtime(), reason="XNNPACK runtime is not enabled")
def test_xnnpack_runtime_registration_available():
    assert tvm.get_global_func("runtime.XNNPACKJSONRuntimeCreate") is not None


if __name__ == "__main__":
    tvm.testing.main()
