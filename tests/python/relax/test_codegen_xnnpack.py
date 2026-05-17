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

import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.backend.pattern_registry import get_patterns_with_prefix
from tvm.script import relax as R


@tvm.script.ir_module
class AddModule:
    @R.function
    def main(x: R.Tensor((4,), "float32"), y: R.Tensor((4,), "float32")):
        with R.dataflow():
            z = relax.op.add(x, y)
            R.output(z)
        return z


def _has_xnnpack_codegen():
    return tvm.get_global_func("relax.ext.xnnpack", allow_missing=True) is not None


def _has_xnnpack_runtime():
    return tvm.get_global_func("runtime.XNNPACKJSONRuntimeCreate", allow_missing=True) is not None


def _has_codegen_attr(mod):
    for func in mod.functions.values():
        if isinstance(func, relax.Function):
            opt_codegen = func.attrs.get("Codegen") if func.attrs else None
            if opt_codegen == "xnnpack":
                return True
    return False


def test_xnnpack_python_module_importable():
    from tvm.relax.backend.xnnpack import partition_for_xnnpack

    assert callable(partition_for_xnnpack)


def test_xnnpack_registers_no_phase1_patterns():
    import tvm.relax.backend.xnnpack  # noqa: F401

    assert len(get_patterns_with_prefix("xnnpack")) == 0


def test_partition_for_xnnpack_does_not_partition_unsupported_ops():
    from tvm.relax.backend.xnnpack import partition_for_xnnpack

    mod = partition_for_xnnpack(AddModule)
    assert mod.same_as(AddModule)
    assert not _has_codegen_attr(mod)

    mod = relax.transform.RunCodegen()(mod)
    assert not mod.attrs or "external_mods" not in mod.attrs


@pytest.mark.skipif(not _has_xnnpack_codegen(), reason="XNNPACK codegen is not enabled")
def test_xnnpack_codegen_registration_accepts_empty_input():
    codegen = tvm.get_global_func("relax.ext.xnnpack")
    assert len(codegen([], {}, {})) == 0


@pytest.mark.skipif(not _has_xnnpack_runtime(), reason="XNNPACK runtime is not enabled")
def test_xnnpack_runtime_registration_available():
    assert tvm.get_global_func("runtime.XNNPACKJSONRuntimeCreate") is not None


if __name__ == "__main__":
    tvm.testing.main()
