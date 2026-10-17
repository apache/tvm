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

"""Compilation through the native TIRx pipeline."""

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import tirx as T


@pytest.mark.skipif(not tvm_tirx.runtime.enabled("llvm"), reason="LLVM is not enabled")
@pytest.mark.parametrize("pipeline", [None, "default", "tirx"])
def test_default_pipeline_allocations(pipeline):
    @T.function
    def add_one(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        temp = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32")])
        for i in range(16):
            temp[i] = A[i] + T.float32(1)
        for i in range(16):
            B[i] = temp[i]

    module = tvm_tirx.tirx.compile(add_one, target="llvm", pipeline=pipeline)
    source = np.arange(16, dtype="float32")
    a = tvm_tirx.runtime.tensor(source)
    b = tvm_tirx.runtime.tensor(np.zeros(16, dtype="float32"))
    module(a, b)
    np.testing.assert_equal(b.numpy(), source + 1)


@pytest.mark.parametrize("print_after_dispatch", [None, "", "1"])
def test_explicit_tile_pass_order_and_diagnostic(monkeypatch, capfd, print_after_dispatch):
    from tvm_tirx.tirx.compilation_pipeline import tirx_pipeline

    if print_after_dispatch is None:
        monkeypatch.delenv("TVM_PRINT_AFTER_TIRX_DISPATCH_OPS", raising=False)
    else:
        monkeypatch.setenv("TVM_PRINT_AFTER_TIRX_DISPATCH_OPS", print_after_dispatch)

    @T.function
    def kernel(A: T.Tensor((2, 4), "float32", layout="default")):
        for i, j in T.grid(2, 4):
            A[i, j] = T.float32(1)

    visited = []
    layouts = {}

    @tvm_tirx.ir.transform.pass_instrument
    class RecordPasses:
        def run_after_pass(self, mod, info):
            visited.append(info.name)
            if info.name in ("tirx.TileDispatch", "tirx.LowerTensorLayouts"):
                layouts[info.name] = mod.script()

    pipeline, _, _ = tirx_pipeline(prepare_only=True, target=tvm_tirx.target.Target("llvm"))
    with tvm_tirx.ir.transform.PassContext(instruments=[RecordPasses()]):
        pipeline(tvm_tirx.IRModule({"main": kernel}))

    start = visited.index("tirx.TileDispatch")
    expected = ["tirx.TileDispatch"]
    if print_after_dispatch is not None:
        expected.append("PrintIR")
    expected.extend(["tirx.LowerTensorLayouts", "tirx.StmtSimplify", "tirx.FlattenTensor"])
    assert visited[start : start + len(expected)] == expected
    assert "TileLayout" in layouts["tirx.TileDispatch"]
    assert "TileLayout" not in layouts["tirx.LowerTensorLayouts"]
    output = capfd.readouterr().err
    assert ("PrintIR(" in output) == (print_after_dispatch is not None)
    if print_after_dispatch is not None:
        assert "TileLayout" in output


if __name__ == "__main__":
    tvm_tirx.testing.main()
