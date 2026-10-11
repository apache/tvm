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
"""Direct variable code generation preserves mutable-cell evaluation order."""

import re

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T


@pytest.mark.parametrize("target", ["c", "llvm"])
def test_mutable_cell_control_flow(target, tmp_path):
    if target == "llvm" and not tvm.runtime.enabled("llvm"):
        pytest.skip("LLVM is not enabled")

    @T.function
    def main(out: T.Tensor((6,), "int32"), n: T.int32):
        x: T.int32 = 1
        captured: T.let = x
        x = 4
        out[0] = captured
        if n > 0:
            x += 2
        else:
            x -= 2
        out[1] = x
        for i in range(3):
            local: T.int32 = i + 10
            local += x
            out[i + 2] = local
            x += 1
        while x < 12:
            x += 1
            if x % 2 == 0:
                continue
            if x > 10:
                break
        out[5] = x

    module = tvm.compile(main, target=target)
    if target == "c":
        source = module.mod.inspect_source()
        assert re.search(r"int32_t x\s*=\s*1;", source)
        assert not re.search(r"x\s*\[", source)
        library = str(tmp_path / "mutable_cells.so")
        module.export_library(library)
        module = tvm.runtime.load_module(library)
    for n, expected in [(1, [1, 6, 16, 18, 20, 11]), (-1, [1, 2, 12, 14, 16, 11])]:
        out = tvm.runtime.tensor(np.zeros(6, dtype="int32"))
        module["main"](out, n)
        np.testing.assert_array_equal(out.numpy(), expected)


def test_mutable_cell_vector_snapshot():
    if not tvm.runtime.enabled("llvm"):
        pytest.skip("LLVM is not enabled")

    @T.function
    def main(out: T.Tensor((4,), "int32")):
        cell = T.alloc_cell("int32x4", T.broadcast(3, 4))
        captured: T.let = cell
        cell = cell + T.broadcast(4, 4)
        out[T.ramp(0, 1, 4)] = captured + cell

    module = tvm.compile(main, target="llvm")
    out = tvm.runtime.tensor(np.zeros(4, dtype="int32"))
    module(out)
    np.testing.assert_array_equal(out.numpy(), [10, 10, 10, 10])


def test_cuda_mutable_cell_return_and_snapshot(monkeypatch):
    from tvm.tirx.mutable_cell import mutable_cell_alloc, mutable_cell_load, mutable_cell_store

    allocation = mutable_cell_alloc("int32", 2)
    cell = tvm.ir.Var("cell", allocation.ty)
    captured = tvm.ir.Var("captured", tvm.ir.PrimType("int32"))
    function = tvm.tirx.Function(
        [],
        [
            tvm.ir.Bind(cell, allocation),
            tvm.ir.Bind(captured, mutable_cell_load(cell)),
            tvm.ir.Evaluate(mutable_cell_store(cell, 9)),
            tvm.ir.Return(captured + mutable_cell_load(cell)),
        ],
        ret_type=tvm.ir.PrimType("int32"),
    ).with_attr("global_symbol", "cell_value")
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    monkeypatch.setenv("TVM_COMPILE_FORCE_FALLBACK", "1")
    source = tvm.get_global_func("target.build.cuda")(
        tvm.IRModule({"cell_value": function}), target
    ).inspect_source()
    assert re.search(r"int cell\s*=\s*2;", source)
    assert "cell = 9;" in source
    assert re.search(r"int cell_value_\d+ = cell;", source)
    assert re.search(r"return \(captured \+ cell_value_\d+\);", source)


def test_cuda_mutable_cell_uint128_zero(monkeypatch):
    from tvm.tirx.mutable_cell import mutable_cell_alloc, mutable_cell_load

    allocation = mutable_cell_alloc("uint128")
    cell = tvm.ir.Var("cell", allocation.ty)
    function = tvm.tirx.Function(
        [],
        [tvm.ir.Bind(cell, allocation), tvm.ir.Return(mutable_cell_load(cell))],
        ret_type=tvm.ir.PrimType("uint128"),
    ).with_attr("global_symbol", "cell_value")
    monkeypatch.setenv("TVM_COMPILE_FORCE_FALLBACK", "1")
    source = tvm.get_global_func("target.build.cuda")(
        tvm.IRModule({"cell_value": function}),
        tvm.target.Target({"kind": "cuda", "arch": "sm_80"}),
    ).inspect_source()
    assert re.search(r"__uint128_t cell\s*=\s*[^;]+;", source)
    assert not re.search(r"cell\s*\[", source)
    snapshot = re.search(r"__uint128_t (\w+) = cell;", source)
    assert snapshot is not None
    assert f"return {snapshot.group(1)};" in source


@pytest.mark.parametrize("attrs", [{"mutable_cell_writes": [0]}, {"mutable_cell_condition": 0}])
def test_cuda_rejects_malformed_cell_operand_metadata(attrs, monkeypatch):
    from tvm.tirx.mutable_cell import mutable_cell_alloc, mutable_cell_load

    allocation = mutable_cell_alloc("int32", 0)
    cell = tvm.ir.Var("cell", allocation.ty)
    call = tvm.ir.Call(
        "tirx.cuda.func_call",
        [
            tvm.ir.StringImm("write_cell"),
            mutable_cell_load(cell),
            tvm.ir.StringImm("__device__ void write_cell(int& value) { value = 1; }"),
        ],
        attrs=attrs,
    )
    function = tvm.tirx.Function(
        [], [tvm.ir.Bind(cell, allocation), tvm.ir.Evaluate(call)], ret_type=tvm.ir.TupleType([])
    ).with_attr("global_symbol", "main")
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    monkeypatch.setenv("TVM_COMPILE_FORCE_FALLBACK", "1")
    with pytest.raises(ValueError, match="mutable_cell_"):
        tvm.get_global_func("target.build.cuda")(tvm.IRModule({"main": function}), target)


if __name__ == "__main__":
    tvm.testing.main()
