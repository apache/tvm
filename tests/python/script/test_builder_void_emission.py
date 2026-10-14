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
"""Core behavior of canonical void operators in an IRBuilder."""

import pytest
import tvm_ffi

from tvm import ir, tirx
from tvm.backend import cuda
from tvm.ir.builder_base import AlreadyEmitted
from tvm.script import ir_builder as I
from tvm.script import tirx as T


@pytest.fixture(scope="module", autouse=True)
def register_cuda():
    cuda.register_backend()


@pytest.mark.parametrize("ty", [ir.PrimType("void"), ir.TupleType([])])
def test_void_emission_and_fallback(ty):
    assert isinstance(tirx.cpu_parallel_barrier(ty=ty), ir.Call)
    with I.IRBuilder() as builder:
        value = tirx.cpu_parallel_barrier(ty="int32")
        assert isinstance(value, ir.Call)
        assert builder.result is None
        receipt = tirx.cpu_parallel_barrier(ty=ty)
        assert isinstance(receipt, AlreadyEmitted)
        assert T.evaluate(receipt) is receipt
    assert builder.get().same_as(receipt.value)
    assert receipt.value.value.ty.same_as(ty)
    assert tvm_ffi.convert(receipt).same_as(receipt.value)


def test_parser_emits_once_with_source_locations():
    @T.function
    def function():
        T.ptx.barrier.sync.aligned(0)
        T.evaluate(T.ptx.barrier.sync.aligned(0))
        T.cpu_parallel_barrier()
        T.evaluate(T.cpu_parallel_barrier())

    assert len(function.body.seq) == 4
    for stmt in function.body.seq:
        assert isinstance(stmt, ir.Evaluate)
        assert not isinstance(stmt.loc, ir.UnknownLoc)
        assert not isinstance(stmt.value.loc, ir.UnknownLoc)


def test_raw_construction_and_codegen_do_not_emit():
    from tvm.backend.cuda.codegen.registry import CODEGEN_REGISTRY

    ptx = T.ptx.barrier.sync.aligned(0)
    assert isinstance(ptx, ir.Call)
    with I.IRBuilder() as builder:
        raw = tirx.call_intrin("void", "tirx.cpu_parallel_barrier")
        assert isinstance(raw, ir.Call)
        helper, _ = CODEGEN_REGISTRY[ptx.op.name](ptx.args)
        assert isinstance(helper, ir.Call)
        ptx.script()
        assert builder.result is None
