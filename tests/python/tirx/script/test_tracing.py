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
# pylint: disable=invalid-name, missing-docstring
"""Native Python tracing, emission and parameter contracts."""

import linecache
import sys
from contextlib import nullcontext

import pytest
from tvm_ffi import structural_walk

import tvm_tirx
from tvm_tirx import ir
from tvm_tirx.script import tirx as T
from tvm_tirx.script.ir_builder import AlreadyEmitted, IRBuilder
from tvm_tirx.tirx.script import ir_builder as I


def _calls(function):
    calls = []
    structural_walk(function, (ir.Call, calls.append))
    return calls


def _leaf(loc):
    while isinstance(loc, ir.CallSiteLoc):
        loc = loc.callee
    return loc


@pytest.mark.parametrize("trace", [False, True])
def test_void_calls_emit_once_and_values_remain_expressions(trace):
    with IRBuilder() as builder:
        with I.function_():
            with builder.line_trace_scope() if trace else nullcontext():
                receipt = T.ptx["bar.sync"](0)
                assert isinstance(receipt, AlreadyEmitted)
                assert T.evaluate(receipt) is receipt
                I.emit_(receipt)
                T.gpu_storage_sync("shared")
                T.cuda.func_call("side_effect", "__device__ void side_effect() {}")
                value = T.cuda.elect_sync()
                assert isinstance(value, ir.Call)
                raw = ir.Call("tirx.gpu_storage_sync", [ir.StringImm("shared")])
                assert isinstance(raw, ir.Call)
                T.evaluate(raw)
    names = [call.op.name for call in _calls(builder.get())]
    assert names.count("tirx.ptx.bar_sync") == 1
    assert names.count("tirx.gpu_storage_sync") == 2
    assert "tirx.cuda.elect_sync" not in names


def test_auto_emit_uses_current_builder_without_tracing():
    with IRBuilder() as outer:
        with I.function_():
            T.ptx["bar.sync"](0)
            with pytest.raises(RuntimeError, match="failed"):
                with IRBuilder() as inner:
                    with I.function_():
                        receipt = T.ptx["bar.sync"](1)
                        assert isinstance(receipt, AlreadyEmitted)
                        raise RuntimeError("failed")
            T.ptx["bar.sync"](2)
    assert isinstance(T.ptx["bar.sync"](3), ir.Call)
    assert len(_calls(outer.get())) == 2
    assert inner.result is None  # Aborted functions are not finalized.


@pytest.mark.parametrize("trace", [False, True])
def test_backend_tile_namespace_emits_once(trace):
    from tvm_tirx.backend.cuda.tile.patterns import apply as raw_apply
    from tvm_tirx.backend.cuda.tile.patterns.ewise import ewise as raw_ewise

    @T.function(private=True)
    def identity(value: T.float32) -> T.float32:
        return value

    with IRBuilder() as builder:
        with I.function_():
            source = I.arg_("source", T.Tensor((1, 4), T.float32))
            dest = I.bind(T.alloc_tensor(ty_args=[T.Tensor((1, 4), T.float32, scope="local")]))
            with builder.line_trace_scope() if trace else nullcontext():
                receipt = T.cuda.tile.apply(T.ptx["ld.global.v4.f32"], dest, source)
                assert isinstance(receipt, AlreadyEmitted)
                T.evaluate(receipt)
                receipt = T.cuda.tile.ewise(identity, dest, dest)
                assert isinstance(receipt, AlreadyEmitted)
                T.evaluate(receipt)
                assert isinstance(raw_apply(T.ptx["ld.global.v4.f32"], dest, source), ir.Call)
                assert isinstance(raw_ewise(identity, dest, dest), ir.Call)
    names = [call.op.name for call in _calls(builder.get())]
    assert names.count("tirx.cuda.tile.apply") == 1
    assert names.count("tirx.cuda.tile.ewise") == 1


def test_ptx_failed_and_ambiguous_candidates_emit_nothing():
    with IRBuilder() as builder:
        with I.function_():
            with pytest.raises(ValueError):
                T.ptx["bar.sync"]()
            chain = T.ptx["bar.sync"]
            ambiguous = type(chain)(chain._cands * 2)
            with pytest.raises(AssertionError, match="ambiguous"):
                ambiguous(0)
            T.ptx["bar.sync"](1)
    assert len(_calls(builder.get())) == 1


def test_line_trace_tracks_direct_calls_helpers_and_frames():
    def helper():
        T.ptx["bar.sync"](1)

    with IRBuilder() as builder:
        with I.function_():
            with builder.line_trace_scope():
                T.ptx["bar.sync"](0)
                helper()
                with T.serial(2):
                    T.ptx["bar.sync"](2)
    calls = _calls(builder.get())
    assert len(calls) == 3
    for call, fragment in zip(calls, ("(0)", "(1)", "(2)")):
        loc = _leaf(call.loc)
        assert loc.source_name.name == __file__
        assert fragment in linecache.getline(__file__, loc.start_line)
    assert isinstance(calls[1].loc, ir.CallSiteLoc)
    assert "helper()" in linecache.getline(__file__, _leaf(calls[1].loc.caller).start_line)


def test_line_trace_preserves_existing_callback_and_restores_on_exception():
    previous = sys.gettrace()
    events = []

    def existing(frame, event, arg):
        if frame.f_code.co_name == "helper":
            events.append(event)
        return existing

    def helper():
        T.ptx["bar.sync"](0)
        raise RuntimeError("failed")

    frame = sys._getframe()
    saved = (frame.f_trace, frame.f_trace_lines, frame.f_trace_opcodes)
    sys.settrace(existing)
    try:
        frame.f_trace = existing
        frame.f_trace_lines = False
        frame.f_trace_opcodes = True
        with IRBuilder() as builder:
            with I.function_():
                with pytest.raises(RuntimeError, match="failed"):
                    with builder.line_trace_scope():
                        helper()
                assert sys.gettrace() is existing
        assert {"call", "line", "return"} <= set(events)
        assert (frame.f_trace, frame.f_trace_lines, frame.f_trace_opcodes) == (
            existing,
            False,
            True,
        )
    finally:
        sys.settrace(previous)
        frame.f_trace, frame.f_trace_lines, frame.f_trace_opcodes = saved


def test_nested_line_trace_restores_outer_builder():
    with IRBuilder() as outer:
        with I.function_():
            with outer.line_trace_scope():
                T.ptx["bar.sync"](0)
                with IRBuilder() as inner:
                    with I.function_():
                        with inner.line_trace_scope():
                            T.ptx["bar.sync"](1)
                T.ptx["bar.sync"](2)
    for function, fragments in ((outer.get(), ("(0)", "(2)")), (inner.get(), ("(1)",))):
        calls = _calls(function)
        assert len(calls) == len(fragments)
        for call, fragment in zip(calls, fragments):
            assert fragment in linecache.getline(__file__, _leaf(call.loc).start_line)


def _source_chain(loc):
    if isinstance(loc, ir.CallSiteLoc):
        return [*_source_chain(loc.caller), *_source_chain(loc.callee)]
    return [loc]


def test_trace_preserves_explicit_scopes_and_deferred_frames():
    from tvm_tirx.script.ir_builder import at_

    caller = ir.SourceLoc(ir.SourceName("caller.py"), 2, 1, 2, 20)
    explicit = ir.SourceLoc(ir.SourceName("definition.py"), 3, 1, 3, 20)
    attached = ir.SourceLoc(ir.SourceName("definition.py"), 3, 5, 3, 10)

    def helper():
        return T.evaluate(1)

    with IRBuilder() as builder:
        with builder.with_loc(caller):
            with builder.line_trace_scope():
                loop_line = sys._getframe().f_lineno + 1
                loop = T.serial(2)
            with loop, builder.line_trace_scope():
                same_line = sys._getframe().f_lineno + 1
                first, second = helper(), T.evaluate(2)
                with builder.with_loc(explicit):
                    third = T.evaluate(3)
                    fourth = at_(attached, T.evaluate(4))
                last = T.evaluate(5)
    result = builder.get()
    assert isinstance(result, ir.For)
    assert result.loc.caller.same_as(caller)
    assert result.loc.callee.start_line == loop_line
    assert len(result.body.seq) == 5
    for receipt, statement, fragment, extra in zip(
        (first, second, third, fourth, last),
        result.body.seq,
        (
            "return T.evaluate(1)",
            "helper(), T.evaluate(2)",
            "T.evaluate(3)",
            "T.evaluate(4)",
            "T.evaluate(5)",
        ),
        (None, None, explicit, attached, None),
    ):
        assert receipt.value.same_as(statement)
        chain = _source_chain(statement.loc)
        assert chain[0].same_as(caller)
        if extra is not None:
            assert chain.pop().same_as(extra)
        assert chain[-1].source_name.name == __file__
        assert fragment in linecache.getline(__file__, chain[-1].start_line)
    # Returning from a helper restores the caller even on the same Python line.
    assert _leaf(second.value.loc).start_line == same_line
    assert _source_chain(first.value.loc)[-2].start_line == same_line
    with builder:
        assert isinstance(T.evaluate(0).value.loc, ir.UnknownLoc)


def test_nested_trace_callback_failure_restores_outer_scope(monkeypatch):
    from tvm_tirx.script.ir_builder import _ffi_api

    original = _ffi_api.IRBuilderUpdateLoc
    fail = [False]

    def update(*args):
        if fail[0]:
            fail[0] = False
            raise RuntimeError("trace callback failed")
        return original(*args)

    monkeypatch.setattr(_ffi_api, "IRBuilderUpdateLoc", update)
    previous = sys.gettrace()
    frame = sys._getframe()
    saved = (frame.f_trace, frame.f_trace_lines, frame.f_trace_opcodes)
    with IRBuilder() as builder:
        with I.function_():
            with builder.line_trace_scope():
                callback = sys.gettrace()
                with pytest.raises(RuntimeError, match="trace callback failed"):
                    with builder.line_trace_scope():
                        fail[0] = True
                        T.evaluate(0)
                assert sys.gettrace() == callback
                receipt = T.evaluate(5)
    assert sys.gettrace() is previous
    assert (frame.f_trace, frame.f_trace_lines, frame.f_trace_opcodes) == saved
    assert "receipt = T.evaluate(5)" in linecache.getline(
        __file__, _leaf(receipt.value.loc).start_line
    )
    assert len(builder.get().body.seq) == 1


def test_shared_tensor_shapes_are_checked_by_compiled_host():
    import torch

    n = T.Var("n", T.int64)
    with IRBuilder() as builder:
        with I.function_():
            I.func_name_("check_shape")
            I.func_attr({"global_symbol": "check_shape"})
            I.func_ret_type_(T.int64)
            I.arg_("x", T.Tensor((n,), T.float32))
            I.arg_("y", T.Tensor((n,), T.float32))
            I.return_(n)
    compiled = tvm_tirx.tirx.compile(builder.get(), target="llvm", pipeline="tirx")
    assert compiled(torch.zeros(3), torch.zeros(3)) == 3
    for other, error, message in (
        (torch.zeros(4), ValueError, "expected to match"),
        (torch.zeros(3, dtype=torch.int32), TypeError, "expected float32"),
        (torch.zeros(1, 3), ValueError, "expected 1"),
    ):
        with pytest.raises(error, match=message):
            compiled(torch.zeros(3), other)
