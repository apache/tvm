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
"""Execution scope utilities for CUDA op dispatches."""

from collections.abc import Callable

from tvm.script import tirx as Tx
from tvm.tirx import PrimFunc
from tvm.tirx.operator.tile_primitive import DispatchContext
from tvm.tirx.stmt import TilePrimitiveCall


def macro_or_prim_func(macro: Callable, need_macro: bool = False) -> Callable:
    """Wrap a macro in a ``prim_func`` unless the caller explicitly wants the macro."""
    if need_macro:
        return macro

    @Tx.prim_func(check_well_formed=False)
    def func():
        macro()

    return func


def thread_selector(sctx: DispatchContext, inner_impl, macro: bool = False) -> Callable:
    """Narrow execution to a single, deterministic thread within ``sctx.exec_scope``.

    The elected thread is stable across invocations so that synchronization
    primitives (for example PTX ``elect_sync``) behave correctly.

    Parameters
    ----------
    sctx : DispatchContext
        The dispatch context. Only ``sctx.scope_kind`` is consulted; the
        caller is responsible for having narrowed into the desired scope via an
        ``if Tx.filter(...):`` guard before reaching here.
    inner_impl : Tx.inline
        The body to execute inside the selected thread.
    macro : bool
        If True, return the macro directly; otherwise wrap it in a ``prim_func``.
    """
    assert not isinstance(inner_impl, PrimFunc), "inner_impl must be a macro, not a PrimFunc"
    name = sctx.scope_kind
    if name == "thread":
        return macro_or_prim_func(inner_impl, need_macro=macro)
    if name == "cta":

        @Tx.inline()
        def impl():
            Tx.lane_id([32])
            if Tx.ptx.elect_sync():
                with Tx.thread():
                    inner_impl()

        return macro_or_prim_func(impl, need_macro=macro)
    if name == "warp":

        @Tx.inline()
        def impl():
            Tx.lane_id([32])
            if Tx.ptx.elect_sync():
                with Tx.thread():
                    inner_impl()

        return macro_or_prim_func(impl, need_macro=macro)
    if name == "warpgroup":

        @Tx.inline()
        def impl():
            warp_id = Tx.warp_id_in_wg([4])
            Tx.lane_id([32])
            if Tx.filter(warp_id, 0, 1):
                with Tx.warp():
                    if Tx.ptx.elect_sync():
                        with Tx.thread():
                            inner_impl()

        return macro_or_prim_func(impl, need_macro=macro)
    raise ValueError(f"thread_selector: unsupported exec_scope {name!r}")


def single_thread(op_call: TilePrimitiveCall, sctx: DispatchContext) -> bool:
    """Predicate for dispatchers that require a single-thread execution scope."""
    del op_call
    return sctx.is_thread


def exec_scope_ok(
    op_call: TilePrimitiveCall, sctx: DispatchContext, expected_scopes: list[str]
) -> tuple[bool, str | None]:
    """Predicate helper: check that ``sctx.scope_kind`` is in *expected_scopes*."""
    del op_call
    ok = sctx.scope_kind in expected_scopes
    return ok, None if ok else f"unsupported exec_scope {sctx.scope_kind}"
