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
# pylint: disable=invalid-name
"""Hexagon-specific IR transformations"""

import functools as ft

import tvm
from ..._ffi.registry import register_func

### VTCM

vtcm_size = 4 * 1024 * 1024  # pylint: disable=invalid-name


@register_func("tvm.info.mem.local.vtcm")
def mem_info_vtcm():
    # pylint: disable=bad-whitespace
    return tvm.ir.make_node(
        "MemoryInfo",
        unit_bits=8,
        max_num_bits=vtcm_size * 8,
        max_simd_bits=128 * 8,
        head_address=tvm.runtime.const(100, "uint32"),
    )


def lower_vtcm_(get_alloc, get_free, def_align, func, mod, ctx):  # pylint: disable=unused-argument

    """Generic VTCM allocation

    Parameters
    ----------
    get_alloc : function: tir.Allocate, int -> tir.expr (dtype='handle')
      The VTCM allocation function. It takes an Allocate statement, and the required
      alignment, and returns a pointer to the allocated VTCM buffer.
    get_free : function: tir.expr (dtype='handle') -> None
      The VTCM deallocation function. It takes the address of the allocated buffer
      and frees it. It returns no value.
    def_align : int
      The default alignment that will be passed to the allocation function, if the
      program does not specify the alignment via a 'storage_alignment' attribute.
    func : tir.PrimFunc
    mod : tvm.IRModule
    ctx : transform.PassContext

    Returns
    -------
    stmt : tvm.stmt
        Transformed function body.
    """

    vtcm_buffers = []
    alignments = {}

    def buf_align(var):
        """Determine the alignment of the buffer with variable 'var'."""
        if var in alignments and alignments[var]:
            return alignments[var][-1]
        return def_align

    def visit(stmt):
        """Collect information about VTCM buffers and their alignments."""
        if isinstance(stmt, tvm.tir.AttrStmt):
            if stmt.attr_key == "storage_alignment":
                if not stmt.node in alignments:
                    alignments[stmt.node] = []
                alignments[stmt.node].append(stmt.value)
        elif isinstance(stmt, tvm.tir.Allocate):
            scope = stmt.buffer_var.type_annotation.storage_scope
            if scope == "local.vtcm":
                vtcm_buffers.append(stmt.buffer_var)

    def mutate(stmt):
        """Insert calls to VTCM allocation and deallocation routines."""
        if isinstance(stmt, tvm.tir.AttrStmt):
            if stmt.attr_key == "storage_alignment":
                alignments[stmt.node].pop()
            return stmt
        if isinstance(stmt, tvm.tir.Allocate):
            var = stmt.buffer_var
            scope = var.type_annotation.storage_scope
            is_vtcm = var in vtcm_buffers
            if scope == "local.vtcm":
                vtcm_buffers.pop()
            if is_vtcm:
                is_null = tvm.tir.call_intrin("bool", tvm.ir.Op.get("tir.isnullptr"), var)
                throw_error = tvm.tir.call_intrin(
                    "int32", tvm.ir.Op.get("tir.tvm_throw_last_error")
                )
                body_w_free = tvm.tir.SeqStmt([stmt.body, tvm.tir.Evaluate(get_free(var))])
                body_w_check = tvm.tir.IfThenElse(
                    is_null, tvm.tir.Evaluate(throw_error), body_w_free
                )
                return tvm.tir.LetStmt(
                    stmt.buffer_var, get_alloc(stmt, buf_align(var)), body_w_check
                )
            return stmt
        raise ValueError("Wrong argument type (" + type(stmt) + ") to 'mutate'")

    f = func.with_body(
        tvm.tir.stmt_functor.ir_transform(
            func.body, visit, mutate, ["tir.Allocate", "tir.AttrStmt"]
        )
    )
    return f


def ir_lower_vtcm():
    """Create a VTCM lowering pass.

    VTCM memory has to be allocated using special functions.
    """

    def get_alloc(stmt, align):
        assert isinstance(stmt, tvm.tir.Allocate)
        return tvm.tir.call_extern(
            "handle",
            "HexagonBackendAllocateVTCM",
            ft.reduce(lambda x, y: x * y, stmt.extents, 1),
            align,
        )

    def get_free(var):
        return tvm.tir.call_extern("handle", "HexagonBackendFreeVTCM", var)

    # pylint: disable=bad-whitespace
    @tvm.tir.transform.prim_func_pass(opt_level=0, name="Lower VTCM pass")
    def transform(func, mod, ctx):
        return lower_vtcm_(get_alloc, get_free, 2048, func, mod, ctx)

    return transform


def ir_lower_vtcm_pass():
    return [(3, ir_lower_vtcm())]
