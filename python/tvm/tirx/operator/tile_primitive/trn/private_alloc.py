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

from typing import Any

from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx import Buffer, FloatImm, Stmt
from tvm.tirx.operator.tile_primitive.dispatch_context import DispatchContext
from tvm.tirx.operator.tile_primitive.ops import (
    BinaryReduce,
    Copy,
    Gemm,
    ReduceOp,
    UnaryOpWithBiasScale,
    UnaryReduce,
)
from tvm.tirx.operator.tile_primitive.registry import f_op_dispatcher
from tvm.tirx.operator.tile_primitive.trn.common import init_analyzer, nki_dim
from tvm.tirx.operator.tile_primitive.trn.dim_utils import get_ewise_dim_map
from tvm.tirx.operator.tile_primitive.trn.instruction_generator import InstructionGenerator
from tvm.tirx.stmt import TilePrimitiveCall


def alloc_const_bias_trn(
    op: TilePrimitiveCall, buffer_dict: dict[Any, tuple[Buffer, Stmt | None]], sctx: DispatchContext
) -> dict[str, Any]:
    bias = op.bias if op.bias is not None else FloatImm(op.dsts[0].buffer.dtype, 0.0)
    if "const_bias" in op.workspace:
        return {}
    if not isinstance(bias, (FloatImm)):
        return {}
    par_size = op.dsts[0].buffer.layout.size("P")
    max_inst_size = op.config.get("max_inst_size", 512)
    if ("const_bias", bias.value) in buffer_dict:
        bias_buffer, bias_init_stmt = buffer_dict[("const_bias", bias.value)]
        old_shape = bias_buffer.shape
        new_shape = [max(par_size, old_shape[0]), max(max_inst_size, old_shape[1])]
        if new_shape[0] == old_shape[0] and new_shape[1] == old_shape[1]:
            return {"const_bias": ("const_bias", bias.value)}
    else:
        new_shape = (par_size, max_inst_size)
    new_buffer = T.buffer(new_shape, dtype=bias.dtype, scope="trn.sbuf", buffer_name="const_bias")

    @T.prim_func
    def const_bias_init():
        with T.attr(0, "tensorized_nki_instruction", 1):
            for p_loop in T.serial(0, par_size, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(0, max_inst_size, annotations={nki_dim: "F"}):
                    T.evaluate(T.nki.memset(new_buffer[p_loop, f_loop], bias))
        Tx.tvm_kernel_replace_point()

    buffer_dict[("const_bias", bias.value)] = (new_buffer, const_bias_init.body)
    return {"const_bias": ("const_bias", bias.value)}


def alloc_partial_reduce_trn(
    op: TilePrimitiveCall, buffer_dict: dict[Any, tuple[Buffer, Stmt | None]], sctx: DispatchContext
) -> dict[str, Any]:
    if "partial_reduce" in op.workspace:
        return {}
    f_op_dispatcher(op, sctx)
    partial_reduce_buffer = None
    if DispatchContext.kPrivateAlloc not in sctx.callbacks:
        return {}
    for buffer in sctx.callbacks[DispatchContext.kPrivateAlloc]:
        if buffer.name == "partial_reduce":
            partial_reduce_buffer = buffer
            break
    if partial_reduce_buffer is None:
        return {}
    # no reuse opportunity
    buffer_dict[partial_reduce_buffer] = (partial_reduce_buffer, None)
    return {"partial_reduce": partial_reduce_buffer}


def alloc_identity_trn(
    op: TilePrimitiveCall, buffer_dict: dict[Any, tuple[Buffer, Stmt | None]], sctx: DispatchContext
) -> dict[str, Any]:
    if "identity" in op.workspace:
        return {}
    par_size = op.srcs[0].buffer.layout.size("P")
    if "identity" in buffer_dict:
        identity_buffer, identity_init_stmt = buffer_dict["identity"]
        old_shape = identity_buffer.shape
        new_shape = [max(par_size, old_shape[0]), max(par_size, old_shape[1])]
        if new_shape[0] == old_shape[0] and new_shape[1] == old_shape[1]:
            return {"identity": "identity"}
    else:
        new_shape = (par_size, par_size)
    new_buffer = T.buffer(
        new_shape, dtype=op.srcs[0].buffer.dtype, scope="trn.sbuf", buffer_name="identity"
    )

    @T.prim_func
    def identity_init():
        with T.attr(0, "tensorized_nki_instruction", 1):
            for p_loop in T.serial(0, par_size, annotations={nki_dim: "P"}):
                for rhs_f_loop in T.serial(0, par_size, annotations={nki_dim: "F"}):
                    T.evaluate(T.nki.identity(new_buffer[p_loop, rhs_f_loop], par_size))
        Tx.tvm_kernel_replace_point()

    buffer_dict["identity"] = (new_buffer, identity_init.body)
    return {"identity": "identity"}


def alloc_acc_psum_trn(
    op: TilePrimitiveCall, buffer_dict: dict[Any, tuple[Buffer, Stmt | None]], sctx: DispatchContext
) -> dict[str, Any]:
    if "acc_psum" in op.workspace or op.dsts[0].buffer.scope() == "trn.psum":
        return {}
    par_size = op.dsts[0].buffer.layout.size("P")
    acc_psum = T.buffer(
        (8, par_size, 512),
        "float32",
        scope="trn.psum",
        allocated_addr=(0, 0),
        buffer_name="acc_psum",
    )
    # no reuse opportunity
    buffer_dict[acc_psum] = (acc_psum, None)
    return {"acc_psum": acc_psum}


def alloc_copy_trn(
    op: TilePrimitiveCall, buffer_dict: dict[Any, tuple[Buffer, Stmt | None]], sctx: DispatchContext
) -> dict[str, Buffer]:
    src_region = op.srcs[0]
    dst_region = op.dsts[0]
    analyzer = init_analyzer(sctx)
    dim_map = get_ewise_dim_map(src_region, dst_region, analyzer)
    inst_gen = InstructionGenerator([src_region, dst_region], analyzer)
    inst_gen.link_buffer_regions(src_region, dst_region, dim_map)
    if inst_gen.check_partition_dim_match(src_region, dst_region):
        return {}

    identity_dict = alloc_identity_trn(op, buffer_dict, sctx)
    acc_psum_dict = alloc_acc_psum_trn(op, buffer_dict, sctx)
    return identity_dict | acc_psum_dict


def alloc_unary_reduce_trn(
    op: TilePrimitiveCall, buffer_dict: dict[Any, tuple[Buffer, Stmt | None]], sctx: DispatchContext
) -> dict[str, Buffer]:
    if "max_inst_size" in op.config:
        partial_reduce_dict = alloc_partial_reduce_trn(op, buffer_dict, sctx)
        const_bias_dict = alloc_const_bias_trn(op, buffer_dict, sctx)
        return partial_reduce_dict | const_bias_dict
    else:
        if "const_bias" in op.workspace and "partial_reduce" in op.workspace:
            return {}
        f_op_dispatcher(op, sctx)
        partial_reduce_buffer = None
        const_bias_buffer = None
        if DispatchContext.kPrivateAlloc not in sctx.callbacks:
            return {}
        for buffer in sctx.callbacks[DispatchContext.kPrivateAlloc]:
            if buffer.name == "partial_reduce":
                partial_reduce_buffer = buffer
            elif buffer.name == "const_bias":
                const_bias_buffer = buffer
        # no reuse opportunity
        workspace_dict = {}
        if partial_reduce_buffer is not None and "partial_reduce" not in op.workspace:
            buffer_dict[partial_reduce_buffer] = (partial_reduce_buffer, None)
            workspace_dict["partial_reduce"] = partial_reduce_buffer
        if const_bias_buffer is not None and "const_bias" not in op.workspace:
            assert len(sctx.callbacks[DispatchContext.kDeviceInitStmt]) == 1, (
                "const_bias should have init"
            )
            init_stmt = sctx.callbacks[DispatchContext.kDeviceInitStmt][0]
            buffer_dict[const_bias_buffer] = (const_bias_buffer, init_stmt)
            workspace_dict["const_bias"] = const_bias_buffer
        return workspace_dict


UnaryOpWithBiasScale.get_private_buffers_trn = alloc_const_bias_trn
ReduceOp.get_private_buffers_trn = alloc_partial_reduce_trn
Copy.get_private_buffers_trn = alloc_copy_trn
Gemm.get_private_buffers_trn = alloc_acc_psum_trn
BinaryReduce.get_private_buffers_trn = alloc_partial_reduce_trn
UnaryReduce.get_private_buffers_trn = alloc_unary_reduce_trn
