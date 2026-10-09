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

from tvm.backend.trn.tile_primitive.common import nki_dim
from tvm.ir import Stmt
from tvm.script import tirx as T
from tvm.tirx import DispatchContext, FloatImm, IntImm, Var
from tvm.tirx.op.tile import (
    BinaryReduce,
    Exp,
    Gemm,
    ReduceOp,
    Sqrt,
    UnaryOpWithScaleBias,
    UnaryReduce,
)
from tvm.tirx.operator.tile_primitive.registry import f_op_dispatcher
from tvm.tirx.tensor_instruction import TensorCall


def _scalar_dtype(scalar) -> str:
    dtype = getattr(scalar, "dtype", None)
    if dtype is not None:
        return str(dtype)
    ty = getattr(scalar, "ty", None)
    dtype = getattr(ty, "dtype", None)
    if dtype is None:
        raise AttributeError(f"{type(scalar).__name__} has no dtype-bearing PrimType")
    return str(dtype)


def alloc_const_bias_trn(
    op: TensorCall, buffer_dict: dict[Any, tuple[Var, Stmt | None]], sctx: DispatchContext
) -> dict[str, Any]:
    bias = getattr(op, "bias", FloatImm(op.dsts[0].source.ty.dtype, 0.0))
    if "const_bias" in op.workspaces:
        return {}
    if not isinstance(bias, (FloatImm)):
        return {}
    bias_key = ("const_bias", _scalar_dtype(bias), float(bias.value).hex())
    par_size = op.dsts[0].source.ty.layout.size("P")
    max_inst_size = op.options.get("max_inst_size", 512)
    if isinstance(max_inst_size, int | IntImm) and int(max_inst_size) == -1:
        raise ValueError("Constant bias workspace allocation requires a finite max_inst_size")
    if bias_key in buffer_dict:
        bias_buffer, bias_init_stmt = buffer_dict[bias_key]
        old_shape = bias_buffer.ty.shape
        new_shape = [max(par_size, old_shape[0]), max(max_inst_size, old_shape[1])]
        if new_shape[0] == old_shape[0] and new_shape[1] == old_shape[1]:
            return {"const_bias": bias_key}
    else:
        new_shape = (par_size, max_inst_size)
    new_buffer = T.Var(
        "const_bias", T.Tensor(new_shape, dtype=_scalar_dtype(bias), scope="trn.sbuf")
    )

    # This fragment captures buffers and indices from its insertion scope.
    @T.function(check_well_formed=False)
    def const_bias_init():
        with T.nki.tensorized_instruction():
            for p_loop in T.serial(0, par_size, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(0, max_inst_size, annotations={nki_dim: "F"}):
                    T.evaluate(T.nki.memset(new_buffer[p_loop, f_loop], bias))
        T.kernel_replace_point()

    buffer_dict[bias_key] = (new_buffer, const_bias_init.body)
    return {"const_bias": bias_key}


def alloc_partial_reduce_trn(
    op: TensorCall, buffer_dict: dict[Any, tuple[Var, Stmt | None]], sctx: DispatchContext
) -> dict[str, Any]:
    if "partial_reduce" in op.workspaces:
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


def alloc_acc_psum_trn(
    op: TensorCall, buffer_dict: dict[Any, tuple[Var, Stmt | None]], sctx: DispatchContext
) -> dict[str, Any]:
    if "acc_psum" in op.workspaces or op.dsts[0].source.scope() == "trn.psum":
        return {}
    par_size = op.dsts[0].source.ty.layout.size("P")
    acc_psum = T.Var(
        "acc_psum", T.Tensor((8, par_size, 512), "float32", scope="trn.psum", allocated_addr=(0, 0))
    )
    # no reuse opportunity
    buffer_dict[acc_psum] = (acc_psum, None)
    return {"acc_psum": acc_psum}


def alloc_unary_reduce_trn(
    op: TensorCall, buffer_dict: dict[Any, tuple[Var, Stmt | None]], sctx: DispatchContext
) -> dict[str, Var]:
    if "max_inst_size" in op.options:
        partial_reduce_dict = alloc_partial_reduce_trn(op, buffer_dict, sctx)
        const_bias_dict = alloc_const_bias_trn(op, buffer_dict, sctx)
        return partial_reduce_dict | const_bias_dict
    else:
        if "const_bias" in op.workspaces and "partial_reduce" in op.workspaces:
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
        if partial_reduce_buffer is not None and "partial_reduce" not in op.workspaces:
            buffer_dict[partial_reduce_buffer] = (partial_reduce_buffer, None)
            workspace_dict["partial_reduce"] = partial_reduce_buffer
        if const_bias_buffer is not None and "const_bias" not in op.workspaces:
            assert len(sctx.callbacks[DispatchContext.kDeviceInitStmt]) == 1, (
                "const_bias should have init"
            )
            init_stmt = sctx.callbacks[DispatchContext.kDeviceInitStmt][0]
            buffer_dict[const_bias_buffer] = (const_bias_buffer, init_stmt)
            workspace_dict["const_bias"] = const_bias_buffer
        return workspace_dict


UnaryOpWithScaleBias.get_private_buffers_trn = alloc_const_bias_trn
Sqrt.get_private_buffers_trn = alloc_const_bias_trn
Exp.get_private_buffers_trn = alloc_const_bias_trn
ReduceOp.get_private_buffers_trn = alloc_partial_reduce_trn
Gemm.get_private_buffers_trn = alloc_acc_psum_trn
BinaryReduce.get_private_buffers_trn = alloc_partial_reduce_trn
UnaryReduce.get_private_buffers_trn = alloc_unary_reduce_trn
