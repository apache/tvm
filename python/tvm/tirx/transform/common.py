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


import tvm_ffi

from tvm.ir import Call, Evaluate, Op, Stmt, Var
from tvm.tirx import Expr, decl_tensor, is_tensor_var
from tvm.tirx.layout import Iter, TileLayout


class TensorReplacer:
    """
    Replace tensor variables with other tensor variables.
    Tensor variables are ordinary Vars, so the same mapping also rewrites
    ``tensor_data_ptr`` projections.
    """

    def __init__(
        self, tensor_map: dict[Var, Var] | None = None, var_map: dict[Var, Var] | None = None
    ):
        super().__init__()
        self.tensor_map = tensor_map if tensor_map is not None else {}
        self.var_map = var_map if var_map is not None else {}
        for old_tensor, new_tensor in self.tensor_map.items():
            self.var_map[old_tensor] = new_tensor

    def __call__(self, node):
        def replace_var(op: Var):
            if is_tensor_var(op):
                return self._mutate_tensor(op)
            return self.var_map.get(op, op)

        return tvm_ffi.structural_map(
            node,
            (Var, replace_var),
            order="post",
        )

    def _replace_expr(self, expr: Expr):
        return tvm_ffi.structural_map(
            expr,
            (Var, lambda var: self.var_map.get(var, var)),
            order="post",
        )

    def _mutate_tensor(self, tensor: Var):
        if tensor in self.tensor_map:
            return self.tensor_map[tensor]

        new_shape = [self._replace_expr(expr) for expr in tensor.ty.shape]
        new_strides = [self._replace_expr(expr) for expr in tensor.ty.strides]
        new_elem_offset = (
            self._replace_expr(tensor.ty.elem_offset) if tensor.ty.elem_offset is not None else None
        )
        if isinstance(tensor.ty.layout, TileLayout):
            new_shard = [
                Iter(self._replace_expr(it.extent), self._replace_expr(it.stride), it.axis)
                for it in tensor.ty.layout.shard
            ]
            new_replicate = [
                Iter(self._replace_expr(it.extent), self._replace_expr(it.stride), it.axis)
                for it in tensor.ty.layout.replica
            ]
            new_layout = TileLayout.from_iters(
                new_shard,
                new_replicate,
                offset=tensor.ty.layout.offset,
            )
        else:
            new_layout = tensor.ty.layout

        unchanged = (
            all(old is new for old, new in zip(tensor.ty.shape, new_shape))
            and all(old is new for old, new in zip(tensor.ty.strides, new_strides))
            and tensor.ty.elem_offset is new_elem_offset
            and tensor.ty.layout is new_layout
        )
        if unchanged:
            return tensor

        new_tensor = decl_tensor(
            new_shape,
            tensor.ty.dtype,
            tensor.name,
            None,
            new_strides,
            new_elem_offset,
            tensor.scope(),
            tensor.ty.data_alignment,
            tensor.ty.offset_factor,
            layout=new_layout,
        )
        self.tensor_map[tensor] = new_tensor
        self.var_map[tensor] = new_tensor
        return new_tensor


def seek_kernel_replace_point(stmt: Stmt, body: Stmt) -> Stmt:
    """Replace the kernel replacement point in ``stmt`` with ``body``."""

    def replace_evaluate(op: Evaluate):
        value = op.value
        if isinstance(value, Call) and value.op.same_as(Op.get("tirx.kernel_replace_point")):
            return body
        return op

    return tvm_ffi.structural_map(stmt, (Evaluate, replace_evaluate), order="post")
