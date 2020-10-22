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
"""TVM Script Parser Special Stmt Classes"""
# pylint: disable=unused-argument, no-self-argument, inconsistent-return-statements
# pylint: disable=relative-beyond-top-level
from typed_ast import ast3 as ast

import tvm.tir
from tvm import te
from .utils import get_param_list
from .registry import register


class SpecialStmt:
    """Base class for all Special Stmts"""

    def __init__(self, func, def_symbol):
        self.func = func
        self.def_symbol = def_symbol
        self.node = None
        self.context = None

    def signature(self):
        return "tir." + self.func.__name__, get_param_list(self.func)

    def handle(self, node, context, arg_list):
        self.node = node
        self.context = context
        return self.func(*arg_list)


@register
class MatchBuffer(SpecialStmt):
    """Special Stmt match_buffer(var, shape, dtype, data, strides, elem_offset, scope, align,
                                 offset_factor, buffer_type)
    Example
    -------
    .. code-block:: python
        A = tir.match_buffer(a, (128, 128), dtype="float32")
    """

    def __init__(self):
        def match_buffer(
            param,
            shape,
            dtype="float32",
            data=None,
            strides=None,
            elem_offset=None,
            scope="global",
            align=-1,
            offset_factor=0,
            buffer_type="default",
        ):
            assert isinstance(self.node, ast.Assign)

            if param not in self.context.func_params:
                self.context.report_error("Can not bind non-input param to buffer")
            if strides is None:
                strides = []
            align = align.value if not isinstance(align, int) else align
            offset_factor = (
                offset_factor.value if not isinstance(offset_factor, int) else offset_factor
            )
            buffer = tvm.tir.decl_buffer(
                shape,
                dtype,
                self.node.targets[0].id,
                data,
                strides,
                elem_offset,
                scope,
                align,
                offset_factor,
                buffer_type,
            )
            self.context.func_buffer_map[param] = buffer
            self.context.update_symbol(self.node.targets[0].id, buffer)

        super().__init__(match_buffer, def_symbol=True)


@register
class BufferDeclare(SpecialStmt):
    """Special Stmt buffer_decl(shape, dtype, data, strides, elem_offset, scope, align,
                                offset_factor, buffer_type)
    Example
    -------
    .. code-block:: python
        A = tir.buffer_decl((128, 128), dtype="float32")
    """

    def __init__(self):
        def buffer_decl(
            shape,
            dtype="float32",
            data=None,
            strides=None,
            elem_offset=None,
            scope="global",
            align=-1,
            offset_factor=0,
            buffer_type="default",
        ):
            assert isinstance(self.node, ast.Assign)

            if strides is None:
                strides = []
            align = align.value if not isinstance(align, int) else align
            offset_factor = (
                offset_factor.value if not isinstance(offset_factor, int) else offset_factor
            )
            buffer = tvm.tir.decl_buffer(
                shape,
                dtype,
                self.node.targets[0].id,
                data,
                strides,
                elem_offset,
                scope,
                align,
                offset_factor,
                buffer_type,
            )
            self.context.update_symbol(self.node.targets[0].id, buffer)
            return buffer

        super().__init__(buffer_decl, def_symbol=True)


@register
class VarDef(SpecialStmt):
    """ Special function for defining a Var"""

    def __init__(self):
        def var(dtype):
            assert isinstance(self.node, ast.Assign)
            v = te.var(self.node.targets[0].id, dtype)
            self.context.update_symbol(v.name, v)

        super().__init__(var, def_symbol=True)


@register
class EnvThread(SpecialStmt):
    """ Bind a var to thread env """

    def __init__(self):
        def env_thread(env_name):
            assert isinstance(self.node, ast.Assign)
            v = te.var(self.node.targets[0].id)
            self.context.func_var_env_dict[v] = env_name
            self.context.update_symbol(v.name, v)

        super().__init__(env_thread, def_symbol=True)


@register
class FuncAttr(SpecialStmt):
    """Special Stmt for declaring the DictAttr of PrimFunc
    Example
    -------
    .. code-block:: python
         tir.func_attr({"tir.noalias": True, "global_symbol"})
    """

    def __init__(self):
        def func_attr(dict_attr):
            self.context.func_dict_attr = dict_attr

        super().__init__(func_attr, def_symbol=False)
