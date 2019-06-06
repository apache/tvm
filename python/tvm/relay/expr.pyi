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

from typing import List
import tvm
from .base import Span, NodeBase
from .ty import Type, TypeParam
from ._analysis import _get_checked_type


class Expr(NodeBase):
    def checked_type(self):
        ...

    def __call__(self, *args):
        ...


class Constant(Expr):
    data = ...  # type: tvm.nd.NDArray

    def __init__(self, data):
        # type: (tvm.nd.NDArray) -> None
        ...


class Tuple(Expr):
    fields = ...  # type: List[Expr]

    def __init__(self, fields):
        # type: (List[Expr]) -> None
        ...


class Var(Expr):
    """A local variable in Relay."""
    name_hint = ...  # type: str

    def __init__(self, name_hint):
        # type: (str) -> None
        ...


class GlobalVar(Expr):
    name_hint = ...  # type: str

    def __init__(self, name_hint):
        # type: (str) -> None
        ...


class Param(Expr):
    var = ...  # type: Var
    type = ...  # type: Type

    def __init__(self, var, ty):
        # type: (Var, Type) -> None
        ...


class Function(Expr):
    """A function in Relay, see tvm/relay/expr.h for more details."""
    type_params = ...  # type: List[TypeParam]
    params = ...  # type: List[Param]
    ret_type = ...  # type: Type
    body = ...  # type: Expr

    def __init__(self,
                 params,  # type: List[Param],
                 ret_type,  # type: Type,
                 body,  # type: Expr,
                 type_params=None,  # type: List[TypeParam]
                 ):
        # type: (...) -> None
        ...


@register_relay_node
class Call(Expr):
    """A function call in Relay, see tvm/relay/expr.h for more details."""
    op = ...  # type: Expr
    args = ...  # type: List[Expr]
    # todo(@jroesch): add attrs. revise attrs type in __init__

    def __init__(self, op, args, attrs=None, ty_args=None):
        # type: (Expr, List[Expr], Optional[List[Any]], Optional[List[Type]]) -> None
        if not ty_args:
            ty_args = []

        self.__init_handle_by_constructor__(
            _make.Call, op, args, attrs, ty_args)


@register_relay_node
class Let(Expr):
    """A variable bindings in Relay, see tvm/relay/expr.h for more details."""
    var = ...  # type: Var
    value = ...  # type: Expr
    body = ...  # type: Expr
    value_type = ...  # type: Type

    def __init__(self, var, value, body, value_type):
        # type: (Var, Expr, Expr, Type) -> None
        ...


@register_relay_node
class If(Expr):
    """A conditional expression in Relay, see tvm/relay/expr.h for more details."""
    cond = ...  # type: Expr
    true_value = ...  # type: Expr
    false_value = ...  # type: Expr
    span = ...  # type: Span

    def __init__(self, cond, true_value, false_value):
        # type: (Expr, Expr, Expr) -> None
        ...
