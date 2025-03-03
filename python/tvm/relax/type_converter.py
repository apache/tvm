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

# pylint: disable=invalid-name,too-many-locals

"""Argument converter utility for Relax

This utility is used to decorate constructors of `tvm.relax.Expr`, and
must be able to be imported before `tvm.relax.Expr` or its subtypes
have been defined.  Neither the class definitions nor any type
signature in this file may reference relax types.  All references must
be exclusively in function bodies to avoid having a circular reference
during module imports.
"""

import functools
import inspect

from typing import List, Optional, Callable, TypeVar, Any

import tvm

FType = TypeVar("FType", bound=Callable[..., "tvm.relax.Expr"])


class _ArgsConverter:
    """A helper class to convert the arguments to Expr."""

    @staticmethod
    def convert(args_to_expr: List[str], args_to_list_expr: List[str]):
        """Convert the arguments to Expr.

        Parameters
        ----------
        args_to_expr : List[str]
            The argument names to be converted to Expr.

        args_to_list_expr : List[str]
            The argument names to be converted to List[Expr].

        Returns
        -------
        output : Callable[[FType], FType]
            The decorator.
        """

        if any([x in args_to_list_expr for x in args_to_expr]):
            raise ValueError("`args_to_expr` and `args_to_list_expr` should be disjoint.")

        def _convert(name: str, value: Any) -> Any:
            if value is None:
                return value
            if name in args_to_expr:
                try:
                    return tvm.relax.utils.convert_to_expr(value)
                except Exception as err:
                    raise TypeError(
                        f"Argument `{name}` is expected to be converted to `Expr`, "
                        f"but failed with input value: {value}"
                    ) from err
            elif name in args_to_list_expr:
                try:
                    return [tvm.relax.utils.convert_to_expr(x) for x in value]
                except Exception as err:
                    raise TypeError(
                        f"Argument `{name}` is expected to be converted to `List[Expr]`, "
                        f"but failed with input value: {value}"
                    ) from err
            else:
                return value

        def inner(func: FType) -> FType:
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            for name in args_to_expr + args_to_list_expr:
                if name not in param_names:
                    raise ValueError(f"Argument `{name}` is not found in function signature.")

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                for param in sig.parameters.values():
                    if param.kind == param.VAR_POSITIONAL:
                        # *args case
                        values = [_convert(param.name, x) for x in bound.arguments[param.name]]
                        bound.arguments[param.name] = tuple(values)
                    elif param.kind == param.VAR_KEYWORD:
                        # **kwargs case
                        key_value = {
                            key: _convert(param.name, value)
                            for key, value in bound.arguments[param.name].items()
                        }
                        bound.arguments[param.name] = key_value
                    else:
                        bound.arguments[param.name] = _convert(
                            param.name, bound.arguments[param.name]
                        )
                return func(*bound.args, **bound.kwargs)

            return wrapper  # type: ignore

        return inner

    @staticmethod
    def to_expr(*arg_names: str) -> Callable:
        """Convert the arguments to Expr.

        Parameters
        ----------
        *arg_names: str
            The list of argument names that need to be converted to Expr.

        Returns
        -------
        output: Callable
            The decorator.
        """

        return _ArgsConverter.convert(args_to_expr=list(arg_names), args_to_list_expr=[])

    @staticmethod
    def to_list_expr(*arg_names: str) -> Callable:
        """Convert the arguments to List of Expr.

        Parameters
        ----------
        *arg_names: str
            The list of argument names that need to be converted to List of Expr.

        Returns
        -------
        output: Callable
            The decorator.
        """

        return _ArgsConverter.convert(args_to_expr=[], args_to_list_expr=list(arg_names))

    @staticmethod
    def auto(func: FType) -> FType:
        """Decorator for automatically convert the arguments to Expr according to type annotation.
        Only two patterns are supported:

        1. The argument is Expr or Optional[Expr].

        2. The argument is List[Expr] or Optional[List[Expr]].

        """
        sig = inspect.signature(func)
        args_to_expr = []
        args_to_list_expr = []

        from . import Expr  # pylint: disable=import-outside-toplevel

        for param in sig.parameters.values():
            anno = param.annotation
            if anno in (Expr, Optional[Expr]):
                args_to_expr.append(param.name)
            if anno in (List[Expr], Optional[List[Expr]]):
                args_to_list_expr.append(param.name)

        return _ArgsConverter.convert(args_to_expr, args_to_list_expr)(func)


args_converter = _ArgsConverter()  # pylint: disable=invalid-name
