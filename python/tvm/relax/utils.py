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
"""Utility functions for Relax"""
import functools
import inspect
from typing import Any, Callable, List, Optional, TypeVar

from .. import tir
from ..runtime import String, convert_to_object
from ..tir import PrimExpr
from . import _ffi_api
from .expr import Expr, Function, PrimValue, ShapeExpr, StringImm
from .expr import Tuple as rx_Tuple


def metadata_partitioner(rx_txt: str) -> List[str]:
    """Extract Relax program and metadata section.

    Parameters
    ----------
    rx_txt : str
        The input relax text.

    Returns
    -------
    output : List[str]
        The result list of partitioned text, the first element
        is the relax program, and the second is metadata section.
    """
    partitions = []
    left_curly = 0
    meta_start = 0
    meta_end = 0
    for i, char in enumerate(rx_txt):
        if i < 0:
            raise ValueError("The program is invalid.")
        if char == "{":
            if meta_start == 0:
                meta_start = i
            left_curly += 1
        elif char == "}":
            left_curly -= 1
            if left_curly == 0:
                meta_end = i + 1
                break

    if meta_end == 0:
        raise ValueError("The metadata section was not found.")
    metadata = rx_txt[meta_start:meta_end]
    rx_program = rx_txt[meta_end:-1]

    partitions.append(rx_program)
    partitions.append(metadata)

    return partitions


def convert_to_expr(value: Any) -> Expr:
    """Helper function to convert the input to Expr, which follows the rules:
    1. Return the input itself if it's already a `relax.Expr`;
    2. Return `relax.PrimValue` if the input is a `PrimExpr`;
    3. Return `relax.StringImm` if the input is `tvm.String` or `str`;
    4. Return `relax.ShapeExpr` if the input is a tuple/list of `PrimExpr` w/ int dtype;
    5. Return `relax.Tuple` if the input is a tuple/list of `Expr`.

    Notes
    -----
    1. `tvm.tir.StringImm` is not allowed because of ambiguity,
       which can be either `relax.StringImm` or `relax.PrimValue`.
    2. We regard empty tuple/list as `relax.Tuple` instead of `relax.ShapeExpr`
    """
    if isinstance(value, int):
        return PrimValue(tir.IntImm("int64", value))

    tvm_value = convert_to_object(value)
    # Case 1
    if isinstance(tvm_value, Expr):  # type: ignore
        return tvm_value
    # Note`` 1
    if isinstance(tvm_value, tir.StringImm):
        raise TypeError(
            "Cannot convert `tir.StringImm` to `relax.Expr` because of ambiguity,"
            "which can be either `relax.StringImm` or `relax.PrimValue` "
        )
    # Case 2
    if isinstance(tvm_value, PrimExpr):
        return PrimValue(value)
    # Case 3
    if isinstance(tvm_value, String):
        return StringImm(value)
    # Case 4 & 5
    if isinstance(value, (tuple, list)):
        # Note 2
        if len(value) == 0:
            return rx_Tuple([])
        # Case 4
        opt_prim_value = [convert_to_object(v) for v in value]
        if all([isinstance(v, PrimExpr) and v.dtype.startswith("int") for v in opt_prim_value]):
            return ShapeExpr(value)
        # Case 5
        # `convert_to_expr` ensures that all elements are `Expr` if no exception raises
        return rx_Tuple([convert_to_expr(v) for v in value])
    raise TypeError(f"Cannot convert {value} with type {type(value)} to `relax.Expr`")


FType = TypeVar("FType", bound=Callable[..., Expr])


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
            raise ValueError(f"`args_to_expr` and `args_to_list_expr` should be disjoint.")

        def _convert(name: str, value: Any) -> Any:
            if value is None:
                return value
            if name in args_to_expr:
                try:
                    return convert_to_expr(value)
                except:
                    raise TypeError(
                        f"Argument `{name}` is expected to be converted to `Expr`, "
                        f"but failed with input value: {value}"
                    )
            elif name in args_to_list_expr:
                try:
                    return [convert_to_expr(x) for x in value]
                except:
                    raise TypeError(
                        f"Argument `{name}` is expected to be converted to `List[Expr]`, "
                        f"but failed with input value: {value}"
                    )
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

        for param in sig.parameters.values():
            anno = param.annotation
            if anno in (Expr, Optional[Expr]):
                args_to_expr.append(param.name)
            if anno in (List[Expr], Optional[List[Expr]]):
                args_to_list_expr.append(param.name)

        return _ArgsConverter.convert(args_to_expr, args_to_list_expr)(func)


args_converter = _ArgsConverter()  # pylint: disable=invalid-name


def copy_with_new_params(func: Function) -> Function:
    """Copy the given function. The parameters of the original function would be copied to
    satisfy the restriction in the well-formed check: any two functions cannot share the same
    parameter variable.

    Parameters
    ----------
    func : Function
        The relax function to copy.

    Returns
    -------
    ret : Function
        The copied function.
    """
    return _ffi_api.CopyWithNewParams(func)  # type: ignore
