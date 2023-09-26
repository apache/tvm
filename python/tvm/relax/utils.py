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
"""Utility functions for Relax"""
import functools
import inspect
from typing import Tuple as typing_Tuple
from typing import Any, Callable, List, Dict, Optional, TypeVar

from .. import tir
from ..tir import PrimExpr
from ..runtime import String, convert_to_object
from . import _ffi_api
from .expr import Tuple as rx_Tuple
from .expr import Expr, ShapeExpr, Function, PrimValue, StringImm, te_tensor
from ..te import Tensor as te_Tensor, create_prim_func
from ..ir import Array, Attrs, Type, Map, VDevice
from .struct_info import PrimStructInfo, ShapeStructInfo, TensorStructInfo


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
    4. Return `relax.Tuple` if the input is a tuple/list of `Expr`.

    Notes
    -----
    1. `tvm.tir.StringImm` is not allowed because of ambiguity,
       which can be either `relax.StringImm` or `relax.PrimValue`.
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
    # Case 4
    if isinstance(value, (tuple, list)):
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
            raise ValueError("`args_to_expr` and `args_to_list_expr` should be disjoint.")

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


def copy_with_new_vars(func: Function) -> Function:
    """Copy the given function. All variables that are bound inside the original function
    would be copied to satisfy the restriction in the well-formed check: Variables in
    Relax must be bound exactly once. This also ensures that both the function and its copy
    can be inserted into the same IRModule, and be asserted on the structural equality
    agaisnt IRModule created by TVMScript.

    Parameters
    ----------
    func : Function
        The relax function to copy.

    Returns
    -------
    ret : Function
        The copied function.
    """
    return _ffi_api.CopyWithNewVars(func)  # type: ignore


def gen_call_tir_inputs(
    func: Callable, *args: Any, **kwargs: Any
) -> typing_Tuple[tir.PrimFunc, Expr, List[TensorStructInfo], Optional[ShapeExpr]]:
    """Generate the inputs for call_tir according to the te function.
    This function converts arguments from relax expression to te tensor,
    The callback func should return a te tensor or a list of te tensors.

    Parameters
    ----------
    func : Callable
        A function that returns a te tensor or a list of te tensors.

    args : Any, optional
        arguments passed to the function.

    kwargs : Any, optional
        The keyword arguments passed to the function.
        Note that the keyword args 'primfunc_attrs' is reserved for passing func
        attributes to be added to the PrimFunc that gets created.

    Returns
    -------
    ret : Tuple[tir.PrimFunc, Expr, List[TensorStructInfo], Optional[ShapeExpr]]
        ret contains the inputs for call_tir, including a tir prim_func, args,
        out_sinfo, and tir_vars.
    """

    def _convert_te_arg(
        te_args: Any, tir_var_map: Dict[tir.Var, tir.PrimExpr]
    ) -> typing_Tuple[Any, List[te_Tensor]]:
        """Helper function used to convert Relax expressions to TE tensor.

        In the common case, the type of te_args is a Relax expression and is converted
        into a TE tensor.
        If te_args is a nested or recursive datatype (i.e list, dict, tvm.ir.Map, tvm.ir.Array),
        we recursive and convert any value of type Relax expression into a TE tensor.
        Common values of type int, float, and str are preserved.

        In dynamic shape cases, the passed in arguments may contain TIR variable.
        For example, the argument can be a Relax Var with TensorStructInfo, which
        has symbolic shape, or the argument can be a ShapeExpr with symbolic variables.
        To make the PrimFunc generated has independent variables with
        the caller Relax function, we will substitute the TIR variables in the input
        arguments with fresh ones, which is done by maintaining a TIR variable mapping.

        Parameters
        ----------
        te_args : Any
            Argument to convert to TE

        tir_var_map : Dict[tir.Var, tir.PrimExpr]
            The TIR variable mapping, which maps TIR variables on the Relax function
            side to the new set of variables used on the PrimFunc side.

        Returns
        -------
        ret : (Any, [tvm.te.Tensor])
            A tuple of the converted te_args, and a list of te tensors for each converted
            Relax expression
        """
        te_args_list = []
        # extra list of tir expression arguments
        # that are not covered by Tensor
        extra_tir_args_list = []

        def _copy_undefined_var(expr: tir.PrimExpr):
            def _visit_expr(e: tir.PrimExpr):
                if isinstance(e, tir.Var) and e not in tir_var_map:
                    new_var = tir.Var(e.name, e.dtype)
                    tir_var_map[e] = new_var

            tir.stmt_functor.post_order_visit(expr, _visit_expr)

        n_tensor = 0

        def _convert_te_arg_helper(arg):
            nonlocal n_tensor
            if isinstance(arg, Expr):  # type: ignore
                if isinstance(arg.struct_info, TensorStructInfo):
                    assert isinstance(
                        arg.struct_info.shape, ShapeExpr
                    ), "emit_te now only supports Tensor that has ShapeExpr shape"
                    for shape_value in arg.struct_info.shape.values:
                        _copy_undefined_var(shape_value)

                    name = chr(ord("A") + n_tensor) if n_tensor < 26 else f"input{n_tensor}"
                    arg = te_tensor(arg, tir_var_map, name)
                    n_tensor += 1
                    te_args_list.append(arg)
                    return arg
                if isinstance(arg.struct_info, ShapeStructInfo):
                    assert isinstance(
                        arg, ShapeExpr
                    ), "For Expr having ShapeStructInfo, emit_te now only supports ShapeExpr"
                    return [_convert_te_arg_helper(val) for val in arg.values]
                if isinstance(arg.struct_info, PrimStructInfo):
                    return arg.value
            elif isinstance(arg, (list, Array)):
                return [_convert_te_arg_helper(x) for x in arg]
            elif isinstance(arg, tuple):
                return tuple(_convert_te_arg_helper(x) for x in arg)
            elif isinstance(arg, (dict, Map)):
                for key in arg:
                    assert isinstance(
                        key, str
                    ), "emit_te only supports dict with string as the key currently"
                return {k: _convert_te_arg_helper(arg[k]) for k in arg}
            elif isinstance(arg, tir.PrimExpr):
                _copy_undefined_var(arg)
                new_arg = tir.stmt_functor.substitute(arg, tir_var_map)
                extra_tir_args_list.append(new_arg)
                return new_arg
            elif isinstance(arg, (int, float, str, Type, Attrs)) or arg is None:
                return arg
            raise TypeError("not supported type in emit_te: {}".format(type(arg)))

        new_arg = _convert_te_arg_helper(te_args)
        return new_arg, te_args_list, extra_tir_args_list

    def _get_unbound_tir_vars(
        args: List[te_Tensor], extra_tir_args: List[PrimExpr]
    ) -> List[tir.Var]:
        """get unbound TIR vars (i.e TIR vars used in the shape but is not
        itself a dimension of a shape)"""
        bound_vars = set()
        used_vars = set()

        def _populate_used_vars(expr):
            if isinstance(expr, tir.Var):
                used_vars.add(expr)

        for val in extra_tir_args:
            tir.stmt_functor.post_order_visit(val, _populate_used_vars)

        for x in args:
            for s in x.shape:
                tir.stmt_functor.post_order_visit(s, _populate_used_vars)
                if isinstance(s, tir.Var):
                    bound_vars.add(s)

        diff = used_vars - bound_vars
        return list(diff)

    def _get_vdevice(arg: Any) -> Optional[VDevice]:
        """get the virtual device from arguments."""
        vdevice = None
        if isinstance(arg, Expr):  # type: ignore
            if isinstance(arg.struct_info, TensorStructInfo):
                vdevice = arg.struct_info.vdevice
        elif isinstance(arg, (list, Array, tuple)):
            for x in arg:
                vdevice = _get_vdevice(x)
                if vdevice is not None:
                    return vdevice
        elif isinstance(arg, (dict, Map)):
            for k in arg:
                vdevice = _get_vdevice(arg[k])
                if vdevice is not None:
                    return vdevice
        return vdevice

    def _shape_with_old_tir_var(
        shape_values: List[tir.PrimExpr], tir_var_inverse_map: Dict[tir.Var, tir.PrimExpr]
    ):
        return ShapeExpr(
            [tir.stmt_functor.substitute(value, tir_var_inverse_map) for value in shape_values]
        )

    primfunc_attrs = kwargs.pop("primfunc_attrs", None)

    tir_var_map: Dict[tir.Var, tir.PrimExpr] = {}
    new_args, te_arg_list, tir_arg_list = _convert_te_arg(args, tir_var_map)
    new_kwargs, te_kwarg_list, tir_kwarg_list = _convert_te_arg(kwargs, tir_var_map)

    te_args = te_arg_list + te_kwarg_list

    te_out = func(*new_args, **new_kwargs)
    assert isinstance(te_out, te_Tensor) or (
        isinstance(te_out, (tuple, list, Array)) and all(isinstance(t, te_Tensor) for t in te_out)
    ), "only support te.tensor or tuple/list/Array of te.tensor as function output"

    outs = [te_out] if isinstance(te_out, te_Tensor) else list(te_out)
    unbound_tir_vars = _get_unbound_tir_vars(te_args + outs, tir_arg_list + tir_kwarg_list)

    inputs = [*te_args] + outs + unbound_tir_vars
    tir_func = create_prim_func(inputs, "int64")

    if primfunc_attrs:
        tir_func = tir_func.with_attrs(primfunc_attrs)

    tir_func = tir_func.without_attr("global_symbol")

    call_tir_args = [x.op.value for x in te_args]

    # Invert the TIR variable mapping, to convert the output shape back
    # with old set of variables.
    tir_var_inverse_map = {v: k for k, v in tir_var_map.items()}

    output_sinfo = [
        TensorStructInfo(
            _shape_with_old_tir_var(out.shape, tir_var_inverse_map),
            out.dtype,
            _get_vdevice(args),
        )
        for out in outs
    ]

    tir_vars = None
    if len(unbound_tir_vars) > 0:
        tir_vars = _shape_with_old_tir_var(unbound_tir_vars, tir_var_inverse_map)

    return (tir_func, call_tir_args, output_sinfo, tir_vars)
