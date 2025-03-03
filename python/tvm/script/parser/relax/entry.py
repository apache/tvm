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
# pylint: disable=missing-docstring, invalid-name
import inspect
from typing import Any
from typing import Callable as _Callable
from typing import Dict, List, Optional, Set, TypeVar, Union

import tvm
from tvm.relax import (
    Expr,
    SeqExpr,
    ShapeExpr,
    FuncStructInfo,
    Function,
    ObjectStructInfo,
    PrimStructInfo,
    ShapeStructInfo,
    StructInfo,
    TensorStructInfo,
    TupleStructInfo,
)
from tvm.relax.expr import Var
from tvm.runtime import ObjectGeneric
from tvm.tir import PrimExpr

from .._core import doc, parse, utils
from ..core.entry import scan_macro
from ..core.parser import Parser, ScriptMacro
from ..ir import lookup_vdevice
from ...ir_builder import relax as R

FType = TypeVar("FType", bound=_Callable)

############################## R.function ##############################


# this formulation allows us to support having @R.function
# appear as a decorator by itself or to have optional arguments
# like @R.function(pure=False)
def function(
    f: Optional[FType] = None, pure: bool = True, private: bool = False, check_well_formed=True
) -> Union[Function, FType]:
    # pylint: disable=unused-argument
    # (pure and private aren't used here, but are used later in parsing)

    # need to inspect the stack first because is_defined_in_class expects the outer class
    # to be in a particular position in the stack
    orig_stack = inspect.stack()

    def decorator_wrapper(f):
        if not inspect.isfunction(f):
            raise TypeError(f"Expect a function, but got: {f}")
        if utils.is_defined_in_class(orig_stack, f):
            return f
        return parse(f, utils.inspect_function_capture(f), check_well_formed=check_well_formed)

    if f is not None:
        # if there are no optional args given, this will directly invoke the wrapper
        return decorator_wrapper(f)
    else:
        # if there is a optional arg given, it returns the wrapper function
        # as a new decorator and applies it
        setattr(decorator_wrapper, "dispatch_token", "relax")
        return decorator_wrapper


setattr(function, "dispatch_token", "relax")


############################## R.macro ##############################


class RelaxMacro(ScriptMacro):
    """Specialization of the ScriptMacro class for Relax."""

    def parse_macro(self, parser: Parser) -> Expr:
        macro_def = self.get_macro_def()
        ret_value = None

        with R.SeqExpr() as seq:
            for idx, stmt in enumerate(macro_def.body):
                # Normally, a "return" statement is only allowed in a R.function. We don't
                # want to parse the macro's body as if it was a body of a function, because
                # the latter imposes some constraints that we want to avoid.
                # At the same time, we want to use "return" to indicate the value of the
                # macro (since in Relax everything is an expression), so add special handling
                # of "return".
                if isinstance(stmt, doc.Return):
                    ret_value = parser.eval_expr(stmt.value)
                    if idx + 1 != len(macro_def.body):
                        parser.report_error(macro_def, "'return' should be the last statement")
                    break
                parser.visit(stmt)

        if ret_value is None:
            parser.report_error(macro_def, "Macros must end with a return statement")

        return SeqExpr(seq.binding_blocks, ret_value)


def macro(*args, hygienic: bool = True) -> _Callable:
    """Decorator for macro definitions.

    Parameters
    ----------
    hygienic: bool
        Specifies whether the macro is hygienic or not.
        A macro is hygienic if all symbols used in the macro's body are resolved
        to values from the location of the macro definition. A non-hygienic macro
        will have its symbols resolved to values at the time of the macro's use.
    """

    def _decorator(func: _Callable) -> ScriptMacro:
        source, closure_vars = scan_macro(func, utils.inspect_function_capture(func))
        obj = RelaxMacro(source, closure_vars, func, hygienic)

        def wrapper(*args, **kwargs):
            return obj(*args, **kwargs)

        return wrapper

    if len(args) == 0:
        return _decorator
    if len(args) == 1 and inspect.isfunction(args[0]):
        return _decorator(args[0])

    raise ValueError(
        "Invalid use of R.macro. Usage: @R.macro, @R.macro(), @R.macro(hygienic=[True|False])"
    )


############################# Struct Info ##############################


class StructInfoProxy(ObjectGeneric):
    def as_struct_info(self, dict_globals: Optional[Dict[str, Any]] = None) -> StructInfo:
        raise NotImplementedError()

    def get_symbolic_vars(self) -> Set[str]:
        return {}

    def asobject(self):
        return self.as_struct_info(None)


############################### R.Object ################################


class ObjectProxy(StructInfoProxy):
    """The proxy fo ObjectStructInfo.

    Parameters
    ----------
    values : Optional[List[PrimExpr]]
       The symbolic shape values if known.

    ndim : Optional[int]
       The size of the shape.
    """

    def __init__(self) -> None:
        pass

    def get_symbolic_vars(self) -> Set[str]:
        return set()

    def as_struct_info(self, dict_globals: Optional[Dict[str, Any]] = None) -> ShapeStructInfo:
        return ObjectStructInfo()


def Object() -> ObjectProxy:
    return ObjectProxy()


############################### R.Tensor ###############################


def _eval_shape(expr: Union[str, PrimExpr], dict_globals: Optional[Dict[str, Any]]) -> PrimExpr:
    if isinstance(expr, str):
        code = compile(expr, "<string>", "eval")
        return eval(code, dict_globals or {})  # pylint: disable=eval-used
    else:
        return expr


class TensorProxy(StructInfoProxy):
    shape: Optional[List[Union[str, PrimExpr]]]
    dtype: str
    vdevice: Optional[str]
    ndim: int

    def __init__(
        self,
        shape: Optional[Union[List[Union[PrimExpr, str]], Expr]] = None,
        dtype: Optional[str] = None,
        vdevice: Optional[str] = None,
        ndim: int = -1,
    ) -> None:
        if isinstance(shape, Expr):
            if not isinstance(shape, (ShapeExpr, Var)):
                raise ValueError(
                    "When the shape is an Expr, it must be a ShapeExpr or a Var with ShapeExpr "
                    f"value. But got: {shape} with type: {type(shape)}"
                )
            if isinstance(shape, Var) and not isinstance(shape.struct_info, ShapeStructInfo):
                raise ValueError(
                    "When the shape is a Var, it must have shape struct_info. But got "
                    f"{shape} with struct_info: {shape.struct_info}"
                )
        self.shape = shape
        self.dtype = dtype
        self.vdevice = vdevice
        self.ndim = ndim

    def get_symbolic_vars(self) -> Set[str]:
        if self.shape is None or isinstance(self.shape, Expr):
            return {}
        else:
            return {s for s in self.shape if isinstance(s, str) and s.isidentifier()}

    def as_struct_info(self, dict_globals: Optional[Dict[str, Any]] = None) -> TensorStructInfo:
        vdev = self.vdevice
        if isinstance(self.vdevice, str):
            if ":" in self.vdevice:
                split_vdev = self.vdevice.split(":")
                vdev = lookup_vdevice(split_vdev[0], int(split_vdev[1]))
            else:
                vdev = lookup_vdevice(self.vdevice, 0)

        if self.shape is None:
            return TensorStructInfo(None, self.dtype, vdev, self.ndim)
        elif isinstance(self.shape, (ShapeExpr, Var)):
            return TensorStructInfo(self.shape, self.dtype, vdev, self.ndim)
        else:
            if dict_globals is None and any([isinstance(s, str) for s in self.shape]):
                raise ValueError(
                    "String-defined shape expr is only allowed when parsing function parameters "
                    "and return annotations for TVMScript."
                )
            shape = [_eval_shape(s, dict_globals) for s in self.shape]
            return TensorStructInfo(shape, self.dtype, vdev, self.ndim)


def Tensor(
    shape: Optional[Union[List[Union[PrimExpr, str]], Expr]] = None,
    dtype: Optional[str] = None,
    vdevice: Optional[str] = None,
    ndim: int = -1,
) -> TensorProxy:
    # scalar tensor case
    if shape is not None and not isinstance(shape, Var) and len(shape) == 0:
        shape = []
    if isinstance(shape, str) and dtype is None:
        dtype = shape
        shape = None

    if shape is not None and not isinstance(shape, (tuple, list)) and not isinstance(shape, Expr):
        raise ValueError(f"shape must be a list/tuple or an Expr, but got: {shape}")
    return TensorProxy(shape, dtype, vdevice, ndim)


############################## R.Callable ##############################


class CallableProxy(StructInfoProxy):
    params: List[StructInfoProxy]
    ret: StructInfoProxy
    purity: bool
    derive_func: Optional[Union[str, tvm.ir.EnvFunc]]

    """Function type.

    A function type consists of a list of type parameters to enable
    the definition of generic functions,
    a set of type constraints which we omit for the time being,
    a sequence of argument types, the purity of the function, and a return type.

    Parameters
    ----------
    params : List[StructInfoProxy]
        The argument StructInfoProxy

    ret : StructInfoProxy
        The return StructInfoProxy.

    purity : bool
        Whether the callable is pure.

    derive_func: Optional[Union[str, tvm.ir.EnvFunc]]
        The derivation function to determine the output StructInfo,
        based on the arguments provided to the function.  The
        specified function should be accessible using
        `tvm.get_global_func`, and should have a signature
        `Callable[[relax.Call, relax.BlockBuilder], relax.StructInfo]`.

    """

    def __init__(
        self,
        params: Optional[Union[StructInfoProxy, List[StructInfoProxy]]] = None,
        ret: Optional[StructInfoProxy] = None,
        purity: Optional[bool] = None,
        derive_func: Optional[Union[str, tvm.ir.EnvFunc]] = None,
    ) -> None:
        if params is None:
            self.params = params
        else:
            if not isinstance(params, (list, tuple)):
                params = [params]
            # convert `R.Callable` to `R.Callable()`
            self.params = [param() if callable(param) else param for param in params]

        # Mimic the C++ defaults, where an opaque function is assumed
        # to be impure, and a non-opaque function is assumed to be
        # pure.
        if purity is None:
            purity = params is not None

        self.ret = ret() if callable(ret) else ret
        self.purity = purity
        self.derive_func = derive_func

    def get_symbolic_vars(self) -> Set[str]:
        if self.params is None:
            return set()
        else:
            return set().union(*[p.get_symbolic_vars() for p in self.params])

    def as_struct_info(self, dict_globals: Optional[Dict[str, Any]] = None) -> FuncStructInfo:
        if self.ret is None:
            ret = None
        else:
            ret = self.ret.as_struct_info(dict_globals)

        if self.params is None:
            params = None
        else:
            params = [param.as_struct_info(dict_globals) for param in self.params]

        if params is None:
            return FuncStructInfo.opaque_func(
                ret=ret, derive_func=self.derive_func, purity=self.purity
            )
        else:
            return FuncStructInfo(params, ret, purity=self.purity)


def Callable(
    params: Optional[Union[StructInfoProxy, List[StructInfoProxy]]] = None,
    ret: Optional[StructInfoProxy] = None,
    purity: Optional[bool] = None,
    derive_func: Optional[Union[str, tvm.ir.EnvFunc]] = None,
) -> CallableProxy:
    return CallableProxy(params, ret, purity=purity, derive_func=derive_func)


############################### R.Tuple ################################


class TupleProxy(StructInfoProxy):
    fields: List[StructInfoProxy]
    """The type of tuple values.

    Parameters
    ----------
    fields : List[StructInfoProxy]
        The fields in the tuple
    """

    def __init__(
        self,
        *fields: List[StructInfoProxy],
    ) -> None:
        if len(fields) == 1 and isinstance(fields[0], (tuple, list)):
            fields = fields[0]
        # convert `R.Tensor` to `R.Tensor()`
        self.fields = [field() if callable(field) else field for field in fields]

    def get_symbolic_vars(self) -> Set[str]:
        return set().union(*[f.get_symbolic_vars() for f in self.fields])

    def as_struct_info(self, dict_globals: Optional[Dict[str, Any]] = None) -> TupleStructInfo:
        fields = [field.as_struct_info(dict_globals) for field in self.fields]
        return TupleStructInfo(fields)


def Tuple(*fields: List[StructInfoProxy]) -> TupleProxy:
    return TupleProxy(*fields)


############################### R.Shape ################################


class ShapeProxy(StructInfoProxy):
    values: Optional[List[PrimExpr]]
    ndim: int
    """The type of shape values.

    Parameters
    ----------
    values : Optional[List[PrimExpr]]
       The symbolic shape values if known.

    ndim : Optional[int]
       The size of the shape.
    """

    def __init__(
        self,
        values: Optional[List[PrimExpr]] = None,
        ndim: int = -1,
    ) -> None:
        self.values = values
        self.ndim = ndim

    def get_symbolic_vars(self) -> Set[str]:
        if self.values is None:
            return set()
        else:
            return {v for v in self.values if isinstance(v, str) and v.isidentifier()}

    def as_struct_info(self, dict_globals: Optional[Dict[str, Any]] = None) -> ShapeStructInfo:
        values = [_eval_shape(v, dict_globals) for v in self.values] if self.values else None
        return ShapeStructInfo(values, self.ndim)


def Shape(values: Optional[List[PrimExpr]] = None, ndim: int = -1) -> ShapeProxy:
    return ShapeProxy(values, ndim)


################################ R.Prim ################################


class PrimProxy(StructInfoProxy):
    dtype: Optional[str]
    value: Optional[Union[int, float, str, PrimExpr]]

    """The type of TIR-representable values.

    Parameters
    ----------
    dtype : Optional[str]
       The data type.

    value: Optional[Union[int, float, str, PrimExpr]]
       The known value
    """

    def __init__(
        self,
        dtype: Optional[str] = None,
        value: Optional[Union[int, float, str, PrimExpr]] = None,
    ) -> None:
        if dtype is None and value is None:
            raise TypeError(
                "R.Prim missing required argument.  " "Must provide either 'dtype' or 'value'"
            )

        self.dtype = dtype
        self.value = value

    def get_symbolic_vars(self) -> Set[str]:
        if isinstance(self.value, str) and self.value.isidentifier():
            return {self.value}
        else:
            return set()

    def as_struct_info(self, dict_globals: Optional[Dict[str, Any]] = None) -> ShapeStructInfo:
        if self.value is None:
            return PrimStructInfo(dtype=self.dtype)
        else:
            value = _eval_shape(self.value, dict_globals)
            return PrimStructInfo(dtype=self.dtype, value=value)


def Prim(
    dtype: Optional[str] = None,
    value: Optional[Union[int, float, str, PrimExpr]] = None,
) -> PrimProxy:
    return PrimProxy(dtype, value)


############################ R.match_cast #############################
class MatchCastPair:
    value: Expr
    struct_info: StructInfo

    def __init__(self, value: Expr, struct_info: StructInfo) -> None:
        self.value = value
        self.struct_info = struct_info


def match_cast(value: Expr, struct_info: StructInfo):
    struct_info = _normalize_struct_info(struct_info)

    if value is None:
        raise ValueError("value of match_cast cannot be None")
    if struct_info is None:
        raise ValueError("struct_info of match_cast cannot be None")
    return MatchCastPair(value, struct_info)


def _normalize_struct_info_proxy(annotation) -> StructInfoProxy:
    if annotation is None:
        return TupleProxy([])
    elif callable(annotation):
        return annotation()
    elif isinstance(annotation, StructInfoProxy):
        return annotation
    else:
        raise TypeError(f"Expected StructInfoProxy but got {type(annotation)}.")


def _normalize_struct_info(
    struct_info, dict_globals: Optional[Dict[str, Any]] = None
) -> StructInfo:
    if isinstance(struct_info, StructInfo):
        return struct_info
    else:
        proxy = _normalize_struct_info_proxy(struct_info)
        return proxy.as_struct_info(dict_globals)
