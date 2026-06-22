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
from collections.abc import Callable as _Callable
from typing import Any, TypeVar

import tvm
from tvm.ir import PrimType
from tvm.relax import (
    Expr,
    Function,
    FuncType,
    ObjectType,
    SeqExpr,
    ShapeExpr,
    ShapeType,
    TensorType,
    TupleType,
    Type,
)
from tvm.relax.expr import Var
from tvm.relax.script import builder as R
from tvm.runtime import ObjectConvertible
from tvm.script.ir_builder.ir import lookup_vdevice
from tvm.script.parser._core import doc, parse, utils
from tvm.script.parser.core.entry import scan_macro
from tvm.script.parser.core.parser import Parser, ScriptMacro
from tvm.tirx import PrimExpr

FType = TypeVar("FType", bound=_Callable)

############################## R.function ##############################


# this formulation allows us to support having @R.function
# appear as a decorator by itself or to have optional arguments
# like @R.function(pure=False)
def function(
    f: FType | None = None, pure: bool = True, private: bool = False, check_well_formed=True
) -> Function | FType:
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


############################# Type ##############################


class TypeProxy(ObjectConvertible):
    def as_ty(self, dict_globals: dict[str, Any] | None = None) -> Type:
        raise NotImplementedError()

    def get_symbolic_vars(self) -> set[str]:
        return {}

    def asobject(self):
        return self.as_ty(None)


############################### R.Object ################################


class ObjectProxy(TypeProxy):
    """The proxy fo ObjectType.

    Parameters
    ----------
    values : Optional[List[PrimExpr]]
       The symbolic shape values if known.

    ndim : Optional[int]
       The size of the shape.
    """

    def __init__(self) -> None:
        pass

    def get_symbolic_vars(self) -> set[str]:
        return set()

    def as_ty(self, dict_globals: dict[str, Any] | None = None) -> ObjectType:
        return ObjectType()


def Object() -> ObjectProxy:
    return ObjectProxy()


############################### R.Tensor ###############################


def _eval_shape(expr: str | PrimExpr, dict_globals: dict[str, Any] | None) -> PrimExpr:
    if isinstance(expr, str):
        code = compile(expr, "<string>", "eval")
        return eval(code, dict_globals or {})  # pylint: disable=eval-used
    else:
        return expr


class TensorProxy(TypeProxy):
    shape: list[str | PrimExpr] | None
    dtype: str
    vdevice: str | None
    ndim: int

    def __init__(
        self,
        shape: list[PrimExpr | str] | Expr | None = None,
        dtype: str | None = None,
        vdevice: str | None = None,
        ndim: int = -1,
    ) -> None:
        if isinstance(shape, Expr):
            if not isinstance(shape, ShapeExpr | Var):
                raise ValueError(
                    "When the shape is an Expr, it must be a ShapeExpr or a Var with ShapeExpr "
                    f"value. But got: {shape} with type: {type(shape)}"
                )
            if isinstance(shape, Var) and not isinstance(shape.ty, ShapeType):
                raise ValueError(
                    "When the shape is a Var, it must have shape ty. But got "
                    f"{shape} with ty: {shape.ty}"
                )
        self.shape = shape
        self.dtype = dtype
        self.vdevice = vdevice
        self.ndim = ndim

    def get_symbolic_vars(self) -> set[str]:
        if self.shape is None or isinstance(self.shape, Expr):
            return {}
        else:
            return {s for s in self.shape if isinstance(s, str) and s.isidentifier()}

    def as_ty(self, dict_globals: dict[str, Any] | None = None) -> TensorType:
        vdev = self.vdevice
        if isinstance(self.vdevice, str):
            if ":" in self.vdevice:
                split_vdev = self.vdevice.split(":")
                vdev = lookup_vdevice(split_vdev[0], int(split_vdev[1]))
            else:
                vdev = lookup_vdevice(self.vdevice, 0)

        if self.shape is None:
            return TensorType(None, self.dtype, vdev, self.ndim)
        elif isinstance(self.shape, ShapeExpr | Var):
            return TensorType(self.shape, self.dtype, vdev, self.ndim)
        else:
            if dict_globals is None and any([isinstance(s, str) for s in self.shape]):
                raise ValueError(
                    "String-defined shape expr is only allowed when parsing function parameters "
                    "and return annotations for TVMScript."
                )
            shape = [_eval_shape(s, dict_globals) for s in self.shape]
            return TensorType(shape, self.dtype, vdev, self.ndim)


def Tensor(
    shape: list[PrimExpr | str] | Expr | None = None,
    dtype: str | None = None,
    vdevice: str | None = None,
    ndim: int = -1,
) -> TensorProxy:
    # scalar tensor case
    if shape is not None and not isinstance(shape, Var) and len(shape) == 0:
        shape = []
    if isinstance(shape, str) and dtype is None:
        dtype = shape
        shape = None

    if shape is not None and not isinstance(shape, tuple | list) and not isinstance(shape, Expr):
        raise ValueError(f"shape must be a list/tuple or an Expr, but got: {shape}")
    return TensorProxy(shape, dtype, vdevice, ndim)


############################## R.Callable ##############################


class CallableProxy(TypeProxy):
    params: list[TypeProxy]
    ret: TypeProxy
    purity: bool
    derive_func: str | tvm.ir.EnvFunc | None

    """Function type.

    A function type consists of a list of type parameters to enable
    the definition of generic functions,
    a set of type constraints which we omit for the time being,
    a sequence of argument types, the purity of the function, and a return type.

    Parameters
    ----------
    params : List[TypeProxy]
        The argument TypeProxy

    ret : TypeProxy
        The return TypeProxy.

    purity : bool
        Whether the callable is pure.

    derive_func: Optional[Union[str, tvm.ir.EnvFunc]]
        The derivation function to determine the output Type,
        based on the arguments provided to the function.  The
        specified function should be accessible using
        `tvm.get_global_func`, and should have a signature
        `Callable[[relax.Call, relax.BlockBuilder], relax.Type]`.

    """

    def __init__(
        self,
        params: TypeProxy | list[TypeProxy] | None = None,
        ret: TypeProxy | None = None,
        purity: bool | None = None,
        derive_func: str | tvm.ir.EnvFunc | None = None,
    ) -> None:
        if params is None:
            self.params = params
        else:
            if not isinstance(params, list | tuple):
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

    def get_symbolic_vars(self) -> set[str]:
        if self.params is None:
            return set()
        else:
            return set().union(*[p.get_symbolic_vars() for p in self.params])

    def as_ty(self, dict_globals: dict[str, Any] | None = None) -> FuncType:
        if self.ret is None:
            ret = None
        else:
            ret = self.ret.as_ty(dict_globals)

        if self.params is None:
            params = None
        else:
            params = [param.as_ty(dict_globals) for param in self.params]

        if params is None:
            return FuncType.opaque_func(ret=ret, derive_func=self.derive_func, purity=self.purity)
        else:
            return FuncType(params, ret, purity=self.purity)


def Callable(
    params: TypeProxy | list[TypeProxy] | None = None,
    ret: TypeProxy | None = None,
    purity: bool | None = None,
    derive_func: str | tvm.ir.EnvFunc | None = None,
) -> CallableProxy:
    return CallableProxy(params, ret, purity=purity, derive_func=derive_func)


############################### R.Tuple ################################


class TupleProxy(TypeProxy):
    fields: list[TypeProxy]
    """The type of tuple values.

    Parameters
    ----------
    fields : List[TypeProxy]
        The fields in the tuple
    """

    def __init__(
        self,
        *fields: list[TypeProxy],
    ) -> None:
        if len(fields) == 1 and isinstance(fields[0], tuple | list):
            fields = fields[0]
        # convert `R.Tensor` to `R.Tensor()`
        self.fields = [field() if callable(field) else field for field in fields]

    def get_symbolic_vars(self) -> set[str]:
        return set().union(*[f.get_symbolic_vars() for f in self.fields])

    def as_ty(self, dict_globals: dict[str, Any] | None = None) -> TupleType:
        fields = [field.as_ty(dict_globals) for field in self.fields]
        return TupleType(fields)


def Tuple(*fields: list[TypeProxy]) -> TupleProxy:
    return TupleProxy(*fields)


############################### R.Shape ################################


class ShapeProxy(TypeProxy):
    values: list[PrimExpr] | None
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
        values: list[PrimExpr] | None = None,
        ndim: int = -1,
    ) -> None:
        self.values = values
        self.ndim = ndim

    def get_symbolic_vars(self) -> set[str]:
        if self.values is None:
            return set()
        else:
            return {v for v in self.values if isinstance(v, str) and v.isidentifier()}

    def as_ty(self, dict_globals: dict[str, Any] | None = None) -> ShapeType:
        values = [_eval_shape(v, dict_globals) for v in self.values] if self.values else None
        return ShapeType(values, self.ndim)


def Shape(values: list[PrimExpr] | None = None, ndim: int = -1) -> ShapeProxy:
    return ShapeProxy(values, ndim)


################################ R.Prim ################################


class PrimProxy(TypeProxy):
    dtype: str | None

    """The type of TIR-representable values.

    Parameters
    ----------
    dtype : Optional[str]
       The data type.

    """

    def __init__(
        self,
        dtype: str | None = None,
        value: int | float | str | PrimExpr | None = None,
    ) -> None:
        if dtype is None:
            if isinstance(value, PrimExpr):
                dtype = value.dtype
            elif isinstance(value, float):
                dtype = "float32"
            elif value is not None:
                dtype = "int64"
            else:
                raise TypeError("R.Prim missing required argument 'dtype'")

        self.dtype = dtype

    def get_symbolic_vars(self) -> set[str]:
        return set()

    def as_ty(self, dict_globals: dict[str, Any] | None = None) -> PrimType:
        return PrimType(self.dtype)


def Prim(
    dtype: str | None = None,
    value: int | float | str | PrimExpr | None = None,
) -> PrimProxy:
    return PrimProxy(dtype, value)


############################ R.match_cast #############################
class MatchCastPair:
    value: Expr
    ty: Type

    def __init__(self, value: Expr, ty: Type) -> None:
        self.value = value
        self.ty = ty


def match_cast(value: Expr, ty: Type):
    ty = _normalize_ty(ty)

    if value is None:
        raise ValueError("value of match_cast cannot be None")
    if ty is None:
        raise ValueError("ty of match_cast cannot be None")
    return MatchCastPair(value, ty)


def _normalize_ty_proxy(annotation) -> TypeProxy:
    if annotation is None:
        return TupleProxy([])
    elif callable(annotation):
        annotation = annotation()
        if isinstance(annotation, PrimExpr):
            return PrimProxy(annotation.dtype)
        return annotation
    elif isinstance(annotation, TypeProxy):
        return annotation
    else:
        raise TypeError(f"Expected TypeProxy but got {type(annotation)}.")


def _normalize_ty(ty, dict_globals: dict[str, Any] | None = None) -> Type:
    if isinstance(ty, Type):
        return ty
    else:
        proxy = _normalize_ty_proxy(ty)
        return proxy.as_ty(dict_globals)
