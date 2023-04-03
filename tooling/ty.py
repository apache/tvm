# pylint: disable=too-few-public-methods
"""Type system for Relax operator schema."""
import typing
from abc import ABC, abstractmethod


class _NotSpecified:
    """A singleton used to indicate that no default value is provided."""


NotSpecified = _NotSpecified()


class _BaseType(ABC):
    default: typing.Any
    _cc_as_relax: typing.Optional[typing.Callable[[str], str]]

    def __init__(
        self,
        *,
        default=NotSpecified,
        cc_as_relax=None,
    ):
        self.default = default
        self._cc_as_relax = cc_as_relax

    @abstractmethod
    def py_type(self) -> str:
        """Return the python type annotation string."""
        raise NotImplementedError

    @abstractmethod
    def py_type_guard(self) -> str:
        """Return the python type checking code."""
        raise NotImplementedError

    @abstractmethod
    def cc_type(self, force_object: bool) -> str:
        """Return the C++ type annotation string."""
        raise NotImplementedError

    @abstractmethod
    def cc_as_relax(self) -> str:
        """Return the C++ converter from the type to relax::Expr."""
        raise NotImplementedError


##### Containers #####


_BaseTypeOrCallable = typing.Union[
    _BaseType,
    typing.Callable[[], _BaseType],
]


def _parse_base_type_or_callable(
    t: _BaseTypeOrCallable,  # pylint: disable=invalid-name
) -> _BaseType:
    try:
        if callable(t):
            t = t()
        assert isinstance(t, _BaseType), t
    except Exception as error:
        raise TypeError(f"Invalid type: {t}") from error
    return t


class Array(_BaseType):
    """A container of elements of the same type."""

    elem: _BaseType
    length: typing.List[int]

    def __init__(
        self,
        elem: _BaseTypeOrCallable,
        *,
        length: typing.Union[None, int, typing.Sequence[int]] = None,
        default=NotSpecified,
        cc_as_relax=None,
    ):
        super().__init__(default=default, cc_as_relax=cc_as_relax)
        self.elem = _parse_base_type_or_callable(elem)
        if length is None:
            self.length = []
        elif isinstance(length, int):
            self.length = [length]
        else:
            self.length = list(length)

    def py_type(self) -> str:
        return f"ty.Array[{self.elem.py_type()}]"

    def py_type_guard(self) -> str:
        return f"tg.Array({self.elem.py_type_guard()}, {self.length})"

    def cc_type(self, _) -> str:
        return f"Array<{self.elem.cc_type(True)}>"

    def cc_as_relax(self) -> str:
        if isinstance(self.elem, _ScalarConstant):
            kind = "FromArrayToOpaque"
        elif isinstance(self.elem, IntPrimExpr):
            kind = "FromArrayToShapeExpr"
        elif isinstance(self.elem, (Tensor, Shape, AnyRelaxExpr)):
            kind = "FromArrayToTuple"
        else:
            raise NotImplementedError
        length = "{" + ", ".join(str(s) for s in self.length) + "}"
        elem = self.elem.cc_as_relax()
        return f"{kind}({elem}, {length})"


class Optional(_BaseType):
    """Optional type."""

    elem: _BaseType

    def __init__(
        self,
        elem: _BaseTypeOrCallable,
        *,
        default=NotSpecified,
        cc_as_relax=None,
    ):
        super().__init__(default=default, cc_as_relax=cc_as_relax)
        self.elem = _parse_base_type_or_callable(elem)

    def py_type(self) -> str:
        return f"ty.Optional[{self.elem.py_type()}]"

    def py_type_guard(self) -> str:
        return f"tg.Optional({self.elem.py_type_guard()})"

    def cc_type(self, _) -> str:
        return f"Optional<{self.elem.cc_type(True)}>"

    def cc_as_relax(self) -> str:
        if isinstance(self.elem, (_ScalarConstant, DType, Str, Axis, Axes)):
            elem = self.elem.cc_as_relax()
            return f"FromOptionalToOpaque({elem})"
        raise NotImplementedError


class Map(_BaseType):
    """A key-value dictionary."""

    key: _BaseType
    value: _BaseType

    def __init__(
        self,
        key: _BaseTypeOrCallable,
        value: _BaseTypeOrCallable,
        *,
        default=NotSpecified,
        cc_as_relax=None,
    ):
        super().__init__(default=default, cc_as_relax=cc_as_relax)
        self.key = _parse_base_type_or_callable(key)
        self.value = _parse_base_type_or_callable(value)

    def py_type(self) -> str:
        return f"ty.Map[{self.key.py_type()}, {self.value.py_type()}]"

    def py_type_guard(self) -> str:
        return f"tg.Map({self.key.py_type_guard()}, {self.value.py_type_guard()})"

    def cc_type(self, _) -> str:
        key = self.key.cc_type(True)
        value = self.value.cc_type(True)
        return f"Map<{key}, {value}>"

    def cc_as_relax(self) -> str:
        raise NotImplementedError


class Union(_BaseType):
    """Union of types."""

    elems: typing.List[_BaseType]

    def __init__(
        self,
        *es: _BaseTypeOrCallable,
        **default,
    ):
        super().__init__(**default)
        self.elems = [_parse_base_type_or_callable(e) for e in es]

    def py_type(self) -> str:
        return f"ty.Union[{', '.join(e.py_type() for e in self.elems)}]"

    def py_type_guard(self) -> str:
        return f"tg.Union({', '.join(e.py_type_guard() for e in self.elems)})"

    def cc_type(self, _) -> str:
        return "ObjectRef"

    def cc_as_relax(self) -> str:
        raise NotImplementedError


##### PrimValues #####


class _Scalar(_BaseType):
    """A base class for scalar types."""


class _ScalarConstant(_Scalar):
    """A base class for scalar constants."""


class Int(_ScalarConstant):
    """An integer scalar."""

    def py_type(self) -> str:
        return "ty.Int"

    def py_type_guard(self) -> str:
        return "tg.Int()"

    def cc_type(self, force_object: bool) -> str:
        return "IntImm" if force_object else "int64_t"

    def cc_as_relax(self) -> str:
        return "FromScalarConstant(DTypeAll)"


class Float(_ScalarConstant):
    """A float scalar."""

    def py_type(self) -> str:
        return "ty.Float"

    def py_type_guard(self) -> str:
        return "tg.Float()"

    def cc_type(self, force_object: bool) -> str:
        return "FloatImm" if force_object else "double"

    def cc_as_relax(self) -> str:
        return "FromScalarConstant(DTypeFloat)"


class Bool(_ScalarConstant):
    """A boolean scalar."""

    def py_type(self) -> str:
        return "ty.Bool"

    def py_type_guard(self) -> str:
        return "tg.Bool()"

    def cc_type(self, force_object: bool) -> str:
        return "Bool" if force_object else "bool"

    def cc_as_relax(self) -> str:
        return "FromScalarConstant(DTypeBool)"


class PrimExpr(_Scalar):
    """A TIR PrimExpr."""

    def py_type(self) -> str:
        return "ty.PrimExpr"

    def py_type_guard(self) -> str:
        return "tg.PrimExpr()"

    def cc_type(self, _: bool) -> str:
        return "PrimExpr"

    def cc_as_relax(self) -> str:
        return "FromPrimExpr(DTypeAll)"


class IntPrimExpr(PrimExpr):
    """A TIR PrimExpr that evaluates to an integer."""

    def py_type(self) -> str:
        return "ty.IntPrimExpr"

    def py_type_guard(self) -> str:
        return "tg.IntPrimExpr()"

    def cc_type(self, _) -> str:
        return "PrimExpr"

    def cc_as_relax(self) -> str:
        return "FromPrimExpr(DTypeInt)"


class FloatPrimExpr(PrimExpr):
    """A TIR PrimExpr that evaluates to a float."""

    def py_type(self) -> str:
        return "ty.FloatPrimExpr"

    def py_type_guard(self) -> str:
        return "tg.FloatPrimExpr()"

    def cc_type(self, _) -> str:
        return "PrimExpr"

    def cc_as_relax(self) -> str:
        return "FromPrimExpr(DTypeFloat)"


class BoolPrimExpr(PrimExpr):
    """A TIR PrimExpr that evaluates to a boolean."""

    def py_type(self) -> str:
        return "ty.BoolPrimExpr"

    def py_type_guard(self) -> str:
        return "tg.BoolPrimExpr()"

    def cc_type(self, _) -> str:
        return "PrimExpr"

    def cc_as_relax(self) -> str:
        return "FromPrimExpr(DTypeBool)"


class TIRVar(PrimExpr):
    """A TIR Var."""

    def py_type(self) -> str:
        return "ty.TIRVar"

    def py_type_guard(self) -> str:
        return "tg.TIRVar()"

    def cc_type(self, _) -> str:
        return "tir::Var"

    def cc_as_relax(self) -> str:
        return "FromTIRVar(DTypeAll)"


class IntTIRVar(TIRVar):
    """A TIR integer Var."""

    def py_type(self) -> str:
        return "ty.IntTIRVar"

    def py_type_guard(self) -> str:
        return "tg.IntTIRVar()"

    def cc_type(self, _) -> str:
        return "tir::Var"

    def cc_as_relax(self) -> str:
        return "FromTIRVar(DTypeInt)"


class FloatTIRVar(TIRVar):
    """A TIR float Var."""

    def py_type(self) -> str:
        return "ty.FloatTIRVar"

    def py_type_guard(self) -> str:
        return "tg.FloatTIRVar()"

    def cc_type(self, _) -> str:
        return "tir::Var"

    def cc_as_relax(self) -> str:
        return "FromTIRVar(DTypeFloat)"


class BoolTIRVar(TIRVar):
    """A TIR boolean Var."""

    def py_type(self) -> str:
        return "ty.BoolTIRVar"

    def py_type_guard(self) -> str:
        return "tg.BoolTIRVar()"

    def cc_type(self, _) -> str:
        return "tir::Var"

    def cc_as_relax(self) -> str:
        return "FromTIRVar(DTypeBool)"


##### Tensor #####


class Tensor(_BaseType):
    """A relax tensor."""

    ndim: typing.List[int]

    def __init__(
        self,
        *,
        ndim: typing.Union[None, int, typing.Sequence[int]] = None,
        default=NotSpecified,
        cc_as_relax=None,
    ):
        super().__init__(default=default, cc_as_relax=cc_as_relax)
        if ndim is None:
            self.ndim = []
        elif isinstance(ndim, int):
            self.ndim = [ndim]
        else:
            self.ndim = list(ndim)

    def py_type(self) -> str:
        return "ty.Tensor"

    def py_type_guard(self) -> str:
        return f"tg.Tensor({self.ndim})"

    def cc_type(self, _) -> str:
        return "relax::Expr"

    def _proxy_cc_as_relax(self, dtype: str) -> str:
        ndim = "{" + ", ".join(str(x) for x in self.ndim) + "}"
        return f"FromTensor({dtype}, {ndim})"

    def cc_as_relax(self) -> str:
        return self._proxy_cc_as_relax("DTypeAll")


class IntTensor(Tensor):
    """A relax integer tensor."""

    def py_type(self) -> str:
        return "ty.IntTensor"

    def py_type_guard(self) -> str:
        return f"tg.IntTensor({self.ndim})"

    def cc_type(self, _) -> str:
        return "relax::Expr"

    def cc_as_relax(self) -> str:
        return self._proxy_cc_as_relax("DTypeInt")


class FloatTensor(Tensor):
    """A relax float tensor."""

    def py_type(self) -> str:
        return "ty.FloatTensor"

    def py_type_guard(self) -> str:
        return f"tg.FloatTensor({self.ndim})"

    def cc_type(self, _) -> str:
        return "relax::Expr"

    def cc_as_relax(self) -> str:
        return self._proxy_cc_as_relax("DTypeFloat")


class BoolTensor(Tensor):
    """A relax boolean tensor."""

    def py_type(self) -> str:
        return "ty.BoolTensor"

    def py_type_guard(self) -> str:
        return f"tg.BoolTensor({self.ndim})"

    def cc_type(self, _) -> str:
        return "relax::Expr"

    def cc_as_relax(self) -> str:
        return self._proxy_cc_as_relax("DTypeBool")


##### Misc #####


class AnyRelaxExpr(_BaseType):
    """A relax expression that can have any type and sinfo."""

    def py_type(self) -> str:
        return "ty.AnyRelaxExpr"

    def py_type_guard(self) -> str:
        return "tg.AnyRelaxExpr()"

    def cc_type(self, _) -> str:
        return "relax::Expr"

    def cc_as_relax(self) -> str:
        return "FromAnyRelaxExpr()"


class TupleExpr(_BaseType):
    """A relax Tuple expression."""

    def py_type(self) -> str:
        return "ty.TupleExpr"

    def py_type_guard(self) -> str:
        return "tg.TupleExpr()"

    def cc_type(self, _) -> str:
        return "relax::Tuple"

    def cc_as_relax(self) -> str:
        return "FromTupleExpr()"


class Str(_BaseType):
    """A string literal."""

    def py_type(self) -> str:
        return "ty.Str"

    def py_type_guard(self) -> str:
        return "tg.Str()"

    def cc_type(self, _) -> str:
        return "String"

    def cc_as_relax(self) -> str:
        return "FromStr()"


class DType(_BaseType):
    """A dtype literal."""

    def py_type(self) -> str:
        return "ty.DType"

    def py_type_guard(self) -> str:
        return "tg.DType()"

    def cc_type(self, force_object: bool) -> str:
        return "DataTypeImm" if force_object else "runtime::DataType"

    def cc_as_relax(self) -> str:
        return "FromDType()"


class Shape(_BaseType):
    """A shape, which can be relax ShapeExpr, or a sequence of PrimExprs."""

    def py_type(self) -> str:
        return "ty.Shape"

    def py_type_guard(self) -> str:
        return "tg.Shape()"

    def cc_type(self, _) -> str:
        return "relax::Expr"

    def cc_as_relax(self) -> str:
        return "FromShape()"


class Axis(_BaseType):
    """An axis attached to an input tensor."""

    of: str
    is_insertion: bool
    normalize: bool

    def __init__(
        self,
        of: str,
        *,
        is_insertion: bool = False,
        normalize: bool = True,
        default=NotSpecified,
        cc_as_relax=None,
    ):
        super().__init__(default=default, cc_as_relax=cc_as_relax)
        self.of = of  # pylint: disable=invalid-name
        self.is_insertion = is_insertion
        self.normalize = normalize

    def py_type(self) -> str:
        return "ty.Axis"

    def py_type_guard(self) -> str:
        return f"tg.Axis({self.of}, {self.is_insertion}, {self.normalize})"

    def cc_type(self, force_object: bool) -> str:
        return "IntImm" if force_object else "int64_t"

    def cc_as_relax(self) -> str:
        of = self.of  # pylint: disable=invalid-name
        is_insertion = str(self.is_insertion).lower()
        normalize = str(self.normalize).lower()
        return f"FromAxis({of}, {is_insertion}, {normalize})"


class Axes(_BaseType):
    """A sequence of axes attached to an input tensor."""

    of: str
    is_insertion: bool
    normalize: bool

    def __init__(
        self,
        of: str,
        *,
        is_insertion: bool = False,
        normalize: bool = True,
        default=NotSpecified,
        cc_as_relax=None,
    ):
        super().__init__(default=default, cc_as_relax=cc_as_relax)
        self.of = of  # pylint: disable=invalid-name
        self.is_insertion = is_insertion
        self.normalize = normalize

    def py_type(self) -> str:
        return "ty.Axes"

    def py_type_guard(self) -> str:
        return f"tg.Axes({self.of}, {self.is_insertion}, {self.normalize})"

    def cc_type(self, _) -> str:
        return "Array<IntImm>"

    def cc_as_relax(self) -> str:
        of = self.of  # pylint: disable=invalid-name
        is_insertion = str(self.is_insertion).lower()
        normalize = str(self.normalize).lower()
        return f"FromAxes({of}, {is_insertion}, {normalize})"


class GlobalVar(_BaseType):
    """A GlobalVar in TVM IRModule."""

    def py_type(self) -> str:
        return "ty.GlobalVar"

    def py_type_guard(self) -> str:
        return "tg.GlobalVar()"

    def cc_type(self, _) -> str:
        return "tvm::GlobalVar"

    def cc_as_relax(self) -> str:
        return "FromGlobalVar()"


class ExternFunc(_BaseType):
    """External function"""

    def py_type(self) -> str:
        return "ty.ExternFunc"

    def py_type_guard(self) -> str:
        return "tg.ExternFunc()"

    def cc_type(self, _) -> str:
        return "relax::ExternFunc"

    def cc_as_relax(self) -> str:
        return "FromExternFunc()"


class IndexMap(_BaseType):
    """An IndexMap in TIR."""

    def py_type(self) -> str:
        return "ty.IndexMap"

    def py_type_guard(self) -> str:
        return "tg.IndexMap()"

    def cc_type(self, _) -> str:
        return "tir::IndexMap"

    def cc_as_relax(self) -> str:
        return "FromIndexMap()"


class StructInfo(_BaseType):
    """Relax StructInfo."""

    def py_type(self) -> str:
        return "ty.StructInfo"

    def py_type_guard(self) -> str:
        return "tg.StructInfo()"

    def cc_type(self, _) -> str:
        return "relax::StructInfo"

    def cc_as_relax(self) -> str:
        return "FromStructInfo()"
