# pylint: disable=too-few-public-methods, missing-docstring
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type, Union

from numpydoc.docscrape import NumpyDocString  # type: ignore

from . import ty


@dataclass
class ParamDef:
    name: str
    type: ty._BaseType
    doc: List[str]

    def py_type(self) -> str:
        """Return the python type annotation string."""
        return self.type.py_type()

    def py_type_guard(self) -> str:
        """Return the python type checking code."""
        return self.type.py_type_guard()

    def cc_type(self, force_object: bool) -> str:
        """Return the C++ type annotation string."""
        return self.type.cc_type(force_object)

    def cc_arg2relax(self, name: str) -> str:
        """Return the C++ code to convert the param to a relax expression."""
        _cc_arg2relax = self.type._cc_arg2relax  # pylint: disable=protected-access
        if _cc_arg2relax is not None:
            func = _cc_arg2relax(name)
        else:
            try:
                func = self.type.cc_arg2relax()
                if func == "nullptr":
                    func = name
                else:
                    func = f"{func}({name})"
            except NotImplementedError:
                raise NotImplementedError(
                    f"Cannot convert {self.py_type()} to relax expr on parameter {name}"
                ) from None
        return f"TVM_RELAX_OP_ARG_CHECK({func}, {name}, _args);"

    def cc_relax2te(self, index: int, name: str) -> str:
        """Return the C++ code to convert Relax param to a param that can be recognized by TE."""
        _cc_relax2te = self.type._cc_relax2te  # pylint: disable=protected-access
        if _cc_relax2te is not None:
            return _cc_relax2te(index, name)
        try:
            func = self.type.cc_relax2te().replace("{name}", name)
            return f"{func}(call->args[{index}])"
        except NotImplementedError:
            raise NotImplementedError(
                f"Cannot convert {self.py_type()} to relax expr on parameter {name}"
            ) from None


@dataclass
class OpSchema:  # pylint: disable=too-many-instance-attributes
    _unparsed: Optional[Type]

    name: str
    category: str
    creator_name: str
    ffi_name: str
    te_func: Optional[str]
    sinfo: Optional[str]
    sinfo_out_dtype: Optional[str]
    sinfo_fallback: Optional[str]
    legalization: Optional[str]
    legalization_fallback: Optional[str]
    legalization_name_hint: Optional[str]
    min_num_args: Optional[int]
    attrs: List[Tuple[str, str]]

    doc: List[str]
    params: List[ParamDef]
    ret: ParamDef

    def parse(self) -> None:
        op_def, self._unparsed = self._unparsed, None
        if op_def is None:
            return
        np_doc = NumpyDocString(op_def.__doc__ or "")
        self.doc = np_doc.get("Summary", [])  # TODO: this doesn't seem to include all doc
        self.params = []
        self.ret = None
        param_doc: Dict[str, List[str]] = {
            p.name: p.desc
            for p in np_doc.get(
                "Attributes",
                [],
            )
        }
        for name, ann in op_def.__dict__.items():
            if name.startswith("__"):
                continue
            param = ParamDef(
                name=name,
                type=ty._parse_base_type_or_callable(ann),  # pylint: disable=protected-access
                doc=param_doc.get(name, None),
            )
            if name == "ret":
                self.ret = param
            else:
                self.params.append(param)
        if self.ret is None:
            raise RuntimeError(f"Missing return type for {self.name}")

    def _te_params(self) -> Tuple[str, str]:
        names = []
        converts = []
        for i, param in enumerate(self.params):
            name = param.name
            cvt = param.cc_relax2te(i, name)
            names.append(name)
            converts.append(f"_RV {name} = {cvt};")
        return "{" + ", ".join(names) + "}", "\n".join(converts)

    def register_sinfo(self) -> str:
        if self.sinfo is None:
            return ""
        if isinstance(self.sinfo, str):
            assert self.sinfo_fallback is None
            return f'.set_attr<FInferStructInfo>("FInferStructInfo", {self.sinfo})'
        assert isinstance(self.sinfo, ty._NotSpecified)  # pylint: disable=protected-access
        if self.te_func is None:
            return ""
        names, converts = self._te_params()
        if self.sinfo_fallback is None:
            handle_non_derivable = "throw e;"
        else:
            handle_non_derivable = f"return {self.sinfo_fallback}(call, _bb);"
        if self.sinfo_out_dtype is None:
            out_dtype = ""
        else:
            out_dtype = f", {self.sinfo_out_dtype}"
        return f"""
.set_attr<FInferStructInfo>("FInferStructInfo",
[](const Call& call, const BlockBuilder& _bb) -> StructInfo {{
  using namespace relax::relax2te;
  using _RV = runtime::TVMRetValue;
  static const auto* _te = runtime::Registry::Get("{self.te_func}");
  ICHECK(_te != nullptr);
  TensorHandler _h;
  try {{
{converts}
    return _h.AddOutput(CallGlobalFunc(_te, {names}){out_dtype});
  }} catch (const NotDerivable& e) {{
    {handle_non_derivable}
  }}
}})"""

    def register_legalization(self) -> str:
        if self.legalization is None:
            return ""
        if isinstance(self.legalization, str):
            assert self.legalization_fallback is None
            return f'.set_attr<FLegalize>("FLegalize", {self.legalization})'
        assert isinstance(self.legalization, ty._NotSpecified)  # pylint: disable=protected-access
        if self.te_func is None:
            return ""
        names, converts = self._te_params()
        if self.legalization_fallback is None:
            handle_non_deducible = "return call;"
        else:
            handle_non_deducible = f"return {self.legalization_fallback}(call, _bb);"
        if self.sinfo_out_dtype is None:
            out_dtype = ""
        else:
            out_dtype = f", {self.sinfo_out_dtype}"
        return f"""
.set_attr<FLegalize>("FLegalize", [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {{
  using namespace relax::relax2te;
  using _RV = runtime::TVMRetValue;
  static const auto* _te = runtime::Registry::Get("{self.te_func}");
  ICHECK(_te != nullptr);
  TensorHandler _h;
  try {{
{converts}
    StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {names}){out_dtype});
    return _h.EmitTE(_bb, "{self.legalization_name_hint}", _sinfo);
  }} catch (const NotDerivable& e) {{
    {handle_non_deducible}
  }}
}})"""

    def register_attrs(self) -> str:
        return "\n".join(
            f'.set_attr<{attr_name}>("{attr_name}", {attr_value})'
            for attr_name, attr_value in self.attrs
        )


class Registry:

    _schemas: List[OpSchema]
    _singleton: "Registry"

    def __init__(self):
        self._schemas = []

    def add(self, schema: OpSchema) -> None:
        self._schemas.append(schema)

    def get_registered_ops(self) -> List[OpSchema]:
        for schema in self._schemas:
            try:
                schema.parse()
            except Exception as error:
                raise RuntimeError(f"Error parsing {schema.name}") from error
        result = list(self._schemas)
        result.sort(key=lambda schema: schema.category + "/" + schema.name)
        return result


Registry._singleton = Registry()  # pylint: disable=protected-access


def register_op(
    name: str,
    *,
    te_func: Optional[str] = None,
    category: Union[str, ty._NotSpecified] = ty.NotSpecified,
    creator_name: Union[str, ty._NotSpecified] = ty.NotSpecified,
    ffi_name: Union[str, ty._NotSpecified] = ty.NotSpecified,
    sinfo: Union[None, str, ty._NotSpecified] = ty.NotSpecified,
    sinfo_out_dtype: Optional[str] = None,
    sinfo_fallback: Optional[str] = None,
    legalization: Union[None, str, ty._NotSpecified] = ty.NotSpecified,
    legalization_fallback: Optional[str] = None,
    legalization_name_hint: Optional[str] = None,
    min_num_args: Optional[int] = None,
    attrs: Optional[List[Tuple[str, str]]] = None,
):
    if isinstance(creator_name, ty._NotSpecified):  # pylint: disable=protected-access
        creator_name = name.split(".")[-1]
    if isinstance(ffi_name, ty._NotSpecified):  # pylint: disable=protected-access
        ffi_name = "relax.op." + name
    if legalization_name_hint is None:
        legalization_name_hint = name.split(".")[-1]
    if attrs is None:
        attrs = []

    def decorator(op_def: Type) -> Type:
        nonlocal category
        if isinstance(category, ty._NotSpecified):  # pylint: disable=protected-access
            category = op_def.__module__.split(".")[-1]
        Registry._singleton.add(  # pylint: disable=protected-access
            OpSchema(
                _unparsed=op_def,
                name=name,
                category=category,
                creator_name=creator_name,
                ffi_name=ffi_name,
                te_func=te_func,
                sinfo=sinfo,
                sinfo_out_dtype=sinfo_out_dtype,
                sinfo_fallback=sinfo_fallback,
                legalization=legalization,
                legalization_fallback=legalization_fallback,
                legalization_name_hint=legalization_name_hint,
                min_num_args=min_num_args,
                attrs=attrs,
                doc=None,
                params=None,
                ret=None,
            )
        )
        return op_def

    return decorator


def get_registered_ops() -> List[OpSchema]:
    return Registry._singleton.get_registered_ops()  # pylint: disable=protected-access
