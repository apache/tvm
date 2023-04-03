# pylint: disable=too-few-public-methods, missing-docstring
from dataclasses import dataclass
from typing import Dict, List, Optional, Type, Union

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

    def cc_as_relax(self, name: str) -> str:
        """Return the C++ code to convert the param to a relax expression."""
        _cc_as_relax = self.type._cc_as_relax  # pylint: disable=protected-access
        if _cc_as_relax is not None:
            return _cc_as_relax(name)
        try:
            return self.type.cc_as_relax() + f"({name})"
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
    sinfo_inference: Optional[str]
    legalization: Optional[str]
    min_num_args: Optional[int]

    doc: List[str]
    params: List[ParamDef]
    ret: ParamDef

    def parse(self) -> None:
        op_def, self._unparsed = self._unparsed, None
        if op_def is None:
            return
        np_doc = NumpyDocString(op_def.__doc__ or "")
        self.doc = np_doc.get("Summary", [])
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
    sinfo_inference: Union[None, str, ty._NotSpecified] = ty.NotSpecified,
    legalization: Union[None, str, ty._NotSpecified] = ty.NotSpecified,
    min_num_args: Optional[int] = None,
):
    if isinstance(creator_name, ty._NotSpecified):  # pylint: disable=protected-access
        creator_name = name.split(".")[-1]
    if isinstance(ffi_name, ty._NotSpecified):  # pylint: disable=protected-access
        ffi_name = "relax.op." + name
    if te_func is not None:
        if isinstance(sinfo_inference, ty._NotSpecified):  # pylint: disable=protected-access
            sinfo_inference = f'InferStructInfoFromTE("{te_func}")'
        if isinstance(legalization, ty._NotSpecified):  # pylint: disable=protected-access
            legalization = f'LegalizeFromTE("{te_func}", "{creator_name}")'
    else:
        if isinstance(sinfo_inference, ty._NotSpecified):  # pylint: disable=protected-access
            sinfo_inference = None
        if isinstance(legalization, ty._NotSpecified):  # pylint: disable=protected-access
            legalization = None

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
                sinfo_inference=sinfo_inference,
                legalization=legalization,
                min_num_args=min_num_args,
                doc=None,
                params=None,
                ret=None,
            )
        )
        return op_def

    return decorator


def get_registered_ops() -> List[OpSchema]:
    return Registry._singleton.get_registered_ops()  # pylint: disable=protected-access
