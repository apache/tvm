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
"""Type checking functionality"""
import collections
import collections.abc
import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import typing


def _is_none_type(type_: Any) -> bool:
    return type_ is None or type_ is type(None)


if hasattr(typing, "_GenericAlias"):
    # For python versions 3.7 onward, check the __origin__ attribute.

    class _Subtype:
        @staticmethod
        def _origin(type_: Any) -> Any:
            if isinstance(type_, typing._GenericAlias):  # type: ignore # pylint: disable=protected-access
                return type_.__origin__
            return None

        @staticmethod
        def list_(type_: Any) -> Any:
            if _Subtype._origin(type_) is list:
                (subtype,) = type_.__args__
                return [subtype]
            return None

        @staticmethod
        def dict_(type_: Any) -> Any:
            if _Subtype._origin(type_) is dict:
                (ktype, vtype) = type_.__args__
                return [ktype, vtype]
            return None

        @staticmethod
        def tuple_(type_: Any) -> Optional[List[type]]:
            if _Subtype._origin(type_) is tuple:
                subtypes = type_.__args__
                return subtypes
            return None

        @staticmethod
        def optional(type_: Any) -> Optional[List[type]]:
            if _Subtype._origin(type_) is Union:
                subtypes = type_.__args__
                if len(subtypes) == 2 and _is_none_type(subtypes[1]):
                    return [subtypes[0]]
            return None

        @staticmethod
        def union(type_: Any) -> Optional[List[type]]:
            if _Subtype._origin(type_) is Union:
                subtypes = type_.__args__
                if len(subtypes) != 2 or not _is_none_type(subtypes[1]):
                    return list(subtypes)
            return None

        @staticmethod
        def callable(type_: Any) -> Optional[List[type]]:
            if _Subtype._origin(type_) is collections.abc.Callable:
                subtypes = type_.__args__
                return subtypes
            return None

elif hasattr(typing, "_Union"):
    # For python 3.6 and below, check the __name__ attribute, or CallableMeta.

    class _Subtype:  # type: ignore
        @staticmethod
        def list_(type_: Any) -> Optional[List[type]]:
            if isinstance(type_, typing.GenericMeta):  # type: ignore # pylint: disable=no-member
                if type_.__name__ == "List":
                    (subtype,) = type_.__args__  # type: ignore # pylint: disable=no-member
                    return [subtype]
            return None

        @staticmethod
        def dict_(type_: Any) -> Optional[List[type]]:
            if isinstance(type_, typing.GenericMeta):  # type: ignore # pylint: disable=no-member
                if type_.__name__ == "Dict":
                    (ktype, vtype) = type_.__args__  # type: ignore # pylint: disable=no-member
                    return [ktype, vtype]
            return None

        @staticmethod
        def tuple_(type_: Any) -> Optional[List[type]]:
            if isinstance(type_, typing.GenericMeta):  # type: ignore # pylint: disable=no-member
                if type_.__name__ == "Tuple":
                    subtypes = type_.__args__  # type: ignore # pylint: disable=no-member
                    return subtypes
            return None

        @staticmethod
        def optional(type_: Any) -> Optional[List[type]]:
            if isinstance(type_, typing._Union):  # type: ignore # pylint: disable=no-member,protected-access
                subtypes = type_.__args__
                if len(subtypes) == 2 and _is_none_type(subtypes[1]):
                    return [subtypes[0]]
            return None

        @staticmethod
        def union(type_: Any) -> Optional[List[type]]:
            if isinstance(type_, typing._Union):  # type: ignore # pylint: disable=no-member,protected-access
                subtypes = type_.__args__
                if len(subtypes) != 2 or not _is_none_type(subtypes[1]):
                    return list(subtypes)
            return None

        @staticmethod
        def callable(type_: Any) -> Optional[List[type]]:
            if isinstance(type_, typing.CallableMeta):  # type: ignore # pylint: disable=no-member,protected-access
                subtypes = type_.__args__
                return subtypes
            return None


def _dispatcher(type_: Any) -> Tuple[str, List[type]]:
    if _is_none_type(type_):
        return "none", []

    subtype = _Subtype.list_(type_)
    if subtype is not None:
        return "list", subtype

    subtype = _Subtype.dict_(type_)
    if subtype is not None:
        return "dict", subtype

    subtype = _Subtype.tuple_(type_)
    if subtype is not None:
        return "tuple", subtype

    subtype = _Subtype.optional(type_)
    if subtype is not None:
        return "optional", subtype

    subtype = _Subtype.union(type_)
    if subtype is not None:
        return "union", subtype

    subtype = _Subtype.callable(type_)
    if subtype is not None:
        return "callable", subtype

    return "atomic", [type_]


def callable_str(subtypes):
    if subtypes:
        *arg_types, return_type = subtypes
        arg_str = ", ".join(_type2str(arg_type) for arg_type in arg_types)
        return_type_str = _type2str(return_type)
        return f"Callable[[{arg_str}], {return_type_str}]"
    else:
        return "Callable"


_TYPE2STR: Dict[Any, Callable] = {
    "none": lambda: "None",
    "atomic": lambda t: str(t.__name__),
    "callable": callable_str,
    "list": lambda t: f"List[{_type2str(t)}]",
    "dict": lambda k, v: f"Dict[{_type2str(k)}, {_type2str(v)}]",
    "tuple": lambda *t: f"Tuple[{', '.join([_type2str(x) for x in t])}]",
    "optional": lambda t: f"Optional[{_type2str(t)}]",
    "union": lambda *t: f"Union[{', '.join([_type2str(x) for x in t])}]",
}


def _type2str(type_: Any) -> str:
    key, subtypes = _dispatcher(type_)
    return _TYPE2STR[key](*subtypes)


def _val2type(value: Any):
    if isinstance(value, list):
        types = set(_val2type(x) for x in value)
        if len(types) == 1:
            return List[types.pop()]  # type: ignore

        return List[Union[tuple(types)]]  # type: ignore

    if isinstance(value, tuple):
        types = tuple(_val2type(x) for x in value)  # type: ignore
        return Tuple[types]

    return type(value)


def _type_check_err(x: Any, name: str, expected: Any) -> str:
    return (
        f'"{name}" has wrong type. '
        f'Expected "{_type2str(expected)}", '
        f'but gets: "{_type2str(_val2type(x))}"'
    )


def _type_check_vtable() -> Dict[str, Callable]:
    def _type_check_none(v: Any, name: str) -> Optional[str]:
        return None if v is None else _type_check_err(v, name, None)

    def _type_check_atomic(v: Any, name: str, type_: Any) -> Optional[str]:
        return None if isinstance(v, type_) else _type_check_err(v, name, type_)

    def _type_check_callable(v: Any, name: str, *_subtypes: Any) -> Optional[str]:
        # Current implementation only validates that the argument is
        # callable, and doesn't validate the arguments accepted by the
        # callable, if any.
        return None if callable(v) else _type_check_err(v, name, Callable)

    def _type_check_list(v: List[Any], name: str, type_: Any) -> Optional[str]:
        if not isinstance(v, (list, tuple)):
            return _type_check_err(v, name, list)
        for i, x in enumerate(v):
            error_msg = _type_check(x, f"{name}[{i}]", type_)
            if error_msg is not None:
                return error_msg
        return None

    def _type_check_dict(dict_obj: Dict[Any, Any], name: str, *types: Any) -> Optional[str]:
        ktype_, vtype_ = types
        if not isinstance(dict_obj, dict):
            return _type_check_err(dict_obj, name, dict)
        for k, v in dict_obj.items():
            error_msg = _type_check(k, f"{name}[{k}]", ktype_)
            if error_msg is not None:
                return error_msg
            error_msg = _type_check(v, f"{name}[{k}]", vtype_)
            if error_msg is not None:
                return error_msg
        return None

    def _type_check_tuple(v: Any, name: str, *types: Any) -> Optional[str]:
        if not isinstance(v, tuple):
            return _type_check_err(v, name, Tuple[types])
        if len(types) != len(v):
            return _type_check_err(v, name, Tuple[types])
        for i, (x, type_) in enumerate(zip(v, types)):
            error_msg = _type_check(x, f"{name}[{i}]", type_)
            if error_msg is not None:
                return error_msg
        return None

    def _type_check_optional(v: Any, name: str, type_: Any) -> Optional[str]:
        return None if v is None else _type_check(v, name, type_)

    def _type_check_union(v: Any, name: str, *types: Any) -> Optional[str]:
        for type_ in types:
            error_msg = _type_check(v, name, type_)
            if error_msg is None:
                return None
        return _type_check_err(v, name, Union[types])

    return {
        "none": _type_check_none,
        "atomic": _type_check_atomic,
        "callable": _type_check_callable,
        "list": _type_check_list,
        "dict": _type_check_dict,
        "tuple": _type_check_tuple,
        "optional": _type_check_optional,
        "union": _type_check_union,
    }


_TYPE_CHECK: Dict[Any, Callable] = _type_check_vtable()


def _type_check(v: Any, name: str, type_: Any) -> Optional[str]:
    key, subtypes = _dispatcher(type_)
    return _TYPE_CHECK[key](v, name, *subtypes)


FType = TypeVar("FType", bound=Callable[..., Any])


def type_checked(func: FType) -> FType:
    """Type check the input arguments of a function."""
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrap(*args, **kwargs):
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        for param in sig.parameters.values():
            if param.annotation != inspect.Signature.empty:
                error_msg = _type_check(
                    bound_args.arguments[param.name],
                    param.name,
                    param.annotation,
                )
                if error_msg is not None:
                    error_msg = f'In "{func.__qualname__}", {error_msg}'
                    raise TypeError(error_msg)
        return func(*args, **kwargs)

    return wrap  # type: ignore
