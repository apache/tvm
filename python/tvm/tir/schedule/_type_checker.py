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
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import typing


def _is_none_type(type_: Any) -> bool:
    return type_ is None or type_ is type(None)


def _list_subtype(type_: Any) -> Optional[List[type]]:
    if isinstance(type_, typing._GenericAlias) and type_.__origin__ is list:  # type: ignore # pylint: disable=protected-access
        (subtype,) = type_.__args__
        return [subtype]
    return None


def _optional_subtype(type_: Any) -> Optional[List[type]]:
    if isinstance(type_, typing._GenericAlias) and type_.__origin__ is Union:  # type: ignore # pylint: disable=protected-access
        subtypes = type_.__args__
        if len(subtypes) == 2 and _is_none_type(subtypes[1]):
            return [subtypes[0]]
    return None


def _union_subtype(type_: Any) -> Optional[List[type]]:
    if isinstance(type_, typing._GenericAlias) and type_.__origin__ is Union:  # type: ignore # pylint: disable=protected-access
        subtypes = type_.__args__
        if len(subtypes) != 2 or not _is_none_type(subtypes[1]):
            return list(subtypes)
    return None


def _dispatcher(type_: Any) -> Tuple[str, List[type]]:
    if _is_none_type(type_):
        return "none", []

    subtype = _list_subtype(type_)
    if subtype is not None:
        return "list", subtype

    subtype = _optional_subtype(type_)
    if subtype is not None:
        return "optional", subtype

    subtype = _union_subtype(type_)
    if subtype is not None:
        return "union", subtype

    return "atomic", [type_]


_TYPE2STR: Dict[Any, Callable] = {
    "none": lambda: "None",
    "atomic": lambda t: str(t.__name__),
    "list": lambda t: f"List[{_type2str(t)}]",
    "optional": lambda t: f"Optional[{_type2str(t)}]",
    "union": lambda *t: f"Union[{', '.join([_type2str(x) for x in t])}]",
}


def _type2str(type_: Any) -> str:
    key, subtypes = _dispatcher(type_)
    return _TYPE2STR[key](*subtypes)


def _type_check_err(x: Any, name: str, expected: Any) -> str:
    return (
        f'"{name}" has wrong type. '
        f'Expected "{_type2str(expected)}", '
        f'but gets: "{_type2str(type(x))}"'
    )


def _type_check_vtable() -> Dict[str, Callable]:
    def _type_check_none(v: Any, name: str) -> Optional[str]:
        return None if v is None else _type_check_err(v, name, None)

    def _type_check_atomic(v: Any, name: str, type_: Any) -> Optional[str]:
        return None if isinstance(v, type_) else _type_check_err(v, name, type_)

    def _type_check_list(v: List[Any], name: str, type_: Any) -> Optional[str]:
        if not isinstance(v, (list, tuple)):
            return _type_check_err(v, name, list)
        for i, x in enumerate(v):
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
        return _type_check_err(v, name, types)

    return {
        "none": _type_check_none,
        "atomic": _type_check_atomic,
        "list": _type_check_list,
        "optional": _type_check_optional,
        "union": _type_check_union,
    }


_TYPE_CHECK: Dict[Any, Callable] = _type_check_vtable()


def _type_check(v: Any, name: str, type_: Any) -> Optional[str]:
    key, subtypes = _dispatcher(type_)
    return _TYPE_CHECK[key](v, name, *subtypes)


def type_checked(func: Callable) -> Callable:
    """Type check the input arguments of a function."""
    sig = inspect.signature(func)

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
                    raise TypeError(error_msg)
        return func(*args, **kwargs)

    return wrap
