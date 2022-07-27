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
"""Doc types for TVMScript Unified Printer"""

from typing import List, Dict, Tuple, Optional, Union, Sequence
from enum import IntEnum, unique

import tvm._ffi
import tvm.ir.container
from tvm.runtime import Object
from tvm.tir import FloatImm, IntImm

from . import _ffi_api


class Doc(Object):
    """Base class of all Docs"""


class ExprDoc(Object):
    """Base class of all expression Docs"""

    def attr(self, name: str) -> "AttrAccessDoc":
        """
        Create a doc that represents attribute access on self.

        Parameters
        ----------
        name : str
            The attribute name to access

        Returns
        -------
        doc : AttrAccessDoc
        """
        return _ffi_api.ExprDocAttr(self, name)  # type: ignore

    def call(self, *args: Tuple["ExprDoc"], **kwargs: Dict[str, "ExprDoc"]) -> "CallDoc":
        """
        Create a doc that represents function call, with self as callee.

        Parameters
        ----------
        *args : ExprDoc
            The positional arguments of the function call.
        **kwargs
            The keyword arguments of the function call.

        Returns
        -------
        doc : CallDoc
        """
        kwargs_keys = list(kwargs.keys())
        kwargs_values = list(kwargs.values())
        return _ffi_api.ExprDocCall(self, args, kwargs_keys, kwargs_values)  # type: ignore

    _IndexType = Union["ExprDoc", "SliceDoc"]

    def __getitem__(self, indices: Union[Tuple[_IndexType], _IndexType]) -> "IndexDoc":
        """
        Create a doc that represents index access on self.

        Parameters
        ----------
        indices : Union[Tuple[Union["ExprDoc", "SliceDoc"]], Union["ExprDoc", "SliceDoc"]]
            The indices to access

        Returns
        -------
        doc : IndexDoc
        """
        if not isinstance(indices, tuple):
            indices = (indices,)
        return _ffi_api.ExprDocIndex(self, indices)  # type: ignore

    def __iter__(self):
        """
        This is implemented to prevent confusing error message when trying to use ExprDoc
        as iterable. According to PEP-234, An object can be iterated over if it
        implements __iter__() or __getitem__(). If an object has only __getitem__
        but not __iter__, interpreter will iterate the object by calling
        __getitem__ with 0, 1, 2, ..., until an IndexError is raised.

        https://peps.python.org/pep-0234/#python-api-specification
        """
        raise RuntimeError(f"{self.__class__} cannot be used as iterable.")


@tvm._ffi.register_object("script.printer.LiteralDoc")
class LiteralDoc(ExprDoc):
    """Doc that represents literal value"""

    value: Union[str, IntImm, FloatImm, None]

    def __init__(self, value: Union[str, float, bool, int, None]):
        if value is None:
            self.__init_handle_by_constructor__(_ffi_api.LiteralDocNone)  # type: ignore
        elif isinstance(value, str):
            self.__init_handle_by_constructor__(_ffi_api.LiteralDocStr, value)  # type: ignore
        elif isinstance(value, float):
            self.__init_handle_by_constructor__(_ffi_api.LiteralDocFloat, value)  # type: ignore
        elif isinstance(value, bool):
            self.__init_handle_by_constructor__(_ffi_api.LiteralDocBoolean, value)  # type: ignore
        elif isinstance(value, int):
            self.__init_handle_by_constructor__(_ffi_api.LiteralDocInt, value)  # type: ignore
        else:
            raise TypeError(f"Unsupported type {type(value)} for LiteralDoc")


@tvm._ffi.register_object("script.printer.IdDoc")
class IdDoc(ExprDoc):
    """Doc that represents identifier"""

    name: str

    def __init__(self, name: str):
        self.__init_handle_by_constructor__(_ffi_api.IdDoc, name)  # type: ignore


@tvm._ffi.register_object("script.printer.AttrAccessDoc")
class AttrAccessDoc(ExprDoc):
    """Doc that represents attribute access on an expression"""

    value: ExprDoc
    name: str

    def __init__(self, value: ExprDoc, name: str):
        self.__init_handle_by_constructor__(_ffi_api.AttrAccessDoc, value, name)  # type: ignore


@tvm._ffi.register_object("script.printer.IndexDoc")
class IndexDoc(ExprDoc):
    """Doc that represents index access on an expression"""

    value: ExprDoc
    indices: Sequence[Union[ExprDoc, "SliceDoc"]]

    def __init__(self, value: ExprDoc, indices: List[Union[ExprDoc, "SliceDoc"]]):
        self.__init_handle_by_constructor__(_ffi_api.IndexDoc, value, indices)  # type: ignore


@tvm._ffi.register_object("script.printer.CallDoc")
class CallDoc(ExprDoc):
    """Doc that represents function call"""

    callee: ExprDoc
    args: Sequence[ExprDoc]
    kwargs_keys: Sequence[str]
    kwargs_values: Sequence[ExprDoc]

    def __init__(self, callee: ExprDoc, *args: Tuple[ExprDoc], **kwargs: Dict[str, ExprDoc]):
        kwargs_keys = list(kwargs.keys())
        kwargs_values = list(kwargs.values())
        self.__init_handle_by_constructor__(
            _ffi_api.CallDoc, callee, args, kwargs_keys, kwargs_values  # type: ignore
        )


@unique
class OperationKind(IntEnum):
    """
    This enum represents the kind of operation (operator) in OpeartionDoc

    It's mirrored from OperationDocNode::Kind at include/tvm/script/printer/doc.h
    """

    # The name convention follows https://docs.python.org/3/library/ast.html
    # pylint: disable=invalid-name

    _UnaryStart = 0
    USub = 1
    Invert = 2
    _UnaryEnd = 3

    _BinaryStart = 4
    Add = 5
    Sub = 6
    Mult = 7
    Div = 8
    FloorDiv = 9
    Mod = 10
    Pow = 11
    LShift = 12
    RShift = 13
    BitAnd = 14
    BitOr = 15
    BitXor = 16
    Lt = 17
    LtE = 18
    Eq = 19
    NotEq = 20
    Gt = 21
    GtE = 22
    _BinaryEnd = 23

    _SpecialStart = 24
    IfThenElse = 25
    _SpecialEnd = 26

    # pylint: enable=invalid-name


@tvm._ffi.register_object("script.printer.OperationDoc")
class OperationDoc(ExprDoc):
    """
    Doc that represents operation

    It can be unary, binary and other special operators (for example, the
    if-then-else expression).
    """

    kind: OperationKind
    operands: Sequence[ExprDoc]

    def __init__(self, kind: OperationKind, operands: List[ExprDoc]):
        self.__init_handle_by_constructor__(_ffi_api.OperationDoc, kind, operands)  # type: ignore


@tvm._ffi.register_object("script.printer.LambdaDoc")
class LambdaDoc(ExprDoc):
    """Doc that represents lambda function"""

    args: Sequence[IdDoc]
    body: ExprDoc

    def __init__(self, args: List[IdDoc], body: ExprDoc):
        self.__init_handle_by_constructor__(_ffi_api.LambdaDoc, args, body)  # type: ignore


@tvm._ffi.register_object("script.printer.TupleDoc")
class TupleDoc(ExprDoc):
    """Doc that represents tuple literal"""

    elements: Sequence[ExprDoc]

    def __init__(self, elements: List[ExprDoc]):
        self.__init_handle_by_constructor__(_ffi_api.TupleDoc, elements)  # type: ignore


@tvm._ffi.register_object("script.printer.ListDoc")
class ListDoc(ExprDoc):
    """Doc that represents list literal"""

    elements: Sequence[ExprDoc]

    def __init__(self, elements: List[ExprDoc]):
        self.__init_handle_by_constructor__(_ffi_api.ListDoc, elements)  # type: ignore


@tvm._ffi.register_object("script.printer.DictDoc")
class DictDoc(ExprDoc):
    """Doc that represents dict literal"""

    keys: Sequence[ExprDoc]
    values: Sequence[ExprDoc]

    def __init__(self, content: Dict[ExprDoc, ExprDoc]):
        keys = list(content.keys())
        values = list(content.values())
        self.__init_handle_by_constructor__(_ffi_api.DictDoc, keys, values)  # type: ignore


@tvm._ffi.register_object("script.printer.SliceDoc")
class SliceDoc(ExprDoc):
    """
    Doc that represents slice in Index expression

    This doc can only appear in `IndexDoc.indices`.
    """

    start: Optional[ExprDoc]
    stop: Optional[ExprDoc]
    step: Optional[ExprDoc]

    def __init__(
        self,
        start: Optional[ExprDoc] = None,
        stop: Optional[ExprDoc] = None,
        step: Optional[ExprDoc] = None,
    ):
        self.__init_handle_by_constructor__(_ffi_api.SliceDoc, start, stop, step)  # type: ignore
