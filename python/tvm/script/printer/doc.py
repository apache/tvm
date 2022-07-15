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

from typing import List, Dict, Tuple, Optional, Union
from enum import IntEnum, auto, unique

import tvm._ffi
import tvm.ir.container
from tvm.runtime import Object
from tvm.tir import FloatImm, IntImm

from . import _ffi_api


class Doc(Object):
    """Base class of all Docs"""


class ExprDoc(Object):
    """Base class of all expression Docs"""

    def attr_access(self, attr: str) -> "AttrAccessDoc":
        """
        Create a doc that represents attribute access on self.

        Parameters
        ----------
        attr : str
            The attribute name to access

        Returns
        -------
        doc : AttrAccessDoc
        """
        return _ffi_api.ExprDocAttr(self, attr)  # type: ignore

    def index_access(self, indices: List[Union["ExprDoc", "SliceDoc"]]) -> "IndexDoc":
        """
        Create a doc that represents index access on self.

        Parameters
        ----------
        indices : List[Union["ExprDoc", "SliceDoc"]]
            The indices to access

        Returns
        -------
        doc : IndexDoc
        """
        return _ffi_api.ExprDocIndex(self, indices)  # type: ignore

    def call_with(self, *args: Tuple["ExprDoc"], **kwargs: Dict[str, "ExprDoc"]) -> "CallDoc":
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
    attr: str

    def __init__(self, value: ExprDoc, attr: str):
        self.__init_handle_by_constructor__(_ffi_api.AttrAccessDoc, value, attr)  # type: ignore


@tvm._ffi.register_object("script.printer.IndexDoc")
class IndexDoc(ExprDoc):
    """Doc that represents index access on an expression"""

    value: ExprDoc
    indices: tvm.ir.container.Array  # actual type:  List[Union[ExprDoc, "SliceDoc"]]

    def __init__(self, value: ExprDoc, indices: List[Union[ExprDoc, "SliceDoc"]]):
        self.__init_handle_by_constructor__(_ffi_api.IndexDoc, value, indices)  # type: ignore


@tvm._ffi.register_object("script.printer.CallDoc")
class CallDoc(ExprDoc):
    """Doc that represents function call"""

    callee: ExprDoc
    args: tvm.ir.container.Array  # actual type: List[ExprDoc]
    kwargs_keys: tvm.ir.container.Array  # actual type: List[str]
    kwargs_values: tvm.ir.container.Array  # actual type: List[ExprDoc]

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
    USub = auto()
    Invert = auto()
    _UnaryEnd = auto()

    _BinaryStart = auto()
    Add = auto()
    Sub = auto()
    Mult = auto()
    Div = auto()
    FloorDiv = auto()
    Mod = auto()
    Pow = auto()
    LShift = auto()
    RShift = auto()
    BitAnd = auto()
    BitOr = auto()
    BitXor = auto()
    Lt = auto()
    LtE = auto()
    Eq = auto()
    NotEq = auto()
    Gt = auto()
    GtE = auto()
    _BinaryEnd = auto()

    _SpecialStart = auto()
    IfThenElse = auto()
    _SpecialEnd = auto()

    # pylint: enable=invalid-name


@tvm._ffi.register_object("script.printer.OperationDoc")
class OperationDoc(ExprDoc):
    """
    Doc that represents operation

    It can be unary, binary and other special operators (for example, the
    if-then-else expression).
    """

    kind: OperationKind
    operands: tvm.ir.container.Array  # actual type: List[ExprDoc]

    def __init__(self, kind: OperationKind, operands: List[ExprDoc]):
        self.__init_handle_by_constructor__(_ffi_api.OperationDoc, kind, operands)  # type: ignore


@tvm._ffi.register_object("script.printer.LambdaDoc")
class LambdaDoc(ExprDoc):
    """Doc that represents lambda function"""

    args: tvm.ir.container.Array  # actual type: List[IdDoc]
    body: ExprDoc

    def __init__(self, args: List[IdDoc], body: ExprDoc):
        self.__init_handle_by_constructor__(_ffi_api.LambdaDoc, args, body)  # type: ignore


@tvm._ffi.register_object("script.printer.TupleDoc")
class TupleDoc(ExprDoc):
    """Doc that represents tuple literal"""

    elements: tvm.ir.container.Array  # actual type: List[ExprDoc]

    def __init__(self, elements: List[ExprDoc]):
        self.__init_handle_by_constructor__(_ffi_api.TupleDoc, elements)  # type: ignore


@tvm._ffi.register_object("script.printer.ListDoc")
class ListDoc(ExprDoc):
    """Doc that represents list literal"""

    elements: tvm.ir.container.Array  # actual type: List[ExprDoc]

    def __init__(self, elements: List[ExprDoc]):
        self.__init_handle_by_constructor__(_ffi_api.ListDoc, elements)  # type: ignore


@tvm._ffi.register_object("script.printer.DictDoc")
class DictDoc(ExprDoc):
    """Doc that represents dict literal"""

    keys: tvm.ir.container.Array  # actual type: List[ExprDoc]
    values: tvm.ir.container.Array  # actual type: List[ExprDoc]

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

    def __init__(self, start: Optional[ExprDoc] = None, stop: Optional[ExprDoc] = None):
        self.__init_handle_by_constructor__(_ffi_api.SliceDoc, start, stop)  # type: ignore
