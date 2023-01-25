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

from enum import IntEnum, unique
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tvm._ffi import register_object
from tvm.runtime import Object, ObjectPath
from tvm.tir import FloatImm, IntImm

from . import _ffi_api


class Doc(Object):
    """Base class of all Docs"""

    @property
    def source_paths(self) -> Sequence[ObjectPath]:
        """
        The list of object paths of the source IR node.

        This is used to trace back to the IR node position where
        this Doc is generated, in order to position the diagnostic
        message.
        """
        return self.__getattr__("source_paths")  # pylint: disable=unnecessary-dunder-call

    @source_paths.setter
    def source_paths(self, value):
        return _ffi_api.DocSetSourcePaths(self, value)  # type: ignore # pylint: disable=no-member


class ExprDoc(Doc):
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
        return _ffi_api.ExprDocAttr(self, name)  # type: ignore # pylint: disable=no-member

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
        return _ffi_api.ExprDocCall(self, args, kwargs_keys, kwargs_values)  # type: ignore # pylint: disable=no-member

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
        return _ffi_api.ExprDocIndex(self, indices)  # type: ignore # pylint: disable=no-member

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


class StmtDoc(Doc):
    """Base class of statement doc"""

    @property
    def comment(self) -> Optional[str]:
        """
        The comment of this doc.

        The actual position of the comment depends on the type of Doc
        and also the DocPrinter implementation. It could be on the same
        line as the statement, or the line above, or inside the statement
        if it spans over multiple lines.
        """
        # It has to call the dunder method to avoid infinite recursion
        return self.__getattr__("comment")  # pylint: disable=unnecessary-dunder-call

    @comment.setter
    def comment(self, value):
        return _ffi_api.StmtDocSetComment(self, value)  # type: ignore # pylint: disable=no-member


@register_object("script.printer.StmtBlockDoc")
class StmtBlockDoc(Doc):
    """The container doc that holds a list of StmtDoc.

    Note: `StmtBlockDoc` is never used in the IR, but a temporary container that allows holding a
    list of StmtDoc.
    """

    stmts: Sequence[StmtDoc]

    def __init__(self, stmts: List[StmtDoc]):
        self.__init_handle_by_constructor__(_ffi_api.StmtBlockDoc, stmts)  # type: ignore # pylint: disable=no-member


@register_object("script.printer.LiteralDoc")
class LiteralDoc(ExprDoc):
    """Doc that represents literal value"""

    value: Union[str, IntImm, FloatImm, None]

    def __init__(
        self,
        value: Union[str, float, bool, int, None],
        path: Optional[ObjectPath] = None,
    ):
        if value is None:
            self.__init_handle_by_constructor__(_ffi_api.LiteralDocNone, path)  # type: ignore # pylint: disable=no-member
        elif isinstance(value, str):
            self.__init_handle_by_constructor__(
                _ffi_api.LiteralDocStr,  # type: ignore # pylint: disable=no-member
                value,
                path,
            )
        elif isinstance(value, float):
            self.__init_handle_by_constructor__(
                _ffi_api.LiteralDocFloat,  # type: ignore # pylint: disable=no-member
                value,
                path,
            )
        elif isinstance(value, bool):
            self.__init_handle_by_constructor__(
                _ffi_api.LiteralDocBoolean,  # type: ignore # pylint: disable=no-member
                value,
                path,
            )
        elif isinstance(value, int):
            self.__init_handle_by_constructor__(
                _ffi_api.LiteralDocInt,  # type: ignore # pylint: disable=no-member
                value,
                path,
            )
        else:
            raise TypeError(f"Unsupported type {type(value)} for LiteralDoc")


@register_object("script.printer.IdDoc")
class IdDoc(ExprDoc):
    """Doc that represents identifier"""

    name: str

    def __init__(self, name: str):
        self.__init_handle_by_constructor__(_ffi_api.IdDoc, name)  # type: ignore # pylint: disable=no-member


@register_object("script.printer.AttrAccessDoc")
class AttrAccessDoc(ExprDoc):
    """Doc that represents attribute access on an expression"""

    value: ExprDoc
    name: str

    def __init__(self, value: ExprDoc, name: str):
        self.__init_handle_by_constructor__(_ffi_api.AttrAccessDoc, value, name)  # type: ignore # pylint: disable=no-member


@register_object("script.printer.IndexDoc")
class IndexDoc(ExprDoc):
    """Doc that represents index access on an expression"""

    value: ExprDoc
    indices: Sequence[Union[ExprDoc, "SliceDoc"]]

    def __init__(self, value: ExprDoc, indices: List[Union[ExprDoc, "SliceDoc"]]):
        self.__init_handle_by_constructor__(_ffi_api.IndexDoc, value, indices)  # type: ignore # pylint: disable=no-member


@register_object("script.printer.CallDoc")
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
            _ffi_api.CallDoc,  # type: ignore # pylint: disable=no-member
            callee,
            args,
            kwargs_keys,
            kwargs_values,
        )


@unique
class OperationKind(IntEnum):
    """
    This enum represents the kind of operation (operator) in OperationDoc

    It's mirrored from OperationDocNode::Kind at include/tvm/script/printer/doc.h
    """

    # The name convention follows https://docs.python.org/3/library/ast.html
    # pylint: disable=invalid-name

    _UnaryStart = 0
    USub = 1
    Invert = 2
    Not = 3
    _UnaryEnd = 4

    _BinaryStart = 5
    Add = 6
    Sub = 7
    Mult = 8
    Div = 9
    FloorDiv = 10
    Mod = 11
    Pow = 12
    LShift = 13
    RShift = 14
    BitAnd = 15
    BitOr = 16
    BitXor = 17
    Lt = 18
    LtE = 19
    Eq = 20
    NotEq = 21
    Gt = 22
    GtE = 23
    And = 24
    Or = 25
    _BinaryEnd = 26

    _SpecialStart = 27
    IfThenElse = 28
    _SpecialEnd = 29

    # pylint: enable=invalid-name


@register_object("script.printer.OperationDoc")
class OperationDoc(ExprDoc):
    """
    Doc that represents operation

    It can be unary, binary and other special operators (for example, the
    if-then-else expression).
    """

    kind: OperationKind
    operands: Sequence[ExprDoc]

    def __init__(self, kind: OperationKind, operands: List[ExprDoc]):
        self.__init_handle_by_constructor__(_ffi_api.OperationDoc, kind, operands)  # type: ignore # pylint: disable=no-member


@register_object("script.printer.LambdaDoc")
class LambdaDoc(ExprDoc):
    """Doc that represents lambda function"""

    args: Sequence[IdDoc]
    body: ExprDoc

    def __init__(self, args: List[IdDoc], body: ExprDoc):
        self.__init_handle_by_constructor__(_ffi_api.LambdaDoc, args, body)  # type: ignore # pylint: disable=no-member


@register_object("script.printer.TupleDoc")
class TupleDoc(ExprDoc):
    """Doc that represents tuple literal"""

    elements: Sequence[ExprDoc]

    def __init__(self, elements: List[ExprDoc]):
        self.__init_handle_by_constructor__(_ffi_api.TupleDoc, elements)  # type: ignore # pylint: disable=no-member


@register_object("script.printer.ListDoc")
class ListDoc(ExprDoc):
    """Doc that represents list literal"""

    elements: Sequence[ExprDoc]

    def __init__(self, elements: List[ExprDoc]):
        self.__init_handle_by_constructor__(_ffi_api.ListDoc, elements)  # type: ignore # pylint: disable=no-member


@register_object("script.printer.DictDoc")
class DictDoc(ExprDoc):
    """Doc that represents dict literal"""

    keys: Sequence[ExprDoc]
    values: Sequence[ExprDoc]

    def __init__(self, content: Dict[ExprDoc, ExprDoc]):
        keys = list(content.keys())
        values = list(content.values())
        self.__init_handle_by_constructor__(_ffi_api.DictDoc, keys, values)  # type: ignore # pylint: disable=no-member


@register_object("script.printer.SliceDoc")
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
        self.__init_handle_by_constructor__(_ffi_api.SliceDoc, start, stop, step)  # type: ignore # pylint: disable=no-member


@register_object("script.printer.AssignDoc")
class AssignDoc(StmtDoc):
    """Doc that represents assign statement."""

    lhs: ExprDoc
    rhs: Optional[ExprDoc]
    annotation: Optional[ExprDoc]

    def __init__(self, lhs: ExprDoc, rhs: Optional[ExprDoc], annotation: Optional[ExprDoc] = None):
        self.__init_handle_by_constructor__(
            _ffi_api.AssignDoc,  # type: ignore # pylint: disable=no-member
            lhs,
            rhs,
            annotation,
        )


@register_object("script.printer.IfDoc")
class IfDoc(StmtDoc):
    """Doc that represent if-then-else statement."""

    predicate: ExprDoc
    then_branch: Sequence[StmtDoc]
    else_branch: Sequence[StmtDoc]

    def __init__(self, predicate: ExprDoc, then_branch: List[StmtDoc], else_branch: List[StmtDoc]):
        self.__init_handle_by_constructor__(
            _ffi_api.IfDoc,  # type: ignore # pylint: disable=no-member
            predicate,
            then_branch,
            else_branch,
        )


@register_object("script.printer.WhileDoc")
class WhileDoc(StmtDoc):
    """Doc that represents while statement."""

    predicate: ExprDoc
    body: Sequence[StmtDoc]

    def __init__(self, predicate: ExprDoc, body: List[StmtDoc]):
        self.__init_handle_by_constructor__(_ffi_api.WhileDoc, predicate, body)  # type: ignore # pylint: disable=no-member


@register_object("script.printer.ForDoc")
class ForDoc(StmtDoc):
    """Doc that represents for statement."""

    lhs: ExprDoc
    rhs: ExprDoc
    body: Sequence[StmtDoc]

    def __init__(self, lhs: ExprDoc, rhs: ExprDoc, body: List[StmtDoc]):
        self.__init_handle_by_constructor__(_ffi_api.ForDoc, lhs, rhs, body)  # type: ignore # pylint: disable=no-member


@register_object("script.printer.ScopeDoc")
class ScopeDoc(StmtDoc):
    """
    Doc that represents special scopes.

    Specifically, this means the with statement in Python:

    with <rhs> as <lhs>:
        <body...>
    """

    lhs: Optional[ExprDoc]
    rhs: ExprDoc
    body: Sequence[StmtDoc]

    def __init__(self, lhs: Optional[ExprDoc], rhs: ExprDoc, body: List[StmtDoc]):
        self.__init_handle_by_constructor__(_ffi_api.ScopeDoc, lhs, rhs, body)  # type: ignore # pylint: disable=no-member


@register_object("script.printer.ExprStmtDoc")
class ExprStmtDoc(StmtDoc):
    """Doc that represents an expression as statement."""

    expr: ExprDoc

    def __init__(self, expr: ExprDoc):
        self.__init_handle_by_constructor__(_ffi_api.ExprStmtDoc, expr)  # type: ignore # pylint: disable=no-member


@register_object("script.printer.AssertDoc")
class AssertDoc(StmtDoc):
    """Doc that represents assert statement."""

    test: ExprDoc
    msg: Optional[ExprDoc]

    def __init__(self, test: ExprDoc, msg: Optional[ExprDoc] = None):
        self.__init_handle_by_constructor__(_ffi_api.AssertDoc, test, msg)  # type: ignore # pylint: disable=no-member


@register_object("script.printer.ReturnDoc")
class ReturnDoc(StmtDoc):
    """Doc that represents return statement."""

    value: ExprDoc

    def __init__(self, value: ExprDoc):
        self.__init_handle_by_constructor__(_ffi_api.ReturnDoc, value)  # type: ignore # pylint: disable=no-member


@register_object("script.printer.FunctionDoc")
class FunctionDoc(StmtDoc):
    """Doc that represents function definition."""

    name: IdDoc
    args: Sequence[AssignDoc]
    decorators: Sequence[ExprDoc]
    return_type: Optional[ExprDoc]
    body: Sequence[StmtDoc]

    def __init__(
        self,
        name: IdDoc,
        args: List[AssignDoc],
        decorators: List[ExprDoc],
        return_type: Optional[ExprDoc],
        body: List[StmtDoc],
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.FunctionDoc,  # type: ignore # pylint: disable=no-member
            name,
            args,
            decorators,
            return_type,
            body,
        )


@register_object("script.printer.ClassDoc")
class ClassDoc(StmtDoc):
    """Doc that represents class definition."""

    name: IdDoc
    decorators: Sequence[ExprDoc]
    body: Sequence[StmtDoc]

    def __init__(self, name: IdDoc, decorators: List[ExprDoc], body: List[StmtDoc]):
        self.__init_handle_by_constructor__(
            _ffi_api.ClassDoc,  # type: ignore # pylint: disable=no-member
            name,
            decorators,
            body,
        )


@register_object("script.printer.CommentDoc")
class CommentDoc(StmtDoc):
    """Doc that represents comment."""

    def __init__(self, comment: str):
        self.__init_handle_by_constructor__(
            _ffi_api.CommentDoc, comment  # type: ignore # pylint: disable=no-member
        )


@register_object("script.printer.DocStringDoc")
class DocStringDoc(StmtDoc):
    """Doc that represents docstring."""

    def __init__(self, docs: str):
        self.__init_handle_by_constructor__(
            _ffi_api.DocStringDoc, docs  # type: ignore # pylint: disable=no-member
        )
