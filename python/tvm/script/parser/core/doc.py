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
"""TVM Script Parser doc AST"""

import ast
import inspect
import typing
from collections import defaultdict

from . import doc_core as doc
from .doc_core import *  # pylint: disable=unused-import,wildcard-import,redefined-builtin,W0614

FnToDoc = typing.Callable[[ast.AST], doc.AST]
FnFromDoc = typing.Callable[[doc.AST], ast.AST]


class Entry:
    """Mapping entry between python AST node type str and doc AST.

    Parameters
    ----------
    to_doc : typing.Optional[FnToDoc]
        The callable methods for converting python AST node to doc AST.

    from_doc : typing.Optional[FnFromDoc]
        The callable methods for converting doc AST to python AST node.
    """

    to_doc: typing.Optional[FnToDoc]
    from_doc: typing.Optional[FnFromDoc]

    def __init__(self):
        self.to_doc = None
        self.from_doc = None


class Registry:
    """Registration map from python AST node type str to methods of conversion
    between python AST node and doc AST node.

    Parameters
    ----------
    _inst : typing.Optional["Registry"]
        The instance of Registry.

    table : typing.Dict[str, Entry]
        The registration map from python AST node type str to methods of conversion
        between python AST node and doc AST node.
    """

    _inst: typing.Optional["Registry"] = None
    table: typing.Dict[str, Entry]

    def __init__(self):
        self.table = defaultdict(Entry)


def register_to_doc(name: str):
    """Register the to_doc method for python AST node type.

    Parameters
    ----------
    name : str
        The type of python AST node.

    Returns
    -------
    f : Callable[[FnToDoc], None]
        The function of registering the to_doc method for python AST node type.
    """

    def f(to_doc: FnToDoc):  # pylint: disable=redefined-outer-name
        reg = Registry._inst  # pylint: disable=protected-access
        reg.table[name].to_doc = to_doc

    return f


def register_from_doc(name: str):
    """Register the from_doc method for python AST node type.

    Parameters
    ----------
    name : str
        The type of python AST node.

    Returns
    -------
    f : Callable[[FnFromDoc], None]
        The function of registering the from_doc method for python AST node type.
    """

    def f(to_doc: FnFromDoc):  # pylint: disable=redefined-outer-name
        reg = Registry._inst  # pylint: disable=protected-access
        reg.table[name].from_doc = to_doc

    return f


def _is_atomic_type(node):
    return (
        node is None
        or node in [..., True, False]
        or isinstance(
            node,
            (
                int,
                float,
                str,
                bool,
                bytes,
                complex,
            ),
        )
    )


def _get_registry_entry(cls_name, attr):
    cls_name = cls_name.split(".")[-1]
    reg = Registry._inst  # pylint: disable=protected-access
    if cls_name in reg.table:
        entry = reg.table[cls_name]
        return getattr(entry, attr, None)
    return None


def from_doc(node):
    """Get original python AST node from doc AST node.

    Parameters
    ----------
    node : doc.AST
        The doc AST node.

    Returns
    -------
    res : ast.AST
        The corresponding AST node.
    """
    if _is_atomic_type(node):
        return node
    if isinstance(node, tuple):
        return tuple(from_doc(n) for n in node)
    if isinstance(node, list):
        return [from_doc(n) for n in node]
    func = _get_registry_entry(node.__class__.__name__, "from_doc")
    if not func:
        raise NotImplementedError(f"from_doc is not implemented for: {node.__class__.__name__}")
    return func(node)


def to_doc(node):
    """Get doc AST node from python AST node.

    Parameters
    ----------
    node : ast.AST
        The AST node.

    Returns
    -------
    res : doc.AST
        The corresponding doc AST node.
    """
    if _is_atomic_type(node):
        return node
    if isinstance(node, tuple):
        return tuple(to_doc(n) for n in node)
    if isinstance(node, list):
        return [to_doc(n) for n in node]
    func = _get_registry_entry(node.__class__.__name__, "to_doc")
    if not func:
        raise NotImplementedError(f"to_doc is not implemented for: {node.__class__.__name__}")
    return func(node)


def parse(
    source: str,
    filename: str = "<unknown>",
    mode: str = "exec",
) -> doc.AST:
    """Parse TVMScript source code str to doc AST.

    Its interface is consistent with python built-in ast.parse.
    And it will parse by python 3.8 first if possible,
    or it will parse with python version in current environment.

    Parameters
    ----------
    source : str
        The TVMScript source code.

    filename : str
        The optional filename of the file where source code locates.

    mode : str
        The parsing mode for ast.parse.

    Returns
    -------
    res : doc.AST
        The parsed doc AST.
    """
    try:
        program = ast.parse(  # pylint: disable=unexpected-keyword-arg
            source=source,
            filename=filename,
            mode=mode,
            feature_version=(3, 8),
        )
    except:  # pylint: disable=bare-except
        program = ast.parse(
            source=source,
            filename=filename,
            mode=mode,
        )
    return to_doc(program)


class NodeVisitor:
    """Node visitor for doc AST"""

    def visit(self, node: doc.AST) -> None:
        if isinstance(node, (list, tuple)):
            for item in node:
                self.visit(item)
            return
        if not isinstance(node, doc.AST):
            return
        getattr(
            self,
            "visit_" + node.__class__.__name__.split(".")[-1],
            self.generic_visit,
        )(node)

    def generic_visit(self, node: doc.AST) -> None:
        for field in node.__class__._FIELDS:  # pylint: disable=protected-access
            value = getattr(node, field, None)
            if value is None:
                pass
            elif isinstance(value, (doc.AST, list, tuple)):
                self.visit(value)


class NodeTransformer:
    """Node transformer for doc AST"""

    def visit(self, node: doc.AST) -> doc.AST:
        if isinstance(node, list):
            return [self.visit(item) for item in node]
        if isinstance(node, tuple):
            return tuple(self.visit(item) for item in node)
        if not isinstance(node, doc.AST):
            return node
        return getattr(
            self,
            "visit_" + node.__class__.__name__.split(".")[-1],
            self.generic_visit,
        )(node)

    def generic_visit(self, node: doc.AST) -> doc.AST:
        kv: typing.Dict[str, typing.Any] = {}
        for field in node.__class__._FIELDS:  # pylint: disable=protected-access
            value = getattr(node, field, None)
            if value is None:
                pass
            elif isinstance(value, (doc.AST, list, tuple)):
                value = self.visit(value)
            kv[field] = value
        return node.__class__(**kv)


def _register_default():
    class DefaultTranslator:
        def __init__(self, doc_cls, func, fields):
            self.doc_cls = doc_cls  # getattr(doc, name)
            self.func = func
            self.fields = fields

        def __call__(self, node):
            kv = {attr: self.func(getattr(node, attr, None)) for attr in self.fields}
            return self.doc_cls(**kv)

    Registry._inst = Registry()  # pylint: disable=protected-access
    for cls_name in dir(doc):
        doc_cls = getattr(doc, cls_name)
        if not hasattr(ast, cls_name):
            continue
        if inspect.isclass(doc_cls) and issubclass(doc_cls, doc.AST):
            assert "." not in cls_name
            register_to_doc(cls_name)(
                DefaultTranslator(
                    getattr(doc, cls_name),
                    to_doc,
                    doc_cls._FIELDS,  # pylint: disable=protected-access
                )
            )
            register_from_doc(cls_name)(
                DefaultTranslator(
                    getattr(ast, cls_name),
                    from_doc,
                    doc_cls._FIELDS,  # pylint: disable=protected-access
                )
            )


_register_default()
