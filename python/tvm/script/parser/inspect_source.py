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
"""Inspect source text, AST coordinates and the explicit definition context.

``Source`` exposes dedented text and relative AST coordinates with conversion to
absolute IR spans. ``acquire_source`` returns an AST whose original coordinates
are already restored for compilation. Both keep inspection in this module;
source ownership and execution remain with the parser entry point.
"""

from __future__ import annotations
import __future__

import ast
import dis
import inspect
import linecache
import textwrap
from collections.abc import Mapping
from types import CodeType, FrameType, FunctionType
from typing import Any

from tvm_ffi.dataclasses import MISSING

from tvm.ir import SourceName, Span

from .annotation import parse_annotation
from .prescan import collect_annotation_free_names


class _AnnotationScope(dict):
    """Snapshot selected bindings; absent names fail only when an annotation reads them."""

    def __init__(self, names, *scopes):
        super().__init__()
        for name in names:
            for scope in scopes:
                if name in scope:
                    value = scope[name]
                    if value is not MISSING:
                        self[name] = value
                    # An explicit absence still shadows the remaining scopes.
                    break

    def __missing__(self, name):
        raise NameError(f"name {name!r} is not defined")


class Source:
    """Source code class for TVMScript.

    It is constructed by source code str or doc AST tree.

    Parameters
    ----------
    source_name : str
        The filename of the file where the source code locates.

    start_line : int
        The first line number of the source code.

    start_column : int
        The first column number of the first line of the source code.

    source : str
        The source code str of source code.

    full_source : str
        The complete source code of the file where the source code locates.
    """

    source_name: str | None
    start_line: int
    start_column: int
    source: str
    full_source: str

    def __init__(self, program: str | FunctionType | type) -> None:
        if isinstance(program, str):
            self.source_name = "<str>"
            self.start_line = 1
            self.start_column = 0
            self.source = program
            self.full_source = program
            return

        self.source_name = inspect.getsourcefile(program)  # type: ignore
        lines, self.start_line = inspect.getsourcelines(program)  # type: ignore
        if lines:
            self.start_column = len(lines[0]) - len(lines[0].lstrip())
        else:
            self.start_column = 0
        if self.start_column and lines:
            self.source = "\n".join([line[self.start_column :].rstrip() for line in lines])
        else:
            self.source = "".join(lines)
        try:
            # It will cause a problem when running in Jupyter Notebook.
            # `mod` will be <module '__main__'>, which is a built-in module
            # and `getsource` will throw a TypeError
            mod = inspect.getmodule(program)
            if mod:
                self.full_source = inspect.getsource(mod)
            else:
                self.full_source = self.source
        except TypeError:
            # It's a work around for Jupyter problem.
            # Since `findsource` is an internal API of inspect, we just use it
            # as a fallback method.
            src, _ = inspect.findsource(program)  # type: ignore
            self.full_source = "".join(src)

    def as_ast(self) -> ast.Module:
        """Parse the source code into AST.

        Returns
        -------
        res : ast.AST
            The AST of source code.
        """
        return ast.parse(self.source)

    def location(self, node: ast.AST) -> tuple[int, int, int, int]:
        """Return the absolute 1-based source range of an AST node."""
        lineno = getattr(node, "lineno", 1) or 1
        col_offset = getattr(node, "col_offset", self.start_column)
        col_offset = self.start_column if col_offset is None else col_offset
        end_lineno = getattr(node, "end_lineno", lineno) or lineno
        end_col_offset = getattr(node, "end_col_offset", col_offset)
        end_col_offset = col_offset if end_col_offset is None else end_col_offset
        lineno += self.start_line - 1
        end_lineno += self.start_line - 1
        col_offset += self.start_column + 1
        end_col_offset += self.start_column + 1
        return lineno, col_offset, end_lineno, end_col_offset

    def to_span(self, node: ast.AST) -> Span:
        """Convert an AST node to the canonical IR source span."""
        lineno, col_offset, end_lineno, end_col_offset = self.location(node)
        return Span(
            SourceName(self.source_name or "<unknown>"),
            lineno,
            end_lineno,
            col_offset,
            end_col_offset,
        )


def capture_lexical_bindings(function: FunctionType) -> dict[str, Any]:
    """Snapshot only actual body global/closure reads for a deferred callable."""
    bindings: dict[str, Any] = {}
    codes: list[CodeType] = [function.__code__]
    while codes:
        code = codes.pop()
        codes.extend(value for value in code.co_consts if isinstance(value, CodeType))
        for instruction in dis.get_instructions(code):
            if instruction.opname == "LOAD_GLOBAL" and instruction.argval in function.__globals__:
                bindings[instruction.argval] = function.__globals__[instruction.argval]
    for name, cell in zip(function.__code__.co_freevars, function.__closure__ or ()):
        try:
            bindings[name] = cell.cell_contents
        except ValueError:
            # Recursive/late-bound cells may be empty at decoration time.
            pass
    return bindings


def capture_annotation_bindings(
    source: FunctionType, definition_scope: Mapping[str, Any]
) -> dict[str, Any]:
    """Retain only definition-time names needed by deferred annotations.

    JIT and macros own this small mapping while their source callable remains
    usable. Unrelated outer locals and frame objects never enter it.
    """
    tree, filename, _ = acquire_source(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        annotation = (
            node.annotation
            if isinstance(node, ast.arg | ast.AnnAssign)
            else node.returns
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            else None
        )
        if annotation is not None:
            names.update(collect_annotation_free_names(parse_annotation(annotation, filename)))
    return {name: definition_scope[name] for name in names if name in definition_scope}


def capture_definition_scope(frame: FrameType) -> dict[str, Any]:
    """Snapshot immediate locals and active enclosing Python function scopes.

    Postponed annotations do not necessarily create closure cells. Retain their
    active lexical ancestors only when each caller directly owns the child's
    code object. An unrelated caller ends this chain, even in the same file.
    Class locals belong only to the immediate definition context; enclosing
    classes are not Python lexical scopes. No frame survives this snapshot.
    """
    scopes = [dict(frame.f_locals)]
    while frame.f_back is not None:
        parent = frame.f_back
        if parent.f_globals is not frame.f_globals or not any(
            constant is frame.f_code for constant in parent.f_code.co_consts
        ):
            break
        if parent.f_code.co_flags & inspect.CO_NEWLOCALS:
            scopes.append(dict(parent.f_locals))
        frame = parent
    return {name: value for scope in reversed(scopes) for name, value in scope.items()}


def _read_source_lines(
    source: FunctionType | type, definition_source: tuple[str, int] | None
) -> tuple[list[str], int, str | None]:
    """Recover a class from its exact decoration site when module inspection fails."""
    try:
        lines, start = inspect.getsourcelines(source)
        return lines, start, inspect.getsourcefile(source)
    except OSError:
        if not inspect.isclass(source) or definition_source is None:
            raise
        filename, lineno = definition_source
        lines = linecache.getlines(filename)
        # Gallery runners may execute the class in a temporary __main__ module
        # without __file__. Its decorator still has the original code location.
        tree = ast.parse("".join(lines), filename)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == source.__name__:
                start = min([node.lineno, *(item.lineno for item in node.decorator_list)])
                if start <= lineno <= node.lineno:
                    return lines[start - 1 : node.end_lineno], start, filename
        raise


def acquire_source(
    source: str | FunctionType | type,
    filename: str | None = None,
    *,
    definition_source: tuple[str, int] | None = None,
) -> tuple[ast.Module, str, int]:
    """Read source into a location-preserving AST, filename and compiler flags.

    Text uses ``<str>`` unless a filename is supplied. Function/class source
    retains its original file, line and UTF-8 column offsets, including the
    decoration-site fallback used by gallery runners. Source inspection and
    parsing errors propagate unchanged to the caller.
    The returned AST is fresh for this invocation; entry owns it directly through
    prescan and rewriting. No mutable source AST is cached or shared between parses.
    """
    members = vars(source).values() if inspect.isclass(source) else (source,)
    flags = 0
    for member in members:
        code = getattr(member, "__code__", None)
        if code is not None:
            flags |= code.co_flags & __future__.annotations.compiler_flag
    if isinstance(source, str):
        text = source
        filename = filename or "<str>"
        start, indent = 1, 0
        linecache.cache[filename] = (len(text), None, text.splitlines(keepends=True), filename)
    else:
        lines, start, source_filename = _read_source_lines(source, definition_source)
        text = "".join(lines)
        filename = filename or source_filename
        indent = len(lines[0]) - len(lines[0].lstrip())
    tree = ast.parse(textwrap.dedent(text), filename)
    if start != 1:
        ast.increment_lineno(tree, start - 1)
    if indent:
        for node in ast.walk(tree):
            if hasattr(node, "col_offset"):
                node.col_offset += indent
                node.end_col_offset += indent
    return tree, filename, flags
