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
"""TVM Script Parser utils"""

import inspect
from collections import ChainMap
from collections.abc import Callable, Mapping
from types import FrameType
from typing import Any

from .diagnostics import findsource


def get_func_nonlocals(func):
    """A modified version of `inspect.getclosurevars`"""

    if inspect.ismethod(func):
        func = func.__func__

    if not inspect.isfunction(func):
        raise TypeError(f"{func!r} is not a Python function")

    code = func.__code__
    # Nonlocal references are named in co_freevars and resolved
    # by looking them up in __closure__ by positional index
    nonlocal_vars = {}
    if func.__closure__ is not None:
        for var, cell in zip(code.co_freevars, func.__closure__):
            try:
                nonlocal_vars[var] = cell.cell_contents
            except ValueError as err:
                # cell_contents may raise ValueError if the cell is empty.
                if "empty" not in str(err):
                    raise
    return nonlocal_vars


def inspect_function_capture(func: Callable) -> dict[str, Any]:
    """Capture function non-locals and global variables.

    Parameters
    ----------
    func : Callable
        The function to inspect.

    Returns
    -------
    res : Dict[str, Any]
        The function variables map with non-local or global variables.
    """
    captured = {
        **func.__globals__,  # type: ignore
        **get_func_nonlocals(func),
    }
    return captured


def inspect_class_capture(cls: type) -> dict[str, Any]:
    """Capture class non-locals and global variables.

    Parameters
    ----------
    cls : type
        The class to inspect.

    Returns
    -------
    res : Dict[str, Any]
        The class variables map with non-local or global variables.
    """
    result: dict[str, Any] = {}
    for _, v in cls.__dict__.items():
        if inspect.isfunction(v):
            func_vars = inspect_function_capture(v)
            result.update(**func_vars)
    return result


def with_caller_frame_fallback(extra_vars: dict[str, Any], outer_stack: list) -> Mapping[str, Any]:
    """Wrap extra_vars with lazy caller-frame fallback for PEP 563 compatibility.

    With ``from __future__ import annotations``, variables used only in
    annotations are not captured in ``__closure__``.  ChainMap defers
    lookup to caller-frame locals only when a key is missing from the
    primary dict (globals + closure).
    """
    return ChainMap(extra_vars, *[f.frame.f_locals for f in outer_stack[1:]])


def resolve_closure_vars(cls: type, extra_vars: dict[str, Any], outer_stack: list) -> None:
    """Resolve closure variables for class methods hidden by PEP 563.

    With ``from __future__ import annotations``, variables used only in
    annotations are not captured in ``__closure__``.  This function parses
    the class source AST to find names used in function annotations, then
    looks them up in enclosing frames.  Only annotation-referenced names
    are added, avoiding namespace pollution from unrelated caller locals.
    """
    import ast
    import textwrap

    # Collect names used in function annotations from source AST
    try:
        source = textwrap.dedent(inspect.getsource(cls))
        tree = ast.parse(source)
    except (OSError, TypeError):
        return

    ann_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                if arg.annotation:
                    for n in ast.walk(arg.annotation):
                        if isinstance(n, ast.Name):
                            ann_names.add(n.id)
            if node.returns:
                for n in ast.walk(node.returns):
                    if isinstance(n, ast.Name):
                        ann_names.add(n.id)

    # Look up missing annotation names in enclosing frames
    for name in ann_names:
        if name not in extra_vars:
            for frame_info in outer_stack[1:]:
                if name in frame_info.frame.f_locals:
                    extra_vars[name] = frame_info.frame.f_locals[name]
                    break


def is_defined_in_class(frames: list[FrameType], obj: Any) -> bool:
    """Check whether a object is defined in a class scope.

    Parameters
    ----------
    frames : List[FrameType]
        The frame stack of the object, obtained by `inspect.stack()`.

    Returns
    -------
    res : bool
        The result if the object is defined in a class scope.
    """

    def _is_tvmscript_class_annotator(line: str) -> bool:
        """Checks if the line contains a TVMScript annotator for a class

        These match either `@I.ir_module` or `@R.rewriter`, or their
        imported names `@ir_module` or `@rewriter`.
        """

        return line.startswith("@") and ("ir_module" in line or "rewriter" in line)

    if len(frames) > 2:
        frame_info = frames[2]
        code_context = frame_info.code_context
        if code_context is None:
            return False
        line = code_context[0].strip()
        if _is_tvmscript_class_annotator(line):
            return True
        if line.startswith("class"):
            lineno = frame_info.lineno
            if lineno >= 2:
                source, _ = findsource(obj)
                line = source[lineno - 2].strip()
                if _is_tvmscript_class_annotator(line):
                    return True
    return False
