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
"""TVM Script Parser Source and diagnostics"""

import inspect
import sys
from typing import Union

from tvm.ir import IRModule, SourceName, Span, diagnostics

from . import doc


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

    source_name: str
    start_line: int
    start_column: int
    source: str
    full_source: str

    def __init__(self, program: Union[str, doc.AST]):
        if isinstance(program, str):
            self.source_name = "<str>"
            self.start_line = 1
            self.start_column = 0
            self.source = program
            self.full_source = program
            return

        self.source_name = inspect.getsourcefile(program)  # type: ignore
        lines, self.start_line = getsourcelines(program)  # type: ignore
        if lines:
            self.start_column = len(lines[0]) - len(lines[0].lstrip())
        else:
            self.start_column = 0
        if self.start_column and lines:
            self.source = "\n".join([l[self.start_column :].rstrip() for l in lines])
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

    def as_ast(self) -> doc.AST:
        """Parse the source code into AST.

        Returns
        -------
        res : doc.AST
            The AST of source code.
        """
        return doc.parse(self.source)


_getfile = inspect.getfile  # pylint: disable=invalid-name
_findsource = inspect.findsource  # pylint: disable=invalid-name


def _patched_inspect_getfile(obj):
    """Work out which source or compiled file an object was defined in."""
    if not inspect.isclass(obj):
        return _getfile(obj)
    mod = getattr(obj, "__module__", None)
    if mod is not None:
        file = getattr(sys.modules[mod], "__file__", None)
        if file is not None:
            return file
    for _, member in inspect.getmembers(obj):
        if inspect.isfunction(member):
            if obj.__qualname__ + "." + member.__name__ == member.__qualname__:
                return inspect.getfile(member)
    raise TypeError(f"Source for {obj:!r} not found")


def findsource(obj):
    """Return the entire source file and starting line number for an object."""
    import linecache  # pylint: disable=import-outside-toplevel

    if not inspect.isclass(obj):
        return _findsource(obj)

    file = inspect.getsourcefile(obj)
    if file:
        linecache.checkcache(file)
    else:
        file = inspect.getfile(obj)
        if not (file.startswith("<") and file.endswith(">")):
            raise OSError("source code not available")

    module = inspect.getmodule(obj, file)
    if module:
        lines = linecache.getlines(file, module.__dict__)
    else:
        lines = linecache.getlines(file)
    if not lines:
        raise OSError("could not get source code")
    qual_names = obj.__qualname__.replace(".<locals>", "<locals>").split(".")
    in_comment = 0
    scope_stack = []
    indent_info = {}
    for i, line in enumerate(lines):
        n_comment = line.count('"""')
        if n_comment:
            # update multi-line comments status
            in_comment = in_comment ^ (n_comment & 1)
            continue
        if in_comment:
            # skip lines within multi-line comments
            continue
        indent = len(line) - len(line.lstrip())
        tokens = line.split()
        if len(tokens) > 1:
            name = None
            if tokens[0] == "def":
                name = tokens[1].split(":")[0].split("(")[0] + "<locals>"
            elif tokens[0] == "class":
                name = tokens[1].split(":")[0].split("(")[0]
            # pop scope if we are less indented
            while scope_stack and indent_info[scope_stack[-1]] >= indent:
                scope_stack.pop()
            if name:
                scope_stack.append(name)
                indent_info[name] = indent
                if scope_stack == qual_names:
                    return lines, i

    raise OSError("could not find class definition")


def getsourcelines(obj):
    """Extract the block of code at the top of the given list of lines."""
    obj = inspect.unwrap(obj)
    lines, l_num = findsource(obj)
    return inspect.getblock(lines[l_num:]), l_num + 1


inspect.getfile = _patched_inspect_getfile


class Diagnostics:
    """Diagnostics class for error reporting in parser.

    Parameters
    ----------
    source : Source
        The source code.

    ctx : diagnostics.DiagnosticContext
        The diagnostic context for diagnostics.
    """

    source: Source
    ctx: diagnostics.DiagnosticContext

    def __init__(self, source: Source):
        mod = IRModule()
        mod.source_map.add(source.source_name, source.full_source)
        self.source = source
        self.ctx = diagnostics.DiagnosticContext(mod, diagnostics.get_renderer())

    def _emit(self, node: doc.AST, message: str, level: diagnostics.DiagnosticLevel) -> None:
        """Emit a diagnostic.

        Parameters
        ----------
        node : doc.AST
            The node with diagnostic information.

        message : str
            The diagnostic message.

        level : diagnostics.DiagnosticLevel
            The diagnostic level.
        """
        lineno = node.lineno or 1
        col_offset = node.col_offset or self.source.start_column
        end_lineno = node.end_lineno or lineno
        end_col_offset = node.end_col_offset or col_offset
        lineno += self.source.start_line - 1
        end_lineno += self.source.start_line - 1
        col_offset += self.source.start_column + 1
        end_col_offset += self.source.start_column + 1
        self.ctx.emit(
            diagnostics.Diagnostic(
                level=level,
                span=Span(
                    source_name=SourceName(self.source.source_name),
                    line=lineno,
                    end_line=end_lineno,
                    column=col_offset,
                    end_column=end_col_offset,
                ),
                message=message,
            )
        )

    def error(self, node: doc.AST, message: str) -> None:
        """Emit a diagnostic error.

        Parameters
        ----------
        node : doc.AST
            The node with diagnostic error.

        message : str
            The diagnostic message.
        """
        self._emit(node, message, diagnostics.DiagnosticLevel.ERROR)
        self.ctx.render()
