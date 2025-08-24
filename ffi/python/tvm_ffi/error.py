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
# pylint: disable=invalid-name
"""Error handling."""
import re
import types
import sys
import ast
from . import core


def _parse_traceback(traceback):
    """Parse the traceback string into a list of (filename, lineno, func)

    Parameters
    ----------
    traceback : str
        The traceback string.

    Returns
    -------
    result : List[Tuple[str, int, str]]
        The list of (filename, lineno, func)
    """
    pattern = r'File "(.+?)", line (\d+), in (.+)'
    result = []
    for line in traceback.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            try:
                filename = match.group(1)
                lineno = int(match.group(2))
                func = match.group(3)
                result.append((filename, lineno, func))
            except ValueError:
                pass
    return result


class TracebackManager:
    """
    Helper to manage traceback generation
    """

    def __init__(self):
        self._code_cache = {}

    def _get_cached_code_object(self, filename, lineno, func):
        # Hack to create a code object that points to the correct
        # line number and function name
        key = (filename, lineno, func)
        # cache the code object to avoid re-creating it
        if key in self._code_cache:
            return self._code_cache[key]
        # Parse to AST and zero out column info
        # since column info are not accurate in original trace
        tree = ast.parse("_getframe()", filename=filename, mode="eval")
        for node in ast.walk(tree):
            if hasattr(node, "col_offset"):
                node.col_offset = 0
            if hasattr(node, "end_col_offset"):
                node.end_col_offset = 0
        # call into get frame, bt changes the context
        code_object = compile(tree, filename, "eval")
        # replace the function name and line number
        code_object = code_object.replace(co_name=func, co_firstlineno=lineno)
        self._code_cache[key] = code_object
        return code_object

    def _create_frame(self, filename, lineno, func):
        """Create a frame object from the filename, lineno, and func"""
        code_object = self._get_cached_code_object(filename, lineno, func)
        # call into get frame, but changes the context so the code
        # points to the correct frame
        context = {"_getframe": sys._getframe}
        # pylint: disable=eval-used
        return eval(code_object, context, context)

    def append_traceback(self, tb, filename, lineno, func):
        """Append a traceback to the given traceback

        Parameters
        ----------
        tb : types.TracebackType
            The traceback to append to.
        filename : str
            The filename of the traceback
        lineno : int
            The line number of the traceback
        func : str
            The function name of the traceback

        Returns
        -------
        new_tb : types.TracebackType
            The new traceback with the appended frame.
        """
        frame = self._create_frame(filename, lineno, func)
        return types.TracebackType(tb, frame, frame.f_lasti, lineno)


_TRACEBACK_MANAGER = TracebackManager()


def _with_append_traceback(py_error, traceback):
    """Append the traceback to the py_error and return it"""
    tb = py_error.__traceback__
    for filename, lineno, func in reversed(_parse_traceback(traceback)):
        tb = _TRACEBACK_MANAGER.append_traceback(tb, filename, lineno, func)
    return py_error.with_traceback(tb)


def _traceback_to_str(tb):
    """Convert the traceback to a string"""
    lines = []
    while tb is not None:
        frame = tb.tb_frame
        lineno = tb.tb_lineno
        filename = frame.f_code.co_filename
        funcname = frame.f_code.co_name
        lines.append(f'  File "{filename}", line {lineno}, in {funcname}\n')
        tb = tb.tb_next
    return "".join(lines)


core._WITH_APPEND_TRACEBACK = _with_append_traceback
core._TRACEBACK_TO_STR = _traceback_to_str


def register_error(name_or_cls=None, cls=None):
    """Register an error class so it can be recognized by the ffi error handler.

    Parameters
    ----------
    name_or_cls : str or class
        The name of the error class.

    cls : class
        The class to register.

    Returns
    -------
    fregister : function
        Register function if f is not specified.

    Examples
    --------
    .. code-block:: python

      @tvm.error.register_error
      class MyError(RuntimeError):
          pass

      err_inst = tvm.error.create_ffi_error("MyError: xyz")
      assert isinstance(err_inst, MyError)
    """
    if callable(name_or_cls):
        cls = name_or_cls
        name_or_cls = cls.__name__

    def register(mycls):
        """internal register function"""
        err_name = name_or_cls if isinstance(name_or_cls, str) else mycls.__name__
        core.ERROR_NAME_TO_TYPE[err_name] = mycls
        core.ERROR_TYPE_TO_NAME[mycls] = err_name
        return mycls

    if cls is None:
        return register
    return register(cls)


register_error("RuntimeError", RuntimeError)
register_error("ValueError", ValueError)
register_error("TypeError", TypeError)
register_error("AttributeError", AttributeError)
register_error("KeyError", KeyError)
register_error("IndexError", IndexError)
register_error("AssertionError", AssertionError)
