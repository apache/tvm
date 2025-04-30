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


def _with_append_traceback(py_error, traceback):
    """Append the traceback to the py_error and return it"""
    tb = py_error.__traceback__
    for filename, lineno, func in reversed(_parse_traceback(traceback)):
        tb = core._append_traceback_frame(tb, filename, lineno, func)
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
