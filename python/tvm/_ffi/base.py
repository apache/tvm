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
# coding: utf-8
# pylint: disable=invalid-name, import-outside-toplevel
"""Base library for TVM FFI."""
import ctypes
import functools
import os
import re
import sys
import types

from typing import Callable, Sequence, Optional

import numpy as np

from . import libinfo

# ----------------------------
# library loading
# ----------------------------
string_types = (str,)
integer_types = (int, np.int32)
numeric_types = integer_types + (float, np.float16, np.float32)

# this function is needed for python3
# to convert ctypes.char_p .value back to python str
if sys.platform == "win32":

    def _py_str(x):
        try:
            return x.decode("utf-8")
        except UnicodeDecodeError:
            encoding = "cp" + str(ctypes.cdll.kernel32.GetACP())
        return x.decode(encoding)

    py_str = _py_str
else:
    py_str = lambda x: x.decode("utf-8")


def _load_lib():
    """Load libary by searching possible path."""
    lib_path = libinfo.find_lib_path()
    # The dll search path need to be added explicitly in
    # windows after python 3.8
    if sys.platform.startswith("win32") and sys.version_info >= (3, 8):
        for path in libinfo.get_dll_directories():
            os.add_dll_directory(path)
    lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_GLOBAL)
    lib.TVMGetLastError.restype = ctypes.c_char_p
    return lib, os.path.basename(lib_path[0])


try:
    # The following import is needed for TVM to work with pdb
    import readline  # pylint: disable=unused-import
except ImportError:
    pass

# version number
__version__ = libinfo.__version__
# library instance
_LIB, _LIB_NAME = _load_lib()

# Whether we are runtime only
_RUNTIME_ONLY = "runtime" in _LIB_NAME

# The FFI mode of TVM
_FFI_MODE = os.environ.get("TVM_FFI", "auto")


# ----------------------------
# helper function in ctypes.
# ----------------------------
def c_str(string):
    """Create ctypes char * from a python string
    Parameters
    ----------
    string : string type
        python string

    Returns
    -------
    str : c_char_p
        A char pointer that can be passed to C API
    """
    return ctypes.c_char_p(string.encode("utf-8"))


def c_array(ctype, values):
    """Create ctypes array from a python array

    Parameters
    ----------
    ctype : ctypes data type
        data type of the array we want to convert to

    values : tuple or list
        data content

    Returns
    -------
    out : ctypes array
        Created ctypes array
    """
    return (ctype * len(values))(*values)


def decorate(func, fwrapped):
    """A wrapper call of decorator package, differs to call time

    Parameters
    ----------
    func : function
        The original function

    fwrapped : function
        The wrapped function
    """
    import decorator

    return decorator.decorate(func, fwrapped)


# -----------------------------------------
# Base code for structured error handling.
# -----------------------------------------
# Maps error type to its constructor
ERROR_TYPE = {}


class TVMError(RuntimeError):
    """Default error thrown by TVM functions.

    TVMError will be raised if you do not give any error type specification,
    """


def register_error(func_name=None, cls=None):
    """Register an error class so it can be recognized by the ffi error handler.

    Parameters
    ----------
    func_name : str or function or class
        The name of the error function.

    cls : function
        The function to create the class

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
    if callable(func_name):
        cls = func_name
        func_name = cls.__name__

    def register(mycls):
        """internal register function"""
        err_name = func_name if isinstance(func_name, str) else mycls.__name__
        ERROR_TYPE[err_name] = mycls
        return mycls

    if cls is None:
        return register
    return register(cls)


def _valid_error_name(name):
    """Check whether name is a valid error name."""
    return all(x.isalnum() or x in "_." for x in name)


def _find_error_type(line):
    """Find the error name given the first line of the error message.

    Parameters
    ----------
    line : str
        The first line of error message.

    Returns
    -------
    name : str The error name
    """
    if sys.platform == "win32":
        # Stack traces aren't logged on Windows due to a DMLC limitation,
        # so we should try to get the underlying error another way.
        # DMLC formats errors "[timestamp] file:line: ErrorMessage"
        # ErrorMessage is usually formatted "ErrorType: message"
        # We can try to extract the error type using the final ":"
        end_pos = line.rfind(":")
        if end_pos == -1:
            return None
        start_pos = line.rfind(":", 0, end_pos)
        if start_pos == -1:
            err_name = line[:end_pos].strip()
        else:
            err_name = line[start_pos + 1 : end_pos].strip()
        if _valid_error_name(err_name):
            return err_name
        return None

    end_pos = line.find(":")
    if end_pos == -1:
        return None
    err_name = line[:end_pos]
    if _valid_error_name(err_name):
        return err_name
    return None


def c2pyerror(err_msg):
    """Translate C API error message to python style.

    Parameters
    ----------
    err_msg : str
        The error message.

    Returns
    -------
    new_msg : str
        Translated message.

    err_type : str
        Detected error type.
    """
    arr = err_msg.split("\n")
    if arr[-1] == "":
        arr.pop()
    err_type = _find_error_type(arr[0])
    trace_mode = False
    stack_trace = []
    message = []
    for line in arr:
        if trace_mode:
            if line.startswith("        ") and len(stack_trace) > 0:
                stack_trace[-1] += "\n" + line
            elif line.startswith("  "):
                stack_trace.append(line)
            else:
                trace_mode = False
        if not trace_mode:
            if line.startswith("Stack trace"):
                trace_mode = True
            else:
                message.append(line)
    out_msg = ""
    if stack_trace:
        out_msg += "Traceback (most recent call last):\n"
        out_msg += "\n".join(reversed(stack_trace)) + "\n"
    out_msg += "\n".join(message)
    return out_msg, err_type


def py2cerror(err_msg):
    """Translate python style error message to C style.

    Parameters
    ----------
    err_msg : str
        The error message.

    Returns
    -------
    new_msg : str
        Translated message.
    """
    arr = err_msg.split("\n")
    if arr[-1] == "":
        arr.pop()
    trace_mode = False
    stack_trace = []
    message = []
    for line in arr:
        if trace_mode:
            if line.startswith("  "):
                stack_trace.append(line)
            else:
                trace_mode = False
        if not trace_mode:
            if line.find("Traceback") != -1:
                trace_mode = True
            else:
                message.append(line)
    # Remove the first error name if there are two of them.
    # RuntimeError: MyErrorName: message => MyErrorName: message
    head_arr = message[0].split(":", 3)
    if len(head_arr) >= 3 and _valid_error_name(head_arr[1].strip()):
        head_arr[1] = head_arr[1].strip()
        message[0] = ":".join(head_arr[1:])
    # reverse the stack trace.
    out_msg = "\n".join(message)
    if stack_trace:
        out_msg += "\nStack trace:\n"
        out_msg += "\n".join(reversed(stack_trace)) + "\n"
    return out_msg


def get_last_ffi_error():
    """Create error object given result of TVMGetLastError.

    Returns
    -------
    err : object
        The error object based on the err_msg
    """
    c_err_msg = py_str(_LIB.TVMGetLastError())
    py_err_msg, err_type = c2pyerror(c_err_msg)
    if err_type is not None and err_type.startswith("tvm.error."):
        err_type = err_type[10:]
    return ERROR_TYPE.get(err_type, TVMError)(py_err_msg)


def _append_traceback_frame(tb, func_name, filepath, lineno: Optional[int]):
    """Append a dummy frame to appear in the Python traceback"""

    # Compile a dummy function to Python bytecode, so that with the
    # filepath that we want to appear in the traceback.  Any external
    # debugger (e.g. pdb) that catches the exception will use the
    # filepath to show code snippets from that FFI file.
    header = "" if lineno is None else "\n" * (lineno - 1)
    code = compile(
        f"{header}def dummy_func(): raise NotImplementedError()",
        filepath,
        "exec",
    )

    # Replacing the name by updating the bytecode allows the function
    # name to be values that would normally be forbidden by python
    # syntax.  For example, "operator()".
    code = code.replace(co_consts=(code.co_consts[0].replace(co_name=func_name), func_name, None))
    namespace = {}
    exec(code, namespace)  # pylint: disable=exec-used
    dummy_func = namespace["dummy_func"]

    # Execute the dummy function in order to generate a stack frame.
    dummy_tb = None
    try:
        dummy_func()
    except NotImplementedError as err:
        dummy_tb = err.__traceback__

    # Insert the dummy function into the stack trace.
    new_frame = dummy_tb.tb_next
    return types.TracebackType(tb, new_frame.tb_frame, new_frame.tb_lasti, new_frame.tb_lineno)


def _filter_traceback_frames(tb, filter_funcs: Sequence[Callable[[types.CodeType], bool]]):
    orig = tb
    filtered_at_least_one = False
    temp_all_frames = []
    filtered_frames = []

    while tb is not None:
        frame_code = tb.tb_frame.f_code
        should_remove = any(filter_func(frame_code) for filter_func in filter_funcs)
        if not should_remove:
            filtered_at_least_one = True
            filtered_frames.append(tb)
        temp_all_frames.append(tb)
        tb = tb.tb_next

    if not filtered_at_least_one:
        return orig

    def _append_frame(tb, next_tb_frame):
        return types.TracebackType(
            tb, next_tb_frame.tb_frame, next_tb_frame.tb_lasti, next_tb_frame.tb_lineno
        )

    new_tb = functools.reduce(_append_frame, reversed(filtered_frames))

    return new_tb


def raise_last_ffi_error():
    """Raise the previous error from FFI

    This should be used instead of `raise get_last_ffi_error()`, as it
    handle propagation of errors across an FFI boundary.  For example,
    if Python passes a callback to a C++ function, and the callback
    raises an exception, the re-thrown exception should contain the
    full stack trace, not just the stack frames that are above the
    outermost FFI call.
    """

    _LIB.TVMGetLastPythonError.restype = ctypes.c_void_p
    _LIB.TVMGetLastBacktrace.restype = ctypes.c_char_p
    py_err = _LIB.TVMGetLastPythonError()
    if py_err is None:
        c_err_msg = py_str(_LIB.TVMGetLastError())
        py_err_msg, err_type = c2pyerror(c_err_msg)
        if err_type is not None and err_type.startswith("tvm.error."):
            err_type = err_type[10:]
        py_err = ERROR_TYPE.get(err_type, TVMError)(py_err_msg)

    else:
        # TVMGetLastPythonError returns a PyObject*, with NULL when
        # there is no such value.  If we annotated the restype as
        # ctypes.py_object, we would need to return Py_None from the
        # C++ implementation.  This would require introducing a
        # dependency on libpython that we want to avoid when not in a
        # Python environment.  Therefore, casting the resulting void*
        # pointer to PyObject* using ctypes.
        py_err = ctypes.cast(ctypes.c_void_p(py_err), ctypes.py_object).value

    tb = py_err.__traceback__

    # The py_err.__traceback__ only goes from the location thrown
    # up to the next FFI handoff.  To have the stacktrace also
    # include the C++ side, we need to adjust the __traceback__
    # before re-throwing.
    backtrace = _LIB.TVMGetLastBacktrace()
    if backtrace:
        frames = re.split(r"\n\W+\d+:\W+", py_str(backtrace))
        frames = frames[1:]  # Skip "Stack trace: "

        for frame in frames:
            if " at " in frame:
                func_name, frame = frame.split(" at ", 1)
                if ":" in frame:
                    filename, lineno = frame.rsplit(":", 1)
                    lineno = int(lineno.strip())
                else:
                    filename = frame
                    lineno = None
                func_name = func_name.strip()
                filename = filename.strip()

                tb = _append_traceback_frame(tb, func_name, filename, lineno)

    # Remove stack frames that provide little benefit to
    # debugging.  These are only removed from the stack frames
    # contained within the exception we are re-raising, and not to
    # the stack frames that it will continue to collect.
    # Therefore, there may still be a single instance of these
    # frames in the outermost Python-to-FFI call.
    filter_funcs = [
        lambda code: "tvm/_ffi/_ctypes/packed_func.py" in code.co_filename,
        lambda code: "tvm/_ffi/base.py" in code.co_filename,
    ]
    tb = _filter_traceback_frames(tb, filter_funcs)

    py_err = py_err.with_traceback(tb)

    # The exception PyObject may contain a large amount of state,
    # including all stack frames that may be inspected in a later
    # PDB post-mortem.  Therefore, we must make sure to remove the
    # underlying PyObject* from the C++ side after we retrieve it.
    _LIB.TVMDropLastPythonError()

    raise py_err


def check_call(ret):
    """Check the return value of C API call

    This function will raise exception when error occurs.
    Wrap every API call with this function

    Parameters
    ----------
    ret : int
        return value from API calls
    """
    if ret != 0:
        raise_last_ffi_error()
