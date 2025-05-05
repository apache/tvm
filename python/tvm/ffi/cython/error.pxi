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
# error handling for FFI

import types
import re

ERROR_NAME_TO_TYPE = {}
ERROR_TYPE_TO_NAME = {}

_WITH_APPEND_TRACEBACK = None
_TRACEBACK_TO_STR = None


cdef inline int _raise_not_implemented_with_extra_frame(
    const char* func_name,
    int lineno,
    const char* file_path
) except -1:
    """Helper util, raise internal """
    PyErr_SetNone(NotImplementedError)
    __Pyx_AddTraceback_(func_name, 0, lineno, file_path)
    return -1


def _append_traceback_frame(tb, filename, lineno, func):
    """Append tracebacks to frame.

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
    """
    try:
        _raise_not_implemented_with_extra_frame(
            c_str(func), lineno, c_str(filename)
        )
    except NotImplementedError as e:
        dummy_tb = e.__traceback__
    # skip the first frame which is the dummy frame
    new_frame = dummy_tb.tb_next
    return types.TracebackType(tb, new_frame.tb_frame, new_frame.tb_lasti, new_frame.tb_lineno)


cdef class Error(Object):
    """Base class for all FFI errors, usually they are attached to errors

    Note
    ----
    Do not directly raise this object, instead use the `py_error` method
    to convert it to a python error then raise it.
    """

    def __init__(self, kind, message, traceback):
        cdef ByteArrayArg kind_arg = ByteArrayArg(c_str(kind))
        cdef ByteArrayArg message_arg = ByteArrayArg(c_str(message))
        cdef ByteArrayArg traceback_arg = ByteArrayArg(c_str(traceback))
        (<Object>self).chandle = TVMFFIErrorCreate(
            kind_arg.cptr(), message_arg.cptr(), traceback_arg.cptr()
        )

    def update_traceback(self, traceback):
        """Update the traceback of the error

        Parameters
        ----------
        traceback : str
            The traceback to update.
        """
        cdef ByteArrayArg traceback_arg = ByteArrayArg(c_str(traceback))
        TVMFFIErrorGetCellPtr(self.chandle).update_traceback(self.chandle, traceback_arg.cptr())

    def py_error(self):
        """
        Convert the FFI error to the python error
        """
        error_cls = ERROR_NAME_TO_TYPE.get(self.kind, RuntimeError)
        py_error = error_cls(self.message)
        py_error = _WITH_APPEND_TRACEBACK(py_error, self.traceback)
        py_error.__tvm_ffi_error__ = self
        return py_error

    @property
    def kind(self):
        return bytearray_to_str(&(TVMFFIErrorGetCellPtr(self.chandle).kind))

    @property
    def message(self):
        return bytearray_to_str(&(TVMFFIErrorGetCellPtr(self.chandle).message))

    @property
    def traceback(self):
        return bytearray_to_str(&(TVMFFIErrorGetCellPtr(self.chandle).traceback))

_register_object_by_index(kTVMFFIError, Error)


cdef inline Error move_from_last_error():
    # raise last error
    error = Error.__new__(Error)
    TVMFFIErrorMoveFromRaised(&(<Object>error).chandle)
    return error


cdef inline int raise_existing_error() except -2:
    return -2


cdef inline int set_last_ffi_error(error) except -1:
    """Set the last FFI error"""
    cdef Error ffi_error

    kind = ERROR_TYPE_TO_NAME.get(type(error), "RuntimeError")
    message = error.__str__()
    py_traceback = _TRACEBACK_TO_STR(error.__traceback__)
    c_traceback = bytearray_to_str(TVMFFITraceback("<unknown>", 0, "<unknown>"))

    # error comes from an exception thrown from C++ side
    if hasattr(error, "__tvm_ffi_error__"):
        # already have stack trace
        ffi_error = error.__tvm_ffi_error__
        # attach the python traceback together with the C++ traceback to get full trace
        ffi_error.update_traceback(c_traceback + py_traceback)
        TVMFFIErrorSetRaised(ffi_error.chandle)
    else:
        ffi_error = Error(kind, message, c_traceback + py_traceback)
        TVMFFIErrorSetRaised(ffi_error.chandle)


def _convert_to_ffi_error(error):
    """Convert the python error to the FFI error"""
    py_traceback = _TRACEBACK_TO_STR(error.__traceback__)
    if hasattr(error, "__tvm_ffi_error__"):
        error.__tvm_ffi_error__.update_traceback(py_traceback)
        return error.__tvm_ffi_error__
    else:
        kind = ERROR_TYPE_TO_NAME.get(type(error), "RuntimeError")
        message = error.__str__()
        return Error(kind, message, py_traceback)


cdef inline int CHECK_CALL(int ret) except -2:
    """Check the return code of the C API function call"""
    if ret == 0:
        return 0
    # -2 brings exception
    if ret == -2:
        raise raise_existing_error()
    raise move_from_last_error().py_error()
