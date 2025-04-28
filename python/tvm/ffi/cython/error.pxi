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


cdef class Error(Object):
    """Base class for all FFI errors, usually they are attached to errors"""
    def update_traceback(self, traceback):
        """Update the traceback of the error

        Parameters
        ----------
        traceback : str
            The traceback to update.
        """
        TVMFFIUpdateErrorTraceback(self.chandle, c_str(traceback))

    @property
    def kind(self):
        return py_str(TVMFFIErrorGetErrorInfoPtr(self.chandle).kind)

    @property
    def message(self):
        return py_str(TVMFFIErrorGetErrorInfoPtr(self.chandle).message)

    @property
    def traceback(self):
        return py_str(TVMFFIErrorGetErrorInfoPtr(self.chandle).traceback)

_register_object_by_index(kTVMFFIError, Error)


cdef inline object move_from_last_error():
    # raise last error
    cdef TVMFFIAny result
    cdef TVMFFIErrorInfo* error_info
    TVMFFIMoveFromLastError(&result)
    if result.type_index != kTVMFFIError:
        if result.type_index >= kTVMFFIStaticObjectBegin:
            TVMFFIObjectFree(result.v_ptr)
        return RuntimeError("Error happened in FFI call type_index=%d" % result.type_index)
    error = Error.__new__(Error)
    (<Object>error).chandle = result.v_ptr
    error_info = TVMFFIErrorGetErrorInfoPtr(result.v_ptr)
    error_cls = ERROR_NAME_TO_TYPE.get(error.kind, RuntimeError)
    py_error = error_cls(error.message)
    py_error = _WITH_APPEND_TRACEBACK(py_error, error.traceback)
    py_error.__tvm_ffi_error__ = error
    return py_error


cdef inline int raise_existing_error() except -2:
    return -2


cdef inline int set_last_ffi_error(error) except -1:
    """Set the last FFI error"""
    cdef Error ffi_error
    cdef TVMFFIAny temp_args

    kind = ERROR_TYPE_TO_NAME.get(type(error), "RuntimeError")
    message = error.__str__()
    py_traceback = _TRACEBACK_TO_STR(error.__traceback__)

    # error comes from an exception thrown from C++ side
    if hasattr(error, "__tvm_ffi_error__"):
        # already have stack trace
        ffi_error = error.__tvm_ffi_error__
        c_traceback = py_str(TVMFFITraceback("<unknown>", 0, "<unknown>"))
        # attach the python traceback together with the C++ traceback to get full trace
        ffi_error.update_traceback(c_traceback + py_traceback)
        temp_args.v_int64 = 0
        temp_args.type_index = kTVMFFIError
        temp_args.v_ptr = ffi_error.chandle
        TVMFFISetLastError(&temp_args)
    else:
        TVMFFISetLastErrorCStr(c_str(kind), c_str(message), c_str(py_traceback))


cdef inline int CHECK_CALL(int ret) except -2:
    """Check the return code of the C API function call"""
    if ret == 0:
        return 0
    # -2 brings exception
    if ret == -2:
        raise raise_existing_error()
    raise move_from_last_error()
