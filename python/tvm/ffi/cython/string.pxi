
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

# helper class for string/bytes handling

cdef inline str _string_obj_get_py_str(obj):
    cdef TVMFFIByteArray* bytes = TVMFFIBytesGetByteArrayPtr((<Object>obj).chandle)
    return py_str(PyBytes_FromStringAndSize(bytes.data, bytes.size))


cdef inline bytes _bytes_obj_get_py_bytes(obj):
    cdef TVMFFIByteArray* bytes = TVMFFIBytesGetByteArrayPtr((<Object>obj).chandle)
    return PyBytes_FromStringAndSize(bytes.data, bytes.size)



class String(str, PyNativeObject):
    __slots__ = ["__tvm_ffi_object__"]
    """String object that is possibly returned by FFI call.

    Note
    ----
    This class subclasses str so it can be directly treated as str.
    There is no need to construct this object explicitly.
    """
    def __new__(cls, value):
        val = str.__new__(cls, value)
        val.__init_tvm_ffi_object_by_constructor__(_STR_CONSTRUCTOR, value)
        return val

    # pylint: disable=no-self-argument
    def __from_tvm_ffi_object__(cls, obj):
        """Construct from a given tvm object."""
        content = _string_obj_get_py_str(obj)
        val = str.__new__(cls, content)
        val.__tvm_ffi_object__ = obj
        return val


_register_object_by_index(kTVMFFIStr, String)


class Bytes(bytes, PyNativeObject):
    """Bytes object that is possibly returned by FFI call.

    Note
    ----
    This class subclasses bytes so it can be directly treated as bytes.
    There is no need to construct this object explicitly.
    """
    def __new__(cls, value):
        val = bytes.__new__(cls, value)
        val.__init_tvm_ffi_object_by_constructor__(_BYTES_CONSTRUCTOR, value)
        return val

    # pylint: disable=no-self-argument
    def __from_tvm_ffi_object__(cls, obj):
        """Construct from a given tvm object."""
        content = _bytes_obj_get_py_bytes(obj)
        val = bytes.__new__(cls, content)
        val.__tvm_ffi_object__ = obj
        return val


_register_object_by_index(kTVMFFIBytes, Bytes)

# We special handle str/bytes constructor in cython to avoid extra cyclic deps
# as the str/bytes construction must be done in the inner loop of function call
_STR_CONSTRUCTOR = None
_BYTES_CONSTRUCTOR = None
