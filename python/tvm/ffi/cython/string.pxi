
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

def _string_obj_get_py_str(obj):
    cdef TVMFFIByteArray* bytes = TVMFFIBytesGetByteArrayPtr((<Object>obj).chandle)
    return py_str(bytes.data)


def _bytes_obj_get_py_bytes(obj):
    cdef TVMFFIByteArray* bytes = TVMFFIBytesGetByteArrayPtr((<Object>obj).chandle)
    py_bytes = PyBytes_FromStringAndSize(NULL, bytes.size)
    if py_bytes is None:
        raise MemoryError("Failed to allocate bytes object")
    memcpy(PyBytes_AsString(py_bytes), bytes.data, bytes.size)
    return py_bytes


cdef class ByteArrayArg:
    cdef TVMFFIByteArray cdata
    cdef object py_data

    def __cinit__(self, py_data):
        if isinstance(py_data, bytearray):
            py_data = bytes(py_data)
        cdef char* data
        cdef Py_ssize_t size
        self.py_data = py_data
        PyBytes_AsStringAndSize(py_data, &data, &size)
        self.cdata.data = data
        self.cdata.size = size

