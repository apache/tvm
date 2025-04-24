
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


def _bytes_obj_get_py_bytearray(obj):
    cdef TVMFFIByteArray* bytes = TVMFFIBytesGetByteArrayPtr((<Object>obj).chandle)
    cdef unsigned long long v_ptr
    cdef unsigned long long size
    res = bytearray(bytes.size)
    v_ptr = res.data
    size = res.size
    memcpy(<void*>v_ptr, bytes.data,size)
    return res


cdef class ByteArrayArg:
    cdef TVMFFIByteArray cdata
    cdef object py_data

    def __cinit__(self, py_data):
        if isinstance(py_data, bytes):
            py_data = bytearray(py_data)
        self.py_data = py_data
        self.cdata.data = <const char*>(py_data.data)
        self.cdata.size = py_data.size

