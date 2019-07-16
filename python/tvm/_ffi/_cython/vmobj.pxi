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

"""Maps object type to its constructor"""
OBJECT_TYPE = []

def _register_object(int index, object cls):
    """register node class"""
    while len(OBJECT_TYPE) <= index:
        OBJECT_TYPE.append(None)
    OBJECT_TYPE[index] = cls


cdef inline object make_ret_object(void* chandle):
    global OBJECT_TYPE
    cdef int tag
    cdef list object_type
    cdef object cls
    cdef object handle
    object_type = OBJECT_TYPE
    handle = ctypes_handle(chandle)
    CALL(TVMGetObjectTag(chandle, &tag))
    if tag < len(object_type):
        cls = object_type[tag]
        if cls is not None:
            obj = cls(handle)
        else:
            obj = ObjectBase(handle)
    else:
        obj = ObjectBase(handle)
    return obj


cdef class ObjectBase:
    cdef ObjectHandle chandle

    cdef inline _set_handle(self, handle):
        if handle is None:
            self.chandle = NULL
        else:
            self.chandle = c_handle(handle)

    property handle:
        def __get__(self):
            if self.chandle == NULL:
                return None
            else:
                return ctypes.cast(<unsigned long long>self.chandle, ctypes.c_void_p)
        def __set__(self, value):
            self._set_handle(value)

    def __init__(self, handle):
        self._set_handle(handle)
