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
from ..node_generic import _set_class_node_base

OBJECT_TYPE = []

def _register_object(int index, object cls):
    """register object class"""
    while len(OBJECT_TYPE) <= index:
        OBJECT_TYPE.append(None)
    OBJECT_TYPE[index] = cls


cdef inline object make_ret_object(void* chandle):
    global OBJECT_TYPE
    global _CLASS_NODE
    cdef unsigned tindex
    cdef list object_type
    cdef object cls
    cdef object handle
    object_type = OBJECT_TYPE
    handle = ctypes_handle(chandle)
    CALL(TVMObjectGetTypeIndex(chandle, &tindex))
    if tindex < len(object_type):
        cls = object_type[tindex]
        if cls is not None:
            obj = cls.__new__(cls)
        else:
            # default use node base class
            # TODO(tqchen) change to object after Node unifies with Object
            obj = _CLASS_NODE.__new__(_CLASS_NODE)
    else:
        obj = _CLASS_NODE.__new__(_CLASS_NODE)
    (<ObjectBase>obj).chandle = chandle
    return obj


cdef class ObjectBase:
    cdef void* chandle

    cdef inline _set_handle(self, handle):
        cdef unsigned long long ptr
        if handle is None:
            self.chandle = NULL
        else:
            ptr = handle.value
            self.chandle = <void*>(ptr)

    property handle:
        def __get__(self):
            if self.chandle == NULL:
                return None
            else:
                return ctypes_handle(self.chandle)

        def __set__(self, value):
            self._set_handle(value)

    def __dealloc__(self):
        CALL(TVMObjectFree(self.chandle))

    def __init_handle_by_constructor__(self, fconstructor, *args):
        """Initialize the handle by calling constructor function.

        Parameters
        ----------
        fconstructor : Function
            Constructor function.

        args: list of objects
            The arguments to the constructor

        Note
        ----
        We have a special calling convention to call constructor functions.
        So the return handle is directly set into the Node object
        instead of creating a new Node.
        """
        # avoid error raised during construction.
        self.chandle = NULL
        cdef void* chandle
        ConstructorCall(
            (<FunctionBase>fconstructor).chandle,
            kObjectHandle, args, &chandle)
        self.chandle = chandle


_set_class_node_base(ObjectBase)
