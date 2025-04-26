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

_CLASS_OBJECT = None
_FUNC_CONVERT_TO_OBJECT = None

def _set_class_object(cls):
    global _CLASS_OBJECT
    _CLASS_OBJECT = cls

def _set_func_convert_to_object(func):
    global _FUNC_CONVERT_TO_OBJECT
    _FUNC_CONVERT_TO_OBJECT = func


class ObjectGeneric:
    """Base class for all classes that can be converted to object."""

    def asobject(self):
        """Convert value to object"""
        raise NotImplementedError()


cdef class Object:
    """Base class of all TVM FFI objects.
    """
    cdef void* chandle

    def __dealloc__(self):
        CHECK_CALL(TVMFFIObjectFree(self.chandle))

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
            (<Object>fconstructor).chandle, args, &chandle)
        self.chandle = chandle

    def same_as(self, other):
        """Check object identity.

        Parameters
        ----------
        other : object
            The other object to compare against.

        Returns
        -------
        result : bool
             The comparison result.
        """
        if not isinstance(other, Object):
            return False
        return self.chandle == (<Object>other).chandle


class PyNativeObject:
    """Base class of all TVM objects that also subclass python's builtin types."""
    __slots__ = []

    def __init_object_by_constructor__(self, fconstructor, *args):
        """Initialize the internal tvm_ffi_object by calling constructor function.

        Parameters
        ----------
        fconstructor : Function
            Constructor function.

        args: list of objects
            The arguments to the constructor

        Note
        ----
        We have a special calling convention to call constructor functions.
        So the return object is directly set into the object
        """
        obj = _CLASS_OBJECT.__new__(_CLASS_OBJECT)
        obj.__init_handle_by_constructor__(fconstructor, *args)
        self.__tvm_ffi_object__ = obj


"""Maps object type index to its constructor"""
cdef list OBJECT_TYPE = []
"""Maps object type to its type index"""
cdef dict OBJECT_INDEX = {}


def _register_object_by_index(int index, object cls):
    """register object class"""
    global OBJECT_TYPE
    while len(OBJECT_TYPE) <= index:
        OBJECT_TYPE.append(None)
    OBJECT_TYPE[index] = cls
    OBJECT_INDEX[cls] = index


def _object_type_key_to_index(str type_key):
    """get the type index of object class"""
    cdef int32_t tidx
    CHECK_CALL(TVMFFITypeKeyToIndex(c_str(type_key), &tidx))
    return tidx


cdef inline object make_ret_object(TVMFFIAny result):
    global OBJECT_TYPE
    cdef int32_t tindex
    cdef object cls
    tindex = result.type_index

    if tindex < len(OBJECT_TYPE):
        cls = OBJECT_TYPE[tindex]
        if cls is not None:
            if issubclass(cls, PyNativeObject):
                obj = Object.__new__(Object)
                (<Object>obj).chandle = result.v_obj
                return cls.__from_tvm_object__(cls, obj)
            obj = cls.__new__(cls)
        else:
            obj = _CLASS_OBJECT.__new__(_CLASS_OBJECT)
    else:
        obj = _CLASS_OBJECT.__new__(_CLASS_OBJECT)
    (<Object>obj).chandle = result.v_obj
    return obj


_set_class_object(Object)
