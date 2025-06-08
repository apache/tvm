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


def __object_repr__(obj):
    """Object repr function that can be overridden by assigning to it"""
    return type(obj).__name__ + "(" + obj.__ctypes_handle__().value + ")"


def __object_save_json__(obj):
    """Object repr function that can be overridden by assigning to it"""
    raise NotImplementedError("JSON serialization depends on downstream init")


def __object_load_json__(json_str):
    """Object repr function that can be overridden by assigning to it"""
    raise NotImplementedError("JSON serialization depends on downstream init")


def __object_dir__(obj):
    """Object dir function that can be overridden by assigning to it"""
    return []


def __object_getattr__(obj, name):
    """Object getattr function that can be overridden by assigning to it"""
    raise AttributeError()


def _new_object(cls):
    """Helper function for pickle"""
    return cls.__new__(cls)


class ObjectGeneric:
    """Base class for all classes that can be converted to object."""

    def asobject(self):
        """Convert value to object"""
        raise NotImplementedError()


class ObjectRValueRef:
    """Represent an RValue ref to an object that can be moved.

    Parameters
    ----------
    obj : tvm.runtime.Object
        The object that this value refers to
    """

    __slots__ = ["obj"]

    def __init__(self, obj):
        self.obj = obj


cdef class Object:
    """Base class of all TVM FFI objects.
    """
    cdef void* chandle

    def __cinit__(self):
        # initialize chandle to NULL to avoid leak in
        # case of error before chandle is set
        self.chandle = NULL

    def __dealloc__(self):
        if self.chandle != NULL:
            CHECK_CALL(TVMFFIObjectFree(self.chandle))
            self.chandle = NULL

    def __ctypes_handle__(self):
        return ctypes_handle(self.chandle)

    def __chandle__(self):
        cdef uint64_t chandle = <uint64_t>self.chandle
        return chandle

    def __reduce__(self):
        cls = type(self)
        return (_new_object, (cls,), self.__getstate__())

    def __getstate__(self):
        if not self.__chandle__() == 0:
            # need to explicit convert to str in case String
            # returned and triggered another infinite recursion in get state
            return {"handle": str(__object_save_json__(self))}
        return {"handle": None}

    def __setstate__(self, state):
        # pylint: disable=assigning-non-slot, assignment-from-no-return
        handle = state["handle"]
        if handle is not None:
            self.__init_handle_by_constructor__(__object_load_json__, handle)
        else:
            self.chandle = NULL

    def __getattr__(self, name):
        if self.chandle == NULL:
            raise AttributeError(f"{type(self)} has no attribute {name}")
        try:
            return __object_getattr__(self, name)
        except AttributeError:
            raise AttributeError(f"{type(self)} has no attribute {name}")

    def __dir__(self):
        # exception safety handling for chandle=None
        if self.chandle == NULL:
            return []
        return __object_dir__(self)

    def __repr__(self):
        # exception safety handling for chandle=None
        if self.chandle == NULL:
            return type(self).__name__ + "(chandle=None)"
        return str(__object_repr__(self))

    def __eq__(self, other):
        return self.same_as(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __init_handle_by_load_json__(self, json_str):
        raise NotImplementedError("JSON serialization depends on downstream init")

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

    def __hash__(self):
        cdef uint64_t hash_value = <uint64_t>self.chandle
        return hash_value

    def _move(self):
        """Create an RValue reference to the object and mark the object as moved.

        This is a advanced developer API that can be useful when passing an
        unique reference to an Object that you no longer needed to a function.

        A unique reference can trigger copy on write optimization that avoids
        copy when we transform an object.

        Note
        ----
        All the reference of the object becomes invalid after it is moved.
        Be very careful when using this feature.

        Returns
        -------
        rvalue : The rvalue reference.
        """
        return ObjectRValueRef(self)

    def __move_handle_from__(self, other):
        """Move the handle from other to self"""
        self.chandle = (<Object>other).chandle
        (<Object>other).chandle = NULL


class PyNativeObject:
    """Base class of all TVM objects that also subclass python's builtin types."""
    __slots__ = []

    def __init_tvm_ffi_object_by_constructor__(self, fconstructor, *args):
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
    type_key_arg = ByteArrayArg(c_str(type_key))
    if TVMFFITypeKeyToIndex(type_key_arg.cptr(), &tidx) == 0:
        return tidx
    return None


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
                return cls.__from_tvm_ffi_object__(cls, obj)
            obj = cls.__new__(cls)
        else:
            obj = _CLASS_OBJECT.__new__(_CLASS_OBJECT)
    else:
        obj = _CLASS_OBJECT.__new__(_CLASS_OBJECT)
    (<Object>obj).chandle = result.v_obj
    return obj


_set_class_object(Object)
