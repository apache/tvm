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
# pylint: disable=invalid-name
"""Runtime Object api"""
import ctypes
from ..base import _LIB, check_call
from .types import ArgTypeCode, RETURN_SWITCH, C_TO_PY_ARG_SWITCH, _wrap_arg_func
from .ndarray import _register_ndarray, NDArrayBase


ObjectHandle = ctypes.c_void_p
__init_by_constructor__ = None

"""Maps object type index to its constructor"""
OBJECT_TYPE = {}

"""Maps object type to its type index"""
OBJECT_INDEX = {}

_CLASS_OBJECT = None


def _set_class_object(object_class):
    global _CLASS_OBJECT
    _CLASS_OBJECT = object_class


def _register_object(index, cls):
    """register object class"""
    if issubclass(cls, NDArrayBase):
        _register_ndarray(index, cls)
        return
    OBJECT_TYPE[index] = cls
    OBJECT_INDEX[cls] = index


def _get_object_type_index(cls):
    """get the type index of object class"""
    return OBJECT_INDEX.get(cls)


def _return_object(x):
    handle = x.v_handle
    if not isinstance(handle, ObjectHandle):
        handle = ObjectHandle(handle)
    tindex = ctypes.c_uint()
    check_call(_LIB.TVMObjectGetTypeIndex(handle, ctypes.byref(tindex)))
    cls = OBJECT_TYPE.get(tindex.value, _CLASS_OBJECT)

    # Handle return values that subclass from both TVM objects and
    # python native objects (e.g. runtime.String, a subclass of str).
    if issubclass(cls, PyNativeObject):
        obj = _CLASS_OBJECT.__new__(_CLASS_OBJECT)
        obj.handle = handle
        return cls.__from_tvm_object__(cls, obj)

    # Avoid calling __init__ of cls, instead directly call __new__
    # This allows child class to implement their own __init__
    obj = cls.__new__(cls)
    obj.handle = handle

    # Handle return values that must be converted from the TVM object
    # to a python native object.  This should be used in cases where
    # subclassing the python native object is forbidden.  For example,
    # `runtime.BoxBool` cannot be a subclass of `bool`, as `bool` does
    # not allow any subclasses.
    #
    # The `hasattr` check is done on the object's class, not the
    # object itself, to avoid edge cases that can result in invalid
    # error messages.  If a C++ `LOG(FATAL) << nested_obj;` statement
    # requires C++ to Python conversions in order to print
    # `nested_obj`, then the `AttributeError` used internally by
    # `hasattr` may overwrite the text being collected by
    # `LOG(FATAL)`.  By checking for the method on the class instead
    # of the instance, we avoid throwing the `AttributeError`.
    # if hasattr(type(obj), "__into_pynative_object__"):
    #     return obj.__into_pynative_object__()

    return obj


RETURN_SWITCH[ArgTypeCode.OBJECT_HANDLE] = _return_object
C_TO_PY_ARG_SWITCH[ArgTypeCode.OBJECT_HANDLE] = _wrap_arg_func(
    _return_object, ArgTypeCode.OBJECT_HANDLE
)

C_TO_PY_ARG_SWITCH[ArgTypeCode.OBJECT_RVALUE_REF_ARG] = _wrap_arg_func(
    _return_object, ArgTypeCode.OBJECT_RVALUE_REF_ARG
)


class PyNativeObject:
    """Base class of all TVM objects that also subclass python's builtin types."""

    __slots__ = []

    def __init_tvm_object_by_constructor__(self, fconstructor, *args):
        """Initialize the internal tvm_object by calling constructor function.

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
        # pylint: disable=assigning-non-slot
        obj = _CLASS_OBJECT.__new__(_CLASS_OBJECT)
        obj.__init_handle_by_constructor__(fconstructor, *args)
        self.__tvm_object__ = obj


class ObjectBase(object):
    """Base object for all object types"""

    __slots__ = ["handle"]

    def __del__(self):
        if _LIB is not None:
            try:
                handle = self.handle
            except AttributeError:
                return

            check_call(_LIB.TVMObjectFree(handle))

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
        # assign handle first to avoid error raising
        # pylint: disable=not-callable
        self.handle = None
        handle = __init_by_constructor__(fconstructor, args)
        if not isinstance(handle, ObjectHandle):
            handle = ObjectHandle(handle)
        self.handle = handle

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
        if not isinstance(other, ObjectBase):
            return False
        if self.handle is None:
            return other.handle is None
        return self.handle.value == other.handle.value
