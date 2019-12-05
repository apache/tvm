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
from __future__ import absolute_import

import ctypes
from ..base import _LIB, check_call
from .types import TypeCode, RETURN_SWITCH, C_TO_PY_ARG_SWITCH, _wrap_arg_func
from ..node_generic import _set_class_node_base


ObjectHandle = ctypes.c_void_p
__init_by_constructor__ = None

"""Maps object type to its constructor"""
OBJECT_TYPE = {}

_CLASS_NODE = None

def _set_class_node(node_class):
    global _CLASS_NODE
    _CLASS_NODE = node_class


def _register_object(index, cls):
    """register object class"""
    OBJECT_TYPE[index] = cls


def _return_object(x):
    handle = x.v_handle
    if not isinstance(handle, ObjectHandle):
        handle = ObjectHandle(handle)
    tindex = ctypes.c_uint()
    check_call(_LIB.TVMObjectGetTypeIndex(handle, ctypes.byref(tindex)))
    cls = OBJECT_TYPE.get(tindex.value, _CLASS_NODE)
    # Avoid calling __init__ of cls, instead directly call __new__
    # This allows child class to implement their own __init__
    obj = cls.__new__(cls)
    obj.handle = handle
    return obj

RETURN_SWITCH[TypeCode.OBJECT_HANDLE] = _return_object
C_TO_PY_ARG_SWITCH[TypeCode.OBJECT_HANDLE] = _wrap_arg_func(
    _return_object, TypeCode.OBJECT_HANDLE)


class ObjectBase(object):
    """Base object for all object types"""
    __slots__ = ["handle"]

    def __del__(self):
        if _LIB is not None:
            check_call(_LIB.TVMObjectFree(self.handle))

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
        self.handle = None
        handle = __init_by_constructor__(fconstructor, args)
        if not isinstance(handle, ObjectHandle):
            handle = ObjectHandle(handle)
        self.handle = handle


_set_class_node_base(ObjectBase)
