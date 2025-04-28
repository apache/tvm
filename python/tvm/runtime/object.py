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
# pylint: disable=invalid-name, unused-import
"""Runtime Object API"""

import tvm.ffi.core

from . import _ffi_api, _ffi_node_api


def _new_object(cls):
    """Helper function for pickle"""
    return cls.__new__(cls)


class Object(tvm.ffi.core.Object):
    """Base class for all tvm's runtime objects."""

    __slots__ = []

    def __repr__(self):
        return _ffi_node_api.AsRepr(self)

    def legacy_repr(self):
        return _ffi_node_api.AsLegacyRepr(self)

    def __dir__(self):
        class_names = dir(self.__class__)
        fnames = _ffi_node_api.NodeListAttrNames(self)
        size = fnames(-1)
        return sorted([fnames(i) for i in range(size)] + class_names)

    def __getattr__(self, name):
        try:
            return _ffi_node_api.NodeGetAttr(self, name)
        except AttributeError:
            raise AttributeError(f"{type(self)} has no attribute {name}") from None

    def __hash__(self):
        return _ffi_api.ObjectPtrHash(self)

    def __eq__(self, other):
        return self.same_as(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __reduce__(self):
        cls = type(self)
        return (_new_object, (cls,), self.__getstate__())

    def __getstate__(self):
        if not self._handle_is_none():
            # need to explicit convert to str in case String
            # returned and triggered another infinite recursion in get state
            return {"handle": str(_ffi_node_api.SaveJSON(self))}
        return {"handle": None}

    def __setstate__(self, state):
        # pylint: disable=assigning-non-slot, assignment-from-no-return
        handle = state["handle"]
        if handle is not None:
            self.__init_handle_by_constructor__(_ffi_node_api.LoadJSON, handle)


tvm.ffi.core._set_class_object(Object)
