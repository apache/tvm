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
import ctypes

from tvm._ffi.base import _FFI_MODE, _RUNTIME_ONLY, check_call, _LIB, c_str
from .. import _api_internal

try:
    # pylint: disable=wrong-import-position,unused-import
    if _FFI_MODE == "ctypes":
        raise ImportError()
    from tvm._ffi._cy3.core import _set_class_object, _set_class_object_generic
    from tvm._ffi._cy3.core import ObjectBase
except (RuntimeError, ImportError):
    # pylint: disable=wrong-import-position,unused-import
    from tvm._ffi._ctypes.packed_func import _set_class_object, _set_class_object_generic
    from tvm._ffi._ctypes.object import ObjectBase


def _new_object(cls):
    """Helper function for pickle"""
    return cls.__new__(cls)


class Object(ObjectBase):
    """Base class for all tvm's runtime objects."""
    def __repr__(self):
        return _api_internal._format_str(self)

    def __dir__(self):
        fnames = _api_internal._NodeListAttrNames(self)
        size = fnames(-1)
        return [fnames(i) for i in range(size)]

    def __getattr__(self, name):
        try:
            return _api_internal._NodeGetAttr(self, name)
        except AttributeError:
            raise AttributeError(
                "%s has no attribute %s" % (str(type(self)), name))

    def __hash__(self):
        return _api_internal._raw_ptr(self)

    def __eq__(self, other):
        return self.same_as(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __reduce__(self):
        cls = type(self)
        return (_new_object, (cls, ), self.__getstate__())

    def __getstate__(self):
        handle = self.handle
        if handle is not None:
            return {'handle': _api_internal._save_json(self)}
        return {'handle': None}

    def __setstate__(self, state):
        # pylint: disable=assigning-non-slot
        handle = state['handle']
        if handle is not None:
            json_str = handle
            other = _api_internal._load_json(json_str)
            self.handle = other.handle
            other.handle = None
        else:
            self.handle = None

    def same_as(self, other):
        """check object identity equality"""
        if not isinstance(other, Object):
            return False
        return self.__hash__() == other.__hash__()


_set_class_object(Object)
