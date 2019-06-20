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

import sys
from .base import _FFI_MODE

IMPORT_EXCEPT = RuntimeError if _FFI_MODE == "cython" else ImportError

try:
    # pylint: disable=wrong-import-position
    if _FFI_MODE == "ctypes":
        raise ImportError()
    if sys.version_info >= (3, 0):
        from ._cy3.core import _set_class_object
        from ._cy3.core import ObjectBase as _ObjectBase
        from ._cy3.core import _register_object
    else:
        from ._cy2.core import _set_class_object
        from ._cy2.core import ObjectBase as _ObjectBase
        from ._cy2.core import _register_object
except IMPORT_EXCEPT:
    # pylint: disable=wrong-import-position
    from ._ctypes.function import _set_class_object
    from ._ctypes.vmobj import ObjectBase as _ObjectBase
    from ._ctypes.vmobj import _register_object


class ObjectTag(object):
    """Type code used in API calls"""
    TENSOR = 0
    CLOSURE = 1
    DATATYPE = 2


class Object(_ObjectBase):
    """The VM Object used in Relay virtual machine."""


def register_object(cls):
    _register_object(cls.tag, cls)
    return cls


_set_class_object(Object)
