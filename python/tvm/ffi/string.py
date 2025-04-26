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

from . import core
from . import registry

__all__ = ["String", "Bytes"]


@registry.register_object("object.String")
class String(str, core.PyNativeObject):
    """String object that is possibly returned by FFI call.

    Note
    ----
    This class subclasses str so it can be directly treated as str.
    There is no need to construct this object explicitly.
    """

    __slots__ = ["__tvm_ffi_object__"]

    # pylint: disable=no-self-argument
    def __from_tvm_object__(cls, obj):
        """Construct from a given tvm object."""
        content = core._string_obj_get_py_str(obj)
        val = str.__new__(cls, content)
        val.__tvm_ffi_object__ = obj
        return val


@registry.register_object("object.Bytes")
class Bytes(bytes, core.PyNativeObject):
    """Bytes object that is possibly returned by FFI call.

    Note
    ----
    This class subclasses bytes so it can be directly treated as bytes.
    There is no need to construct this object explicitly.
    """

    # pylint: disable=no-self-argument
    def __from_tvm_object__(cls, obj):
        """Construct from a given tvm object."""
        content = core._bytes_obj_get_py_bytes(obj)
        val = bytes.__new__(cls, content)
        val.__tvm_ffi_object__ = obj
        return val
