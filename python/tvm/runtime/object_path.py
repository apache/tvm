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

"""
ObjectPath class that represents a path from a root object to one of its descendants
via attribute access, array indexing etc.
"""

from typing import Optional

import tvm._ffi
from tvm.runtime import Object
from . import _ffi_node_api


__all__ = (
    "ObjectPath",
    "RootPath",
    "AttributeAccessPath",
    "UnknownAttributeAccessPath",
    "ArrayIndexPath",
    "MissingArrayElementPath",
    "MapValuePath",
    "MissingMapEntryPath",
    "ObjectPathPair",
)


@tvm._ffi.register_object("ObjectPath")
class ObjectPath(Object):
    """
    Path to an object from some root object.
    """

    def __init__(self) -> None:
        super().__init__()
        raise ValueError(
            "ObjectPath can't be initialized directly. "
            "Use ObjectPath.root() to create a path to the root object"
        )

    @staticmethod
    def root(root_name: Optional[str] = None) -> "ObjectPath":
        return _ffi_node_api.ObjectPathRoot(root_name)

    def __eq__(self, other):
        return _ffi_node_api.ObjectPathEqual(self, other)

    def __ne__(self, other):
        return not _ffi_node_api.ObjectPathEqual(self, other)

    @property
    def parent(self) -> "ObjectPath":
        return _ffi_node_api.ObjectPathGetParent(self)

    def __len__(self) -> int:
        return _ffi_node_api.ObjectPathLength(self)

    def get_prefix(self, length) -> "ObjectPath":
        return _ffi_node_api.ObjectPathGetPrefix(self, length)

    def is_prefix_of(self, other) -> "ObjectPath":
        return _ffi_node_api.ObjectPathIsPrefixOf(self, other)

    def attr(self, attr_key) -> "ObjectPath":
        return _ffi_node_api.ObjectPathAttr(self, attr_key)

    def array_index(self, index) -> "ObjectPath":
        return _ffi_node_api.ObjectPathArrayIndex(self, index)

    def missing_array_element(self, index) -> "ObjectPath":
        return _ffi_node_api.ObjectPathMissingArrayElement(self, index)

    def map_value(self, key) -> "ObjectPath":
        return _ffi_node_api.ObjectPathMapValue(self, tvm.runtime.convert(key))

    def missing_map_entry(self) -> "ObjectPath":
        return _ffi_node_api.ObjectPathMissingMapEntry(self)

    __hash__ = Object.__hash__


@tvm._ffi.register_object("RootPath")
class RootPath(ObjectPath):
    pass


@tvm._ffi.register_object("AttributeAccessPath")
class AttributeAccessPath(ObjectPath):
    pass


@tvm._ffi.register_object("UnknownAttributeAccessPath")
class UnknownAttributeAccessPath(ObjectPath):
    pass


@tvm._ffi.register_object("ArrayIndexPath")
class ArrayIndexPath(ObjectPath):
    pass


@tvm._ffi.register_object("MissingArrayElementPath")
class MissingArrayElementPath(ObjectPath):
    pass


@tvm._ffi.register_object("MapValuePath")
class MapValuePath(ObjectPath):
    pass


@tvm._ffi.register_object("MissingMapEntryPath")
class MissingMapEntryPath(ObjectPath):
    pass


@tvm._ffi.register_object("ObjectPathPair")
class ObjectPathPair(Object):
    """
    Pair of ObjectPaths, one for each object being tested for structural equality.
    """

    @property
    def lhs_path(self) -> ObjectPath:
        return _ffi_node_api.ObjectPathPairLhsPath(self)

    @property
    def rhs_path(self) -> ObjectPath:
        return _ffi_node_api.ObjectPathPairRhsPath(self)
