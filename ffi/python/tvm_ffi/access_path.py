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
"""Access path classes."""

from enum import IntEnum
from typing import List, Any
from . import core
from .registry import register_object


class AccessKind(IntEnum):
    ATTR = 0
    ARRAY_ITEM = 1
    MAP_ITEM = 2
    ATTR_MISSING = 3
    ARRAY_ITEM_MISSING = 4
    MAP_ITEM_MISSING = 5


@register_object("ffi.reflection.AccessStep")
class AccessStep(core.Object):
    """Access step container"""


@register_object("ffi.reflection.AccessPath")
class AccessPath(core.Object):
    """Access path container"""

    def __init__(self) -> None:
        super().__init__()
        raise ValueError(
            "AccessPath can't be initialized directly. "
            "Use AccessPath.root() to create a path to the root object"
        )

    @staticmethod
    def root() -> "AccessPath":
        """Create a root access path"""
        return AccessPath._root()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, AccessPath):
            return False
        return self._path_equal(other)

    def __ne__(self, other: Any) -> bool:
        if not isinstance(other, AccessPath):
            return True
        return not self._path_equal(other)

    def is_prefix_of(self, other: "AccessPath") -> bool:
        """Check if this access path is a prefix of another access path

        Parameters
        ----------
        other : AccessPath
            The access path to check if it is a prefix of this access path

        Returns
        -------
        bool
            True if this access path is a prefix of the other access path, False otherwise
        """
        return self._is_prefix_of(other)

    def attr(self, attr_key: str) -> "AccessPath":
        """Create an access path to the attribute of the current object

        Parameters
        ----------
        attr_key : str
            The key of the attribute to access

        Returns
        -------
        AccessPath
            The extended access path
        """
        return self._attr(attr_key)

    def attr_missing(self, attr_key: str) -> "AccessPath":
        """Create an access path that indicate an attribute is missing

        Parameters
        ----------
        attr_key : str
            The key of the attribute to access

        Returns
        -------
        AccessPath
            The extended access path
        """
        return self._attr_missing(attr_key)

    def array_item(self, index: int) -> "AccessPath":
        """Create an access path to the item of the current array

        Parameters
        ----------
        index : int
            The index of the item to access

        Returns
        -------
        AccessPath
            The extended access path
        """
        return self._array_item(index)

    def array_item_missing(self, index: int) -> "AccessPath":
        """Create an access path that indicate an array item is missing

        Parameters
        ----------
        index : int
            The index of the item to access

        Returns
        -------
        AccessPath
            The extended access path
        """
        return self._array_item_missing(index)

    def map_item(self, key: Any) -> "AccessPath":
        """Create an access path to the item of the current map

        Parameters
        ----------
        key : Any
            The key of the item to access

        Returns
        -------
        AccessPath
            The extended access path
        """
        return self._map_item(key)

    def map_item_missing(self, key: Any) -> "AccessPath":
        """Create an access path that indicate a map item is missing

        Parameters
        ----------
        key : Any
            The key of the item to access

        Returns
        -------
        AccessPath
            The extended access path
        """
        return self._map_item_missing(key)

    def to_steps(self) -> List["AccessStep"]:
        """Convert the access path to a list of access steps

        Returns
        -------
        List[AccessStep]
            The list of access steps
        """
        return self._to_steps()

    __hash__ = core.Object.__hash__
