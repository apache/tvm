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

import pytest
from tvm_ffi.access_path import AccessPath, AccessKind


def test_root_path():
    root = AccessPath.root()
    assert isinstance(root, AccessPath)
    steps = root.to_steps()
    assert len(steps) == 0
    assert root == AccessPath.root()


def test_path_attr():
    path = AccessPath.root().attr("foo")
    assert isinstance(path, AccessPath)
    steps = path.to_steps()
    assert len(steps) == 1
    assert steps[0].kind == AccessKind.ATTR
    assert steps[0].key == "foo"
    assert path.parent == AccessPath.root()


def test_path_array_item():
    path = AccessPath.root().array_item(2)
    assert isinstance(path, AccessPath)
    steps = path.to_steps()
    assert len(steps) == 1
    assert steps[0].kind == AccessKind.ARRAY_ITEM
    assert steps[0].key == 2
    assert path.parent == AccessPath.root()


def test_path_missing_array_element():
    path = AccessPath.root().array_item_missing(2)
    assert isinstance(path, AccessPath)
    steps = path.to_steps()
    assert len(steps) == 1
    assert steps[0].kind == AccessKind.ARRAY_ITEM_MISSING
    assert steps[0].key == 2
    assert path.parent == AccessPath.root()


def test_path_map_item():
    path = AccessPath.root().map_item("foo")
    assert isinstance(path, AccessPath)
    steps = path.to_steps()
    assert len(steps) == 1
    assert steps[0].kind == AccessKind.MAP_ITEM
    assert steps[0].key == "foo"
    assert path.parent == AccessPath.root()


def test_path_missing_map_item():
    path = AccessPath.root().map_item_missing("foo")
    assert isinstance(path, AccessPath)
    steps = path.to_steps()
    assert len(steps) == 1
    assert steps[0].kind == AccessKind.MAP_ITEM_MISSING
    assert steps[0].key == "foo"
    assert path.parent == AccessPath.root()


def test_path_is_prefix_of():
    # Root is prefix of root
    assert AccessPath.root().is_prefix_of(AccessPath.root())

    # Root is prefix of any path
    assert AccessPath.root().is_prefix_of(AccessPath.root().attr("foo"))

    # Non-root is not prefix of root
    assert not AccessPath.root().attr("foo").is_prefix_of(AccessPath.root())

    # Path is prefix of itself
    assert AccessPath.root().attr("foo").is_prefix_of(AccessPath.root().attr("foo"))

    # Different attrs are not prefixes of each other
    assert not AccessPath.root().attr("bar").is_prefix_of(AccessPath.root().attr("foo"))

    # Shorter path is prefix of longer path with same start
    assert AccessPath.root().attr("foo").is_prefix_of(AccessPath.root().attr("foo").array_item(2))

    # Longer path is not prefix of shorter path
    assert (
        not AccessPath.root().attr("foo").array_item(2).is_prefix_of(AccessPath.root().attr("foo"))
    )

    # Different paths are not prefixes
    assert (
        not AccessPath.root().attr("foo").is_prefix_of(AccessPath.root().attr("bar").array_item(2))
    )


def test_path_equal():
    # Root equals root
    assert AccessPath.root() == AccessPath.root()

    # Root does not equal non-root paths
    assert not (AccessPath.root() == AccessPath.root().attr("foo"))

    # Non-root does not equal root
    assert not (AccessPath.root().attr("foo") == AccessPath.root())

    # Path equals itself
    assert AccessPath.root().attr("foo") == AccessPath.root().attr("foo")

    # Different attrs are not equal
    assert not (AccessPath.root().attr("bar") == AccessPath.root().attr("foo"))

    # Shorter path does not equal longer path
    assert not (AccessPath.root().attr("foo") == AccessPath.root().attr("foo").array_item(2))

    # Longer path does not equal shorter path
    assert not (AccessPath.root().attr("foo").array_item(2) == AccessPath.root().attr("foo"))

    # Different paths are not equal
    assert not (AccessPath.root().attr("foo") == AccessPath.root().attr("bar").array_item(2))
