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
import tvm
from tvm.runtime import object_path
from tvm.runtime.object_path import ObjectPath


def test_root_path():
    root = ObjectPath.root()
    assert isinstance(root, object_path.RootPath)
    assert str(root) == "<root>"
    assert len(root) == 1
    assert root == ObjectPath.root()
    assert root.parent is None


def test_named_root_path():
    root = ObjectPath.root("base_name")
    assert isinstance(root, object_path.RootPath)
    assert str(root) == "base_name"
    assert len(root) == 1
    assert root != ObjectPath.root()
    assert root == ObjectPath.root("base_name")
    assert root.parent is None


def test_path_attr():
    path = ObjectPath.root().attr("foo")
    assert isinstance(path, object_path.AttributeAccessPath)
    assert str(path) == "<root>.foo"
    assert len(path) == 2
    assert path.parent == ObjectPath.root()


def test_path_attr_unknown():
    path = ObjectPath.root().attr(None)
    assert isinstance(path, object_path.UnknownAttributeAccessPath)
    assert str(path) == "<root>.<unknown attribute>"
    assert len(path) == 2
    assert path.parent == ObjectPath.root()


def test_path_array_index():
    path = ObjectPath.root().array_index(2)
    assert isinstance(path, object_path.ArrayIndexPath)
    assert str(path) == "<root>[2]"
    assert len(path) == 2
    assert path.parent == ObjectPath.root()


def test_path_missing_array_element():
    path = ObjectPath.root().missing_array_element(2)
    assert isinstance(path, object_path.MissingArrayElementPath)
    assert str(path) == "<root>[<missing element #2>]"
    assert len(path) == 2
    assert path.parent == ObjectPath.root()


def test_path_map_value():
    path = ObjectPath.root().map_value("foo")
    assert isinstance(path, object_path.MapValuePath)
    assert str(path) == '<root>["foo"]'
    assert len(path) == 2
    assert path.parent == ObjectPath.root()


def test_path_missing_map_entry():
    path = ObjectPath.root().missing_map_entry()
    assert isinstance(path, object_path.MissingMapEntryPath)
    assert str(path) == "<root>[<missing entry>]"
    assert len(path) == 2
    assert path.parent == ObjectPath.root()


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (ObjectPath.root(), ObjectPath.root(), True),
        (ObjectPath.root(), ObjectPath.root().attr("foo"), True),
        (ObjectPath.root().attr("foo"), ObjectPath.root(), False),
        (ObjectPath.root().attr("foo"), ObjectPath.root().attr("foo"), True),
        (ObjectPath.root().attr("bar"), ObjectPath.root().attr("foo"), False),
        (ObjectPath.root().attr("foo"), ObjectPath.root().attr("foo").array_index(2), True),
        (ObjectPath.root().attr("foo").array_index(2), ObjectPath.root().attr("foo"), False),
        (ObjectPath.root().attr("foo"), ObjectPath.root().attr("bar").array_index(2), False),
    ],
)
def test_path_is_prefix_of(a, b, expected):
    assert a.is_prefix_of(b) == expected


paths_for_equality_test = [
    ObjectPath.root(),
    ObjectPath.root().attr("foo"),
    ObjectPath.root().attr("bar"),
    ObjectPath.root().array_index(3),
    ObjectPath.root().array_index(4),
    ObjectPath.root().missing_array_element(3),
    ObjectPath.root().missing_array_element(4),
    ObjectPath.root().map_value("foo"),
    ObjectPath.root().map_value("bar"),
    ObjectPath.root().missing_map_entry(),
    ObjectPath.root().attr("foo").missing_map_entry(),
]


def make_test_params_for_eq_test():
    return [
        pytest.param(idx, path, id="path{}".format(idx))
        for idx, path in enumerate(paths_for_equality_test)
    ]


@pytest.mark.parametrize("a_idx, a_path", make_test_params_for_eq_test())
@pytest.mark.parametrize("b_idx, b_path", make_test_params_for_eq_test())
def test_path_equal(a_idx, a_path, b_idx, b_path):
    expected = a_idx == b_idx
    result = a_path == b_path
    assert result == expected


def test_path_get_prefix():
    p1 = ObjectPath.root()
    p2 = p1.attr("foo")
    p3 = p2.array_index(5)

    assert p3.parent == p2
    assert p2.parent == p1
    assert p1.parent is None

    assert p2.get_prefix(1) == p1

    assert p3.get_prefix(1) == p1
    assert p3.get_prefix(2) == p2
    assert p3.get_prefix(3) == p3

    with pytest.raises(IndexError) as e:
        p3.get_prefix(0)
    assert "Prefix length must be at least 1" in str(e.value)

    with pytest.raises(IndexError) as e:
        p3.get_prefix(4)
    assert "Attempted to get a prefix longer than the path itself" in str(e.value)
