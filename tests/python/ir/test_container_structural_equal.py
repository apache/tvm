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
import tvm.testing
from tvm.ir.base import get_first_structural_mismatch
from tvm.runtime import ObjectPath


def get_first_mismatch_ensure_symmetry(a, b):
    mismatch = get_first_structural_mismatch(a, b)
    mismatch_swapped = get_first_structural_mismatch(b, a)

    if mismatch is None and mismatch_swapped is None:
        return None

    if (
        mismatch is None
        or mismatch_swapped is None
        or mismatch[0] != mismatch_swapped[1]
        or mismatch[1] != mismatch_swapped[0]
    ):
        raise AssertionError(
            "get_first_structural_mismatch(a, b) and get_first_structural_mismatch(b, a) returned"
            " inconsistent results '{}' and '{}' for a='{}', b='{}'".format(
                mismatch, mismatch_swapped, a, b
            )
        )

    a_path, b_path = mismatch
    b_path_swapped, a_path_swapped = mismatch_swapped
    assert a_path == a_path_swapped
    assert b_path == b_path_swapped

    return mismatch


@pytest.mark.parametrize(
    "a, b, expected_a_path, expected_b_path",
    [
        (
            [1, 2, 3],
            [1, 4, 3],
            ObjectPath.root().array_index(1),
            ObjectPath.root().array_index(1),
        ),
        (
            [1, 2, 3],
            [10, 2, 30],
            ObjectPath.root().array_index(0),
            ObjectPath.root().array_index(0),
        ),
        (
            [1, 3, 4],
            [1, 2, 3, 4],
            ObjectPath.root().array_index(1),
            ObjectPath.root().array_index(1),
        ),
        (
            [1, 2, 3],
            [1, 2, 3, 4],
            ObjectPath.root().missing_array_element(3),
            ObjectPath.root().array_index(3),
        ),
        (
            [],
            [1],
            ObjectPath.root().missing_array_element(0),
            ObjectPath.root().array_index(0),
        ),
    ],
)
def test_array_structural_mismatch(a, b, expected_a_path, expected_b_path):
    a = tvm.runtime.convert(a)
    b = tvm.runtime.convert(b)
    a_path, b_path = get_first_mismatch_ensure_symmetry(a, b)
    assert a_path == expected_a_path
    assert b_path == expected_b_path


@pytest.mark.parametrize(
    "contents",
    [
        [],
        [1],
        [1, 2, 3],
    ],
)
def test_array_structural_equal_to_self(contents):
    a = tvm.runtime.convert(list(contents))
    b = tvm.runtime.convert(list(contents))
    assert get_first_mismatch_ensure_symmetry(a, b) is None


@pytest.mark.parametrize(
    "contents",
    [
        [],
        [1],
        [1, 2, 3],
    ],
)
def test_shape_tuple_structural_equal_to_self(contents):
    a = tvm.runtime.ShapeTuple(list(contents))
    b = tvm.runtime.ShapeTuple(list(contents))
    assert get_first_mismatch_ensure_symmetry(a, b) is None


@pytest.mark.parametrize(
    "contents",
    [
        {},
        {"a": 1, "b": 2},
        {"a": True, "b": False},
    ],
)
def test_string_map_structural_equal_to_self(contents):
    a = tvm.runtime.convert({**contents})
    b = tvm.runtime.convert({**contents})
    assert get_first_mismatch_ensure_symmetry(a, b) is None


@pytest.mark.parametrize(
    "a, b, expected_a_path, expected_b_path",
    [
        (
            dict(a=3, b=4),
            dict(a=3, b=5),
            ObjectPath.root().map_value("b"),
            ObjectPath.root().map_value("b"),
        ),
        (
            dict(a=3, b=4),
            dict(a=3, b=4, c=5),
            ObjectPath.root().missing_map_entry(),
            ObjectPath.root().map_value("c"),
        ),
    ],
)
def test_string_map_structural_mismatch(a, b, expected_a_path, expected_b_path):
    a = tvm.runtime.convert(a)
    b = tvm.runtime.convert(b)
    a_path, b_path = get_first_mismatch_ensure_symmetry(a, b)
    assert a_path == expected_a_path
    assert b_path == expected_b_path


@pytest.mark.parametrize(
    "contents",
    [
        dict(),
        dict(a=1),
        dict(a=3, b=4, c=5),
    ],
)
def test_string_structural_equal_to_self(contents):
    a = tvm.runtime.convert(dict(contents))
    b = tvm.runtime.convert(dict(contents))
    assert get_first_mismatch_ensure_symmetry(a, b) is None


# The behavior of structural equality for maps with non-string keys is fairly specific
# to IR variables because it assumes that map keys have been "mapped" using
# `SEqualReducer::FreeVarEqualImpl()`. So we leave this case to TIR tests.


if __name__ == "__main__":
    tvm.testing.main()
