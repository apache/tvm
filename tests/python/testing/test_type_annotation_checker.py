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
"""Test type checker based on python's type annotations"""

import sys
from typing import Dict, List, Tuple, Union, Callable

import pytest
import _pytest

import tvm
from tvm.tir.schedule._type_checker import type_checked


def int_func(x: int) -> int:
    return 2 * x


def str_func(x: str) -> str:
    return 2 * x


test_cases = [
    {
        "type_annotation": int,
        "positive_cases": [5],
        "negative_cases": ["5"],
    },
    {
        "type_annotation": List[int],
        "positive_cases": [
            [5],
            [],
            # Tuples are allowed to be used as lists, because both are
            # represented in FFI as tvm::runtime::Array.
            (1, 2, 3),
        ],
        "negative_cases": [
            None,
            5,
            ["5"],
        ],
    },
    {
        "type_annotation": Dict[str, int],
        "positive_cases": [
            {"key1": 0, "key2": 1, "key3": -1},
        ],
        "negative_cases": [None, [1], {1: "1"}],
    },
    {
        "type_annotation": Tuple[int],
        "positive_cases": [
            (5,),
        ],
        "negative_cases": [
            None,
            (1, 2, 3),
            [1],
            5,
            ["5"],
        ],
    },
    {
        "type_annotation": Tuple[str, int],
        "positive_cases": [
            ("x", 5),
        ],
        "negative_cases": [
            42,
            ("x", 5, 6),
            ("x", 5, "y"),
            ("x", 5.0),
            (None, 5),
        ],
    },
    {
        "type_annotation": Union[str, int],
        "positive_cases": [
            "x",
            5,
        ],
        "negative_cases": [
            5.0,
            ("x", 5, 6),
            None,
        ],
    },
    {
        "type_annotation": Callable,
        "positive_cases": [str_func, int_func],
        "negative_cases": [
            None,
            "x",
            42,
        ],
    },
    {
        "type_annotation": Callable[[int], int],
        "positive_cases": [int_func],
        "negative_cases": [
            None,
            "x",
            42,
            pytest.param(
                str_func,
                marks=pytest.mark.xfail(
                    reason="Signature of Callable arguments not currently checked"
                ),
            ),
        ],
    },
]


def make_parametrization(type_annotation, case):
    if isinstance(case, _pytest.mark.structures.ParameterSet):
        marks = case.marks
        (case,) = case.values
    else:
        marks = []

    try:
        annotation_name = type_annotation.__name__
    except AttributeError:
        annotation_name = str(type_annotation).replace("typing.", "")

    if hasattr(case, "__name__"):
        case_name = case.__name__
    else:
        case_name = str(case)

    name = f"{annotation_name}, {case_name}"

    return pytest.param(type_annotation, case, marks=marks, id=name)


positive_cases = [
    make_parametrization(config["type_annotation"], case)
    for config in test_cases
    for case in config["positive_cases"]
]

negative_cases = [
    make_parametrization(config["type_annotation"], case)
    for config in test_cases
    for case in config["negative_cases"]
]


@pytest.mark.parametrize(
    ["type_annotation", "case"],
    positive_cases,
)
def test_matches_type(type_annotation, case):
    @type_checked
    def func(_: type_annotation):
        pass

    func(case)


@pytest.mark.parametrize(
    ["type_annotation", "case"],
    negative_cases,
)
def test_not_matches(type_annotation, case):
    @type_checked
    def func(_: type_annotation):
        pass

    with pytest.raises(TypeError):
        func(case)


if __name__ == "__main__":
    tvm.testing.main()
