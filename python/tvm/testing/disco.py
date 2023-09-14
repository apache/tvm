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
# pylint: disable=invalid-name, missing-function-docstring, missing-class-docstring
"""Common utilities for testing disco"""
from tvm._ffi import register_func
from tvm.runtime import NDArray, ShapeTuple, String
from tvm.runtime.ndarray import array


@register_func("tests.disco.add_one")
def add_one(x: int) -> int:  # pylint: disable=invalid-name
    return x + 1


@register_func("tests.disco.add_one_float", override=True)
def add_one_float(x: float):  # pylint: disable=invalid-name
    return x + 0.5


@register_func("tests.disco.add_one_ndarray", override=True)
def add_one_ndarray(x: NDArray) -> NDArray:  # pylint: disable=invalid-name
    return array(x.numpy() + 1)


@register_func("tests.disco.str", override=True)
def str_func(x: str):  # pylint: disable=invalid-name
    return x + "_suffix"


@register_func("tests.disco.str_obj", override=True)
def str_obj_func(x: String):  # pylint: disable=invalid-name
    assert isinstance(x, String)
    return String(x + "_suffix")


@register_func("tests.disco.shape_tuple", override=True)
def shape_tuple_func(x: ShapeTuple):  # pylint: disable=invalid-name
    assert isinstance(x, ShapeTuple)
    return ShapeTuple(list(x) + [4, 5])
