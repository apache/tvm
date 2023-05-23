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

import argparse

import pytest
from tvm.driver.tvmc.shape_parser import parse_shape_string


def test_shape_parser():
    # Check that a valid input is parsed correctly
    shape_string = "input:[10,10,10]"
    shape_dict = parse_shape_string(shape_string)
    assert shape_dict == {"input": [10, 10, 10]}


def test_alternate_syntax():
    shape_string = "input:0:[10,10,10] input2:[20,20,20,20]"
    shape_dict = parse_shape_string(shape_string)
    assert shape_dict == {"input:0": [10, 10, 10], "input2": [20, 20, 20, 20]}


@pytest.mark.parametrize(
    "shape_string",
    [
        "input:[10,10,10] input2:[20,20,20,20]",
        "input: [10, 10, 10] input2: [20, 20, 20, 20]",
        "input:[10,10,10],input2:[20,20,20,20]",
    ],
)
def test_alternate_syntaxes(shape_string):
    shape_dict = parse_shape_string(shape_string)
    assert shape_dict == {"input": [10, 10, 10], "input2": [20, 20, 20, 20]}


def test_negative_dimensions():
    # Check that negative dimensions parse to Any correctly.
    shape_string = "input:[-1,3,224,224]"
    shape_dict = parse_shape_string(shape_string)
    # Convert to strings to allow comparison with Any.
    assert str(shape_dict) == "{'input': [T.Any(), 3, 224, 224]}"


def test_multiple_valid_gpu_inputs():
    # Check that multiple valid gpu inputs are parsed correctly.
    shape_string = "gpu_0/data_0:[1, -1,224,224] gpu_1/data_1:[7, 7]"
    shape_dict = parse_shape_string(shape_string)
    expected = "{'gpu_0/data_0': [1, T.Any(), 224, 224], 'gpu_1/data_1': [7, 7]}"
    assert str(shape_dict) == expected


def test_invalid_pattern():
    shape_string = "input:[a,10]"
    with pytest.raises(argparse.ArgumentTypeError):
        parse_shape_string(shape_string)


def test_invalid_separators():
    shape_string = "input:5,10 input2:10,10"
    with pytest.raises(argparse.ArgumentTypeError):
        parse_shape_string(shape_string)


def test_invalid_colon():
    shape_string = "gpu_0/data_0:5,10 :test:10,10"
    with pytest.raises(argparse.ArgumentTypeError):
        parse_shape_string(shape_string)


@pytest.mark.parametrize(
    "shape_string",
    [
        "gpu_0/data_0:5,10 /:10,10",
        "gpu_0/data_0:5,10 data/:10,10",
        "gpu_0/data_0:5,10 /data:10,10",
        "gpu_0/invalid/data_0:5,10 data_1:10,10",
    ],
)
def test_invalid_slashes(shape_string):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_shape_string(shape_string)


def test_dot():
    # Check dot in input name
    shape_string = "input.1:[10,10,10]"
    shape_dict = parse_shape_string(shape_string)
    assert shape_dict == {"input.1": [10, 10, 10]}
