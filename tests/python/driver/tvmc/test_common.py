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
import os
from os import path

import pytest

from tvm.driver import tvmc


def test_parse_input_shapes__list_lengths():
    shape_string = "(1,224,224,3)"
    sut = tvmc.common.parse_input_shapes(shape_string)

    # output is a list with a list [[1, 224, 224, 3]]
    assert type(sut) is list
    assert len(sut) == 1
    assert type(sut[0]) is list
    assert len(sut[0]) == 4


def test_parse_input_shapes__lists_match():
    shape_string = "(1,224,224,3)"
    sut = tvmc.common.parse_input_shapes(shape_string)

    assert sut[0] == [1, 224, 224, 3]


def test_parse_input_shapes__spaces_are_ignored():
    shape_string = "(1,  224, 224,   3)"
    sut = tvmc.common.parse_input_shapes(shape_string)

    assert type(sut) is list
    assert len(sut) == 1
    assert type(sut[0]) is list
    assert len(sut[0]) == 4


def test_parse_input_shapes__missing():
    shape_string = "(1,224,,3)"
    with pytest.raises(argparse.ArgumentTypeError) as e:
        def f():
            _ = tvmc.common.parse_input_shapes(shape_string)
        f()

    assert 'expected numbers in shape' in str(e.value)


def test_parse_input_shapes_no_brackets():
    shape_string = "1,224,224,3"
    with pytest.raises(argparse.ArgumentTypeError) as e:
        def f():
            _ = tvmc.common.parse_input_shapes(shape_string)
        f()

    assert 'missing brackets around shape' in str(e.value)
