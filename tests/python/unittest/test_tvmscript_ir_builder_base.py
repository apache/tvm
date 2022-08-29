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
"""Unittests for tvm.script.ir_builder.base"""
import pytest
from tvm.script.ir_builder import IRBuilder


def test_ir_builder_scope():
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        assert IRBuilder.current() == ib


def test_ir_builder_multi_scope():
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with IRBuilder() as ib2:  # pylint: disable=invalid-name
            assert IRBuilder.current() == ib2
        assert IRBuilder.current() == ib


def test_ir_builder_no_scope():
    with pytest.raises(ValueError):
        IRBuilder.current()


if __name__ == "__main__":
    test_ir_builder_scope()
    test_ir_builder_multi_scope()
    test_ir_builder_no_scope()
