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
"""Unittests for tvm.script.parser.tir"""

import pytest
import inspect
import tvm.testing
from tvm.script.parser import tir as T
from tvm import ir, tir


def test_tir_buffer_proxy():
    buffer_0 = T.Buffer((128, 128), "float32")
    assert (
        isinstance(buffer_0, tir.Buffer)
        and list(buffer_0.shape) == [128, 128]
        and buffer_0.dtype == "float32"
    )

    buffer_1 = T.Buffer[(64, 64, 64), "int32"]
    assert (
        isinstance(buffer_1, tir.Buffer)
        and list(buffer_1.shape) == [64, 64, 64]
        and buffer_1.dtype == "int32"
    )


def test_tir_ptr_proxy():
    ptr_0 = T.Ptr("int32", "global")
    assert (
        isinstance(ptr_0, tir.Var)
        and ptr_0.dtype == "handle"
        and isinstance(ptr_0.type_annotation, ir.PointerType)
        and ptr_0.type_annotation.element_type == ir.PrimType("int32")
        and ptr_0.type_annotation.storage_scope == "global"
    )

    ptr_1 = T.Ptr["float32", "shared"]
    assert (
        isinstance(ptr_1, tir.Var)
        and ptr_1.dtype == "handle"
        and isinstance(ptr_1.type_annotation, ir.PointerType)
        and ptr_1.type_annotation.element_type == ir.PrimType("float32")
        and ptr_1.type_annotation.storage_scope == "shared"
    )


if __name__ == "__main__":
    tvm.testing.main()
