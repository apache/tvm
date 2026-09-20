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
from tvm import tirx
from tvm.script import tirx as T
from tvm.target.codegen import llvm_version_major

"""
Tests for scalable data types.
"""


def test_create_scalable_data_type_python_api():
    dtype = tvm.DataType("float32xvscalex4")
    assert str(dtype) == "float32xvscalex4"


# LLVM 20 renamed llvm.experimental.stepvector to llvm.stepvector and dropped
# the old name from the intrinsic table:
# https://releases.llvm.org/20.1.0/docs/ReleaseNotes.html
_STEPVECTOR_NAME = (
    "llvm.stepvector" if llvm_version_major() >= 20 else "llvm.experimental.stepvector"
)


@pytest.mark.skipif(llvm_version_major() < 13, reason="Stepvector intrinsic was added in LLVM 13.")
def test_create_scalable_tir_intrin():
    intrin = tirx.call_llvm_intrin("int32xvscalex4", _STEPVECTOR_NAME)
    assert intrin.ty.dtype == "int32xvscalex4"
    assert str(intrin) == f'T.call_llvm_intrin("int32xvscalex4", "{_STEPVECTOR_NAME}")'


@pytest.mark.skipif(llvm_version_major() < 13, reason="Stepvector intrinsic was added in LLVM 13.")
def test_tvm_script_create_scalable_tir_intrin():
    @T.prim_func(s_tir=True)
    def my_func():
        T.call_llvm_intrin("int32xvscalex4", _STEPVECTOR_NAME)

    assert f'T.call_llvm_intrin("int32xvscalex4", "{_STEPVECTOR_NAME}")' in my_func.script()


def test_invalid_data_type():
    with pytest.raises(ValueError):
        tvm.DataType("float32x4xvscale")


if __name__ == "__main__":
    tvm.testing.main()
