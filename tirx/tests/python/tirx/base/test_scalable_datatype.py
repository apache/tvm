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
import tvm_ffi

import tvm_tirx
from tvm_tirx import tirx
from tvm_tirx.codegen import llvm
from tvm_tirx.script import tirx as T

"""
Tests for scalable data types.
"""


def test_create_scalable_data_type_python_api():
    dtype = tvm_ffi.dtype("float32xvscalex4")
    assert str(dtype) == "float32xvscalex4"


# LLVM 20 renamed llvm.experimental.stepvector to llvm.stepvector and dropped
# the old name from the intrinsic table:
# https://releases.llvm.org/20.1.0/docs/ReleaseNotes.html
_STEPVECTOR_NAME = (
    "llvm.stepvector" if llvm.version_major() >= 20 else "llvm.experimental.stepvector"
)


@pytest.mark.skipif(llvm.version_major() < 13, reason="Stepvector intrinsic was added in LLVM 13.")
def test_create_scalable_tir_intrin():
    intrin = tirx.call_llvm_intrin(_STEPVECTOR_NAME, ty="int32xvscalex4")
    assert intrin.ty.dtype == "int32xvscalex4"
    assert (
        str(intrin) == f'T.call_llvm_intrin("{_STEPVECTOR_NAME}", ty=T.PrimType("int32xvscalex4"))'
    )


@pytest.mark.skipif(llvm.version_major() < 13, reason="Stepvector intrinsic was added in LLVM 13.")
def test_tvm_script_create_scalable_tir_intrin():
    @T.function
    def my_func():
        T.call_llvm_intrin(_STEPVECTOR_NAME, ty="int32xvscalex4")

    assert (
        f'T.call_llvm_intrin("{_STEPVECTOR_NAME}", ty=T.PrimType("int32xvscalex4"))'
        in my_func.script()
    )


def test_invalid_data_type():
    with pytest.raises(ValueError):
        tvm_ffi.dtype("float32x4xvscale")


if __name__ == "__main__":
    tvm_tirx.testing.main()
