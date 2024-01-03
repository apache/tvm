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

import tvm
from tvm import tir
from tvm.script import tir as T

"""
Tests for scalable data types.
"""


def test_create_scalable_data_type_python_api():
    dtype = tvm.DataType("float32x4xvscale")
    assert str(dtype) == "float32x4xvscale"


def test_create_scalable_tir_intrin():
    intrin = tir.call_llvm_intrin("int32x4xvscale", "llvm.experimental.stepvector")
    assert intrin.dtype == "int32x4xvscale"
    assert str(intrin) == 'T.call_llvm_intrin("int32x4xvscale", "llvm.experimental.stepvector")'


def test_tvm_script_create_scalable_tir_intrin():
    @T.prim_func
    def my_func():
        T.call_llvm_intrin("int32x4xvscale", "llvm.experimental.stepvector")

    assert (
        'T.call_llvm_intrin("int32x4xvscale", "llvm.experimental.stepvector")' in my_func.script()
    )


if __name__ == "__main__":
    tvm.testing.main()
