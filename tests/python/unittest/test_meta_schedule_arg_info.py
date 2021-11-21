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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring

from tvm.meta_schedule.arg_info import ArgInfo, TensorInfo
from tvm.script import tir as T

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument
# fmt: off

@T.prim_func
def Matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    T.func_attr({"global_symbol": "main"})
    A = T.match_buffer(a, (128, 256), "float32")
    B = T.match_buffer(b, (256, 512), "float32")
    C = T.match_buffer(c, (128, 512), "float32")
    for i, j, k in T.grid(128, 256, 512):
        with T.block("matmul"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument


def test_meta_schedule_tensor_info_creation():
    info = TensorInfo("float32", [1, 224, 224, 3])
    info = str(info)
    assert info == 'TensorInfo("float32", [1, 224, 224, 3])'


def test_meta_schedule_tensor_info_as_json():
    info = TensorInfo("float32", [1, 224, 224, 3])
    info = info.as_json()
    assert info == ["TENSOR", "float32", [1, 224, 224, 3]]


def test_meta_schedule_tensor_info_from_json():
    info = ["TENSOR", "float32", [1, 224, 224, 3]]
    info = TensorInfo.from_json(info)
    assert str(info) == 'TensorInfo("float32", [1, 224, 224, 3])'


def test_meta_schedule_arg_info_from_prim_func():
    a_info, b_info, c_info = ArgInfo.from_prim_func(Matmul)
    assert str(a_info) == 'TensorInfo("float32", [128, 256])'
    assert str(b_info) == 'TensorInfo("float32", [256, 512])'
    assert str(c_info) == 'TensorInfo("float32", [128, 512])'


if __name__ == "__main__":
    test_meta_schedule_tensor_info_creation()
    test_meta_schedule_tensor_info_as_json()
    test_meta_schedule_tensor_info_from_json()
    test_meta_schedule_arg_info_from_prim_func()
