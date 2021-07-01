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
import tvm.relay
import tvm.contrib.target.android_nnapi
from . import infrastructure


def test_codegen_nchw_conv2d():
    data_t = tvm.relay.TensorType((1, 1, 4, 4), "float32")
    data_v = tvm.relay.var("data", data_t)
    data_a = tvm.relay.annotation.compiler_begin(data_v, "android_nnapi")
    weight_t = tvm.relay.TensorType((1, 1, 2, 2), "float32")
    weight_v = tvm.relay.var("weight", weight_t)
    weight_a = tvm.relay.annotation.compiler_begin(weight_v, "android_nnapi")
    conv_c = tvm.relay.nn.conv2d(data=data_a, weight=weight_a)
    conv_a = tvm.relay.annotation.compiler_end(conv_c, "android_nnapi")
    func = tvm.relay.Function([data_v, weight_v], conv_a)
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.relay.transform.PartitionGraph()(mod)
    mod = infrastructure.annotate_for_android_nnapi(mod, 28)

    exe = tvm.relay.backend.vm.compile(
        mod, target="llvm -mtriple=aarch64-linux-android28", params={}
    )
    _, lib = exe.save()
    c_mod = lib.imported_modules[1]
    assert infrastructure.is_compilable(c_mod, 28)


def test_codegen_nchw_conv2d_on_api29():
    data_t = tvm.relay.TensorType((1, 1, 4, 4), "float32")
    data_v = tvm.relay.var("data", data_t)
    data_a = tvm.relay.annotation.compiler_begin(data_v, "android_nnapi")
    weight_t = tvm.relay.TensorType((1, 1, 2, 2), "float32")
    weight_v = tvm.relay.var("weight", weight_t)
    weight_a = tvm.relay.annotation.compiler_begin(weight_v, "android_nnapi")
    conv_c = tvm.relay.nn.conv2d(data=data_a, weight=weight_a)
    conv_a = tvm.relay.annotation.compiler_end(conv_c, "android_nnapi")
    func = tvm.relay.Function([data_v, weight_v], conv_a)
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.relay.transform.PartitionGraph()(mod)
    mod = infrastructure.annotate_for_android_nnapi(mod, 29)

    exe = tvm.relay.backend.vm.compile(
        mod, target="llvm -mtriple=aarch64-linux-android29", params={}
    )
    _, lib = exe.save()
    c_mod = lib.imported_modules[1]
    assert infrastructure.is_compilable(c_mod, 29)


if __name__ == "__main__":
    test_codegen_nchw_conv2d()
    test_codegen_nchw_conv2d_on_api29()
