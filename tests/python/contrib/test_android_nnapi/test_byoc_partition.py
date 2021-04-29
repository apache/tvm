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

import unittest.mock
import tvm
import tvm.relay
import tvm.relay.op.contrib.android_nnapi
import tvm.contrib.target.android_nnapi.relayir_to_nnapi_converter.operation_utils.relay_op as relay_op_handler_root


def test_byoc_partition():
    data = tvm.relay.var("data", shape=(1, 3, 4, 4), dtype="float32")
    kernel = tvm.relay.var("kernel", shape=(2, 3, 4, 4), dtype="float32")
    bias = tvm.relay.var("bias", shape=(2,), dtype="float32")
    mod = tvm.IRModule.from_expr(tvm.relay.nn.bias_add(tvm.relay.nn.conv2d(data, kernel), bias))
    mock_root_handler = lambda: None
    mock_root_handler.nn = lambda: None
    mock_root_handler.nn.conv2d = lambda: None
    mock_root_handler.nn.conv2d.handler = relay_op_handler_root.nn.conv2d.handler
    with unittest.mock.patch(
        "tvm.contrib.target.android_nnapi.relayir_to_nnapi_converter.operation_utils.relay_op",
        new=mock_root_handler,
    ):
        mod, _ = tvm.relay.op.contrib.android_nnapi.byoc_partition_for_android_nnapi(mod, {}, 29)
    assert len(mod.get_global_vars()) == 2
    ext_func_gv = next(filter(lambda gv: gv.name_hint != "main", mod.get_global_vars()))
    ext_func = mod[ext_func_gv]
    assert ext_func.body.op == tvm.relay.op.get("nn.conv2d")


if __name__ == "__main__":
    test_byoc_partition()
