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
"""Test task extraction for autotvm"""
import tvm.relay.testing
from tvm import relay
from tvm import autotvm


def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)

    if name == "resnet-18":
        mod, params = relay.testing.resnet.get_workload(num_layers=18, batch_size=batch_size)
    elif name == "resnet3d-18":
        mod, params = relay.testing.resnet_3d.get_workload(num_layers=18, batch_size=batch_size)
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size)
    elif name == "dcgan":
        mod, params = relay.testing.dcgan.get_workload(batch_size=batch_size)
        input_shape = (batch_size, 100)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape


def test_task_extraction():
    target = "llvm"
    mod_list = []
    params_list = []
    conv2d = relay.op.get("nn.conv2d")
    conv3d = relay.op.get("nn.conv3d")
    conv2d_transpose = relay.op.get("nn.conv2d_transpose")
    dense = relay.op.get("nn.dense")

    mod, params, _ = get_network("resnet-18", batch_size=1)
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(conv2d,)
    )
    assert len(tasks) == 12
    tasks = autotvm.task.extract_from_program(mod, target=target, params=params, ops=(conv2d,))
    assert len(tasks) == 12

    mod, params, _ = get_network("resnet-18", batch_size=1)
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(dense,)
    )
    assert len(tasks) == 2
    tasks = autotvm.task.extract_from_program(mod, target=target, params=params, ops=(dense,))
    assert len(tasks) == 2

    mod, params, _ = get_network("resnet-18", batch_size=1)
    mod_list.append(mod)
    params_list.append(params)
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(conv2d, dense)
    )
    assert len(tasks) == 14
    tasks = autotvm.task.extract_from_program(
        mod, target=target, params=params, ops=(conv2d, dense)
    )
    assert len(tasks) == 14
    tasks = autotvm.task.extract_from_program(mod, target=target, params=params)
    assert len(tasks) == 14

    mod, params, _ = get_network("resnet3d-18", batch_size=1)
    tasks = autotvm.task.extract_from_program(mod, target=target, params=params, ops=(conv3d,))
    assert len(tasks) == 12

    mod, params, _ = get_network("mobilenet", batch_size=1)
    mod_list.append(mod)
    params_list.append(params)
    tasks = autotvm.task.extract_from_program(
        mod, target=target, params=params, ops=(conv2d, dense)
    )
    assert len(tasks) == 21

    mod, params, _ = get_network("dcgan", batch_size=1)
    tasks = autotvm.task.extract_from_program(
        mod, target=target, params=params, ops=(conv2d_transpose,)
    )
    assert len(tasks) == 4

    tasks = autotvm.task.extract_from_multiple_program(
        mod_list, params_list, target=target, ops=(conv2d,)
    )
    assert len(tasks) == 31


def test_task_extraction_for_dense_int8_cuda():
    target = "cuda"
    dense = relay.op.get("nn.dense")

    def get_net(batch, in_dim, out_dim, dtype, out_dtype):
        data = tvm.relay.var("data", shape=[batch, in_dim], dtype=dtype)
        weight = tvm.relay.var("weight", shape=[out_dim, in_dim], dtype=dtype)
        out = relay.nn.dense(data, weight, out_dtype=out_dtype)
        mod, params = relay.testing.create_workload(out)
        return mod, params

    mod, params = get_net(1, 16, 32, "float32", "float32")
    tasks = autotvm.task.extract_from_program(mod, target=target, params=params, ops=(dense,))
    assert len(tasks) == 1 and tasks[0].name == "dense_small_batch.gpu"

    mod, params = get_net(1, 16, 32, "int8", "int32")
    tasks = autotvm.task.extract_from_program(mod, target=target, params=params, ops=(dense,))
    assert len(tasks) == 1 and tasks[0].name == "dense_int8.cuda"


if __name__ == "__main__":
    test_task_extraction()
    test_task_extraction_for_dense_int8_cuda()
