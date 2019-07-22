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

    if name == 'resnet-18':
        mod, params = relay.testing.resnet.get_workload(num_layers=18, batch_size=batch_size)
    elif name == 'mobilenet':
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size)
    elif name == 'dcgan':
        mod, params = relay.testing.dcgan.get_workload(batch_size=batch_size)
        input_shape = (batch_size, 100)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape

def test_task_extraction():
    target = 'llvm'

    mod, params, input_shape = get_network('resnet-18', batch_size=1)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.nn.conv2d,))
    assert len(tasks) == 12

    mod, params, input_shape = get_network('resnet-18', batch_size=1)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.nn.dense,))
    assert len(tasks) == 1

    mod, params, input_shape = get_network('resnet-18', batch_size=1)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.nn.conv2d, relay.op.nn.dense))
    assert len(tasks) == 13

    mod, params, input_shape = get_network('mobilenet', batch_size=1)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.nn.conv2d, relay.op.nn.dense))
    assert len(tasks) == 20

    mod, params, input_shape = get_network('dcgan', batch_size=1)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.nn.conv2d_transpose,))
    assert len(tasks) == 4

if __name__ == '__main__':
    test_task_extraction()
