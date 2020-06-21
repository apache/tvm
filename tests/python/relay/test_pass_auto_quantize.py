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
import numpy as np
import pytest

import tvm
from tvm import te
from tvm import relay
from tvm.relay import testing
from tvm.relay.expr import Call


def quantize_and_build(out):
    f = relay.Function(relay.analysis.free_vars(out), out)
    mod, params = testing.create_workload(f)

    with relay.quantize.qconfig(skip_conv_layers=[]):
        qmod = relay.quantize.quantize(mod, params)

    relay.build(qmod, "llvm", params=params)

    return qmod

def test_mul_rewrite():
    """a test case where rhs of mul is not constant"""
    data = relay.var("data", shape=(1, 16, 64, 64))
    multiplier = relay.sigmoid(relay.var("data", shape=(1, 16, 1, 1)))
    conv = relay.nn.conv2d(data, relay.var("weight"),
                           kernel_size=(3, 3),
                           padding=(1, 1),
                           channels=16)
    act = relay.nn.relu(data=conv)

    quantize_and_build(act * multiplier)

    pool = relay.nn.global_avg_pool2d(data=act)

    quantize_and_build(act * pool)

def test_batch_flatten_rewrite():

    data = relay.var("data", shape=(1, 16, 64, 64), dtype="float32")

    out = relay.nn.conv2d(data, relay.var("weight"),
                          kernel_size=(3, 3),
                          padding=(1, 1),
                          channels=16)

    out = relay.nn.batch_flatten(out)

    qmod = quantize_and_build(out)

    def _check_batch_flatten(node):
        if isinstance(node, Call):
            if(node.op.name == "nn.batch_flatten"):
               assert node.checked_type.dtype == "int8"

    # check if batch_flatten is quantized
    relay.analysis.post_order_visit(qmod["main"], _check_batch_flatten)

def get_calibration_dataset(input_name):
    dataset = []
    for i in range(5):
        data = np.random.uniform(size=(1, 3, 224, 224))
        dataset.append({input_name: data})
    return dataset


@pytest.mark.parametrize("create_target", [True, False])
def test_calibrate_target(create_target):
    mod, params = testing.resnet.get_workload(num_layers=18)
    dataset = get_calibration_dataset("data")
    with relay.quantize.qconfig(calibrate_mode="kl_divergence"):
        if create_target:
            with tvm.target.create("llvm"):
                relay.quantize.quantize(mod, params, dataset)
        else:
            # current_target = None
            relay.quantize.quantize(mod, params, dataset)


def test_calibrate_memory_bound():
    mod, params = testing.resnet.get_workload(num_layers=18)
    dataset = get_calibration_dataset("data")
    import multiprocessing
    num_cpu = multiprocessing.cpu_count()
    with relay.quantize.qconfig(calibrate_mode="kl_divergence",
                                calibrate_chunk_by=num_cpu):
        relay.quantize.quantize(mod, params, dataset)


if __name__ == "__main__":
    test_mul_rewrite()
    test_batch_flatten_rewrite()
    test_calibrate_target(False)
    test_calibrate_target(True)
    test_calibrate_memory_bound()
