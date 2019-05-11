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
"""Benchmarking Relay VM using models from MXNet."""
import numpy as np

import tvm
from tvm.contrib import graph_runtime
from tvm import relay
from tvm.relay import testing


def benchmark_execution(net,
                        params,
                        measure=False,
                        data_shape=(1, 3, 224, 224),
                        out_shape=(1, 1000),
                        dtype='float32'):
    def get_tvm_output(net, data, params, target, ctx, dtype='float32'):
        with relay.build_config(opt_level=1):
            graph, lib, params = relay.build(net, target, params=params)

        m = graph_runtime.create(graph, lib, ctx)
        # set inputs
        m.set_input("data", data)
        m.set_input(**params)
        m.run()
        out = m.get_output(0, tvm.nd.empty(out_shape, dtype))

        if measure:
            print("Evaluate graph runtime inference time cost...")
            ftimer = m.module.time_evaluator("run", ctx, number=1, repeat=20)
            # Measure in millisecond.
            prof_res = np.array(ftimer().results) * 1000
            print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
                  (np.mean(prof_res), np.std(prof_res)))

        return out.asnumpy()

    def get_tvm_vm_output(net, data, params, target, ctx, dtype='float32'):
        ex = relay.create_executor('vm', mod=relay.Module(), ctx=ctx)
        result = ex.evaluate(net)(data, **params)
        return result.asnumpy().astype(dtype)

    # random input
    data = np.random.uniform(size=data_shape).astype(dtype)
    target = "llvm"
    ctx = tvm.cpu(0)

    tvm_out = get_tvm_output(net, tvm.nd.array(data.astype(dtype)), params,
                             target, ctx, dtype)
    vm_out = get_tvm_vm_output(net, tvm.nd.array(data.astype(dtype)), params,
                               target, ctx, dtype)
    tvm.testing.assert_allclose(vm_out, tvm_out, rtol=1e-5, atol=1e-5)


def test_mlp():
    image_shape = (1, 28, 28)
    net, params = testing.mlp.get_workload(1)
    benchmark_execution(net, params, data_shape=image_shape, out_shape=(1, 10))


def test_vgg():
    for n in [11, 16]:
        net, params = testing.vgg.get_workload(1, num_layers=n)
        benchmark_execution(net, params)


def test_resnet():
    for n in [18, 50]:
        net, params = testing.resnet.get_workload(batch_size=1, num_layers=n)
        benchmark_execution(net, params, True)


def test_squeezenet():
    for version in ['1.0', '1.1']:
        net, params = testing.squeezenet.get_workload(version=version)
        benchmark_execution(net, params)


def test_inception_v3():
    image_shape = (3, 299, 299)
    net, params = testing.inception_v3.get_workload(image_shape=image_shape)
    benchmark_execution(net, params, data_shape=image_shape)


def test_dqn():
    image_shape = (4, 84, 84)
    net, params = testing.dqn.get_workload(
        batch_size=1, image_shape=image_shape)
    benchmark_execution(net, params, data_shape=image_shape, out_shape=(1, 18))


def test_dcgan():
    image_shape = (1, 100)
    net, params = testing.dcgan.get_workload(batch_size=1)
    benchmark_execution(net, params, data_shape=image_shape)


def test_mobilenet():
    net, params = testing.mobilenet.get_workload(batch_size=1)
    benchmark_execution(net, params)


def test_densenet():
    net, params = testing.densenet.get_workload(batch_size=1)
    benchmark_execution(net, params)


if __name__ == '__main__':
    test_resnet()
    test_vgg()
    test_squeezenet()
    test_mobilenet()
    test_densenet()
    # The following networks fail
    # test_inception_v3()
    # test_mlp()
    # test_dqn()
    # test_dcgan()
