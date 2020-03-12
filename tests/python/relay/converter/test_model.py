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

"""Relay to ONNX serialization test cases"""
from collections import OrderedDict
import numpy as np
import tvm
from tvm import relay
from tvm.relay.converter import to_onnx
import onnxruntime as rt
import tvm.relay.testing


def func_to_onnx(mod, params, name):
    onnx_model = to_onnx(mod, params, name, path=None)
    return onnx_model.SerializeToString()


def run_onnx(mod, params, name, input_data):
    onnx_model = func_to_onnx(mod, params, name)
    sess = rt.InferenceSession(onnx_model)
    input_names = {}
    for input, data in zip(sess.get_inputs(), input_data):
        input_names[input.name] = data
    output_names = [output.name for output in sess.get_outputs()]
    res = sess.run(output_names, input_names)
    return res[0]


def get_data(in_data_shapes, dtype='float32'):
    in_data = OrderedDict()
    for name, shape in in_data_shapes.items():
        in_data[name] = np.random.uniform(size=shape).astype(dtype)
    return in_data


def run_relay(mod, params, in_data):
    target = 'llvm'
    ctx = tvm.context('llvm', 0)
    intrp = relay.create_executor("graph", mod, ctx=ctx, target=target)
    in_data = [tvm.nd.array(value) for value in in_data.values()]
    return intrp.evaluate()(*in_data, **params).asnumpy()


def _verify_results(mod, params, in_data):
    a = run_relay(mod, params, in_data)
    b = run_onnx(mod, params, 'test_resent', in_data.values())
    np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-7)


def test_resnet():
    num_class = 1000
    in_data_shapes = OrderedDict({"data": (1, 3, 224, 224)})
    in_data = get_data(in_data_shapes, dtype="float32")
    for n in [18, 34, 50, 101]:
        mod, params = tvm.relay.testing.resnet.get_workload(
            1, num_class, num_layers=n)
        _verify_results(mod, params, in_data)


def test_squeezenet():
    in_data_shapes = OrderedDict({"data": (1, 3, 224, 224)})
    in_data = get_data(in_data_shapes, dtype="float32")
    for version in ['1.0', '1.1']:
        mod, params = tvm.relay.testing.squeezenet.get_workload(1, version=version)
        _verify_results(mod, params, in_data)


if __name__ == '__main__':
    test_resnet()
    test_squeezenet()
