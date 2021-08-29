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

"""Relay to ONNX target test cases"""
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from collections import OrderedDict
import numpy as np
import onnxruntime as rt
import tvm
from tvm import relay
from tvm.contrib.target.onnx import to_onnx
import tvm.relay.testing
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.ir import IRModule
from tvm.relay import transform


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


def get_data(in_data_shapes, dtype="float32"):
    in_data = OrderedDict()
    for name, shape in in_data_shapes.items():
        in_data[name] = np.random.uniform(size=shape).astype(dtype)
    return in_data


def run_relay(mod, params, in_data):
    target = "llvm"
    dev = tvm.device("llvm", 0)
    in_data = [tvm.nd.array(value) for value in in_data.values()]
    return (
        relay.create_executor("graph", mod, device=dev, target=target)
        .evaluate()(*in_data, **params)
        .numpy()
    )


def _verify_results(mod, params, in_data):
    a = run_relay(mod, params, in_data)
    b = run_onnx(mod, params, "test_resent", in_data.values())
    np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-7)


def test_resnet():
    num_class = 1000
    in_data_shapes = OrderedDict({"data": (1, 3, 224, 224)})
    in_data = get_data(in_data_shapes, dtype="float32")
    for n in [18, 34, 50, 101]:
        mod, params = tvm.relay.testing.resnet.get_workload(1, num_class, num_layers=n)
        _verify_results(mod, params, in_data)


def test_squeezenet():
    in_data_shapes = OrderedDict({"data": (1, 3, 224, 224)})
    in_data = get_data(in_data_shapes, dtype="float32")
    for version in ["1.0", "1.1"]:
        mod, params = tvm.relay.testing.squeezenet.get_workload(1, version=version)
        _verify_results(mod, params, in_data)


@pytest.mark.skip("USE_TARGET_ONNX should be ON")
def test_partition():
    in_1 = relay.var("in_1", shape=(10, 10), dtype="float32")
    in_2 = relay.var("in_2", shape=(10, 10), dtype="float32")
    in_3 = relay.var("in_3", shape=(10, 10), dtype="float32")
    in_4 = relay.var("in_4", shape=(10, 10), dtype="float32")
    in_5 = relay.var("in_5", shape=(10, 10), dtype="float32")
    in_6 = relay.var("in_6", shape=(10, 10), dtype="float32")
    in_7 = relay.var("in_7", shape=(10, 10), dtype="float32")
    in_8 = relay.var("in_8", shape=(10, 10), dtype="float32")
    in_9 = relay.var("in_9", shape=(10, 10), dtype="float32")
    in_10 = relay.var("in_10", shape=(10, 10), dtype="float32")

    begin0 = compiler_begin(in_1, "onnx")
    begin1 = compiler_begin(in_2, "onnx")
    begin2 = compiler_begin(in_3, "onnx")
    begin3 = compiler_begin(in_4, "onnx")
    node0 = relay.add(begin0, begin1)
    node1 = relay.add(begin2, begin3)
    end0 = compiler_end(node0, "onnx")
    end1 = compiler_end(node1, "onnx")
    begin4 = compiler_begin(end0, "onnx")
    begin5 = compiler_begin(end1, "onnx")
    node2 = relay.add(begin4, begin5)
    end2 = compiler_end(node2, "onnx")

    dbegin0 = compiler_begin(in_5, "default")
    dbegin1 = compiler_begin(in_6, "default")
    node3 = relay.subtract(dbegin0, dbegin1)
    dbegin2 = compiler_begin(in_7, "default")
    dend1 = compiler_end(node3, "default")
    dbegin3 = compiler_begin(dend1, "default")
    node4 = relay.subtract(dbegin2, dbegin3)
    dend2 = compiler_end(node4, "default")

    begin6 = compiler_begin(end2, "onnx")
    begin7 = compiler_begin(dend2, "onnx")
    node5 = relay.add(begin6, begin7)
    end3 = compiler_end(node5, "onnx")
    end4 = compiler_end(node5, "onnx")
    dbegin4 = compiler_begin(in_8, "default")
    dbegin5 = compiler_begin(end3, "default")
    node6 = relay.subtract(dbegin4, dbegin5)
    begin8 = compiler_begin(in_9, "onnx")
    begin9 = compiler_begin(end4, "onnx")
    node7 = relay.multiply(begin8, begin9)
    end5 = compiler_end(node7, "onnx")

    dend3 = compiler_end(node6, "default")
    begin10 = compiler_begin(dend3, "onnx")
    begin11 = compiler_begin(end5, "onnx")
    node8 = relay.add(begin10, begin11)
    end6 = compiler_end(node8, "onnx")
    begin12 = compiler_begin(in_10, "onnx")
    begin13 = compiler_begin(end6, "onnx")
    node9 = relay.add(begin12, begin13)
    end7 = compiler_end(node9, "onnx")

    func = relay.Function([in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10], end7)

    target = "llvm"
    mod = IRModule.from_expr(func)
    mod = transform.PartitionGraph()(mod)

    with tvm.transform.PassContext(opt_level=3, disabled_pass=["FuseOps"]):
        graph_json, mod1, params = relay.build(mod, target)

    assert mod1.type_key == "metadata"
    assert mod1.imported_modules[0].type_key == "llvm"
    assert mod1.imported_modules[0].get_source()
    assert mod1.imported_modules[1].type_key == "onnx"
    assert mod1.imported_modules[1].get_source()


if __name__ == "__main__":
    test_resnet()
    test_squeezenet()
    # test_partition needs USE_TARGET_ONNX to be ON
    test_partition()
