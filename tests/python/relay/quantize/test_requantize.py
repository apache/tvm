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
from tvm import relay
from tvm.relay.transform.quantize import Requantizer
from tvm.relay.frontend.common import infer_type

import numpy as np


def check_requantize(pre_graph, expected_graph):
    post_graph = Requantizer().requantize(pre_graph)

    post_graph = infer_type(post_graph)
    expected_graph = infer_type(expected_graph)

    assert tvm.ir.structural_equal(post_graph, expected_graph)


def test_simple_requantize():
    data_shape = (1, 2, 3, 4)
    int8_data = relay.var("int8_data", relay.TensorType(data_shape, dtype="int8"))
    scale1, zp1 = relay.const(np.array(1).astype("float32")), relay.const(
        np.array(2).astype("int32")
    )

    deq_data = relay.qnn.op.dequantize(int8_data, scale1, zp1)
    scale2, zp2 = relay.const(np.array(3).astype("float32")), relay.const(
        np.array(4).astype("int32")
    )
    pre_graph = relay.Function([int8_data], relay.qnn.op.quantize(deq_data, scale2, zp2))

    expected_graph = relay.Function(
        [int8_data], relay.qnn.op.requantize(int8_data, scale1, zp1, scale2, zp2)
    )
    check_requantize(pre_graph, expected_graph)


def test_int8_requantize():
    data_shape = (1, 2, 3, 4)
    int8_data = relay.var("int8_data", relay.TensorType(data_shape, dtype="int8"))
    scale1, zp1 = relay.const(np.array(1).astype("float32")), relay.const(
        np.array(2).astype("int32")
    )
    deq = relay.qnn.op.dequantize(int8_data, scale1, zp1)
    int8_op = relay.op.nn.relu(deq)
    scale2, zp2 = relay.const(np.array(3).astype("float32")), relay.const(
        np.array(0).astype("int32")
    )
    quantize = relay.qnn.op.quantize(int8_op, scale2, zp2)
    pre_graph = relay.Function([int8_data], quantize)

    requantize = relay.qnn.op.requantize(int8_data, scale1, zp1, scale2, zp2)
    int8_op = relay.op.nn.relu(requantize)
    expected_graph = relay.Function([int8_data], int8_op)

    check_requantize(pre_graph, expected_graph)


def test_int8_requantize_zp():
    data_shape = (1, 2, 3, 4)
    int8_data = relay.var("int8_data", relay.TensorType(data_shape, dtype="int8"))
    scale1, zp1 = relay.const(np.array(1).astype("float32")), relay.const(
        np.array(2).astype("int32")
    )
    deq = relay.qnn.op.dequantize(int8_data, scale1, zp1)
    int8_op = relay.op.nn.relu(deq)
    scale2, zp2 = relay.const(np.array(3).astype("float32")), relay.const(
        np.array(4).astype("int32")
    )
    quantize = relay.qnn.op.quantize(int8_op, scale2, zp2)
    pre_graph = relay.Function([int8_data], quantize)

    requantize = relay.qnn.op.requantize(int8_data, scale1, zp1, scale2, zp2)
    zp = relay.op.cast(zp2, dtype="int8")
    int8_op = relay.op.maximum(requantize, zp)
    expected_graph = relay.Function([int8_data], int8_op)

    check_requantize(pre_graph, expected_graph)


def test_chain_removal():
    data_shape = (1, 2, 3, 4)
    int8_data = relay.var("int8_data", relay.TensorType(data_shape, dtype="int8"))
    scale1, zp1 = relay.const(np.array(1).astype("float32")), relay.const(
        np.array(2).astype("int32")
    )
    scale2, zp2 = relay.const(np.array(3).astype("float32")), relay.const(
        np.array(4).astype("int32")
    )
    requantize = relay.qnn.op.requantize(int8_data, scale1, zp1, scale2, zp2)

    scale3, zp3 = relay.const(np.array(5).astype("float32")), relay.const(
        np.array(6).astype("int32")
    )
    requantize2 = relay.qnn.op.requantize(requantize, scale2, zp2, scale3, zp3)

    scale4, zp4 = relay.const(np.array(7).astype("float32")), relay.const(
        np.array(8).astype("int32")
    )
    requantize3 = relay.qnn.op.requantize(requantize2, scale3, zp3, scale4, zp4)
    pre_graph = relay.Function([int8_data], requantize3)

    expected_graph = relay.Function(
        [int8_data], relay.qnn.op.requantize(int8_data, scale1, zp1, scale4, zp4)
    )

    check_requantize(pre_graph, expected_graph)


def test_consolidate():
    data_shape = (1, 2, 3, 4)
    data = relay.var("data", relay.TensorType(data_shape, dtype="float32"))
    scale1, zp1 = relay.const(np.array(1).astype("float32")), relay.const(
        np.array(2).astype("int32")
    )
    quantize = relay.qnn.op.quantize(data, scale1, zp1)

    scale2, zp2 = relay.const(np.array(3).astype("float32")), relay.const(
        np.array(4).astype("int32")
    )
    requantize = relay.qnn.op.requantize(quantize, scale1, zp1, scale2, zp2)
    pre_graph = relay.Function([data], requantize)

    expected_graph = relay.Function([data], relay.qnn.op.quantize(data, scale2, zp2))

    check_requantize(pre_graph, expected_graph)


if __name__ == "__main__":
    test_simple_requantize()
    test_int8_requantize()
    test_int8_requantize_zp()
    test_chain_removal()
    test_consolidate()
