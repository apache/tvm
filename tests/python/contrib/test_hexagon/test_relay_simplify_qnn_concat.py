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
# pylint: disable=unused-wildcard-import, invalid-name

"""
Test hexagon relay transform - qnn.concat optimization
"""
import tvm
from tvm import relay, testing
from tvm.contrib.hexagon.transform import simplify_qnn_concat


def get_test_module():
    """Creates a test relay module and returns it."""
    q1 = relay.var("q1", shape=(1, 64, 35, 35), dtype="uint8")
    q2 = relay.var("q2", shape=(1, 64, 35, 35), dtype="uint8")
    q3 = relay.var("q3", shape=(1, 32, 35, 35), dtype="uint8")
    s2 = relay.const(0.000109401, dtype="float32")
    s3 = relay.const(0.0486874, dtype="float32")
    s4 = relay.const(0.0425042, dtype="float32")
    s5 = relay.const(0.00345, dtype="float32")
    z1 = relay.const(0, dtype="int32")
    r1 = relay.op.nn.max_pool2d(
        q1,
        pool_size=[3, 3],
        strides=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        ceil_mode=False,
        layout="NHWC",
    )
    r2 = relay.qnn.requantize(q2, s2, z1, s5, z1, axis=1, out_dtype="uint8")
    q_tuple = relay.expr.Tuple([r1, r2, q3])
    s_tuple = relay.expr.Tuple([s4, s5, s3])
    z_tuple = relay.expr.Tuple([z1, z1, z1])
    graph = relay.qnn.concatenate(q_tuple, s_tuple, z_tuple, s3, z1, axis=1)

    func = relay.Function(relay.analysis.free_vars(graph), graph)
    mod = tvm.IRModule.from_expr(func)
    return mod


def get_expected_output_module():
    """Returns manually created expected output module."""
    out_q1 = relay.var("q1", shape=(1, 64, 35, 35), dtype="uint8")
    out_q2 = relay.var("q2", shape=(1, 64, 35, 35), dtype="uint8")
    out_q3 = relay.var("q3", shape=(1, 32, 35, 35), dtype="uint8")
    out_s2 = relay.const(0.000109401, dtype="float32")
    out_s3 = relay.const(0.0486874, dtype="float32")
    out_s4 = relay.const(0.0425042, dtype="float32")
    out_z1 = relay.const(0, dtype="int32")
    nn_max_pool = relay.op.nn.max_pool2d(
        out_q1,
        pool_size=[3, 3],
        strides=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        ceil_mode=False,
        layout="NHWC",
    )
    out_r1 = relay.qnn.requantize(
        nn_max_pool, out_s4, out_z1, out_s3, out_z1, axis=1, out_dtype="uint8"
    )
    out_r2 = relay.qnn.requantize(out_q2, out_s2, out_z1, out_s3, out_z1, axis=1, out_dtype="uint8")
    out_q_tuple = relay.expr.Tuple([out_r1, out_r2, out_q3])
    out_graph = relay.op.concatenate(out_q_tuple, axis=1)

    out_func = relay.Function(relay.analysis.free_vars(out_graph), out_graph)
    out_mod = tvm.IRModule.from_expr(out_func)
    return out_mod


def test_simplify_qnn_concat():
    mod = get_test_module()
    mod = tvm.relay.transform.InferType()(mod)
    mod = simplify_qnn_concat(mod)

    out_mod = get_expected_output_module()
    out_mod = tvm.relay.transform.InferType()(out_mod)

    tvm.ir.assert_structural_equal(mod["main"], out_mod["main"])


if __name__ == "__main__":
    testing.main()
