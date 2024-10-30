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
# pylint: disable=unused-wildcard-import
import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import relay
from tvm.contrib import graph_executor


def compare_expected_fac(expr, expected_expr, args):
    mod_def = tvm.relay.transform.InferType()(tvm.IRModule.from_expr(expr))
    mod_flat = tvm.relay.transform.FlattenAtrousConv()(mod_def)
    mod_exp = tvm.relay.transform.InferType()(tvm.IRModule.from_expr(expected_expr))

    assert expr is expected_expr or not tvm.ir.structural_equal(mod_def, mod_flat)
    tvm.ir.assert_structural_equal(mod_flat, mod_exp)

    result_def = (
        relay.create_executor("vm", mod=mod_def, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    result_flat = (
        relay.create_executor("vm", mod=mod_flat, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    result_exp = (
        relay.create_executor("vm", mod=mod_exp, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )

    assert np.array_equal(result_def, result_flat)
    assert np.array_equal(result_flat, result_exp)


def test_fac_block_shape_2():
    # pattern entry with block_shape=[2, 2]
    shape_x = [1, 5, 5, 4]
    shape_w = [3, 3, 4, 1]

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8").astype("float32")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8").astype("float32")

    weight = relay.const(w_np)
    data = relay.var("data", shape=shape_x, dtype="float32")
    op1 = relay.nn.space_to_batch_nd(data, block_shape=[2, 2], paddings=[[2, 3], [2, 3]])
    op2 = relay.nn.conv2d(
        op1,
        weight,
        padding=[0, 0, 0, 0],
        groups=4,
        channels=4,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="HWOI",
    )
    expr = relay.nn.batch_to_space_nd(op2, block_shape=[2, 2], crops=[[0, 1], [0, 1]])

    expected_expr = relay.nn.conv2d(
        data,
        weight,
        padding=[2, 2, 2, 2],
        dilation=[2, 2],
        groups=4,
        channels=4,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="HWOI",
    )

    compare_expected_fac(expr, expected_expr, [x_np])


def test_fac_block_shape_4():
    # pattern entry with block_shape=[4, 4]
    shape_x = [1, 5, 5, 4]
    shape_w = [3, 3, 4, 1]

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8").astype("float32")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8").astype("float32")

    weight = relay.const(w_np)
    data = relay.var("data", shape=shape_x, dtype="float32")
    op1 = relay.nn.space_to_batch_nd(data, block_shape=[4, 4], paddings=[[4, 7], [4, 7]])
    op2 = relay.nn.conv2d(
        op1,
        weight,
        padding=[0, 0, 0, 0],
        groups=4,
        channels=4,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="HWOI",
    )
    expr = relay.nn.batch_to_space_nd(op2, block_shape=[4, 4], crops=[[0, 3], [0, 3]])

    expected_expr = relay.nn.conv2d(
        data,
        weight,
        padding=[4, 4, 4, 4],
        dilation=[4, 4],
        groups=4,
        channels=4,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="HWOI",
    )

    compare_expected_fac(expr, expected_expr, [x_np])


def test_fac_quantize():
    # quantize pattern entry
    shape_x = [1, 5, 5, 4]
    shape_w = [3, 3, 4, 1]

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8")

    weight = relay.const(w_np)
    data = relay.var("data", shape=shape_x, dtype="int8")
    op1 = relay.nn.space_to_batch_nd(data, block_shape=[2, 2], paddings=[[2, 3], [2, 3]])
    op2 = relay.qnn.op.conv2d(
        op1,
        weight,
        input_zero_point=relay.const(0),
        kernel_zero_point=relay.const(0),
        input_scale=relay.const(2.0),
        kernel_scale=relay.const(1.0),
        padding=[0, 0, 0, 0],
        groups=4,
        channels=4,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="HWOI",
    )
    expr = relay.nn.batch_to_space_nd(op2, block_shape=[2, 2], crops=[[0, 1], [0, 1]])

    expected_expr = relay.qnn.op.conv2d(
        data,
        weight,
        input_zero_point=relay.const(0),
        kernel_zero_point=relay.const(0),
        input_scale=relay.const(2.0),
        kernel_scale=relay.const(1.0),
        padding=[2, 2, 2, 2],
        dilation=[2, 2],
        groups=4,
        channels=4,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="HWOI",
    )

    compare_expected_fac(expr, expected_expr, [x_np])


def test_fac_surrounding():
    # pattern entry with surrounding operations add
    shape_x = [1, 5, 5, 4]
    shape_w = [3, 3, 4, 1]

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8").astype("float32")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8").astype("float32")

    weight = relay.const(w_np)
    data = relay.var("data", shape=shape_x, dtype="float32")
    op0 = relay.op.add(data, relay.const(1.0))
    op1 = relay.nn.space_to_batch_nd(op0, block_shape=[2, 2], paddings=[[2, 3], [2, 3]])
    op2 = relay.nn.conv2d(
        op1,
        weight,
        padding=[0, 0, 0, 0],
        groups=4,
        channels=4,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="HWOI",
    )
    op3 = relay.nn.batch_to_space_nd(op2, block_shape=[2, 2], crops=[[0, 1], [0, 1]])
    expr = relay.op.add(op3, relay.const(-1.0))

    op0 = relay.op.add(data, relay.const(1.0))
    op1 = relay.nn.conv2d(
        op0,
        weight,
        padding=[2, 2, 2, 2],
        dilation=[2, 2],
        groups=4,
        channels=4,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="HWOI",
    )
    expected_expr = relay.op.add(op1, relay.const(-1.0))

    compare_expected_fac(expr, expected_expr, [x_np])


def test_fac_several():
    # several pattern entries
    shape_x = [1, 5, 5, 4]
    shape_w = [3, 3, 4, 1]

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8").astype("float32")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8").astype("float32")

    weight = relay.const(w_np)
    data = relay.var("data", shape=shape_x, dtype="float32")
    op1 = relay.nn.space_to_batch_nd(data, block_shape=[2, 2], paddings=[[2, 3], [2, 3]])
    op2 = relay.nn.conv2d(
        op1,
        weight,
        padding=[0, 0, 0, 0],
        groups=4,
        channels=4,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="HWOI",
    )
    op3 = relay.nn.batch_to_space_nd(op2, block_shape=[2, 2], crops=[[0, 1], [0, 1]])
    op4 = relay.nn.space_to_batch_nd(op3, block_shape=[4, 4], paddings=[[4, 7], [4, 7]])
    op5 = relay.nn.conv2d(
        op4,
        weight,
        padding=[0, 0, 0, 0],
        groups=4,
        channels=4,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="HWOI",
    )
    expr = relay.nn.batch_to_space_nd(op5, block_shape=[4, 4], crops=[[0, 3], [0, 3]])

    op1 = relay.nn.conv2d(
        data,
        weight,
        padding=[2, 2, 2, 2],
        dilation=[2, 2],
        groups=4,
        channels=4,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="HWOI",
    )

    expected_expr = relay.nn.conv2d(
        op1,
        weight,
        padding=[4, 4, 4, 4],
        dilation=[4, 4],
        groups=4,
        channels=4,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="HWOI",
    )

    compare_expected_fac(expr, expected_expr, [x_np])


def test__fac_only_s2b_conv():
    # negative case, only operations space_to_batch_nd-conv2d
    shape_x = [1, 5, 5, 4]
    shape_w = [3, 3, 4, 1]

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8").astype("float32")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8").astype("float32")

    weight = relay.const(w_np)
    data = relay.var("data", shape=shape_x, dtype="float32")
    op1 = relay.nn.space_to_batch_nd(data, block_shape=[2, 2], paddings=[[2, 3], [2, 3]])
    expr = relay.nn.conv2d(
        op1,
        weight,
        padding=[0, 0, 0, 0],
        groups=4,
        channels=4,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="HWOI",
    )

    expected_expr = expr

    compare_expected_fac(expr, expected_expr, [x_np])


def test_fac_only_s2b():
    # negative case, only operation space_to_batch_nd
    shape_x = [1, 5, 5, 4]
    shape_w = [3, 3, 4, 1]

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8").astype("float32")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8").astype("float32")

    weight = relay.const(w_np)
    data = relay.var("data", shape=shape_x, dtype="float32")
    expr = relay.nn.space_to_batch_nd(data, block_shape=[2, 2], paddings=[[2, 3], [2, 3]])

    expected_expr = expr

    compare_expected_fac(expr, expected_expr, [x_np])


def test_fac_only_conv_b2s():
    # negative case, only operations conv2d-batch_to_space_nd
    shape_x = [1, 5, 5, 4]
    shape_w = [3, 3, 4, 1]

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8").astype("float32")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8").astype("float32")

    weight = relay.const(w_np)
    data = relay.var("data", shape=shape_x, dtype="float32")
    op1 = relay.nn.conv2d(
        data,
        weight,
        padding=[0, 0, 0, 0],
        groups=4,
        channels=4,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="HWOI",
    )
    expr = relay.nn.batch_to_space_nd(op1, block_shape=[2, 2], crops=[[0, 1], [0, 1]])

    expected_expr = expr

    compare_expected_fac(expr, expected_expr, [x_np])


def test_fac_only_b2s():
    # negative case, only operation batch_to_space_nd
    shape_x = [1, 5, 5, 4]
    shape_w = [3, 3, 4, 1]

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8").astype("float32")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8").astype("float32")

    weight = relay.const(w_np)
    data = relay.var("data", shape=shape_x, dtype="float32")
    expr = relay.nn.batch_to_space_nd(data, block_shape=[2, 2], crops=[[0, 1], [0, 1]])

    expected_expr = expr

    compare_expected_fac(expr, expected_expr, [x_np])


def test_fac_op_btwn_s2b_conv():
    # negative case, add operation between space_to_batch_nd-conv2d
    shape_x = [1, 5, 5, 4]
    shape_w = [3, 3, 4, 1]

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8").astype("float32")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8").astype("float32")

    weight = relay.const(w_np)
    data = relay.var("data", shape=shape_x, dtype="float32")
    op1 = relay.nn.space_to_batch_nd(data, block_shape=[2, 2], paddings=[[2, 3], [2, 3]])
    op_1_5 = relay.op.add(op1, relay.const(1.0))
    op2 = relay.nn.conv2d(
        op_1_5,
        weight,
        padding=[0, 0, 0, 0],
        groups=4,
        channels=4,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="HWOI",
    )
    expr = relay.nn.batch_to_space_nd(op2, block_shape=[2, 2], crops=[[0, 1], [0, 1]])

    expected_expr = expr

    compare_expected_fac(expr, expected_expr, [x_np])


def test_fac_op_btwn_conv_b2s():
    # negative case, add operation between conv2d-batch_to_space_nd
    shape_x = [1, 5, 5, 4]
    shape_w = [3, 3, 4, 1]

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8").astype("float32")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8").astype("float32")

    weight = relay.const(w_np)
    data = relay.var("data", shape=shape_x, dtype="float32")
    op1 = relay.nn.space_to_batch_nd(data, block_shape=[2, 2], paddings=[[2, 3], [2, 3]])
    op2 = relay.nn.conv2d(
        op1,
        weight,
        padding=[0, 0, 0, 0],
        groups=4,
        channels=4,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="HWOI",
    )
    op_2_5 = relay.op.add(op2, relay.const(1.0))
    expr = relay.nn.batch_to_space_nd(op_2_5, block_shape=[2, 2], crops=[[0, 1], [0, 1]])

    expected_expr = expr

    compare_expected_fac(expr, expected_expr, [x_np])


def test_fac_relay_build():
    #  Check the default optimize pipeline
    shape_x = [1, 5, 5, 4]
    shape_w = [3, 3, 4, 1]

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8").astype("float32")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8").astype("float32")

    weight = relay.const(w_np)
    data = relay.var("data", shape=shape_x, dtype="float32")
    op1 = relay.nn.space_to_batch_nd(data, block_shape=[2, 2], paddings=[[2, 3], [2, 3]])
    op2 = relay.nn.conv2d(
        op1,
        weight,
        padding=[0, 0, 0, 0],
        groups=4,
        channels=4,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="HWOI",
    )
    expr = relay.nn.batch_to_space_nd(op2, block_shape=[2, 2], crops=[[0, 1], [0, 1]])

    mod_def = tvm.relay.transform.InferType()(tvm.IRModule.from_expr(expr))
    result_def = (
        relay.create_executor("vm", mod=mod_def, device=tvm.cpu(), target="llvm")
        .evaluate()(x_np)
        .numpy()
    )

    graph, lib, params = relay.build(mod_def, "llvm", params=None)
    rt_mod = graph_executor.create(graph, lib, device=tvm.cpu())
    rt_mod.set_input("data", x_np)
    rt_mod.set_input(**params)
    rt_mod.run()
    result_flat = rt_mod.get_output(0).numpy()

    assert "space_to_batch_nd" not in graph
    assert "conv2d" in graph
    assert "batch_to_space_nd" not in graph

    assert np.array_equal(result_def, result_flat)


if __name__ == "__main__":
    tvm.testing.main()
