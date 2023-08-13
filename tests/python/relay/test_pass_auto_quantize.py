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
from tvm.topi.utils import get_const_tuple


def quantize_and_build(out, skip_conv_layers=[]):
    f = relay.Function(relay.analysis.free_vars(out), out)
    mod, params = testing.create_workload(f)

    with relay.quantize.qconfig(skip_conv_layers=skip_conv_layers):
        qmod = relay.quantize.quantize(mod, params)

    relay.build(qmod, "llvm", params=params)

    return qmod


def test_mul_rewrite():
    """a test case where rhs of mul is not constant"""
    data = relay.var("data", shape=(1, 16, 64, 64))
    multiplier = relay.sigmoid(relay.var("data", shape=(1, 16, 1, 1)))
    conv = relay.nn.conv2d(
        data, relay.var("weight"), kernel_size=(3, 3), padding=(1, 1), channels=16
    )
    act = relay.nn.relu(data=conv)

    quantize_and_build(act * multiplier)

    pool = relay.nn.global_avg_pool2d(data=act)

    quantize_and_build(act * pool)


def test_skip_conv():
    data = relay.var("data", shape=(1, 16, 64, 64))
    np_weight = np.random.rand(16, 16, 3, 3)
    conv0_weight = relay.Constant(tvm.nd.array(np_weight)).astype("float32")
    conv1_weight = relay.Constant(tvm.nd.array(np_weight)).astype("float32")
    multiplier = relay.sigmoid(relay.var("data", shape=(1, 16, 1, 1)))

    conv0 = relay.nn.conv2d(data, conv0_weight, kernel_size=(3, 3), padding=(1, 1), channels=16)
    act0 = relay.nn.relu(data=conv0)
    conv1 = relay.nn.conv2d(act0, conv1_weight, kernel_size=(3, 3), padding=(1, 1), channels=16)
    act1 = relay.nn.relu(data=conv1)

    quantize_and_build(act1 * multiplier)
    quantize_and_build(act1 * multiplier, skip_conv_layers=[0])
    quantize_and_build(act1 * multiplier, skip_conv_layers=[1])
    quantize_and_build(act1 * multiplier, skip_conv_layers=[0, 1])


def test_stop_quantize():
    data = relay.var("data", shape=(1, 16, 64, 64))
    np_weight0 = np.random.rand(16, 16, 3, 3)
    conv0_weight = relay.Constant(tvm.nd.array(np_weight0)).astype("float32")
    np_weight1 = np.random.rand(16, 16, 1, 1)
    conv1_weight = relay.Constant(tvm.nd.array(np_weight1)).astype("float32")
    multiplier = relay.sigmoid(relay.var("data", shape=(1, 16, 1, 1)))

    conv0 = relay.nn.conv2d(data, conv0_weight, kernel_size=(3, 3), padding=(1, 1), channels=16)
    act0 = relay.nn.relu(data=conv0)

    pool = relay.nn.global_avg_pool2d(data=act0)

    conv1 = relay.nn.conv2d(pool, conv1_weight, kernel_size=(1, 1), padding=(0, 0), channels=16)
    act1 = relay.nn.relu(data=conv1)

    quantize_and_build(act1 * multiplier)


def test_batch_flatten_rewrite():

    data = relay.var("data", shape=(1, 16, 64, 64), dtype="float32")

    out = relay.nn.conv2d(
        data, relay.var("weight"), kernel_size=(3, 3), padding=(1, 1), channels=16
    )

    out = relay.nn.batch_flatten(out)

    qmod = quantize_and_build(out)

    def _check_batch_flatten(node):
        if isinstance(node, Call):
            if node.op.name == "nn.batch_flatten":
                assert node.checked_type.dtype == "int8"

    # check if batch_flatten is quantized
    relay.analysis.post_order_visit(qmod["main"], _check_batch_flatten)


def test_batch_matmul_rewrite():
    data = relay.var("data", shape=(1, 4, 16, 16))
    data2 = relay.sigmoid(relay.var("data", shape=(4, 16, 64)))
    out = relay.nn.conv2d(data, relay.var("weight"), kernel_size=(3, 3), padding=(1, 1), channels=8)

    out = relay.nn.batch_flatten(out)
    out = relay.reshape(out, [1, 32, 64])
    out = relay.nn.batch_matmul(out, data2)

    qmod = quantize_and_build(out)

    def _check_batch_matmul(node):
        if isinstance(node, Call):

            if node.op.name in ["nn.batch_matmul", "nn.conv2d"]:
                assert node.checked_type.dtype == "int32"
            elif node.op.name == "nn.batch_flatten":
                assert node.checked_type.dtype == "int8"

    # check if batch_matmul is quantized
    relay.analysis.post_order_visit(qmod["main"], _check_batch_matmul)


def get_calibration_dataset(mod, input_name):
    dataset = []
    input_shape = [int(x) for x in mod["main"].checked_type.arg_types[0].shape]
    for i in range(5):
        data = np.random.uniform(size=input_shape)
        dataset.append({input_name: data})
    return dataset


@pytest.mark.parametrize("create_target", [True, False])
def test_calibrate_target(create_target):
    mod, params = testing.synthetic.get_workload()
    dataset = get_calibration_dataset(mod, "data")
    with relay.quantize.qconfig(calibrate_mode="kl_divergence"):
        if create_target:
            with tvm.target.Target("llvm"):
                relay.quantize.quantize(mod, params, dataset)
        else:
            # current_target = None
            relay.quantize.quantize(mod, params, dataset)


def test_calibrate_memory_bound():
    mod, params = testing.synthetic.get_workload()
    dataset = get_calibration_dataset(mod, "data")
    import multiprocessing

    num_cpu = multiprocessing.cpu_count()
    with relay.quantize.qconfig(calibrate_mode="kl_divergence", calibrate_chunk_by=num_cpu):
        relay.quantize.quantize(mod, params, dataset)


def test_calibrate_percentile():
    mod, params = testing.synthetic.get_workload()
    dataset = get_calibration_dataset(mod, "data")
    with relay.quantize.qconfig(calibrate_mode="percentile"):
        relay.quantize.quantize(mod, params, dataset)


####################################
# Quant/Dequant Partitioning Tests #
####################################

BASE_CFG = {
    "skip_conv_layers": [],
    "skip_dense_layers": False,
    "dtype_input": "int8",
    "dtype_weight": "int8",
    "dtype_activation": "int32",
}


def gen_rand_tvm(tt, low, high):
    if "int" in tt.dtype:
        data_np = np.random.randint(low, high, size=get_const_tuple(tt.shape), dtype=tt.dtype)
    elif "float" in tt.dtype:
        data_np = np.random.uniform(low, high, size=get_const_tuple(tt.shape)).astype(tt.dtype)
    else:
        assert False, "unknown dtype"
    return tvm.nd.array(data_np, device=tvm.cpu(0))


def verify_partition_fails(mod, params):
    # standard partition should always succeed
    with relay.quantize.qconfig(**BASE_CFG, partition_conversions="enabled"):
        partitioned_mod = relay.quantize.quantize(mod, params)

    try:
        with relay.quantize.qconfig(**BASE_CFG, partition_conversions="fully_integral"):
            partitioned_mod = relay.quantize.quantize(mod, params)
        raise RuntimeError("partitioning should have failed")
    except AssertionError:
        pass


def verify_partition(mod, params):
    with relay.quantize.qconfig(**BASE_CFG, paritition_conversions="disabled"):
        unpartitioned_mod = relay.quantize.quantize(mod, params)
        assert (
            len(unpartitioned_mod.get_global_vars()) == 1
        ), "unpartitioned module should only have one function"
    with relay.quantize.qconfig(**BASE_CFG, partition_conversions="fully_integral"):
        partitioned_mod = relay.quantize.quantize(mod, params)

    # ensure partitioned and unpartitioned results agree
    params = [gen_rand_tvm(param.type_annotation, 0, 1) for param in partitioned_mod["main"].params]

    def _eval_mod(mod):
        return relay.create_executor("vm", device=tvm.cpu(0), target="llvm", mod=mod).evaluate()(
            *params
        )

    partitioned_mod_result = _eval_mod(partitioned_mod)
    unpartitioned_mod_result = _eval_mod(unpartitioned_mod)
    tvm.testing.assert_allclose(unpartitioned_mod_result.numpy(), partitioned_mod_result.numpy())


def test_add_partition():
    mod = tvm.relay.parse(
        """
    #[version = "0.0.5"]
    def @main(
        %x: Tensor[(10, 10), float32],
        %y: Tensor[(10, 10), float32]) {
      add(%x, %y)
    }
    """
    )
    params = {}
    verify_partition_fails(mod, params)


def test_conv2d_partition():
    mod = tvm.relay.parse(
        """
    #[version = "0.0.5"]
    def @main(
        %x: Tensor[(1, 4, 16, 16), float32],
        %w: Tensor[(4, 4, 3, 3), float32]) -> Tensor[(1, 4, 16, 16), float32] {
      nn.conv2d(%x, %w,
        padding=[1, 1, 1, 1],
        channels=4,
        kernel_size=[3, 3])
    }
    """
    )
    weight_ty = mod["main"].params[1].checked_type
    params = {"w": gen_rand_tvm(weight_ty, 0, 1)}
    verify_partition(mod, params)


def test_multiple_arg_conversions_partition():
    mod = tvm.relay.parse(
        """
    #[version = "0.0.5"]
    def @main(
        %x1: Tensor[(1, 4, 16, 16), float32],
        %w1: Tensor[(4, 4, 3, 3), float32],
        %x2: Tensor[(1, 4, 16, 16), float32],
        %w2: Tensor[(4, 4, 3, 3), float32]
        ) -> Tensor[(1, 4, 16, 16), float32] {
      %0 = nn.conv2d(%x1, %w1,
        padding=[1, 1, 1, 1],
        channels=4,
        kernel_size=[3, 3]);
      %1 = nn.conv2d(%x2, %w2,
        padding=[1, 1, 1, 1],
        channels=4,
        kernel_size=[3, 3]);
      add(%0, %1)
    }
    """
    )

    w1_ty = mod["main"].params[1].checked_type
    w2_ty = mod["main"].params[3].checked_type
    params = {"w1": gen_rand_tvm(w1_ty, 0, 1), "w2": gen_rand_tvm(w2_ty, 0, 1)}
    verify_partition(mod, params)


def test_unquantizable_prefix_partition():
    mod = tvm.relay.parse(
        """
    #[version = "0.0.5"]
    def @main(
        %x: Tensor[(1, 4, 16, 16), float32],
        %b: Tensor[(4), float32],
        %w: Tensor[(4, 4, 3, 3), float32]) -> Tensor[(1, 4, 16, 16), float32] {
      // NOTE bias_add isn't currently quantizable
      %0 = nn.bias_add(%x, %b);
      nn.conv2d(%0, %w,
        padding=[1, 1, 1, 1],
        channels=4,
        kernel_size=[3, 3])
    }
    """
    )
    bias_ty = mod["main"].params[1].checked_type
    weight_ty = mod["main"].params[2].checked_type
    params = {"b": gen_rand_tvm(bias_ty, 0, 1), "w": gen_rand_tvm(weight_ty, 0, 1)}
    verify_partition_fails(mod, params)


def test_unquantizable_core_partition():
    mod = tvm.relay.parse(
        """
    #[version = "0.0.5"]
    def @main(
        %x1: Tensor[(1, 4, 16, 16), float32],
        %w1: Tensor[(4, 4, 3, 3), float32],
        %b: Tensor[(4), float32],
        %w2: Tensor[(4, 4, 3, 3), float32]) -> Tensor[(1, 4, 16, 16), float32] {
      %0 = nn.conv2d(%x1, %w1,
        padding=[1, 1, 1, 1],
        channels=4,
        kernel_size=[3, 3]);
      // NOTE bias_add isn't currently quantizable
      %1 = nn.bias_add(%0, %b);
      nn.conv2d(%1, %w2,
        padding=[1, 1, 1, 1],
        channels=4,
        kernel_size=[3, 3])
    }
    """
    )
    w1_ty = mod["main"].params[1].checked_type
    bias_ty = mod["main"].params[2].checked_type
    w2_ty = mod["main"].params[3].checked_type
    params = {
        "w1": gen_rand_tvm(w1_ty, 0, 1),
        "w2": gen_rand_tvm(w2_ty, 0, 1),
        "b": gen_rand_tvm(bias_ty, 0, 1),
    }
    verify_partition_fails(mod, params)


def test_unquantizable_suffix_partition():
    mod = tvm.relay.parse(
        """
    #[version = "0.0.5"]
    def @main(
        %x: Tensor[(1, 4, 16, 16), float32],
        %w: Tensor[(4, 4, 3, 3), float32],
        %b: Tensor[(4), float32]) -> Tensor[(1, 4, 16, 16), float32] {
      %0 = nn.conv2d(%x, %w,
        padding=[1, 1, 1, 1],
        channels=4,
        kernel_size=[3, 3]);
      // NOTE bias_add isn't currently quantizable
      nn.bias_add(%0, %b)
    }
    """
    )
    weight_ty = mod["main"].params[1].checked_type
    bias_ty = mod["main"].params[2].checked_type
    params = {"w": gen_rand_tvm(weight_ty, 0, 1), "b": gen_rand_tvm(bias_ty, 0, 1)}
    verify_partition_fails(mod, params)


def test_left_shift_negative():
    data = relay.var("data", shape=(1, 16, 64, 64))
    weight = relay.const(np.full((16, 16, 3, 3), 256.0))
    conv2d = relay.nn.conv2d(data, weight, kernel_size=(3, 3), padding=(1, 1), channels=16)
    relu = relay.nn.relu(conv2d)

    mod = tvm.IRModule.from_expr(relu)

    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(
            calibrate_mode="global_scale", global_scale=8.0, skip_conv_layers=None
        ):
            qnn_mod = relay.quantize.quantize(mod)

    class OpFinder(relay.ExprVisitor):
        def __init__(self, op_name):
            super(OpFinder, self).__init__()
            self._op_name = op_name
            self.ops = list()

        def visit_call(self, call):
            super().visit_call(call)
            if call.op.name == self._op_name:
                self.ops.append(call)

    opf = OpFinder("left_shift")
    opf.visit(qnn_mod["main"])
    assert len(opf.ops) > 0, 'Broken case, can\'t find any "left_shift" operators.'
    for left_shift_op in opf.ops:
        shift_amount = left_shift_op.args[1].data.numpy()
        assert shift_amount >= 0, "Shift amount must be non-negative."


def test_dense_conv2d_rewrite():
    n, c, h, w = 1, 16, 64, 64
    data = relay.var("data", relay.TensorType((n, c, h, w)))
    inp = relay.var("inp", relay.TensorType((n, c * h * w)))
    weight_T = relay.const(np.random.random((n, c * h * w)), dtype="float32")
    bias = relay.const(np.random.random((n,)), dtype="float32")
    conv_w = relay.const(np.random.random((16, 16, 3, 3)), dtype="float32")

    dense_o = relay.nn.dense(inp, weight_T)
    linear_o = relay.nn.bias_add(dense_o, bias)
    conv2d_o = relay.nn.conv2d(data, conv_w, kernel_size=(3, 3), padding=(1, 1), channels=16)
    result = relay.Tuple((linear_o, conv2d_o))

    mod = tvm.IRModule.from_expr(result)
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(
            calibrate_mode="global_scale", global_scale=8.0, skip_dense_layer=False
        ):
            qnn_mod = relay.quantize.quantize(mod)

    def _check_dense(node):
        if isinstance(node, Call):
            if node.op.name == "nn.dense":
                assert node.args[0].checked_type.dtype == "int8"
                assert node.args[1].checked_type.dtype == "int8"
                assert node.checked_type.dtype == "int32"
            if node.op.name == "nn.conv2d":
                assert node.args[0].checked_type.dtype == "float32"
                assert node.args[1].checked_type.dtype == "float32"
                assert node.checked_type.dtype == "float32"

    relay.analysis.post_order_visit(qnn_mod["main"], _check_dense)


def test_add_lhs_is_none_annotate():
    data_conv = relay.var("data_conv", shape=(1, 16, 64, 64))
    conv2d_w = relay.const(np.random.random((16, 16, 3, 3)))
    conv2d = relay.nn.conv2d(data_conv, conv2d_w, padding=(1, 1), kernel_size=(3, 3))
    data_add = relay.var("data_add", shape=(16, 1, 1))
    add = relay.add(data_add, conv2d)
    global_avg_pool2d = relay.nn.global_avg_pool2d(add)
    mod = tvm.IRModule.from_expr(global_avg_pool2d)

    calibrate_data = [
        {"data_conv": np.random.random((1, 16, 64, 64)), "data_add": np.random.random((16, 1, 1))}
    ]

    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(calibrate_mode="kl_divergence", skip_conv_layers=None):
            qmod = relay.quantize.quantize(mod, dataset=calibrate_data)

    params = [gen_rand_tvm(param.type_annotation, 0, 1) for param in mod["main"].params]

    def _eval_mod(mod):
        return relay.create_executor("vm", device=tvm.cpu(0), target="llvm", mod=mod).evaluate()(
            *params
        )

    mod_result = _eval_mod(mod)
    qmod_result = _eval_mod(qmod)
    tvm.testing.assert_allclose(mod_result.numpy(), qmod_result.numpy(), rtol=1e-1, atol=1e-1)


def test_add_lhs_rhs_is_input_annotate():
    data_conv_r = relay.var("data_conv_r", shape=(1, 16, 64, 64))
    conv2d_r = relay.nn.conv2d(
        data_conv_r,
        relay.const(np.random.random((16, 16, 3, 3))),
        padding=(1, 1),
        kernel_size=(3, 3),
    )
    data_conv_l = relay.var("data_conv_l", shape=(1, 16, 64, 64))
    conv2d_l = relay.nn.conv2d(
        data_conv_l,
        relay.const(np.random.random((16, 16, 3, 3))),
        padding=(1, 1),
        kernel_size=(3, 3),
    )
    add = relay.add(conv2d_l, conv2d_r)
    global_avg_pool2d = relay.nn.global_avg_pool2d(add)
    mod = tvm.IRModule.from_expr(global_avg_pool2d)

    calibrate_data = [
        {
            "data_conv_l": np.random.random((1, 16, 64, 64)),
            "data_conv_r": np.random.random((1, 16, 64, 64)),
            "data_add": np.random.random((16, 1, 1)),
        }
    ]

    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(calibrate_mode="kl_divergence", skip_conv_layers=None):
            qmod = relay.quantize.quantize(mod, dataset=calibrate_data)

    params = [gen_rand_tvm(param.type_annotation, 0, 1) for param in mod["main"].params]

    def _eval_mod(mod):
        return relay.create_executor("vm", device=tvm.cpu(0), target="llvm", mod=mod).evaluate()(
            *params
        )

    mod_result = _eval_mod(mod)
    qmod_result = _eval_mod(qmod)
    tvm.testing.assert_allclose(mod_result.numpy(), qmod_result.numpy(), rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
    test_mul_rewrite()
    test_batch_flatten_rewrite()
    test_batch_matmul_rewrite()
    test_calibrate_target(False)
    test_calibrate_target(True)
    test_calibrate_memory_bound()
    test_calibrate_percentile()

    test_add_partition()
    test_conv2d_partition()
    test_multiple_arg_conversions_partition()
    test_unquantizable_prefix_partition()
    test_unquantizable_core_partition()
    test_unquantizable_suffix_partition()
    test_left_shift_negative()
    test_dense_conv2d_rewrite()

    test_skip_conv()
    test_stop_quantize()

    test_add_lhs_is_none_annotate()
    test_add_lhs_rhs_is_input_annotate()
