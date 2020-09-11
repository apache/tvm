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
from tvm.topi.util import get_const_tuple


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
    conv = relay.nn.conv2d(
        data, relay.var("weight"), kernel_size=(3, 3), padding=(1, 1), channels=16
    )
    act = relay.nn.relu(data=conv)

    quantize_and_build(act * multiplier)

    pool = relay.nn.global_avg_pool2d(data=act)

    quantize_and_build(act * pool)


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
    return tvm.nd.array(data_np, ctx=tvm.cpu(0))


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
        vm = relay.create_executor("vm", ctx=tvm.cpu(0), target="llvm", mod=mod)
        return vm.evaluate()(*params)

    partitioned_mod_result = _eval_mod(partitioned_mod)
    unpartitioned_mod_result = _eval_mod(unpartitioned_mod)
    tvm.testing.assert_allclose(
        unpartitioned_mod_result.asnumpy(), partitioned_mod_result.asnumpy()
    )


def test_add_partition():
    mod = tvm.parser.parse(
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
    mod = tvm.parser.parse(
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
    mod = tvm.parser.parse(
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
    mod = tvm.parser.parse(
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
    mod = tvm.parser.parse(
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
    mod = tvm.parser.parse(
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


if __name__ == "__main__":
    test_mul_rewrite()
    test_batch_flatten_rewrite()
    test_calibrate_target(False)
    test_calibrate_target(True)
    test_calibrate_memory_bound()

    test_add_partition()
    test_conv2d_partition()
    test_multiple_arg_conversions_partition()
    test_unquantizable_prefix_partition()
    test_unquantizable_core_partition()
    test_unquantizable_suffix_partition()
