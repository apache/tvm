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
"""Unit tests for testing ToMixedPrecision pass"""
from typing import Any, Dict, List

import numpy as np
import pytest
import tvm
from tvm import relay
from tvm.relay.testing import lstm
from tvm.relay.transform import InferType, ToMixedPrecision, mixed_precision

target_precision = tvm.testing.parameter(
    pytest.param("float16"),
    pytest.param("bfloat16"),
    ids=["float16", "bfloat16"],
)


def run_module(mod: tvm.runtime.Module, mod_params: Dict[str, Any]) -> List:
    dev = tvm.device("llvm", 0)
    result = relay.create_executor("debug", mod, device=dev, target="llvm").evaluate()(**mod_params)
    if isinstance(result, tvm.runtime.container.ADT):
        result = [r.numpy() for r in result]
        return result
    else:
        return [result.numpy()]


def verify_mixed_precision_output_close(
    mod: tvm.runtime.Module,
    mod_params: Dict[str, Any],
    mixed_precision_dtype="float16",
    rtol: float = 1e-3,
    atol: float = 0,
    keep_orig_output_dtype=False,
) -> tvm.runtime.Module:
    mod = InferType()(mod)
    result_fp32 = run_module(mod, mod_params)

    if not keep_orig_output_dtype:
        amp_mod = ToMixedPrecision(mixed_precision_dtype)(mod)
        result_amp = run_module(amp_mod, mod_params)
    else:
        with tvm.transform.PassContext(
            config={"relay.ToMixedPrecision.keep_orig_output_dtype": True}
        ):
            amp_mod = ToMixedPrecision(mixed_precision_dtype)(mod)
            result_amp = run_module(amp_mod, mod_params)

    # Ensure the results are close
    if mixed_precision_dtype != "bfloat16":
        for fp32, amp in zip(result_fp32, result_amp):
            np.testing.assert_allclose(fp32, amp, rtol=rtol, atol=atol)

    if keep_orig_output_dtype:
        assert (
            np.array(result_amp).dtype == np.array(result_fp32).dtype
        ), "output type and original type mismatch"

    return amp_mod


def test_lstm(target_precision):
    """A small stress test on a single unrolled lstm unit.

    Has internal functions and let statements the pass must work on.
    """
    # TODO(AndrewZhaoLuo): investigate why non-even units cause failure in codegen for CUDA
    # See discussion here: https://github.com/apache/tvm/issues/8294#issuecomment-866190408
    units = 4
    iterations = 5
    mod, mod_params = lstm.get_workload(iterations=iterations, num_hidden=units)

    # This is an unrolled lstm so each data should be the previous results but
    # we don't care, we just want to stress test things.
    for i in range(iterations):
        mod_params["data" if i == 0 else f"data{i}"] = np.random.uniform(
            -10, 10, (1, units)
        ).astype("float32")

    verify_mixed_precision_output_close(
        mod, mod_params, mixed_precision_dtype=target_precision, rtol=0.01, atol=0.01
    )


@pytest.mark.skip(reason="Flaky test")
def test_lstm_float64():
    """Tests if can handle other mixed precision types.

    As a toy example show can convert graph to float64 and have it run.

    It doesn't really make sense to do it, this just shows we can change
    the target mixed_precision_dtype.
    """
    units = 3
    iterations = 5
    mod, mod_params = lstm.get_workload(iterations=iterations, num_hidden=units)

    # This is an unrolled lstm so each data should be the previous results but
    # we don't care, we just want to stress test things.
    for i in range(iterations):
        mod_params["data" if i == 0 else f"data{i}"] = np.random.uniform(
            -10, 10, (1, units)
        ).astype("float32")

    verify_mixed_precision_output_close(
        mod, mod_params, mixed_precision_dtype="float64", rtol=0.01, atol=0.01
    )


def test_convert_single_conv(target_precision):
    """Conv is a green listed operation meaning it will always use fp16 workload.

    By default it accumulates to fp32 and outputs fp16.
    """
    data_shape = (1, 3, 32, 32)
    weight_shape = (5, 3, 3, 3)
    data = relay.var("data", shape=data_shape, dtype="float32")
    weight = relay.var("weight", shape=weight_shape, dtype="float32")
    conv = relay.nn.conv2d(data, weight, strides=(1, 1), padding=(1, 1), out_dtype="float32")
    mod = tvm.IRModule.from_expr(conv)
    mod = tvm.relay.transform.InferType()(mod)

    mod_params = {
        "data": np.random.uniform(-1, 1, size=data_shape).astype("float32"),
        "weight": np.random.uniform(-1, 1, size=weight_shape).astype("float32"),
    }
    amp_mod = verify_mixed_precision_output_close(
        mod,
        mod_params,
        mixed_precision_dtype=target_precision,
        atol=0.01,
        rtol=1e-3,
        keep_orig_output_dtype=True,
    )

    expected_mod = tvm.IRModule.from_expr(
        relay.cast(
            relay.nn.conv2d(
                relay.cast(data, target_precision),
                relay.cast(weight, target_precision),
                strides=(1, 1),
                padding=(1, 1),
                out_dtype=target_precision,
            ),
            "float32",
        )
    )
    expected_mod = tvm.relay.transform.InferType()(expected_mod)

    assert not tvm.ir.structural_equal(amp_mod, mod)
    tvm.ir.assert_structural_equal(amp_mod, expected_mod)


def test_convert_single_conv_fp64():
    """As above but checks choosing a mixed_precision_type other than FP16 works"""
    data_shape = (1, 3, 32, 32)
    weight_shape = (5, 3, 3, 3)
    data = relay.var("data", shape=data_shape, dtype="float32")
    weight = relay.var("weight", shape=weight_shape, dtype="float32")
    conv = relay.nn.conv2d(data, weight, strides=(1, 1), padding=(1, 1), out_dtype="float32")
    mod = tvm.IRModule.from_expr(conv)
    mod = tvm.relay.transform.InferType()(mod)

    mod_params = {
        "data": np.random.uniform(-1, 1, size=data_shape).astype("float32"),
        "weight": np.random.uniform(-1, 1, size=weight_shape).astype("float32"),
    }
    amp_mod = verify_mixed_precision_output_close(
        mod, mod_params, mixed_precision_dtype="float64", atol=0.01, rtol=1e-3
    )

    # Note we still accumulate to FP32 by default, a user would need to overwrite default
    # behavior to make this make more sense.
    expected_mod = tvm.IRModule.from_expr(
        relay.nn.conv2d(
            relay.cast(data, "float64"),
            relay.cast(weight, "float64"),
            strides=(1, 1),
            padding=(1, 1),
            out_dtype="float64",
        ),
    )
    expected_mod = tvm.relay.transform.InferType()(expected_mod)

    assert not tvm.ir.structural_equal(amp_mod, mod)
    tvm.ir.assert_structural_equal(amp_mod, expected_mod)


def test_convert_conv_bn(target_precision):
    """Conv is green and batch norm is gray. As Conv should output fp16 batch_norm should be green."""
    data_shape = (1, 3, 32, 32)
    weight_shape = (5, 3, 3, 3)
    data = relay.var("data", shape=data_shape, dtype="float32")
    weight = relay.var("weight", shape=weight_shape, dtype="float32")
    conv = relay.nn.conv2d(data, weight, strides=(1, 1), padding=(1, 1), out_dtype="float32")

    bn_shape = [5]
    gamma = relay.var("gamma", shape=bn_shape)
    beta = relay.var("beta", shape=bn_shape)
    moving_mean = relay.var("moving_mean", shape=bn_shape)
    moving_var = relay.var("moving_var", shape=bn_shape)
    bn = relay.nn.batch_norm(conv, gamma, beta, moving_mean, moving_var)
    mod = tvm.IRModule.from_expr(bn[0])
    mod = tvm.relay.transform.InferType()(mod)

    mod_params = {
        "data": np.random.uniform(-1, 1, size=data_shape).astype("float32"),
        "weight": np.random.uniform(-1, 1, size=weight_shape).astype("float32"),
        "gamma": np.random.uniform(-1, 1, size=bn_shape).astype("float32"),
        "beta": np.random.uniform(-1, 1, size=bn_shape).astype("float32"),
        "moving_mean": np.random.uniform(-1, 1, size=bn_shape).astype("float32"),
        "moving_var": np.random.uniform(-1, 1, size=bn_shape).astype("float32"),
    }
    amp_mod = verify_mixed_precision_output_close(
        mod, mod_params, mixed_precision_dtype=target_precision, atol=0.025, rtol=0.01
    )

    # Creating expected module
    data = relay.cast(relay.var("data", shape=data_shape), target_precision)
    weight = relay.cast(relay.var("weight", shape=weight_shape), target_precision)
    conv = relay.nn.conv2d(data, weight, strides=(1, 1), padding=(1, 1), out_dtype=target_precision)

    bn_shape = [5]
    gamma = relay.cast(relay.var("gamma", shape=bn_shape), target_precision)
    beta = relay.cast(relay.var("beta", shape=bn_shape), target_precision)
    moving_mean = relay.cast(relay.var("moving_mean", shape=bn_shape), target_precision)
    moving_var = relay.cast(relay.var("moving_var", shape=bn_shape), target_precision)
    bn = relay.nn.batch_norm(conv, gamma, beta, moving_mean, moving_var)

    expected_mod = tvm.IRModule.from_expr(bn[0])
    expected_mod = tvm.relay.transform.InferType()(expected_mod)
    assert not tvm.ir.structural_equal(amp_mod, mod)
    tvm.ir.assert_structural_equal(amp_mod, expected_mod)


def test_do_not_convert_softmax(target_precision):
    """Softmax is a red listed operation and therefore should never be fp16."""
    shape = [1, 2, 3]
    a = relay.var("a", shape=shape)
    b = relay.nn.softmax(a)
    mod = tvm.IRModule.from_expr(b)
    mod = tvm.relay.transform.InferType()(mod)
    out_mod = ToMixedPrecision(target_precision)(mod)
    orig_mod = tvm.relay.transform.InferType()(mod)
    tvm.ir.assert_structural_equal(orig_mod, out_mod)


def test_do_not_convert_arange(target_precision):
    """Arange is a red listed operation and therefore should never be fp16."""
    dtype = "float32"
    arange = relay.arange(relay.const(1, dtype), relay.const(128, dtype))
    mod = tvm.IRModule.from_expr(arange)
    out_mod = ToMixedPrecision(target_precision)(mod)
    orig_mod = tvm.relay.transform.InferType()(mod)
    tvm.ir.assert_structural_equal(orig_mod, out_mod)


def test_do_not_convert_summation(target_precision):
    """Ops that could involve a large summation are not allowed in fp16."""
    shape = [1, 3, 16, 16]
    a = relay.var("a", shape=shape)
    ops = [
        relay.sum,
        relay.mean,
        relay.nn.global_avg_pool2d,
        lambda inp: relay.nn.adaptive_avg_pool2d(inp, (1, 1)),
    ]
    for op in ops:
        mod = tvm.IRModule.from_expr(op(a))
        out_mod = ToMixedPrecision(target_precision)(mod)
        orig_mod = tvm.relay.transform.InferType()(mod)
        tvm.ir.assert_structural_equal(orig_mod, out_mod)


def test_green_gray_propagates_simple(target_precision):
    """Conv is a green listed operation, while addition is gray.

    As Conv outputs fp16 the add should be done in fp16.
    """
    data_shape = (1, 3, 32, 32)
    weight_shape = (5, 3, 3, 3)
    data = relay.var("data", shape=data_shape, dtype="float32")
    weight = relay.var("weight", shape=weight_shape, dtype="float32")
    conv = relay.nn.conv2d(data, weight, strides=(1, 1), padding=(1, 1), out_dtype="float32")
    conv = conv + conv
    mod = tvm.IRModule.from_expr(conv)
    mod = tvm.relay.transform.InferType()(mod)

    mod_params = {
        "data": np.random.uniform(-1, 1, size=data_shape).astype("float32"),
        "weight": np.random.uniform(-1, 1, size=weight_shape).astype("float32"),
    }
    amp_mod = verify_mixed_precision_output_close(
        mod, mod_params, mixed_precision_dtype=target_precision, atol=0.01, rtol=0.01
    )

    conv_expr = relay.nn.conv2d(
        relay.cast(data, target_precision),
        relay.cast(weight, target_precision),
        strides=(1, 1),
        padding=(1, 1),
        out_dtype=target_precision,
    )
    expected_mod = tvm.IRModule.from_expr(conv_expr + conv_expr)
    expected_mod = tvm.relay.transform.InferType()(expected_mod)

    assert not tvm.ir.structural_equal(amp_mod, mod)
    tvm.ir.assert_structural_equal(amp_mod, expected_mod)


def test_green_red_not_use_extraneous_cast(target_precision):
    """Conv. is a green listed operation, while softmax is red.

    Conv. also by default accumulates to fp32 but outputs fp16.

    We want to avoid a situation where we have extraneous casts.
    E.g. because softmax wants to operate on FP32 we might have

    conv (FP32) -> cast (FP16) -> cast (FP32) -> softmax (FP32)

    To get around this internally when we cast in the pass we cache
    the output nodes and the reverse of the cast back to the original
    node. For example casting the `conv (FP32)` to FP16 would produce:

    `conv (FP32) -> cast (FP16)`

    As the outputs. Now anytime we try to cast the `conv (FP32)` node
    to FP16 it would return the cached result instead of a new cast node:

    `conv (FP32) -> cast (FP16)`

    Furthermore, if we try to cast the `cast (FP16)` node back to FP32 it
    would just return

    `conv (FP32)`.

    This test makes sure this behavior occurs.
    """
    data_shape = (1, 3, 32, 32)
    weight_shape = (5, 3, 3, 3)
    data = relay.var("data", shape=data_shape, dtype="float32")
    weight = relay.var("weight", shape=weight_shape, dtype="float32")
    conv = relay.nn.conv2d(data, weight, strides=(1, 1), padding=(1, 1), out_dtype="float32")
    result = relay.nn.softmax(conv)
    mod = tvm.IRModule.from_expr(result)

    mod_params = {
        "data": np.random.uniform(-1, 1, size=data_shape).astype("float32"),
        "weight": np.random.uniform(-1, 1, size=weight_shape).astype("float32"),
    }
    amp_mod = verify_mixed_precision_output_close(
        mod, mod_params, mixed_precision_dtype=target_precision, atol=0.01, rtol=1e-3
    )

    # Construct expected structure
    conv = relay.cast(
        relay.nn.conv2d(
            relay.cast(data, target_precision),
            relay.cast(weight, target_precision),
            strides=(1, 1),
            padding=(1, 1),
            out_dtype=target_precision,
        ),
        "float32",
    )
    result = relay.nn.softmax(conv)
    expected_mod = tvm.IRModule.from_expr(result)
    expected_mod = InferType()(expected_mod)

    tvm.ir.assert_structural_equal(expected_mod, amp_mod)


def test_red_gray_propagates_simple(target_precision):
    """Everything after a softmax should be in FP32 (exception green colored ops)"""
    shape = [1, 2, 3]
    a = relay.var("a", shape=shape)
    b = relay.nn.softmax(a)
    c = b + b
    mod = tvm.IRModule.from_expr(c)
    mod = tvm.relay.transform.InferType()(mod)

    mod_params = {
        "a": np.random.uniform(-1, 1, size=shape).astype("float32"),
    }
    output_mod = verify_mixed_precision_output_close(
        mod, mod_params, mixed_precision_dtype=target_precision, atol=0.0, rtol=0.0
    )

    tvm.ir.assert_structural_equal(mod, output_mod)


def test_let_statement_simple(target_precision):
    """A 'simple' let statement example.

    Noticeable is the mutation of the bound variable types.
    """
    var1 = relay.var("var1", shape=[1, 20])
    var2 = relay.var("var2", shape=[1, 20])

    data = relay.var("data", shape=[1, 20])
    weight = relay.var("weight", shape=[20, 20])

    r1 = var1 + var1

    r2 = var2 + var2
    let2 = relay.Let(var2, relay.nn.dense(r1, weight, units=20), r2)
    let1 = relay.Let(var1, relay.nn.dense(data, weight, units=20), let2)

    mod = tvm.IRModule.from_expr(let1)
    mod_params = {
        "data": np.random.uniform(-1, 1, size=[1, 20]).astype("float32"),
        "weight": np.random.uniform(-1, 1, size=[20, 20]).astype("float32"),
    }
    output_mod = verify_mixed_precision_output_close(
        mod, mod_params, mixed_precision_dtype=target_precision, atol=0.05, rtol=0.15
    )

    # Construct expected structure
    var1 = relay.var("var1", shape=[1, 20], dtype=target_precision)
    var2 = relay.var("var2", shape=[1, 20], dtype=target_precision)
    data = relay.cast(relay.var("data", shape=[1, 20]), target_precision)
    weight = relay.cast(relay.var("weight", shape=[20, 20]), target_precision)
    r1 = var1 + var1
    r2 = var2 + var2
    let2 = relay.Let(
        var2,
        relay.nn.dense(r1, weight, units=20, out_dtype=target_precision),
        r2,
    )
    let1 = relay.Let(
        var1,
        relay.nn.dense(data, weight, units=20, out_dtype=target_precision),
        let2,
    )
    expected_mod = tvm.IRModule.from_expr(let1)
    expected_mod = InferType()(expected_mod)

    tvm.ir.assert_structural_equal(expected_mod, output_mod)


def test_where_simple(target_precision):
    data = relay.var("data", shape=[1, 20])
    weight = relay.var("weight", shape=[20, 20])
    a = relay.nn.dense(data, weight, units=20)
    b = relay.where(data, a, a)
    mod = tvm.IRModule.from_expr(b)
    mod_params = {
        "data": np.random.uniform(-1, 1, size=[1, 20]).astype("float32"),
        "weight": np.random.uniform(-1, 1, size=[20, 20]).astype("float32"),
    }

    output_mod = verify_mixed_precision_output_close(
        mod, mod_params, mixed_precision_dtype=target_precision, atol=0.01, rtol=0.01
    )

    # Create expected module
    data = relay.cast(relay.var("data", shape=[1, 20]), target_precision)
    weight = relay.cast(relay.var("weight", shape=[20, 20]), target_precision)
    a = relay.nn.dense(data, weight, units=20, out_dtype=target_precision)
    b = relay.where(data, a, a)
    expected_mod = tvm.IRModule.from_expr(b)
    expected_mod = InferType()(expected_mod)

    tvm.ir.assert_structural_equal(expected_mod, output_mod)


def test_batch_matmul_simple(target_precision):
    """Batch matmul is a special case where we try to accumulate to fp16.

    This is due to the fact heterogenous accumulation dtypes does not work
    on all platforms at the moment.
    """
    data = relay.var("data", shape=[1, 1, 20])
    weight = relay.var("weight", shape=[1, 20, 20])
    a = relay.nn.batch_matmul(data, weight)
    mod = tvm.IRModule.from_expr(a)
    mod_params = {
        "data": np.random.uniform(-1, 1, size=[1, 1, 20]).astype("float32"),
        "weight": np.random.uniform(-1, 1, size=[1, 20, 20]).astype("float32"),
    }
    output_mod = verify_mixed_precision_output_close(
        mod, mod_params, mixed_precision_dtype=target_precision, atol=0.01, rtol=0.01
    )
    # Create expected module
    data = relay.cast(relay.var("data", shape=[1, 1, 20]), target_precision)
    weight = relay.cast(relay.var("weight", shape=[1, 20, 20]), target_precision)
    a = relay.nn.batch_matmul(data, weight, out_dtype=target_precision)
    expected_mod = tvm.IRModule.from_expr(a)
    expected_mod = InferType()(expected_mod)
    tvm.ir.assert_structural_equal(expected_mod, output_mod)


def test_convert_follow_node_with_integer_arguments(target_precision):
    """Tests the conversion of a follow op with integer arguments + constant float args.

    The follow op should convert the floating point argument into fp16 as constants/vars
    will always be converted if safe to do so.
    """

    data = relay.var("data", shape=[1, 10], dtype="float32")

    # We use an addition to make sure the input indices are not a var
    # (which are always casted if safe)
    indices = relay.var("indices", shape=[1, 1], dtype="int32") + relay.const(0, dtype="int32")
    take = relay.take(data, indices, axis=0)
    mod = tvm.IRModule.from_expr(take)

    mod_params = {
        "data": np.random.uniform(-1, 1, size=[1, 10]).astype("float32"),
        "indices": np.array([[0]]).astype("int32"),
    }
    output_mod = verify_mixed_precision_output_close(
        mod, mod_params, mixed_precision_dtype=target_precision, atol=0.01, rtol=0.01
    )

    # Create expected module
    data = relay.cast(relay.var("data", shape=[1, 10]), target_precision)
    take = relay.take(data, indices, axis=0)
    expected_mod = tvm.IRModule.from_expr(take)
    expected_mod = InferType()(expected_mod)
    tvm.ir.assert_structural_equal(expected_mod, output_mod)


def test_clip(target_precision):
    data = relay.var("data", shape=[1, 10], dtype="float32")
    res = relay.clip(data, a_min=-128000, a_max=128000)

    mod = tvm.IRModule.from_expr(res)

    mod_params = {
        "data": np.random.uniform(-1, 1, size=[1, 10]).astype("float32"),
    }
    output_mod = verify_mixed_precision_output_close(
        mod, mod_params, mixed_precision_dtype=target_precision, atol=0.01, rtol=0.01
    )

    # Create expected module
    if target_precision == "bfloat16":
        data = relay.cast(relay.var("data", shape=[1, 10]), target_precision)
    res = relay.clip(data, a_min=-128000, a_max=128000)
    expected_mod = tvm.IRModule.from_expr(res)
    expected_mod = InferType()(expected_mod)
    tvm.ir.assert_structural_equal(expected_mod, output_mod)


def test_clip_with_pre_op(target_precision):
    data = relay.var("data", shape=[1, 10], dtype="float32")
    const = relay.const(5, "float32")
    res = relay.divide(data, const)
    res = relay.clip(res, a_min=-128000, a_max=128000)

    mod = tvm.IRModule.from_expr(res)

    mod_params = {
        "data": np.random.uniform(-1, 1, size=[1, 10]).astype("float32"),
    }
    output_mod = verify_mixed_precision_output_close(
        mod, mod_params, mixed_precision_dtype=target_precision, atol=0.01, rtol=0.01
    )

    # Create expected module
    data = relay.cast(relay.var("data", shape=[1, 10]), target_precision)
    const = relay.cast(relay.const(5, "float32"), target_precision)
    res = relay.divide(data, const)
    if target_precision == "float16":
        res = relay.cast(res, "float32")
    res = relay.clip(res, a_min=-128000, a_max=128000)
    expected_mod = tvm.IRModule.from_expr(res)
    expected_mod = InferType()(expected_mod)
    tvm.ir.assert_structural_equal(expected_mod, output_mod)


def test_loop(target_precision):
    i = relay.var("i", shape=(), dtype="int32")
    st = relay.var("st", shape=(relay.Any(), 1), dtype="int32")

    def int32(val):
        return relay.const(val, "int32")

    def _cond(i, st):
        return relay.op.min(relay.op.less(i, int32(10)))

    def _body(i, st):
        i_vec = relay.op.reshape(i, (1, 1))
        ret = relay.op.concatenate([st, i_vec], axis=0)
        return i + int32(1), ret

    loop = relay.loops.while_loop(_cond, [i, st], _body)
    start = relay.var("start", shape=(), dtype="int32")
    body = loop(start, relay.op.reshape(relay.const(0), newshape=(1, 1)))
    func = relay.Function([start], relay.TupleGetItem(body, 1))
    mod = tvm.IRModule()
    mod["main"] = func

    mod_params = {
        "start": np.random.uniform(-1, 1, size=()).astype("int32"),
    }
    output_mod = verify_mixed_precision_output_close(
        mod, mod_params, mixed_precision_dtype=target_precision, atol=0.01, rtol=0.01
    )

    # Create expected module
    expected_mod = InferType()(mod)
    tvm.ir.assert_structural_equal(expected_mod, output_mod)


if __name__ == "__main__":
    tvm.testing.main()
