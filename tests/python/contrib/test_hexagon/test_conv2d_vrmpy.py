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

import tvm.testing
from tvm import relay


def get_conv2d_nchw(
    d_shape,
    w_shape,
    padding,
    strides=(1, 1),
    data_dtype = "int8",
    weight_dtype = "int8"
):
    out_dtype = "int32"

    data = relay.var("data", shape=d_shape, dtype=data_dtype)
    weight = relay.var("weight", shape=w_shape, dtype=weight_dtype)
    out_channel = w_shape[0]
    return relay.nn.conv2d(
        data=data,
        weight=weight,
        kernel_size=w_shape[2:],
        channels=out_channel,
        padding=padding,
        strides=strides,
        out_dtype=out_dtype,
    )


@tvm.testing.requires_hexagon
def test_conv2d_u8u8i32_vrmpy(hexagon_session):
    target_hexagon = tvm.target.hexagon("v68")
    target = tvm.target.Target(target_hexagon, host=target_hexagon)
    I = 64
    O = 256
    H = 56
    W = 56
    kH = 3
    kW = 3
    padding = (1, 1)
    strides = (1, 1)

    data_shape = (1, I, H, W)
    weight_shape = (O, I, kH, kW)
    bias_shape = (weight_shape[0],)

    bias = relay.var("bias", shape=bias_shape, dtype="int32")

    data_dtype = "uint8"
    weight_dtype = "int8"
    conv2d = get_conv2d_nchw(data_shape, weight_shape, padding, strides=strides, data_dtype=data_dtype, weight_dtype=weight_dtype)
    bias_add = relay.nn.bias_add(conv2d, bias)

    use_bias = True

    if use_bias:
        out = bias_add
    else:
        out = conv2d

    mod = tvm.IRModule.from_expr(out)

    if data_dtype == "uint8":
        data_np = np.random.uniform(0, 255, size=data_shape).astype("uint8")
    else:
        data_np = np.random.uniform(-128, 127, size=data_shape).astype("int8")

    if weight_dtype == "uint8":
        weight_np = np.random.uniform(0, 255, size=weight_shape).astype("uint8")
    else:
        weight_np = np.random.uniform(-128, 127, size=weight_shape).astype("int8")

    bias_np = np.random.randint(low=-127, high=128, size=bias_shape).astype("int32")
    params = {"weight": weight_np, "bias": bias_np}

    out_ty = relay.transform.InferType()(mod)

    _, _, P, Q = out_ty["main"].body.checked_type.shape

    target_llvm = tvm.target.Target("llvm")

    with tvm.transform.PassContext(
        opt_level=3,
    ):
        lib_ref = relay.build(mod, target=target_llvm, params=params)

    # return

    with tvm.transform.PassContext(
        opt_level=3,
    ):
        # opt_mod, _ = relay.optimize(mod, target=target, params=params)
        # print(opt_mod)
        # return
        executor = relay.backend.Executor("graph", {"link-params": True})
        lib = relay.build(mod, target=target, params=params, executor=executor)

    asm = lib.lib.get_source("asm")
    assert "vrmpy" in asm

    rt_mod = hexagon_session.get_executor_from_factory(lib)

    rt_mod.set_input("data", data_np)

    rt_mod.run()

    out = rt_mod.get_output(0).numpy()

    rt_mod_ref = tvm.contrib.graph_executor.GraphModule(lib_ref["default"](tvm.cpu(0)))

    rt_mod_ref.set_input("data", data_np)

    rt_mod_ref.run()

    ref = rt_mod_ref.get_output(0).numpy()

    np.testing.assert_equal(out, ref)
