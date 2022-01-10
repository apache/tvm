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
import itertools
import tvm
import tvm.relay.testing
from tvm import relay
from tvm.relay.op.contrib import dnnl
import tvm.testing

has_dnnl_codegen = pytest.mark.skipif(
    not tvm.get_global_func("relay.ext.dnnl", True), reason="DNNL codegen not available"
)

run_module = tvm.testing.parameter(
    pytest.param(False, marks=[has_dnnl_codegen, *tvm.testing.requires_llvm()]),
    pytest.param(True, marks=[has_dnnl_codegen, *tvm.testing.requires_llvm()]),
    ids=["compile", "run"],
)


def vmobj_to_list(o):
    if isinstance(o, tvm.nd.NDArray):
        return [o.numpy()]
    elif isinstance(o, tvm.runtime.container.ADT) or isinstance(o, list):
        return [vmobj_to_list(f) for f in o]
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))


def assert_result_dict_holds(result_dict):
    for k1, k2 in itertools.combinations(result_dict, 2):
        res1 = vmobj_to_list(result_dict[k1])
        res2 = vmobj_to_list(result_dict[k2])
        for r1, r2 in zip(res1, res2):
            tvm.testing.assert_allclose(r1, r2, rtol=1e-3, atol=1e-3)


def run_and_verify(mod, input, params, target, run_module):
    def check_dnnl_used(mod):
        num_dnnl_subgraphs = sum(
            [1 if "dnnl" in gv.name_hint else 0 for gv in mod.get_global_vars()]
        )
        assert num_dnnl_subgraphs >= 1

    dev = tvm.cpu()
    result_dict = dict()
    for mode in ["graph", "vm"]:
        for use_dnnl in [False, True]:
            result_key = mode + ("_dnnl" if use_dnnl else "")
            if use_dnnl:
                mod = dnnl.partition_for_dnnl(mod, params)
            with tvm.transform.PassContext(opt_level=3):
                func = relay.create_executor(mode, mod=mod, device=dev, target=target).evaluate()
            if run_module:
                if isinstance(input, dict):
                    result_dict[result_key] = func(**input, **params)
                else:
                    result_dict[result_key] = func(input, **params)

    if run_module:
        assert_result_dict_holds(result_dict)


def run_and_verify_func(config, run_module, target="llvm", dtype="float32"):
    """Test a Relay func by compiling, running, and comparing TVM and DNNL outputs.

    Parameters
    ----------
    config : Tuple[relay.Function, Dict[str, NDArray], List[str]]
        A tuple containing 1) The function to test, 2) A dictionary of var names to input shapes and
        3) A list of which vars should be considered params.

    run_module: bool
        If True, the built module will be run after being compiled.
    """
    f, input_shapes, is_param = config
    params = {x: np.random.uniform(-1, 1, input_shapes[x]).astype(dtype) for x in is_param}
    input_dict = {
        k: np.random.uniform(-1, 1, v).astype(dtype)
        for k, v in input_shapes.items()
        if k not in is_param
    }
    run_and_verify(f, input_dict, params, target, run_module)


def get_conv2d(
    x_shape=(1, 32, 8, 8),
    k_shape=(16, 32, 3, 3),
    groups=1,
    padding=(0, 0),
    strides=(1, 1),
    dilation=(1, 1),
    activation=None,
    dtype="float32",
):
    x = relay.var("x", shape=(x_shape), dtype=dtype)
    kernel = relay.var("kernel", shape=(k_shape), dtype=dtype)
    out = relay.nn.conv2d(
        x,
        kernel,
        kernel_size=k_shape[2:4],
        groups=groups,
        padding=padding,
        strides=strides,
        dilation=dilation,
        channels=k_shape[0],
    )
    dic = {"x": x_shape, "kernel": k_shape}
    param_lst = ["kernel"]

    if activation == "relu":
        return relay.nn.relu(out), dic, param_lst
    elif activation == "tanh":
        return relay.tanh(out), dic, param_lst
    elif activation == "sigmoid":
        return relay.sigmoid(out), dic, param_lst
    else:
        return out, dic, param_lst


def get_conv2d_weights_const(
    x_shape=(1, 32, 8, 8),
    k_shape=(16, 32, 3, 3),
    groups=1,
    padding=(0, 0),
    strides=(1, 1),
    dilation=(1, 1),
    dtype="float32",
):
    x = relay.var("x", shape=(x_shape), dtype=dtype)
    kernel = relay.const(np.ones(k_shape).astype(dtype))
    out = relay.nn.conv2d(
        x,
        kernel,
        channels=k_shape[0],
        kernel_size=k_shape[2:4],
        groups=groups,
        padding=padding,
        strides=strides,
        dilation=dilation,
    )
    dic = {"x": x_shape}
    param_lst = []
    return out, dic, param_lst


def get_conv2d_bias(
    x_shape=(1, 32, 8, 8), k_shape=(16, 32, 3, 3), activation=None, dtype="float32"
):
    conv, dic, param_lst = get_conv2d(x_shape=x_shape, k_shape=k_shape, dtype=dtype)
    bias = relay.var("bias", shape=(k_shape[0],), dtype=dtype)
    out = relay.nn.bias_add(conv, bias)
    dic["bias"] = (k_shape[0],)
    param_lst += ["bias"]

    if activation == "relu":
        return relay.nn.relu(out), dic, param_lst
    elif activation == "tanh":
        return relay.tanh(out), dic, param_lst
    elif activation == "sigmoid":
        return relay.sigmoid(out), dic, param_lst
    else:
        return out, dic, param_lst


def get_conv2d_bias_bn_relu(x_shape=(1, 32, 8, 8), k_shape=(16, 32, 3, 3), dtype="float32"):
    conv2d_bias, dic, param_lst = get_conv2d_bias(x_shape, k_shape, dtype=dtype)
    beta = relay.const(np.zeros(k_shape[0]).astype(dtype))
    gamma = relay.const(np.ones(k_shape[0]).astype(dtype))
    moving_mean = relay.const(np.zeros(k_shape[0]).astype(dtype))
    moving_var = relay.const(np.ones(k_shape[0]).astype(dtype))
    conv2d_bias_bn, _, _ = relay.nn.batch_norm(
        conv2d_bias,
        gamma=gamma,
        beta=beta,
        moving_mean=moving_mean,
        moving_var=moving_var,
        axis=1,
        center=True,
        scale=True,
        epsilon=1e-5,
    )
    return relay.nn.relu(conv2d_bias_bn), dic, param_lst


def get_dense(x_shape=(1, 16), k_shape=(32, 16), activation=None, dtype="float32"):
    x = relay.var("x", shape=(x_shape), dtype=dtype)
    kernel = relay.var("kernel", shape=(k_shape), dtype=dtype)
    out = relay.nn.dense(x, kernel, units=k_shape[0])
    dic = {"x": x_shape, "kernel": k_shape}
    param_lst = ["kernel"]
    return out, dic, param_lst


def get_dense_bias(x_shape=(1, 16), k_shape=(32, 16), activation=None, dtype="float32"):
    dense, dic, param_lst = get_dense(x_shape=x_shape, k_shape=k_shape, dtype=dtype)
    bias = relay.var("bias", shape=(k_shape[0],), dtype=dtype)
    out = relay.nn.bias_add(dense, bias)
    dic["bias"] = (k_shape[0],)
    param_lst += ["bias"]
    return out, dic, param_lst


def test_dnnl_not_compatible(run_module, target="llvm", dtype="float32"):
    xshape = (1, 32, 14, 14)
    x_data = np.random.uniform(-1, 1, xshape).astype(dtype)

    x = relay.var("x", shape=(xshape), dtype=dtype)
    y = relay.add(x, x)
    z = relay.cast(relay.cast(y, "int32"), "float32")
    out = relay.nn.relu(z)
    f = relay.Function([x], out)
    mod = tvm.IRModule()
    mod["main"] = f
    mod = dnnl.partition_for_dnnl(mod)
    for mode in ["graph", "vm"]:
        with tvm.transform.PassContext(opt_level=3):
            func = relay.create_executor(mode, mod=mod, device=tvm.cpu(0), target=target).evaluate()
            if run_module:
                results = func(x_data)


def test_multiple_outputs(run_module, dtype="float32"):
    def get_graph():
        x = relay.var("x", shape=(1, 3), dtype=dtype)
        y = relay.var("y", shape=(1, 3), dtype=dtype)
        z = relay.add(x, y)
        w = relay.add(z, y)
        out = relay.Tuple((z, w))
        f = tvm.IRModule.from_expr(out)
        return f, {"x": (1, 3), "y": (1, 3)}, []

    run_and_verify_func(get_graph(), run_module=run_module, dtype=dtype)


def test_unary(run_module):
    def get_graph(op, x_shape=(1, 8, 3, 3)):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        out = op(x)
        f = tvm.IRModule.from_expr(out)
        return f, {"x": x_shape}, []

    for op in [
        relay.nn.relu,
        relay.tanh,
        relay.sigmoid,
    ]:
        run_and_verify_func(get_graph(op), run_module=run_module)


def test_conv2d(run_module, dtype="float32"):
    x_shape = (1, 32, 8, 8)
    for k_shape, groups in [((16, 32, 3, 3), 1), ((32, 1, 3, 3), 32)]:
        for padding in [(0, 0), (1, 1)]:
            for strides in [(1, 1), (2, 2)]:
                for dilation in [(1, 1), (2, 2)]:
                    conv2d, dic, param_lst = get_conv2d(
                        x_shape=x_shape,
                        k_shape=k_shape,
                        groups=groups,
                        padding=padding,
                        strides=strides,
                        dilation=dilation,
                        dtype=dtype,
                    )
                    conv2d = tvm.IRModule.from_expr(conv2d)
                    config = conv2d, dic, param_lst
                    run_and_verify_func(config, run_module=run_module, dtype=dtype)


def test_conv2d_weights_const(run_module, dtype="float32"):
    x_shape = (1, 32, 8, 8)
    k_shape = (16, 32, 3, 3)
    conv2d, dic, param_lst = get_conv2d_weights_const(x_shape, k_shape, dtype=dtype)
    conv2d = tvm.IRModule.from_expr(conv2d)
    config = conv2d, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)


def test_conv2d_pattern(run_module, dtype="float32"):
    x_shape = (1, 32, 8, 8)
    k_shape = (16, 32, 3, 3)
    activation_lst = [None, "relu", "tanh", "sigmoid"]
    for a in activation_lst:
        conv2d, dic, param_lst = get_conv2d(x_shape, k_shape, activation=a, dtype=dtype)
        conv2d = tvm.IRModule.from_expr(conv2d)
        config = conv2d, dic, param_lst
        run_and_verify_func(config, run_module=run_module, dtype=dtype)

        conv2d_bias, dic, param_lst = get_conv2d_bias(x_shape, k_shape, activation=a, dtype=dtype)
        conv2d_bias = tvm.IRModule.from_expr(conv2d_bias)
        config = conv2d_bias, dic, param_lst
        run_and_verify_func(config, run_module=run_module, dtype=dtype)

    conv2d_bias_bn_relu, dic, param_lst = get_conv2d_bias_bn_relu(x_shape, k_shape, dtype=dtype)
    conv2d_bias_bn_relu = tvm.IRModule.from_expr(conv2d_bias_bn_relu)
    config = conv2d_bias_bn_relu, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)


def test_dense(run_module, dtype="float32"):
    x_shape = (1, 16)
    k_shape = (32, 16)

    dense, dic, param_lst = get_dense(x_shape, k_shape, dtype=dtype)
    dense = tvm.IRModule.from_expr(dense)
    config = dense, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)

    dense, dic, param_lst = get_dense(x_shape, k_shape=(1, 16), dtype=dtype)
    dense = tvm.IRModule.from_expr(dense)
    config = dense, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)


def test_dense_pattern(run_module, dtype="float32"):
    x_shape = (1, 16)
    k_shape = (32, 16)

    dense, dic, param_lst = get_dense(x_shape, k_shape, dtype=dtype)
    dense = tvm.IRModule.from_expr(dense)
    config = dense, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)

    dense_bias, dic, param_lst = get_dense_bias(x_shape, k_shape, dtype=dtype)
    dense_bias = tvm.IRModule.from_expr(dense_bias)
    config = dense_bias, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__] + sys.argv[1:]))
