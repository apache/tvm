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
import pytest
import itertools
import numpy as np
import sys
import subprocess
import math

import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.op.contrib import dnnl
import tvm.testing


has_dnnl_codegen = pytest.mark.skipif(
    not tvm.get_global_func("relay.ext.dnnl", True), reason="DNNL codegen not available"
)

run_module = tvm.testing.parameter(
    pytest.param(False, marks=[has_dnnl_codegen, *tvm.testing.requires_llvm.marks()]),
    pytest.param(True, marks=[has_dnnl_codegen, *tvm.testing.requires_llvm.marks()]),
    ids=["compile", "run"],
)

_bf16_supported = None


def bf16_supported():
    global _bf16_supported
    if _bf16_supported is None:
        _bf16_supported = False
        if sys.platform.startswith("darwin"):
            cpu_info = subprocess.check_output("sysctl -a", shell=True).strip().decode()
            for line in cpu_info.split("\n"):
                if line.startswith("hw.optional.avx512f"):
                    _bf16_supported = bool(line.split(":", 1)[1])
        elif sys.platform.startswith("linux"):
            _bf16_supported = "avx512" in open("/proc/cpuinfo", "r").read()
    return _bf16_supported


def partition_for_dnnl(mod, params=None, alter_layout=True, prune_subgraphs=True):
    """Partition the graph greedily offloading supported operators to DNNL.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.
    Returns
    -------
    mod : Module
        Annotated and partitioned module.
    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    with TempOpAttr("nn.conv2d", "FTVMLegalize", dnnl.legalize_group_conv):
        with TempOpAttr("nn.conv2d_transpose", "FTVMLegalize", dnnl.legalize_group_conv):
            seq = tvm.transform.Sequential(
                [
                    transform.CanonicalizeOps(),
                    transform.InferType(),
                    transform.SimplifyInference(),
                    transform.FoldConstant(),
                    transform.FoldScaleAxis(),
                    # fold consecutive add ops to simplify pattern `conv2d-bias_add-bn-relu`
                    transform.SimplifyExpr(),
                    transform.FoldConstant(),
                    # alter group conv /conv_transpose layout to `GOIHW` / `GIOHW`
                    transform.Legalize(),
                    transform.FoldConstant(),
                ]
            )
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)
    if alter_layout:
        with TempOpAttr("nn.conv1d", "FTVMAlterOpLayout", dnnl.alter_conv):
            with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", dnnl.alter_conv):
                with TempOpAttr("nn.conv3d", "FTVMAlterOpLayout", dnnl.alter_conv):
                    with TempOpAttr(
                        "nn.conv2d_transpose", "FTVMAlterOpLayout", dnnl.alter_conv_transpose
                    ):
                        with TempOpAttr(
                            "nn.conv3d_transpose", "FTVMAlterOpLayout", dnnl.alter_conv_transpose
                        ):
                            alter_layout_seq = tvm.transform.Sequential(
                                [
                                    transform.AlterOpLayout(),
                                    transform.FoldConstant(),
                                ]
                            )
                            with tvm.transform.PassContext(opt_level=3):
                                mod = alter_layout_seq(mod)

    mod = dnnl.rewrite_layer_norm(mod)
    mod = dnnl.rewrite_dense_bias_gelu_reshape_last(mod)

    byoc_seq = tvm.transform.Sequential(
        [
            transform.MergeComposite(dnnl.pattern_table()),
            transform.AnnotateTarget("dnnl"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )

    with tvm.transform.PassContext(opt_level=3):
        mod = byoc_seq(mod)
        if prune_subgraphs:
            mod = dnnl.prune_dnnl_subgraphs(mod)
    return mod


def vmobj_to_list(o):
    if isinstance(o, tvm.nd.NDArray):
        o_np = o.numpy()
        if o_np.dtype == np.uint16:
            o_np = np.left_shift(o_np.astype("uint32"), 16).view("<f4")
        return [o_np]
    elif isinstance(o, tvm.runtime.container.ADT) or isinstance(o, list):
        return [vmobj_to_list(f) for f in o]
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))


def assert_result_dict_holds(result_dict):
    for k1, k2 in itertools.combinations(result_dict, 2):
        res1 = vmobj_to_list(result_dict[k1])
        res2 = vmobj_to_list(result_dict[k2])
        for r1, r2 in zip(res1, res2):
            if "bf16" in k1 or "bf16" in k2:
                np.testing.assert_array_almost_equal(r1, r2, decimal=1)
            else:
                tvm.testing.assert_allclose(r1, r2, rtol=1e-3, atol=1e-3)


def check_dnnl_used(mod, subgraph_num=None):
    num_dnnl_subgraphs = sum([1 if "dnnl" in gv.name_hint else 0 for gv in mod.get_global_vars()])
    if subgraph_num:
        assert num_dnnl_subgraphs == subgraph_num
    else:
        assert num_dnnl_subgraphs >= 1


def run_and_verify(mod, input, params, target, run_module, subgraph_num=None, test_bf16=True):
    dev = tvm.cpu()
    result_dict = dict()
    for mode in ["graph", "vm"]:
        configs = [
            (False, False, False),
            (True, False, False),
            (True, True, False),
        ]
        if test_bf16 and bf16_supported():
            configs += [(True, False, True), (True, True, True)]

        for use_dnnl, alter_layout, use_bf16 in configs:
            result_key = (
                mode
                + ("_dnnl" if use_dnnl else "")
                + ("_layout" if alter_layout else "")
                + ("_bf16" if use_bf16 else "_fp32")
            )
            processed_mod = mod
            if use_bf16:
                processed_mod = relay.transform.ToMixedPrecision("bfloat16")(processed_mod)
                if tvm.ir.structural_equal(processed_mod, mod):
                    print("can not convert to bfloat16, skipping...")
                    continue
            if use_dnnl:
                processed_mod = partition_for_dnnl(processed_mod, params, alter_layout)
                check_dnnl_used(processed_mod)

            with tvm.transform.PassContext(opt_level=3):
                func = relay.create_executor(
                    mode, mod=processed_mod, device=dev, target=target
                ).evaluate()
            if run_module:
                if isinstance(input, dict):
                    result_dict[result_key] = func(**input, **params)
                else:
                    result_dict[result_key] = func(input, **params)

    if run_module:
        assert_result_dict_holds(result_dict)


def run_and_verify_func(
    config, run_module, subgraph_num=None, target="llvm", dtype="float32", test_bf16=True
):
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
    run_and_verify(
        f,
        input_dict,
        params,
        subgraph_num=subgraph_num,
        target=target,
        run_module=run_module,
        test_bf16=test_bf16,
    )


def get_conv1d(
    x_shape=((1, 3, 224)),
    k_shape=(16, 3, 3),
    groups=1,
    padding=(1, 1),
    strides=(1),
    dilation=(1),
    channels=None,
    activation=None,
    dtype="float32",
):
    x = relay.var("x", shape=(x_shape), dtype=dtype)
    kernel = relay.var("kernel", shape=(k_shape), dtype=dtype)
    out = relay.nn.conv1d(
        x,
        kernel,
        kernel_size=k_shape[2:3],
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


def get_conv1d_bias(x_shape=(1, 3, 224), k_shape=(10, 3, 3), activation=None, dtype="float32"):
    conv, dic, param_lst = get_conv1d(x_shape=x_shape, k_shape=k_shape, dtype=dtype)
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


def get_conv1d_bias_bn_relu(x_shape=(1, 3, 224), k_shape=(10, 3, 3), dtype="float32"):
    conv1d_bias, dic, param_lst = get_conv1d_bias(x_shape, k_shape, dtype=dtype)
    beta = relay.const(np.zeros(k_shape[0]).astype(dtype))
    gamma = relay.const(np.ones(k_shape[0]).astype(dtype))
    moving_mean = relay.const(np.zeros(k_shape[0]).astype(dtype))
    moving_var = relay.const(np.ones(k_shape[0]).astype(dtype))
    conv1d_bias_bn, _, _ = relay.nn.batch_norm(
        conv1d_bias,
        gamma=gamma,
        beta=beta,
        moving_mean=moving_mean,
        moving_var=moving_var,
        axis=1,
        center=True,
        scale=True,
        epsilon=1e-5,
    )
    return relay.nn.relu(conv1d_bias_bn), dic, param_lst


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


def get_conv2d_transpose(
    x_shape=(1, 32, 8, 8),
    k_shape=(32, 16, 3, 3),
    groups=1,
    padding=(0, 0),
    strides=(1, 1),
    activation=None,
    dtype="float32",
):
    x = relay.var("x", shape=(x_shape), dtype=dtype)
    kernel = relay.var("kernel", shape=(k_shape), dtype=dtype)
    out = relay.nn.conv2d_transpose(
        x,
        kernel,
        channels=k_shape[1] * groups,
        kernel_size=k_shape[2:4],
        groups=groups,
        padding=padding,
        strides=strides,
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
    kernel = relay.const(np.random.randint(0, 1, k_shape).astype(dtype))
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
    conv, dic, param_lst = get_conv2d_weights_const(x_shape=x_shape, k_shape=k_shape, dtype=dtype)
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


def get_conv2d_transpose_bias(
    x_shape=(1, 32, 8, 8), k_shape=(32, 16, 3, 3), activation=None, dtype="float32"
):
    conv, dic, param_lst = get_conv2d_transpose(x_shape=x_shape, k_shape=k_shape, dtype=dtype)
    bias = relay.var("bias", shape=(k_shape[1],), dtype=dtype)
    out = relay.nn.bias_add(conv, bias)
    dic["bias"] = (k_shape[1],)
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


def get_layer_norm(x_shape=(1, 49, 64), dtype="float32"):
    dic = {"input": x_shape}
    param_lst = []
    input = relay.var("input", shape=x_shape)
    beta = relay.const(np.zeros(x_shape[2]).astype(dtype))
    gamma = relay.const(np.ones(x_shape[2]).astype(dtype))
    out = relay.nn.layer_norm(input, gamma=gamma, beta=beta)
    return out, dic, param_lst


def get_conv2d_bias_sum_relu(x_shape=(1, 32, 8, 8), k_shape=(16, 32, 3, 3), dtype="float32"):
    conv2d_bias, dic, param_lst = get_conv2d_bias(x_shape, k_shape, dtype=dtype)
    sum_data = relay.const(np.random.randint(x_shape).astype(dtype))
    conv2d_bias_sum = relay.add(sum_data, conv2d_bias)
    return relay.nn.relu(conv2d_bias_sum), dic, param_lst


def get_conv3d(
    x_shape=(1, 32, 8, 8, 8),
    k_shape=(16, 32, 3, 3, 3),
    groups=1,
    padding=(0, 0, 0),
    strides=(1, 1, 1),
    dilation=(1, 1, 1),
    activation=None,
    dtype="float32",
):
    x = relay.var("x", shape=(x_shape), dtype=dtype)
    kernel = relay.const(np.random.randint(0, 1, k_shape).astype(dtype))
    out = relay.nn.conv3d(
        x,
        kernel,
        channels=k_shape[0],
        kernel_size=k_shape[2:],
        groups=groups,
        padding=padding,
        strides=strides,
        dilation=dilation,
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


def get_conv3d_transpose(
    x_shape=(1, 32, 8, 8, 8),
    k_shape=(32, 16, 3, 3, 3),
    groups=1,
    padding=(0, 0, 0),
    strides=(1, 1, 1),
    output_padding=(0, 0, 0),
    activation=None,
    dtype="float32",
    data_layout="NCDHW",
    kernel_layout="OIDHW",
):
    x = relay.var("x", shape=(x_shape), dtype=dtype)
    kernel = relay.const(np.random.randint(0, 1, k_shape).astype(dtype))
    out = relay.nn.conv3d_transpose(
        x,
        kernel,
        channels=k_shape[1],
        kernel_size=k_shape[2:5],
        groups=groups,
        padding=padding,
        strides=strides,
        output_padding=output_padding,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
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


def get_conv3d_bias(
    x_shape=(1, 32, 8, 8, 8), k_shape=(16, 32, 3, 3, 3), activation=None, dtype="float32"
):
    conv, dic, param_lst = get_conv3d(x_shape=x_shape, k_shape=k_shape, dtype=dtype)
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


def get_conv3d_transpose_bias(
    x_shape=(1, 32, 8, 8, 8), k_shape=(32, 16, 3, 3, 3), activation=None, dtype="float32"
):
    conv, dic, param_lst = get_conv3d_transpose(x_shape=x_shape, k_shape=k_shape, dtype=dtype)
    bias = relay.var("bias", shape=(k_shape[1],), dtype=dtype)
    out = relay.nn.bias_add(conv, bias)
    dic["bias"] = (k_shape[1],)
    param_lst += ["bias"]

    if activation == "relu":
        return relay.nn.relu(out), dic, param_lst
    elif activation == "tanh":
        return relay.tanh(out), dic, param_lst
    elif activation == "sigmoid":
        return relay.sigmoid(out), dic, param_lst
    else:
        return out, dic, param_lst


def gelu_helper(data):
    const1 = relay.const(math.sqrt(2.0))
    const2 = relay.const(1.0)
    const3 = relay.const(0.5)
    divisor = relay.op.divide(data, const1)
    val_erf = relay.op.erf(divisor)
    added_erf = relay.op.add(val_erf, const2)
    mul1 = relay.op.multiply(data, added_erf)
    out = relay.op.multiply(mul1, const3)
    return out


def get_dense(
    x_shape=(1, 16), k_shape=(32, 16), activation=None, has_reshape=False, dtype="float32"
):
    x = relay.var("x", shape=(x_shape), dtype=dtype)
    kernel = relay.var("kernel", shape=(k_shape), dtype=dtype)
    out = relay.nn.dense(x, kernel, units=k_shape[0])
    # out = relay.nn.dense(x, kernel, units=None)
    if has_reshape:
        out = relay.reshape(out, newshape=(1, x_shape[0], k_shape[0]))
    dic = {"x": x_shape, "kernel": k_shape}
    param_lst = ["kernel"]

    if activation == "gelu":
        out = gelu_helper(out)
    return out, dic, param_lst


def get_dense_bias(
    x_shape=(1, 16),
    k_shape=(32, 16),
    activation=None,
    has_reshape=False,
    use_add=False,
    dtype="float32",
):
    dense, dic, param_lst = get_dense(
        x_shape=x_shape, k_shape=k_shape, has_reshape=has_reshape, dtype=dtype
    )
    bias = relay.var("bias", shape=(k_shape[0],), dtype=dtype)
    if use_add:
        out = relay.add(dense, bias)
    else:
        out = relay.nn.bias_add(dense, bias)
    dic["bias"] = (k_shape[0],)
    param_lst += ["bias"]

    if activation == "gelu":
        out = gelu_helper(out)
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
    mod = partition_for_dnnl(mod)
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


def test_elementwise(run_module, dtype="float32"):
    def get_graph(op, x_shape=(1, 8, 3, 3)):
        x = relay.var("x", shape=(x_shape), dtype=dtype)
        out = op(x)
        f = tvm.IRModule.from_expr(out)
        return f, {"x": x_shape}, []

    for op in [
        relay.abs,
        relay.exp,
        relay.log,
        relay.sqrt,
        relay.nn.relu,
        relay.tanh,
        relay.sigmoid,
    ]:
        run_and_verify_func(get_graph(op), run_module=run_module)


def test_clip(run_module, dtype="float32"):
    def get_graph(x_shape=(1, 8, 3, 3)):
        x = relay.var("x", shape=(x_shape), dtype=dtype)
        out = relay.clip(x, a_min=-0.2, a_max=0.4)
        f = tvm.IRModule.from_expr(out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph(), run_module=run_module)


def test_leaky_relu(run_module, dtype="float32"):
    def get_graph(x_shape=(1, 8, 3, 3)):
        x = relay.var("x", shape=(x_shape), dtype=dtype)
        out = relay.nn.leaky_relu(x, alpha=0.1)
        f = tvm.IRModule.from_expr(out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph(), run_module=run_module)


def test_softmax(run_module, dtype="float32"):
    def get_graph(x_shape, axis):
        x = relay.var("x", shape=(x_shape), dtype=dtype)
        out = relay.nn.softmax(x, axis=axis)
        f = tvm.IRModule.from_expr(out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph((1, 1000), axis=1), run_module=run_module)
    run_and_verify_func(get_graph((1, 1000), axis=-1), run_module=run_module)
    run_and_verify_func(get_graph((1, 3, 4), axis=-2), run_module=run_module)
    run_and_verify_func(get_graph((1, 3, 4), axis=1), run_module=run_module)


def test_conv1d(run_module, dtype="float32"):
    conv1d, dic, param_lst = get_conv1d(channels=16, dtype=dtype)
    conv1d = tvm.IRModule.from_expr(conv1d)
    config = conv1d, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)

    x_shape = (1, 32, 224)
    k_shape = (16, 32, 3)
    conv1d_bias, dic, param_lst = get_conv1d(x_shape, k_shape, dtype=dtype)
    conv1d_bias = tvm.IRModule.from_expr(conv1d_bias)
    config = conv1d_bias, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)


def test_conv1d_pattern(run_module, dtype="float32"):
    x_shape = (1, 3, 224)
    k_shape = (16, 3, 3)
    activation_lst = [None, "relu", "tanh", "sigmoid"]
    for a in activation_lst:
        conv1d, dic, param_lst = get_conv1d(x_shape, k_shape, activation=a, dtype=dtype)
        conv1d = tvm.IRModule.from_expr(conv1d)
        config = conv1d, dic, param_lst
        run_and_verify_func(config, run_module=run_module, dtype=dtype)

        conv1d_bias, dic, param_lst = get_conv1d_bias(x_shape, k_shape, activation=a, dtype=dtype)
        conv1d_bias = tvm.IRModule.from_expr(conv1d_bias)
        config = conv1d_bias, dic, param_lst
        run_and_verify_func(config, run_module=run_module, dtype=dtype)


def test_conv2d(run_module, dtype="float32"):
    x_shape = (1, 32, 8, 8)
    for k_shape, groups in [((16, 32, 3, 3), 1), ((32, 1, 3, 3), 32), ((32, 2, 3, 3), 16)]:
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

    x_shape = (1, 3, 8, 8)
    k_shape = (16, 3, 3, 3)
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

    conv2d_bias_bn_relu, dic, param_lst = get_conv2d_bias_bn_relu(x_shape, k_shape, dtype=dtype)
    conv2d_bias_bn_relu = tvm.IRModule.from_expr(conv2d_bias_bn_relu)
    config = conv2d_bias_bn_relu, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)


def test_conv2d_transpose(run_module, dtype="float32"):
    x_shape = (1, 32, 8, 8)
    for k_shape, groups in [((32, 16, 3, 3), 1), ((32, 1, 3, 3), 32), ((32, 4, 3, 3), 16)]:
        for padding in [(0, 0), (1, 1)]:
            for strides in [(1, 1), (2, 2)]:
                conv2d_transpose, dic, param_lst = get_conv2d_transpose(
                    x_shape=x_shape,
                    k_shape=k_shape,
                    groups=groups,
                    padding=padding,
                    strides=strides,
                    dtype=dtype,
                )
                conv2d_transpose = tvm.IRModule.from_expr(conv2d_transpose)
                config = conv2d_transpose, dic, param_lst
                run_and_verify_func(config, run_module=run_module, dtype=dtype)


def test_conv2d_transpose_pattern(run_module, dtype="float32"):
    activation_lst = [None, "relu", "tanh", "sigmoid"]
    for a in activation_lst:
        conv2d, dic, param_lst = get_conv2d_transpose(activation=a, dtype=dtype)
        conv2d = tvm.IRModule.from_expr(conv2d)
        config = conv2d, dic, param_lst
        run_and_verify_func(config, run_module=run_module, dtype=dtype)

        conv2d_bias, dic, param_lst = get_conv2d_transpose_bias(activation=a, dtype=dtype)
        conv2d_bias = tvm.IRModule.from_expr(conv2d_bias)
        config = conv2d_bias, dic, param_lst
        run_and_verify_func(config, run_module=run_module, dtype=dtype)


def test_conv3d(run_module, dtype="float32"):
    conv3d, dic, param_lst = get_conv3d(dtype=dtype)
    conv3d = tvm.IRModule.from_expr(conv3d)
    config = conv3d, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)

    conv3d, dic, param_lst = get_conv3d(padding=(0, 0, 0, 1, 1, 1), dtype=dtype)
    conv3d = tvm.IRModule.from_expr(conv3d)
    config = conv3d, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)

    conv3d, dic, param_lst = get_conv3d(
        x_shape=(1, 3, 8, 8, 8), k_shape=(16, 3, 3, 3, 3), dtype=dtype
    )
    conv3d = tvm.IRModule.from_expr(conv3d)
    config = conv3d, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)


def test_conv3d_pattern(run_module, dtype="float32"):
    activation_lst = [None, "relu", "tanh", "sigmoid"]
    for a in activation_lst:
        conv3d, dic, param_lst = get_conv3d(activation=a, dtype=dtype)
        conv3d = tvm.IRModule.from_expr(conv3d)
        config = conv3d, dic, param_lst
        run_and_verify_func(config, run_module=run_module, dtype=dtype)

        conv3d_bias, dic, param_lst = get_conv3d_bias(activation=a, dtype=dtype)
        conv3d_bias = tvm.IRModule.from_expr(conv3d_bias)
        config = conv3d_bias, dic, param_lst
        run_and_verify_func(config, run_module=run_module, dtype=dtype)


def test_conv3d_transpose(run_module, dtype="float32"):
    conv3d_transpose, dic, param_lst = get_conv3d_transpose(dtype=dtype)
    conv3d_transpose = tvm.IRModule.from_expr(conv3d_transpose)
    config = conv3d_transpose, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)

    conv3d_transpose, dic, param_lst = get_conv3d_transpose(strides=(2, 2, 2), dtype=dtype)
    conv3d_transpose = tvm.IRModule.from_expr(conv3d_transpose)
    config = conv3d_transpose, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)

    conv3d_transpose, dic, param_lst = get_conv3d_transpose(
        strides=(2, 2, 2), output_padding=(1, 1, 1), dtype=dtype
    )
    conv3d_transpose = tvm.IRModule.from_expr(conv3d_transpose)
    config = conv3d_transpose, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)


def test_conv3d_transpose_pattern(run_module, dtype="float32"):
    activation_lst = [None, "relu", "tanh", "sigmoid"]
    for a in activation_lst:
        conv3d, dic, param_lst = get_conv3d_transpose(activation=a, dtype=dtype)
        conv3d = tvm.IRModule.from_expr(conv3d)
        config = conv3d, dic, param_lst
        run_and_verify_func(config, run_module=run_module, dtype=dtype)

        conv3d_bias, dic, param_lst = get_conv3d_transpose_bias(activation=a, dtype=dtype)
        conv3d_bias = tvm.IRModule.from_expr(conv3d_bias)
        config = conv3d_bias, dic, param_lst
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

    dense, dic, param_lst = get_dense(x_shape, k_shape, activation="gelu", dtype=dtype)
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

    dense_bias, dic, param_lst = get_dense_bias(x_shape, k_shape, activation="gelu", dtype=dtype)
    dense_bias = tvm.IRModule.from_expr(dense_bias)
    config = dense_bias, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)


def test_pool2d(run_module, dtype="float32"):
    def get_graph(
        op,
        x_shape=(1, 3, 32, 32),
        pool_size=(2, 2),
        strides=(2, 2),
        padding=(0, 0),
        ceil_mode=False,
        count_include_pad=None,
    ):
        x = relay.var("x", shape=(x_shape), dtype=dtype)
        if count_include_pad is not None:
            out = op(
                x,
                pool_size=pool_size,
                strides=strides,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
            )
        else:
            out = op(
                x,
                pool_size=pool_size,
                strides=strides,
                padding=padding,
                ceil_mode=ceil_mode,
            )
        out = tvm.IRModule.from_expr(out)
        return out, {"x": x_shape}, []

    for pool_size in [(2, 2), (3, 3)]:
        for strides in [(1, 1), (2, 2)]:
            for padding in [(0, 0), (1, 1), (0, 0, 1, 1)]:
                for ceil_mode in [False]:
                    # Skip "the padding size is larger than or equal to the filter size for exclusive-counting pooling"
                    if pool_size == (2, 2) and padding == (0, 0, 1, 1):
                        continue
                    for count_include_pad in [False, True]:
                        # Skip "inclusive-counted blended or average pooling is not supported in combination with asymmetric padding"
                        if count_include_pad and (padding == (0, 0, 1, 1) or strides == (2, 2)):
                            continue
                        run_and_verify_func(
                            get_graph(
                                relay.nn.avg_pool2d,
                                pool_size=pool_size,
                                strides=strides,
                                padding=padding,
                                ceil_mode=ceil_mode,
                                count_include_pad=count_include_pad,
                            ),
                            run_module=run_module,
                        )
                    run_and_verify_func(
                        get_graph(
                            relay.nn.max_pool2d,
                            pool_size=pool_size,
                            strides=strides,
                            padding=padding,
                            ceil_mode=ceil_mode,
                        ),
                        run_module=run_module,
                    )


def test_pool3d(run_module, dtype="float32"):
    def get_graph(
        op,
        x_shape=(1, 3, 8, 32, 32),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        padding=(0, 0, 0),
        ceil_mode=False,
        count_include_pad=None,
        dtype="float32",
    ):
        x = relay.var("x", shape=(x_shape), dtype=dtype)
        if count_include_pad is not None:
            out = op(
                x,
                pool_size=pool_size,
                strides=strides,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
            )
        else:
            out = op(
                x,
                pool_size=pool_size,
                strides=strides,
                padding=padding,
                ceil_mode=ceil_mode,
            )
        out = tvm.IRModule.from_expr(out)
        return out, {"x": x_shape}, []

    run_and_verify_func(get_graph(relay.nn.avg_pool3d), run_module=run_module)
    run_and_verify_func(get_graph(relay.nn.max_pool3d), run_module=run_module)
    run_and_verify_func(
        get_graph(relay.nn.max_pool3d, padding=(0, 0, 0, 1, 1, 1)), run_module=run_module
    )
    run_and_verify_func(get_graph(relay.nn.max_pool3d, strides=(1, 1, 1)), run_module=run_module)


def test_prune_dnnl_subgraph(run_module):
    """In this test, OP "add" should be offloaded from dnnl codegen."""

    def get_graph():
        x1 = relay.var("x1", shape=(1, 32, 56, 56))
        x2 = relay.var("x2", shape=(1, 32, 56, 56))
        bias = relay.var("bias", shape=(32,))
        weight = relay.var("weight", shape=(32, 32, 3, 3))
        y = relay.nn.conv2d(
            x1,
            weight,
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        y = relay.nn.bias_add(y, bias)
        y = relay.nn.relu(y)
        y = relay.nn.global_max_pool2d(y)
        y = relay.add(y, x2)
        dic = {
            "x1": (1, 32, 56, 56),
            "x2": (1, 32, 56, 56),
            "weight": (32, 32, 3, 3),
            "bias": (32,),
        }
        param_lst = ["weight", "bias"]
        out = tvm.IRModule.from_expr(y)
        return out, dic, param_lst

    run_and_verify_func(get_graph(), subgraph_num=1, run_module=run_module, test_bf16=False)


def test_layer_norm(run_module, dtype="float32"):
    x_shape = (1, 49, 64)

    ln, dic, param_lst = get_layer_norm(x_shape, dtype=dtype)
    ln = tvm.IRModule.from_expr(ln)
    config = ln, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)


def test_rewrite_dense_bias_gelu_reshape_last(run_module, dtype="float32"):
    def get_graph(act=None):
        x_shape = (1, 16)
        k_shape = (32, 16)

        dense_bias, dic, param_lst = get_dense_bias(
            x_shape, k_shape, activation=act, has_reshape=True, use_add=True, dtype=dtype
        )
        dense_bias = tvm.IRModule.from_expr(dense_bias)
        processed_dense_bias = partition_for_dnnl(
            dense_bias, params=None, alter_layout=False, prune_subgraphs=False
        )
        check_dnnl_used(processed_dense_bias, 1)

        return dense_bias, dic, param_lst

    run_and_verify_func(
        get_graph("gelu"), subgraph_num=1, run_module=run_module, dtype=dtype, test_bf16=False
    )
    run_and_verify_func(
        get_graph(), subgraph_num=1, run_module=run_module, dtype=dtype, test_bf16=False
    )


if __name__ == "__main__":
    tvm.testing.main()
