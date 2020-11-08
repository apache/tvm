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
import time
import pytest

import tvm
import tvm.relay.testing
from tvm import relay
from tvm.relay.op.contrib import tensorrt
from tvm.contrib import graph_runtime


def skip_codegen_test():
    """Skip test if TensorRT and CUDA codegen are not present"""
    if not tvm.runtime.enabled("cuda") or not tvm.gpu(0).exist:
        print("Skip because CUDA is not enabled.")
        return True
    if not tvm.get_global_func("relay.ext.tensorrt", True):
        print("Skip because TensorRT codegen is not available.")
        return True
    return False


def skip_runtime_test():
    if not tvm.runtime.enabled("cuda") or not tvm.gpu(0).exist:
        print("Skip because CUDA is not enabled.")
        return True
    if not tensorrt.is_tensorrt_runtime_enabled():
        print("Skip because TensorRT runtime is not available.")
        return True
    return False


def run_and_verify_func(config, target="cuda"):
    """Test a Relay func by compiling, running, and comparing TVM and TRT outputs.

    Parameters
    ----------
    config : Tuple[relay.Function, Dict[str, NDArray], List[str]]
        A tuple containing 1) The function to test, 2) A dictionary of var names to input shapes and
        3) A list of which vars should be considered params.
    """
    if skip_codegen_test():
        return
    f, input_shapes, is_param = config
    params = {x: np.random.uniform(-1, 1, input_shapes[x]).astype(np.float32) for x in is_param}
    input_dict = {
        k: np.random.uniform(-1, 1, v).astype(np.float32)
        for k, v in input_shapes.items()
        if k not in is_param
    }

    # Run TRT
    mod = tvm.IRModule()
    mod["main"] = f
    mod, config = tensorrt.partition_for_tensorrt(mod, params)
    with tvm.transform.PassContext(opt_level=3, config={"relay.ext.tensorrt.options": config}):
        graph, lib, graph_params = relay.build(mod, target, params=params)
    if skip_runtime_test():
        return
    ctx = tvm.context(target)
    mod = graph_runtime.create(graph, lib, ctx=ctx)
    mod.set_input(**graph_params)
    mod.run(**input_dict)
    results = [mod.get_output(i) for i in range(mod.get_num_outputs())]

    # Run reference
    mod = tvm.IRModule()
    mod["main"] = f
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, graph_params = relay.build(mod, target, params=params)
    mod = graph_runtime.create(graph, lib, ctx=ctx)
    mod.set_input(**graph_params)
    mod.run(**input_dict)
    ref_results = [mod.get_output(i) for i in range(mod.get_num_outputs())]

    assert len(results) == len(ref_results)
    for i in range(len(results)):
        res = results[i].asnumpy()
        ref_res = ref_results[i].asnumpy()
        assert res.shape == ref_res.shape
        tvm.testing.assert_allclose(res, ref_res, rtol=1e-3, atol=1e-3)


def run_and_verify_model(model):
    if skip_codegen_test():
        return

    def compile_and_run(i_data, input_shape, dtype, use_trt=True, num_iteration=1):
        import mxnet as mx
        from mxnet.gluon.model_zoo.vision import get_model

        def check_trt_used(graph):
            import json

            graph = json.loads(graph)
            num_trt_subgraphs = sum(
                [
                    1
                    for n in graph["nodes"]
                    if n.get("attrs", {}).get("func_name", "").startswith("tensorrt_")
                ]
            )
            assert num_trt_subgraphs >= 1

        block = get_model(model, pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)

        if use_trt:
            mod, config = tensorrt.partition_for_tensorrt(mod, params)
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.ext.tensorrt.options": config}
            ):
                graph, lib, params = relay.build(mod, "cuda", params=params)
            check_trt_used(graph)
        else:
            with tvm.transform.PassContext(opt_level=3):
                graph, lib, params = relay.build(mod, "cuda", params=params)

        if skip_runtime_test():
            return
        mod = graph_runtime.create(graph, lib, ctx=tvm.gpu(0))
        mod.set_input(**params)
        # Warmup
        for i in range(10):
            mod.run(data=i_data)
        # Time
        times = []
        for i in range(num_iteration):
            start_time = time.time()
            mod.run(data=i_data)
            res = mod.get_output(0)
            times.append(time.time() - start_time)
        latency = 1000.0 * np.mean(times)
        print(model, latency)
        return res

    dtype = "float32"
    input_shape = (1, 3, 224, 224)
    i_data = np.random.uniform(-1, 1, input_shape).astype(dtype)
    res = compile_and_run(i_data, input_shape, dtype, use_trt=True)
    if skip_runtime_test():
        return
    ref_res = compile_and_run(i_data, input_shape, dtype, use_trt=False)
    tvm.testing.assert_allclose(res.asnumpy(), ref_res.asnumpy(), rtol=1e-3, atol=1e-3)


def test_tensorrt_simple():
    if skip_codegen_test():
        return
    dtype = "float32"
    xshape = (1, 3, 2, 2)
    yshape = (1, 3, 1, 1)
    zshape = (1, 1, 1, 1)
    x = relay.var("x", shape=(xshape), dtype=dtype)
    y = relay.var("y", shape=(yshape), dtype=dtype)
    z = relay.var("z", shape=(zshape), dtype=dtype)
    w = z * (x + y)
    out = relay.nn.relu(w)
    f = relay.Function([x, y, z], out)

    mod = tvm.IRModule()
    mod["main"] = f
    mod, config = tensorrt.partition_for_tensorrt(mod)
    with tvm.transform.PassContext(opt_level=3, config={"relay.ext.tensorrt.options": config}):
        graph, lib, params = relay.build(mod, "cuda")
    if skip_runtime_test():
        return
    mod = graph_runtime.create(graph, lib, ctx=tvm.gpu(0))
    x_data = np.random.uniform(-1, 1, xshape).astype(dtype)
    y_data = np.random.uniform(-1, 1, yshape).astype(dtype)
    z_data = np.random.uniform(-1, 1, zshape).astype(dtype)
    mod.run(x=x_data, y=y_data, z=z_data)
    results = [mod.get_output(i).asnumpy() for i in range(mod.get_num_outputs())]


def test_tensorrt_simple_cpu_io():
    def get_graph():
        dtype = "float32"
        x_shape = (1, 3, 2, 2)
        y_shape = (1, 3, 1, 1)
        z_shape = (1, 1, 1, 1)
        x = relay.var("x", shape=(x_shape), dtype=dtype)
        y = relay.var("y", shape=(y_shape), dtype=dtype)
        z = relay.var("z", shape=(z_shape), dtype=dtype)
        w = z * (x + y)
        out = relay.nn.relu(w)
        f = relay.Function([x, y, z], out)
        return f, {"x": x_shape, "y": y_shape, "z": z_shape}, ["y"]

    run_and_verify_func(get_graph(), target="llvm")


def test_tensorrt_not_compatible():
    if skip_codegen_test():
        return
    dtype = "float32"
    xshape = (1, 32, 14, 14)
    x = relay.var("x", shape=(xshape), dtype=dtype)
    y = relay.add(x, x)
    z = relay.erf(y)
    out = relay.nn.relu(z)
    f = relay.Function([x], out)
    mod = tvm.IRModule()
    mod["main"] = f
    mod, config = tensorrt.partition_for_tensorrt(mod)
    with tvm.transform.PassContext(opt_level=3, config={"relay.ext.tensorrt.options": config}):
        graph, lib, params = relay.build(mod, "cuda")
    if skip_runtime_test():
        return
    mod = graph_runtime.create(graph, lib, ctx=tvm.gpu(0))
    x_data = np.random.uniform(-1, 1, xshape).astype(dtype)
    mod.run(x=x_data)
    results = [mod.get_output(i).asnumpy() for i in range(mod.get_num_outputs())]


def test_tensorrt_serialize():
    if skip_codegen_test():
        return
    import mxnet
    from mxnet.gluon.model_zoo.vision import get_model

    block = get_model("resnet18_v1", pretrained=True)
    mod, params = relay.frontend.from_mxnet(
        block, shape={"data": (1, 3, 224, 224)}, dtype="float32"
    )
    # Compile
    mod, config = tensorrt.partition_for_tensorrt(mod, params)
    with tvm.transform.PassContext(opt_level=3, config={"relay.ext.tensorrt.options": config}):
        lib = relay.build(mod, "cuda", params=params)
    # Serialize
    lib.export_library("compiled.so")
    # Deserialize
    loaded_lib = tvm.runtime.load_module("compiled.so")
    # Run
    if skip_runtime_test():
        return
    gen_module = tvm.contrib.graph_runtime.GraphModule(loaded_lib["default"](tvm.gpu(0)))
    i_data = np.random.uniform(0, 1, (1, 3, 224, 224)).astype("float32")
    gen_module.run(data=i_data)


def test_conv2d():
    def get_graph(
        x_shape=(1, 32, 8, 8),
        k_shape=(16, 32, 3, 3),
        groups=1,
        padding=(0, 0),
        strides=(1, 1),
        dilation=(1, 1),
    ):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        kernel = relay.var("kernel", shape=(k_shape), dtype="float32")
        out = relay.nn.conv2d(
            x,
            kernel,
            kernel_size=k_shape[2:4],
            groups=groups,
            padding=padding,
            strides=strides,
            dilation=dilation,
        )
        f = relay.Function([x, kernel], out)
        return f, {"x": x_shape, "kernel": k_shape}, ["kernel"]

    for k_shape, groups in [((16, 32, 3, 3), 1), ((32, 1, 3, 3), 32)]:
        for padding in [(0, 0), (1, 1)]:
            for strides in [(1, 1), (2, 2)]:
                for dilation in [(1, 1), (2, 2)]:
                    run_and_verify_func(
                        get_graph(
                            k_shape=k_shape,
                            groups=groups,
                            padding=padding,
                            strides=strides,
                            dilation=dilation,
                        )
                    )


def test_conv2d_nhwc():
    def get_graph(x_shape=(1, 8, 8, 32), k_shape=(3, 3, 32, 16)):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        kernel = relay.var("kernel", shape=(k_shape), dtype="float32")
        out = relay.nn.conv2d(
            x,
            kernel,
            channels=16,
            kernel_size=(3, 3),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        f = relay.Function([x, kernel], out)
        return f, {"x": x_shape, "kernel": k_shape}, ["kernel"]

    run_and_verify_func(get_graph())


def test_conv2d_weights_const():
    def get_graph(
        x_shape=(1, 32, 8, 8),
        k_shape=(16, 32, 3, 3),
        groups=1,
        padding=(0, 0),
        strides=(1, 1),
        dilation=(1, 1),
    ):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        kernel = relay.const(np.ones(k_shape).astype("float32"))
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
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph())


def test_conv2d_weights_transposed():
    def get_graph(x_shape=(1, 32, 9, 9), k_shape=(3, 3, 32, 16), order=(3, 2, 0, 1)):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        kernel = relay.var("kernel", shape=(k_shape), dtype="float32")
        kernel_t = relay.transpose(kernel, order)
        # Conv2d requires constant weights in TensorRT, so the weights should be transposed by
        # FoldConstant.
        out = relay.nn.conv2d(x, kernel_t, channels=k_shape[order[0]], kernel_size=(3, 3))
        f = relay.Function([x, kernel], out)
        return f, {"x": x_shape, "kernel": k_shape}, ["kernel"]

    run_and_verify_func(get_graph())


def test_dense():
    def get_graph(x_shape=(1, 16), k_shape=(32, 16)):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        kernel = relay.var("kernel", shape=(k_shape), dtype="float32")
        # Dense requires constant weights in TensorRT, so the weights are transposed by us.
        out = relay.nn.dense(x, kernel, units=k_shape[0])
        f = relay.Function([x, kernel], out)
        return f, {"x": x_shape, "kernel": k_shape}, ["kernel"]

    run_and_verify_func(get_graph())


def test_bias_add():
    def get_graph(x_shape=(1, 16), channels=16):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        bias = relay.var("bias", shape=(channels,), dtype="float32")
        out = relay.nn.bias_add(x, bias)
        f = relay.Function([x, bias], out)
        return f, {"x": x_shape, "bias": (channels,)}, ["bias"]

    run_and_verify_func(get_graph())
    run_and_verify_func(get_graph((1, 6, 3, 4), 6))


def test_pool2d():
    def get_graph(
        op,
        x_shape=(1, 3, 32, 32),
        pool_size=(2, 2),
        strides=(2, 2),
        padding=(0, 0),
        ceil_mode=False,
        count_include_pad=None,
    ):
        x = relay.var("x", shape=(x_shape), dtype="float32")
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
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    for pool_size in [(2, 2), (3, 3)]:
        for strides in [(1, 1), (2, 2)]:
            for padding in [(0, 0), (1, 1), (0, 0, 1, 1)]:
                for ceil_mode in [False, True]:
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
                            )
                        )
                    run_and_verify_func(
                        get_graph(
                            relay.nn.max_pool2d,
                            pool_size=pool_size,
                            strides=strides,
                            padding=padding,
                            ceil_mode=ceil_mode,
                        )
                    )


def test_global_pool2d():
    def get_graph(op, x_shape=(1, 3, 32, 32)):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        out = op(x)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph(relay.nn.global_max_pool2d))
    run_and_verify_func(get_graph(relay.nn.global_avg_pool2d))


def test_batch_flatten():
    def get_graph(x_shape=(1, 3, 4, 6)):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        out = relay.nn.batch_flatten(x)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph())


def test_expand_dims():
    def get_graph(x_shape=(1, 3), axis=1, num_newaxis=1):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        out = relay.expand_dims(x, axis, num_newaxis)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph())


def test_squeeze():
    def get_graph(x_shape, axis):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        out = relay.squeeze(x, axis=axis)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph((1, 5, 1, 1), (2, 3)))
    run_and_verify_func(get_graph((1, 3, 1), (-1,)))


def test_concatenate():
    def get_graph(input_shapes, axis):
        concat_inputs = []
        shapes_dict = {}
        for i in range(len(input_shapes)):
            name = "input_{}".format(i)
            concat_inputs.append(relay.var(name, shape=(input_shapes[i]), dtype="float32"))
            shapes_dict[name] = input_shapes[i]
        out = relay.concatenate(concat_inputs, axis)
        f = relay.Function(concat_inputs, out)
        return f, shapes_dict, []

    run_and_verify_func(get_graph([(1, 2, 6, 6), (1, 3, 6, 6)], axis=1))


def test_conv2d_transpose():
    def get_graph(
        x_shape=(1, 32, 8, 8),
        k_shape=(32, 16, 3, 3),
        groups=1,
        padding=(0, 0),
        strides=(1, 1),
    ):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        kernel = relay.var("kernel", shape=(k_shape), dtype="float32")
        out = relay.nn.conv2d_transpose(
            x,
            kernel,
            channels=k_shape[1],
            kernel_size=k_shape[2:4],
            groups=groups,
            padding=padding,
            strides=strides,
        )
        f = relay.Function([x, kernel], out)
        return f, {"x": x_shape, "kernel": k_shape}, ["kernel"]

    for padding in [(0, 0), (1, 1)]:
        for strides in [(1, 1), (2, 2)]:
            run_and_verify_func(get_graph(padding=padding, strides=strides))


def test_reshape():
    def get_graph(x_shape, new_shape):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        out = relay.reshape(x, new_shape)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph((1, 1, 1, 10), (-1, 10)))
    run_and_verify_func(get_graph((1, 10, 2, 3), (1, -1)))
    run_and_verify_func(get_graph((1, 1, 2, 3), (1, 6)))


def test_transpose():
    def get_graph(x_shape, order):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        out = relay.transpose(x, order)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph((1, 16, 7, 7), [0, 2, 3, 1]))
    run_and_verify_func(get_graph((1, 7, 7, 16), [0, 3, 1, 2]))


def test_float_const():
    def get_graph(x_shape=(1, 16)):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        beta = relay.const(1, dtype="float32")
        out = relay.multiply(x, beta)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph())


def test_pad():
    def get_graph(x_shape, pad_width):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        out = relay.nn.pad(x, pad_width=pad_width)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph((1, 8, 16, 16), [[0, 0], [0, 0], [0, 0], [0, 0]]))
    run_and_verify_func(get_graph((1, 8, 16, 16), [[0, 0], [0, 0], [1, 1], [1, 1]]))
    run_and_verify_func(get_graph((1, 8, 16, 16), [[0, 0], [0, 0], [0, 1], [2, 0]]))
    run_and_verify_func(get_graph((1, 8, 3, 16, 16), [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]))


def test_softmax():
    def get_graph(x_shape, axis):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        out = relay.nn.softmax(x, axis=axis)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph((1, 1000), axis=1))
    run_and_verify_func(get_graph((1, 1000), axis=-1))
    run_and_verify_func(get_graph((1, 3, 4), axis=-2))
    run_and_verify_func(get_graph((1, 3, 4), axis=1))


def test_batch_norm():
    def get_graph(x_shape, param_shape, axis=1, epsilon=1e-5):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        beta = relay.var("beta", shape=(param_shape), dtype="float32")
        gamma = relay.var("gamma", shape=(param_shape), dtype="float32")
        moving_mean = relay.var("moving_mean", shape=(param_shape), dtype="float32")
        moving_var = relay.var("moving_var", shape=(param_shape), dtype="float32")
        out, _, _ = relay.nn.batch_norm(
            x,
            gamma=gamma,
            beta=beta,
            moving_mean=moving_mean,
            moving_var=moving_var,
            axis=axis,
            center=True,
            scale=True,
            epsilon=epsilon,
        )
        f = relay.Function([x, gamma, beta, moving_mean, moving_var], out)
        return (
            f,
            {
                "x": x_shape,
                "beta": param_shape,
                "gamma": param_shape,
                "moving_mean": param_shape,
                "moving_var": param_shape,
            },
            ["beta", "gamma", "moving_mean", "moving_var"],
        )

    run_and_verify_func(get_graph((1, 64, 56, 56), (64,)))
    run_and_verify_func(get_graph((1, 56, 56, 64), (64,), axis=3, epsilon=1.001e-05))


def test_unary():
    def get_graph(op, x_shape=(1, 8, 3, 3)):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        out = op(x)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    for op in [
        relay.nn.relu,
        relay.sigmoid,
        relay.tanh,
        relay.exp,
        relay.log,
        relay.sqrt,
        relay.abs,
        relay.negative,
        relay.sin,
        relay.cos,
        relay.atan,
        relay.ceil,
        relay.floor,
    ]:
        run_and_verify_func(get_graph(op))


def test_clip():
    def get_graph(x_shape=(1, 8, 3, 3)):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        out = relay.clip(x, a_min=-0.2, a_max=0.4)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph())


def test_leaky_relu():
    def get_graph(x_shape=(1, 8, 3, 3)):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        out = relay.nn.leaky_relu(x, alpha=0.1)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph())


def test_binary():
    def get_graph(op, x_shape, y_shape, y_is_const=False):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        if y_is_const:
            y = relay.const(np.ones(y_shape).astype("float32"))
            out = op(x, y)
            f = relay.Function([x], out)
            return f, {"x": x_shape}, []
        y = relay.var("y", shape=(y_shape), dtype="float32")
        out = op(x, y)
        f = relay.Function([x, y], out)
        return f, {"x": x_shape, "y": y_shape}, []

    for op in [relay.add, relay.subtract, relay.multiply, relay.divide, relay.power]:
        for y_is_const in [True, False]:
            run_and_verify_func(get_graph(op, (1, 8, 3, 3), (1, 8, 3, 3), y_is_const))
            run_and_verify_func(get_graph(op, (1, 8, 1, 3), (1, 8, 3, 1), y_is_const))
            run_and_verify_func(get_graph(op, (1, 10), (10,), y_is_const))
            run_and_verify_func(get_graph(op, (1, 1, 1, 10), (10,), y_is_const))
            run_and_verify_func(get_graph(op, (1, 1, 1), (3,), y_is_const))


def test_reduce():
    def get_graph(op, x_shape=(1, 2, 3, 4), axis=(2, 3), keepdims=False):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        out = op(x, axis=axis, keepdims=keepdims)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    for op in [relay.sum, relay.prod, relay.max, relay.min, relay.mean]:
        for keepdims in [True, False]:
            run_and_verify_func(get_graph(op, axis=(1), keepdims=keepdims))
            run_and_verify_func(get_graph(op, axis=(2, 3), keepdims=keepdims))
            run_and_verify_func(get_graph(op, axis=(1, 2), keepdims=keepdims))
            run_and_verify_func(get_graph(op, axis=(1, 2, 3), keepdims=keepdims))


def test_strided_slice():
    def get_graph(x_shape, begin, end, strides=None):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        if strides:
            out = relay.strided_slice(
                x,
                relay.expr.const(begin, dtype="int32"),
                relay.expr.const(end, dtype="int32"),
                relay.expr.const(strides, dtype="int32"),
            )
        else:
            out = relay.strided_slice(
                x,
                relay.expr.const(begin, dtype="int32"),
                relay.expr.const(end, dtype="int32"),
            )
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph((1, 3, 6, 7), [0, 0, 0, 0], [1, 1, 6, 7]))
    run_and_verify_func(get_graph((1, 3, 6, 7), [0, 1, 0, 0], [1, 2, 6, 6]))
    run_and_verify_func(get_graph((1, 10), [0, 0], [1, 10], [1, 2]))


def test_adaptive_pool2d():
    def get_graph(op, x_shape=(1, 3, 32, 32), out_size=(1, 1)):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        out = op(x, out_size)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph(relay.nn.adaptive_max_pool2d))
    run_and_verify_func(get_graph(relay.nn.adaptive_avg_pool2d))


def test_multiple_outputs():
    def get_graph():
        x = relay.var("x", shape=(1, 3), dtype="float32")
        y = relay.var("y", shape=(1, 3), dtype="float32")
        z = relay.add(x, y)
        w = relay.add(z, y)
        out = relay.Tuple((z, w))
        f = relay.Function([x, y], out)
        return f, {"x": (1, 3), "y": (1, 3)}, []

    run_and_verify_func(get_graph())


def test_conv3d():
    def get_graph(
        x_shape=(1, 32, 8, 8, 8),
        k_shape=(16, 32, 3, 3, 3),
        groups=1,
        padding=(0, 0, 0),
        strides=(1, 1, 1),
        dilation=(1, 1, 1),
    ):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        kernel = relay.var("kernel", shape=(k_shape), dtype="float32")
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
        f = relay.Function([x, kernel], out)
        return f, {"x": x_shape, "kernel": k_shape}, ["kernel"]

    run_and_verify_func(get_graph())
    run_and_verify_func(get_graph(padding=(0, 0, 0, 1, 1, 1)))


def test_pool3d():
    def get_graph(
        op,
        x_shape=(1, 3, 8, 32, 32),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        padding=(0, 0, 0),
        ceil_mode=False,
        count_include_pad=None,
    ):
        x = relay.var("x", shape=(x_shape), dtype="float32")
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
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph(relay.nn.avg_pool3d))
    run_and_verify_func(get_graph(relay.nn.max_pool3d))
    run_and_verify_func(get_graph(relay.nn.max_pool3d, padding=(0, 0, 0, 1, 1, 1)))
    run_and_verify_func(get_graph(relay.nn.max_pool3d, strides=(1, 1, 1)))


def test_conv3d_transpose():
    def get_graph(
        x_shape=(1, 32, 8, 8, 8),
        k_shape=(32, 16, 3, 3, 3),
        groups=1,
        padding=(0, 0, 0),
        strides=(1, 1, 1),
        output_padding=(0, 0, 0),
    ):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        kernel = relay.var("kernel", shape=(k_shape), dtype="float32")
        out = relay.nn.conv3d_transpose(
            x,
            kernel,
            channels=k_shape[1],
            kernel_size=k_shape[2:5],
            groups=groups,
            padding=padding,
            strides=strides,
            output_padding=output_padding,
        )
        f = relay.Function([x, kernel], out)
        return f, {"x": x_shape, "kernel": k_shape}, ["kernel"]

    run_and_verify_func(get_graph())
    run_and_verify_func(get_graph(strides=(2, 2, 2)))
    run_and_verify_func(get_graph(strides=(2, 2, 2), output_padding=(1, 1, 1)))


def test_alexnet():
    run_and_verify_model("alexnet")


def test_resnet18_v1():
    run_and_verify_model("resnet18_v1")


def test_resnet18_v2():
    run_and_verify_model("resnet18_v2")


def test_squeezenet():
    run_and_verify_model("squeezenet1.0")


def test_mobilenet():
    run_and_verify_model("mobilenet0.25")


def test_mobilenet_v2():
    run_and_verify_model("mobilenetv2_0.25")


def test_vgg11():
    run_and_verify_model("vgg11")


def test_densenet121():
    run_and_verify_model("densenet121")


if __name__ == "__main__":
    test_tensorrt_not_compatible()
    test_tensorrt_simple()
    test_tensorrt_simple_cpu_io()
    test_tensorrt_serialize()

    # Op tests
    test_conv2d()
    test_conv2d_nhwc()
    test_conv2d_weights_const()
    test_conv2d_weights_transposed()
    test_dense()
    test_bias_add()
    test_pool2d()
    test_global_pool2d()
    test_batch_flatten()
    test_expand_dims()
    test_squeeze()
    test_concatenate()
    test_conv2d_transpose()
    test_reshape()
    test_transpose()
    test_float_const()
    test_pad()
    test_softmax()
    test_batch_norm()
    test_unary()
    test_clip()
    test_leaky_relu()
    test_binary()
    test_reduce()
    test_strided_slice()
    test_adaptive_pool2d()
    test_multiple_outputs()
    test_conv3d()
    test_pool3d()
    test_conv3d_transpose()

    # Integration tests
    test_alexnet()
    test_resnet18_v1()
    test_resnet18_v2()
    test_squeezenet()
    test_mobilenet()
    test_mobilenet_v2()
    test_vgg11()
    test_densenet121()
