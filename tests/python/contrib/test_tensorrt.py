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
import itertools
import mxnet as mx
from mxnet.gluon.model_zoo.vision import get_model

import tvm
import tvm.relay.testing
from tvm import relay
from tvm.relay.op.contrib import tensorrt
from tvm.contrib import graph_runtime, utils
from tvm.runtime.vm import VirtualMachine
from tvm.relay import Any, GlobalVar, transform


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


def vmobj_to_list(o):
    if isinstance(o, tvm.nd.NDArray):
        return [o.asnumpy()]
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
    ctx = tvm.context(target)

    result_dict = dict()
    for mode in ["graph", "vm"]:
        for use_trt in [False, True]:
            mod = tvm.IRModule()
            mod["main"] = f
            result_key = mode + ("_trt" if use_trt else "")
            if use_trt:
                mod, config = tensorrt.partition_for_tensorrt(mod, params)
                with tvm.transform.PassContext(
                    opt_level=3, config={"relay.ext.tensorrt.options": config}
                ):
                    exec = relay.create_executor(mode, mod=mod, ctx=ctx, target=target)
            else:
                with tvm.transform.PassContext(opt_level=3):
                    exec = relay.create_executor(mode, mod=mod, ctx=ctx, target=target)
            if not skip_runtime_test():
                result_dict[result_key] = exec.evaluate()(**input_dict, **params)

    if not skip_runtime_test():
        assert_result_dict_holds(result_dict)


def run_and_verify_model(model):
    if skip_codegen_test():
        return

    def check_trt_used(mod):
        num_trt_subgraphs = sum(
            [1 if gv.name_hint == "tensorrt_0" else 0 for gv in mod.get_global_vars()]
        )
        assert num_trt_subgraphs == 1

    def compile_and_run(mod, params, i_data, mode="vm", use_trt=True):
        assert mode in ["graph", "vm"]

        if use_trt:
            mod, config = tensorrt.partition_for_tensorrt(mod, params)
            check_trt_used(mod)
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.ext.tensorrt.options": config}
            ):
                exec = relay.create_executor(mode, mod=mod, ctx=tvm.gpu(0), target="cuda")
        else:
            with tvm.transform.PassContext(opt_level=3):
                exec = relay.create_executor(mode, mod=mod, ctx=tvm.gpu(0), target="cuda")

        res = exec.evaluate()(i_data, **params) if not skip_runtime_test() else None
        return res

    dtype = "float32"
    input_shape = (1, 3, 224, 224)
    i_data = np.random.uniform(-1, 1, input_shape).astype(dtype)
    block = get_model(model, pretrained=True)
    mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)

    result_dict = dict()
    for mode in ["vm", "graph"]:
        for use_trt in [True, False]:
            result_key = mode + ("_trt" if use_trt else "")
            result_dict[result_key] = compile_and_run(
                mod, params, i_data, mode=mode, use_trt=use_trt
            )

    if not skip_runtime_test():
        assert_result_dict_holds(result_dict)


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

    x_data = np.random.uniform(-1, 1, xshape).astype(dtype)
    y_data = np.random.uniform(-1, 1, yshape).astype(dtype)
    z_data = np.random.uniform(-1, 1, zshape).astype(dtype)

    result_dict = dict()
    for mode in ["vm", "graph"]:
        for use_trt in [True, False]:
            mod = tvm.IRModule()
            mod["main"] = f
            result_key = mode + ("_trt" if use_trt else "")
            if use_trt:
                mod, config = tensorrt.partition_for_tensorrt(mod)
                with tvm.transform.PassContext(
                    opt_level=3, config={"relay.ext.tensorrt.options": config}
                ):
                    relay_exec = relay.create_executor(mode, mod=mod, ctx=tvm.gpu(0), target="cuda")
            else:
                with tvm.transform.PassContext(opt_level=3):
                    relay_exec = relay.create_executor(mode, mod=mod, ctx=tvm.gpu(0), target="cuda")
            if not skip_runtime_test():
                result_dict[result_key] = relay_exec.evaluate()(x_data, y_data, z_data)

    if not skip_runtime_test():
        assert_result_dict_holds(result_dict)


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
    x_data = np.random.uniform(-1, 1, xshape).astype(dtype)

    x = relay.var("x", shape=(xshape), dtype=dtype)
    y = relay.add(x, x)
    z = relay.erf(y)
    out = relay.nn.relu(z)
    f = relay.Function([x], out)
    mod = tvm.IRModule()
    mod["main"] = f
    mod, config = tensorrt.partition_for_tensorrt(mod)
    for mode in ["graph", "vm"]:
        with tvm.transform.PassContext(opt_level=3, config={"relay.ext.tensorrt.options": config}):
            exec = relay.create_executor(mode, mod=mod, ctx=tvm.gpu(0), target="cuda")
            if not skip_runtime_test():
                results = exec.evaluate()(x_data)


def test_tensorrt_serialize_graph_runtime():
    if skip_codegen_test():
        return

    data_shape = (1, 3, 224, 224)
    data_type = "float32"
    i_data = np.random.uniform(0, 1, data_shape).astype(data_type)
    block = get_model("resnet18_v1", pretrained=True)
    mod, params = relay.frontend.from_mxnet(block, shape={"data": data_shape}, dtype=data_type)
    mod, config = tensorrt.partition_for_tensorrt(mod)
    tmpdir = utils.tempdir()

    def compile_graph(mod, params):
        with tvm.transform.PassContext(opt_level=3, config={"relay.ext.tensorrt.options": config}):
            graph, lib, params = relay.build(mod, params=params, target="cuda")
            params = relay.save_param_dict(params)
        return graph, lib, params

    def run_graph(graph, lib, params):
        mod_ = graph_runtime.create(graph, lib, ctx=tvm.gpu(0))
        mod_.load_params(params)
        mod_.run(data=i_data)
        res = mod_.get_output(0)
        return res

    def save_graph(graph, lib, params):
        # Serialize
        with open(tmpdir.relpath("compiled.json"), "w") as f_graph_json:
            f_graph_json.write(graph)
        with open(tmpdir.relpath("compiled.params"), "wb") as f_params:
            f_params.write(params)
        lib.export_library(tmpdir.relpath("compiled.so"))

    def load_graph():
        # Deserialize
        with open(tmpdir.relpath("compiled.json"), "r") as f_graph_json:
            graph = f_graph_json.read()
        with open(tmpdir.relpath("compiled.params"), "rb") as f_params:
            params = bytearray(f_params.read())
        lib = tvm.runtime.load_module(tmpdir.relpath("compiled.so"))
        return graph, lib, params

    # Test serialization with graph runtime
    graph, lib, graph_params = compile_graph(mod, params)
    save_graph(graph, lib, graph_params)
    loaded_graph, loaded_lib, loaded_params = load_graph()

    if not skip_runtime_test():
        result_dict = dict()
        result_dict["graph"] = run_graph(graph, lib, graph_params)
        result_dict["graph_ref"] = run_graph(loaded_graph, loaded_lib, loaded_params)
        assert_result_dict_holds(result_dict)


def test_tensorrt_serialize_vm():
    if skip_codegen_test():
        return

    data_shape = (1, 3, 224, 224)
    data_type = "float32"
    i_data = np.random.uniform(0, 1, data_shape).astype(data_type)
    block = get_model("resnet18_v1", pretrained=True)
    mod, params = relay.frontend.from_mxnet(block, shape={"data": data_shape}, dtype=data_type)
    mod, config = tensorrt.partition_for_tensorrt(mod)
    tmpdir = utils.tempdir()

    def compile_vm(mod, params):
        with tvm.transform.PassContext(opt_level=3, config={"relay.ext.tensorrt.options": config}):
            vm_exec = relay.vm.compile(mod, target="cuda", params=params)
            code, lib = vm_exec.save()
        return code, lib

    def run_vm(code, lib):
        vm_exec = tvm.runtime.vm.Executable.load_exec(code, lib)
        vm = VirtualMachine(vm_exec, tvm.gpu(0))
        result = vm.invoke("main", data=i_data)
        return result

    def save_vm(code, lib):
        # save and load the code and lib file.
        lib.export_library(tmpdir.relpath("path_lib.so"))
        with open(tmpdir.relpath("path_code.ro"), "wb") as fo:
            fo.write(code)

    def load_vm():
        lib = tvm.runtime.load_module(tmpdir.relpath("path_lib.so"))
        code = bytearray(open(tmpdir.relpath("path_code.ro"), "rb").read())
        return lib, code

    # Test serialization with VM
    code_vm, lib_vm = compile_vm(mod, params)
    save_vm(code_vm, lib_vm)
    loaded_lib_vm, loaded_code_vm = load_vm()

    if not skip_runtime_test():
        result_dict = dict()
        result_dict["vm"] = run_vm(code_vm, lib_vm)
        result_dict["vm_ref"] = run_vm(loaded_code_vm, loaded_lib_vm)
        assert_result_dict_holds(result_dict)


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
    def get_graph(x_shape, begin, end, strides=None, slice_mode="size"):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        if strides:
            out = relay.strided_slice(
                x,
                begin,
                end,
                strides,
                slice_mode=slice_mode,
            )
        else:
            out = relay.strided_slice(
                x,
                begin,
                end,
                slice_mode=slice_mode,
            )
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    for slice_mode in ["size", "end"]:
        run_and_verify_func(
            get_graph((1, 3, 6, 7), (0, 0, 0, 0), (1, 1, 6, 7), slice_mode=slice_mode)
        )
        run_and_verify_func(
            get_graph((1, 3, 6, 7), [0, 1, 0, 0], [1, 2, 6, 6], slice_mode=slice_mode)
        )
        run_and_verify_func(
            get_graph((2, 3, 6, 7), [0, 0, 0, 0], [-1, -1, -1, -1], slice_mode=slice_mode)
        )
        run_and_verify_func(
            get_graph((2, 3, 6, 7), [0, 1, 0, 0], [-1, -1, -1, -1], slice_mode=slice_mode)
        )
        run_and_verify_func(get_graph((1, 6), [0, 1], [1, 3], slice_mode=slice_mode))


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


def test_tensorrt_integration():
    # Integration tests
    test_alexnet()
    test_resnet18_v1()
    test_resnet18_v2()
    test_squeezenet()
    test_mobilenet()
    test_mobilenet_v2()
    test_vgg11()
    test_densenet121()


def test_dynamic_offload():
    """
    This test checks for proper dynamic offloading of relay graphs. An addition between
    the outputs of two conv2d's is performed, one of them having all static args whereas
    the other has a arg with dynamic shape. It is expected for the TRT partitioner to
    offload the conv2d with dynamic arg to TVM while running the other in TRT.
    """

    if skip_codegen_test():
        return

    data_shape = (1, 32, 8, 8)
    k_shape = (1, 32, 3, 3)

    x = relay.var("x", shape=(data_shape[0], data_shape[1], Any(), Any()), dtype="float32")
    y = relay.var("y", shape=(data_shape), dtype="float32")
    kernel = relay.var("kernel", shape=(k_shape), dtype="float32")

    def get_expected():
        def set_func_attr(func, compile_name, symbol_name):
            func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
            func = func.with_attr("Inline", tvm.tir.IntImm("int32", 1))
            func = func.with_attr("Compiler", compile_name)
            func = func.with_attr("global_symbol", symbol_name)
            return func

        # Create a nested TRT function that matches the expected output
        mod = tvm.IRModule()
        var1 = relay.var("tensorrt_0_i0", shape=(data_shape), dtype="float32")
        kernel_trt = relay.var("tensorrt_0_i1", shape=(k_shape), dtype="float32")
        out1 = relay.nn.conv2d(var1, kernel_trt, channels=k_shape[0], kernel_size=k_shape[2:4])
        f1 = GlobalVar("tensorrt_0")
        func = relay.Function([var1, kernel_trt], out1)
        func = set_func_attr(func, "tensorrt", "tensorrt_0")
        mod[f1] = func
        mod = relay.transform.InferType()(mod)

        # Create the main function
        out1 = relay.nn.conv2d(x, kernel, channels=k_shape[0], kernel_size=k_shape[2:4])
        out = relay.add(out1, f1(y, kernel))
        f = relay.Function([x, y, kernel], out)
        mod["main"] = f
        mod = relay.transform.InferType()(mod)
        return mod

    # Create relay function that will be offloaded to TRT
    out1 = relay.nn.conv2d(x, kernel, channels=k_shape[0], kernel_size=k_shape[2:4])
    out2 = relay.nn.conv2d(y, kernel, channels=k_shape[0], kernel_size=k_shape[2:4])
    out = relay.add(out1, out2)
    f = relay.Function([x, y, kernel], out)

    # Pass the function to TRT compilation
    mod = tvm.IRModule()
    mod["main"] = f
    mod = relay.transform.InferType()(mod)
    mod_trt, config = tensorrt.partition_for_tensorrt(mod, params={})

    # Get the expected relay graph and compare
    mod_exp = get_expected()
    tvm.ir.assert_structural_equal(mod_trt, mod_exp, map_free_vars=True)


if __name__ == "__main__":
    pytest.main([__file__])
