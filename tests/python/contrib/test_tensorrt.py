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

import itertools
import logging
from typing import Tuple

import numpy as np
import pytest

try:
    # See issue #9362.
    import torch
except:
    pass

import tvm
import tvm.relay.testing
import tvm.testing
from tvm import relay
from tvm.contrib.download import download
from tvm.relay import Any, GlobalVar
from tvm.relay.expr_functor import ExprVisitor
from tvm.relay.op.contrib import tensorrt

SUPPORTED_DTYPES = ["float16", "float32"]

has_tensorrt_codegen = pytest.mark.skipif(
    not tensorrt.is_tensorrt_compiler_enabled(), reason="TensorRT codegen not available"
)

# CAUTION: Currently always false in CI since adds tens of minutes to test time and depends
# on TensorRT installation. See https://github.com/apache/tvm/issues/11765
has_tensorrt_runtime = pytest.mark.skipif(
    not tensorrt.is_tensorrt_runtime_enabled(), reason="TensorRT runtime not available"
)

run_module = tvm.testing.parameter(
    pytest.param(False, marks=[has_tensorrt_codegen, *tvm.testing.requires_cuda.marks()]),
    pytest.param(
        True, marks=[has_tensorrt_runtime, has_tensorrt_codegen, *tvm.testing.requires_cuda.marks()]
    ),
    ids=["compile", "run"],
)


def vmobj_to_list(o):
    if isinstance(o, tvm.nd.NDArray):
        return [o.numpy()]
    elif isinstance(o, tvm.runtime.container.ADT) or isinstance(o, list):
        return [vmobj_to_list(f) for f in o]
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))


def assert_result_dict_holds(result_dict, dtype="float16"):
    for k1, k2 in itertools.combinations(result_dict, 2):
        res1 = vmobj_to_list(result_dict[k1])
        res2 = vmobj_to_list(result_dict[k2])
        for r1, r2 in zip(res1, res2):
            if dtype == "float16":
                tvm.testing.assert_allclose(r1, r2, rtol=1e-1, atol=1e-1)
            else:
                tvm.testing.assert_allclose(r1, r2, rtol=1e-3, atol=5e-3)


def set_outer_func_attr(func, compile_name, symbol_name):
    func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Inline", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Compiler", compile_name)
    func = func.with_attr("global_symbol", symbol_name)
    return func


def set_inner_func_attr(func, pattern_name, composite_name):
    func = func.with_attr("PartitionedFromPattern", pattern_name)
    func = func.with_attr("Composite", composite_name)
    return func


def run_and_verify_func(config, target="cuda", run_module=True, data_type="float32"):
    """Test a Relay func by compiling, running, and comparing TVM and TRT outputs.

    Parameters
    ----------
    config : Tuple[relay.Function, Dict[str, NDArray], List[str]]
        A tuple containing 1) The function to test, 2) A dictionary of var names to input shapes and
        3) A list of which vars should be considered params.

    run_module: bool

        If True, the built module will be run after being compiled.

    data_type: str
        Check between single and double floating precision
    """
    np.random.seed(42)
    f, input_shapes, is_param = config
    params = {
        x: np.random.uniform(-1, 1, input_shapes[x]).astype(dtype=data_type) for x in is_param
    }
    input_dict = {
        k: np.random.uniform(-1, 1, v).astype(dtype=data_type)
        for k, v in input_shapes.items()
        if k not in is_param
    }
    dev = tvm.device(target)

    result_dict = dict()
    for mode in ["vm", "graph"]:
        for use_trt in [True, False]:
            mod = tvm.IRModule()
            mod["main"] = f
            result_key = mode + ("_trt" if use_trt else "")
            if use_trt:
                use_fp16 = data_type == "float16"
                trt_target = tvm.target.Target(f"tensorrt -use_fp16={use_fp16}")
                mod = relay.transform.InferType()(mod)
                mod = tensorrt.partition_for_tensorrt(mod, params=params, target=trt_target)
                with tvm.transform.PassContext(opt_level=3):
                    func = relay.create_executor(
                        mode, mod=mod, device=dev, target=[target, trt_target]
                    ).evaluate()
            else:
                mod = relay.transform.InferType()(mod)
                with tvm.transform.PassContext(opt_level=3):
                    func = relay.create_executor(
                        mode, mod=mod, device=dev, target=target
                    ).evaluate()

            if run_module:
                result_dict[result_key] = func(**input_dict, **params)

            if run_module:
                assert_result_dict_holds(result_dict, data_type)


def test_tensorrt_simple(run_module):
    for dtype in SUPPORTED_DTYPES:
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
            for use_trt in [False, True]:
                mod = tvm.IRModule()
                mod["main"] = f
                result_key = mode + ("_trt" if use_trt else "")
                if use_trt:
                    mod = relay.transform.InferType()(mod)
                    mod = tensorrt.partition_for_tensorrt(mod)
                    with tvm.transform.PassContext(opt_level=3):
                        func = relay.create_executor(
                            mode, mod=mod, device=tvm.cuda(0), target="cuda"
                        ).evaluate()
                else:
                    mod = relay.transform.InferType()(mod)
                    with tvm.transform.PassContext(opt_level=3):
                        func = relay.create_executor(
                            mode, mod=mod, device=tvm.cuda(0), target="cuda"
                        ).evaluate()
                if run_module:
                    result_dict[result_key] = func(x_data, y_data, z_data)

        if run_module:
            assert_result_dict_holds(result_dict)


def test_tensorrt_simple_cpu_io(run_module):
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

    run_and_verify_func(get_graph(), target="llvm", run_module=run_module)


def test_tensorrt_not_compatible(run_module):
    dtype = "float32"
    xshape = (1, 32, 14, 14)
    x_data = np.random.uniform(-1, 1, xshape).astype(dtype)

    x = relay.var("x", shape=(xshape), dtype=dtype)
    y = relay.add(x, x)
    z = relay.cast(relay.cast(y, "int32"), "float32")
    out = relay.nn.relu(z)
    f = relay.Function([x], out)
    mod = tvm.IRModule()
    mod["main"] = f
    mod = tensorrt.partition_for_tensorrt(mod)
    for mode in ["graph", "vm"]:
        with tvm.transform.PassContext(opt_level=3):
            func = relay.create_executor(
                mode, mod=mod, device=tvm.cuda(0), target="cuda"
            ).evaluate()
            if run_module:
                results = func(x_data)


def test_conv1d(run_module):
    def get_graph(
        x_shape=((1, 3, 224)),
        k_shape=(10, 3, 3),
        groups=1,
        padding=(1, 1),
        strides=(1),
        dilation=(1),
        channels=None,
        d_type="float16",
    ):
        x = relay.var("x", shape=(x_shape), dtype=d_type)
        kernel = relay.var("kernel", shape=(k_shape), dtype=d_type)
        out = relay.nn.conv1d(
            x,
            kernel,
            kernel_size=k_shape[2:3],
            groups=groups,
            padding=padding,
            strides=strides,
            dilation=dilation,
            channels=channels,
            out_dtype="float16",
        )
        f = relay.Function([x, kernel], out)
        return f, {"x": x_shape, "kernel": k_shape}, ["kernel"]

    for d_type in ["float16"]:
        run_and_verify_func(
            get_graph(channels=10, d_type=d_type), run_module=run_module, data_type=d_type
        )


def test_conv2d(run_module):
    def get_graph(
        x_shape=(1, 32, 8, 8),
        k_shape=(16, 32, 3, 3),
        groups=1,
        padding=(0, 0),
        strides=(1, 1),
        dilation=(1, 1),
        channels=None,
        data_type="float16",
    ):
        x = relay.var("x", shape=(x_shape), dtype=data_type)
        kernel = relay.var("kernel", shape=(k_shape), dtype=data_type)
        out = relay.nn.conv2d(
            x,
            kernel,
            kernel_size=k_shape[2:4],
            groups=groups,
            padding=padding,
            strides=strides,
            dilation=dilation,
            channels=channels,
            out_dtype=data_type,
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
                        ),
                        run_module=run_module,
                        data_type="float16",
                    )
    run_and_verify_func(
        get_graph(
            (1, 3, 16, 16), (3, 8, 7, 7), 3, [2, 2, 3, 3], [2, 2], [1, 1], 24, data_type="float16"
        ),
        run_module=run_module,
        data_type="float16",
    )

    run_and_verify_func(
        get_graph((1, 3, 16, 16), (1, 3, 1, 1), channels=1, data_type="float32"),
        run_module=run_module,
        data_type="float32",
    )


def test_conv2d_nhwc(run_module):
    def get_graph(x_shape=(1, 8, 8, 32), k_shape=(3, 3, 32, 16)):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        kernel = relay.var("kernel", shape=(k_shape), dtype="float32")
        out = relay.nn.conv2d(
            x, kernel, channels=16, kernel_size=(3, 3), data_layout="NHWC", kernel_layout="HWIO"
        )
        f = relay.Function([x, kernel], out)
        return f, {"x": x_shape, "kernel": k_shape}, ["kernel"]

    run_and_verify_func(get_graph(), run_module=run_module)


def test_conv2d_weights_const(run_module):
    def get_graph(
        x_shape=(1, 32, 8, 8),
        k_shape=(16, 32, 3, 3),
        groups=1,
        padding=(0, 0),
        strides=(1, 1),
        dilation=(1, 1),
        data_type="float16",
    ):
        x = relay.var("x", shape=(x_shape), dtype=data_type)
        kernel = relay.const(np.ones(k_shape).astype(dtype=data_type))
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

    for tp in ["float16"]:
        run_and_verify_func(get_graph(data_type=tp), run_module=run_module, data_type=tp)


def test_conv2d_weights_transposed(run_module):
    def get_graph(x_shape=(1, 32, 9, 9), k_shape=(3, 3, 32, 16), order=(3, 2, 0, 1)):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        kernel = relay.var("kernel", shape=(k_shape), dtype="float32")
        kernel_t = relay.transpose(kernel, order)
        # Conv2d requires constant weights in TensorRT, so the weights should be transposed by
        # FoldConstant.
        out = relay.nn.conv2d(x, kernel_t, channels=k_shape[order[0]], kernel_size=(3, 3))
        f = relay.Function([x, kernel], out)
        return f, {"x": x_shape, "kernel": k_shape}, ["kernel"]

    run_and_verify_func(get_graph(), run_module=run_module)


def test_dense(run_module):
    def get_graph(x_shape=(1, 16), k_shape=(32, 16), dtp="float16"):
        x = relay.var("x", shape=(x_shape), dtype=dtp)
        kernel = relay.var("kernel", shape=(k_shape), dtype=dtp)
        # Dense requires constant weights in TensorRT, so the weights are transposed by us.
        out = relay.nn.dense(x, kernel, units=k_shape[0])
        f = relay.Function([x, kernel], out)
        return f, {"x": x_shape, "kernel": k_shape}, ["kernel"]

    for tp in ["float32"]:
        run_and_verify_func(get_graph(dtp=tp), run_module=run_module, data_type=tp)
        run_and_verify_func(get_graph(k_shape=(1, 16), dtp=tp), run_module=run_module, data_type=tp)


def test_batch_matmul(run_module):
    def get_graph(x_shape=(12, 128, 64), y_shape=(12, 128, 64), transa=False, transb=True):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        y = relay.var("y", shape=(y_shape), dtype="float32")
        out = relay.nn.batch_matmul(x, y, transpose_a=transa, transpose_b=transb)
        f = relay.Function([x, y], out)
        return f, {"x": x_shape, "y": y_shape}, []

    run_and_verify_func(
        get_graph(x_shape=(12, 64, 128), y_shape=(12, 128, 64), transa=True, transb=True),
        run_module=run_module,
    )
    run_and_verify_func(
        get_graph(x_shape=(12, 64, 128), y_shape=(12, 64, 128), transa=True, transb=False),
        run_module=run_module,
    )
    run_and_verify_func(
        get_graph(x_shape=(12, 128, 64), y_shape=(12, 128, 64), transa=False, transb=True),
        run_module=run_module,
    )
    run_and_verify_func(
        get_graph(x_shape=(12, 128, 64), y_shape=(12, 64, 128), transa=False, transb=False),
        run_module=run_module,
    )


def test_bias_add(run_module):
    def get_graph(x_shape=(1, 16), channels=16, axis=1):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        bias = relay.var("bias", shape=(channels,), dtype="float32")
        out = relay.nn.bias_add(x, bias, axis)
        f = relay.Function([x, bias], out)
        return f, {"x": x_shape, "bias": (channels,)}, ["bias"]

    run_and_verify_func(get_graph(), run_module=run_module)
    run_and_verify_func(get_graph((1, 6, 3, 4), 6), run_module=run_module)
    run_and_verify_func(get_graph((1, 6, 3, 4), 4, -1), run_module=run_module)


def test_pool2d(run_module):
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
            out = op(x, pool_size=pool_size, strides=strides, padding=padding, ceil_mode=ceil_mode)
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


def test_global_pool2d(run_module):
    def get_graph(op, x_shape=(1, 3, 32, 32)):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        out = op(x)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph(relay.nn.global_max_pool2d), run_module=run_module)
    run_and_verify_func(get_graph(relay.nn.global_avg_pool2d), run_module=run_module)


def test_batch_flatten(run_module):
    def get_graph(x_shape=(1, 3, 4, 6), data_type="float16"):
        x = relay.var("x", shape=(x_shape), dtype=data_type)
        out = relay.nn.batch_flatten(x)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    for dtp in ["float16", "float32"]:
        run_and_verify_func(get_graph(data_type=dtp), run_module=run_module, data_type=dtp)


def test_expand_dims(run_module):
    def get_graph(x_shape=(1, 3), axis=1, num_newaxis=1):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        out = relay.expand_dims(x, axis, num_newaxis)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph(), run_module=run_module)


def test_squeeze(run_module):
    def get_graph(x_shape, axis, dtype):
        x = relay.var("x", shape=(x_shape), dtype=dtype)
        out = relay.squeeze(x, axis=axis)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    for dtype in SUPPORTED_DTYPES:
        run_and_verify_func(
            get_graph((1, 5, 1, 1), (2, 3), dtype=dtype), run_module=run_module, data_type=dtype
        )
        run_and_verify_func(
            get_graph((1, 3, 1), (-1,), dtype=dtype), run_module=run_module, data_type=dtype
        )


def test_concatenate(run_module):
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

    run_and_verify_func(get_graph([(1, 2, 6, 6), (1, 3, 6, 6)], axis=1), run_module=run_module)


def test_split(run_module):
    def get_graph(x_shape, indices_or_sections, axis):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        out = relay.split(x, indices_or_sections=indices_or_sections, axis=axis)
        f = relay.Function([x], out.astuple())
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph((1, 16), indices_or_sections=2, axis=1), run_module=run_module)
    run_and_verify_func(get_graph((1, 16), indices_or_sections=4, axis=1), run_module=run_module)
    run_and_verify_func(get_graph((1, 16), indices_or_sections=[8], axis=1), run_module=run_module)
    run_and_verify_func(
        get_graph((1, 16), indices_or_sections=[2, 3, 6, 10, 14], axis=1), run_module=run_module
    )


def test_conv2d_transpose(run_module):
    def get_graph(
        x_shape=(1, 32, 8, 8), k_shape=(32, 16, 3, 3), groups=1, padding=(0, 0), strides=(1, 1)
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
            run_and_verify_func(get_graph(padding=padding, strides=strides), run_module=run_module)


def test_reshape(run_module):
    def get_graph(x_shape, new_shape):
        x = relay.var("x", shape=(x_shape), dtype="float16")
        out = relay.reshape(x, new_shape)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(
        get_graph((1, 1, 1, 10), (-1, 10)), run_module=run_module, data_type="float16"
    )
    run_and_verify_func(
        get_graph((1, 10, 2, 3), (1, -1)), run_module=run_module, data_type="float16"
    )
    run_and_verify_func(get_graph((1, 1, 2, 3), (1, 6)), run_module=run_module, data_type="float16")


class AreOpsOnGraph(ExprVisitor):
    """
    Visits the Graph recursively and checks if it contains ops in the op_list
    """

    def __init__(self, op_list):
        ExprVisitor.__init__(self)
        self.op_list = op_list
        self.on_graph = False

    def visit_call(self, call):
        if isinstance(call.op, tvm.tir.op.Op):
            if str(call.op.name) in self.op_list:
                self.on_graph = True

        return super().visit_call(call)

    def are_ops_on_graph(self, subgraph) -> bool:
        """
        This function recursively visits the graph and checks if op_list ops are ongraph"
        """
        self.visit(subgraph)
        return self.on_graph


def are_ops_on_trt(mod, op_list):
    op_on_trt = False
    op_on_tvm = False
    for subgraph in mod.get_global_vars():
        name = subgraph.name_hint
        if mod[name].attrs and mod[name].attrs["Compiler"] == "tensorrt":
            op_on_trt |= AreOpsOnGraph(op_list).are_ops_on_graph(mod[name].body)
        else:
            op_on_tvm |= AreOpsOnGraph(op_list).are_ops_on_graph(mod[name].body)

    return op_on_trt and not op_on_tvm


def test_dynamic_reshape(run_module):
    def test_run(x_data_list, x_shape, new_shape, should_offload_to_trt):
        result_arr = [{} for _ in range(len(x_data_list))]
        for use_trt in [True, False]:
            x = relay.var("x", shape=x_shape, dtype="float32")
            out = relay.reshape(x, new_shape)
            f = relay.Function([x], out)
            mod = tvm.IRModule()
            mod["main"] = f
            if use_trt:
                logging.info("Before partitioning:\n%s", mod)
                mod = tensorrt.partition_for_tensorrt(mod)
                logging.info("After partitioning:\n%s", mod)
                assert are_ops_on_trt(mod, op_list=["reshape"]) == should_offload_to_trt
            if run_module:
                with relay.build_config(opt_level=3):
                    func = relay.create_executor(
                        "vm", mod=mod, device=tvm.cpu(0), target="llvm"
                    ).evaluate()

                for i, x_data in enumerate(x_data_list):
                    result_arr[i][use_trt] = func(x_data)

        if run_module:
            for i in range(len(x_data_list)):
                assert_result_dict_holds(result_arr[i])

    dim_values = [1, 1, 0, 2, 3, 0, 1, 3, 2]
    x_shape = (relay.Any(), 3, 2, 3)
    x_data_list = [
        np.ones([dim_value] + list(x_shape)[1:]).astype("float32") for dim_value in dim_values
    ]
    new_shape = (-1, 3, 2, 3)
    should_offload_to_trt = True
    test_run(x_data_list, x_shape, new_shape, should_offload_to_trt)

    dim_values = [1, 1, 0, 2, 3, 0, 1, 3, 2]
    x_shape = (relay.Any(), 3, 2, 3)
    x_data_list = [
        np.ones([dim_value] + list(x_shape)[1:]).astype("float32") for dim_value in dim_values
    ]
    new_shape = (-1, 1, 2, 3)
    should_offload_to_trt = False
    test_run(x_data_list, x_shape, new_shape, should_offload_to_trt)

    dim_values = [1, 1, 0, 2, 3, 0, 1, 3, 2]
    x_shape = (1, relay.Any(), 2, 3)
    x_data_list = [
        np.ones(list(x_shape[:1]) + [dim_value] + list(x_shape)[2:]).astype("float32")
        for dim_value in dim_values
    ]
    new_shape = (1, -1, 2, 3)
    should_offload_to_trt = False
    test_run(x_data_list, x_shape, new_shape, should_offload_to_trt)


def test_transpose(run_module):
    def get_graph(x_shape, order):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        out = relay.transpose(x, order)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph((1, 16, 7, 7), [0, 2, 3, 1]), run_module=run_module)
    run_and_verify_func(get_graph((1, 7, 7, 16), [0, 3, 1, 2]), run_module=run_module)


def test_float_const(run_module):
    def get_graph(x_shape=(1, 16)):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        beta = relay.const(1, dtype="float32")
        out = relay.multiply(x, beta)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph(), run_module=run_module, data_type="float32")


def test_float_const16(run_module):
    def get_graph(x_shape=(1, 16)):
        x = relay.var("x", shape=(x_shape), dtype="float16")
        beta = relay.const(1, dtype="float16")
        out = relay.multiply(x, beta)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph(), run_module=run_module, data_type="float16")


def test_pad(run_module):
    def get_graph(x_shape, pad_width):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        out = relay.nn.pad(x, pad_width=pad_width)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(
        get_graph((1, 8, 16, 16), [[0, 0], [0, 0], [0, 0], [0, 0]]), run_module=run_module
    )
    run_and_verify_func(
        get_graph((1, 8, 16, 16), [[0, 0], [0, 0], [1, 1], [1, 1]]), run_module=run_module
    )
    run_and_verify_func(
        get_graph((1, 8, 16, 16), [[0, 0], [0, 0], [0, 1], [2, 0]]), run_module=run_module
    )
    run_and_verify_func(
        get_graph((1, 8, 3, 16, 16), [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),
        run_module=run_module,
    )


def test_add(run_module):
    def get_graph(x_shape):
        x = relay.var("x", shape=(x_shape), dtype="float16")
        y = relay.var("y", shape=(x_shape), dtype="float16")
        out = relay.add(x, y)
        f = relay.Function([x, y], out)
        return f, {"x": x_shape, "y": x_shape}, []

    run_and_verify_func(get_graph((1, 1000)), run_module=run_module, data_type="float16")


def test_softmax(run_module):
    def get_graph(x_shape, axis, data_type="float32"):
        x = relay.var("x", shape=(x_shape), dtype=data_type)
        out = relay.nn.softmax(x, axis=axis)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(
        get_graph((1, 1000), axis=1, data_type="float32"),
        run_module=run_module,
        data_type="float32",
    )
    run_and_verify_func(
        get_graph((1, 1000), axis=-1, data_type="float32"),
        run_module=run_module,
        data_type="float32",
    )
    run_and_verify_func(
        get_graph((1, 3, 4), axis=-2, data_type="float16"),
        run_module=run_module,
        data_type="float16",
    )
    run_and_verify_func(
        get_graph((1, 3, 4), axis=1, data_type="float16"),
        run_module=run_module,
        data_type="float16",
    )


def test_batch_norm(run_module):
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

    run_and_verify_func(get_graph((1, 64, 56, 56), (64,)), run_module=run_module)
    run_and_verify_func(
        get_graph((1, 56, 56, 64), (64,), axis=3, epsilon=1.001e-05), run_module=run_module
    )
    run_and_verify_func(get_graph((1, 4, 8, 4), (8,), axis=2), run_module=run_module)
    run_and_verify_func(get_graph((1, 8, 4, 4, 4), (8,), axis=1), run_module=run_module)
    run_and_verify_func(get_graph((1, 4, 8, 4, 4), (8,), axis=2), run_module=run_module)
    run_and_verify_func(get_graph((1, 4, 4, 4, 8), (8,), axis=4), run_module=run_module)
    run_and_verify_func(get_graph((1, 8), (8,), axis=1), run_module=run_module)
    run_and_verify_func(get_graph((1, 3, 8), (8,), axis=2), run_module=run_module)


def test_layer_norm(run_module):
    def get_graph(x_shape, param_shape, axis=1, epsilon=1e-5):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        gamma = relay.var("gamma", shape=(param_shape), dtype="float32")
        beta = relay.var("beta", shape=(param_shape), dtype="float32")
        out = relay.nn.layer_norm(
            x, gamma=gamma, beta=beta, axis=axis, epsilon=epsilon, center=True, scale=True
        )
        f = relay.Function([x, gamma, beta], out)
        return (f, {"x": x_shape, "beta": param_shape, "gamma": param_shape}, ["beta", "gamma"])

    run_and_verify_func(get_graph((1, 32, 8, 8), (32,)), run_module=run_module)
    run_and_verify_func(
        get_graph((1, 8, 8, 32), (32,), axis=3, epsilon=1.001e-05), run_module=run_module
    )
    run_and_verify_func(get_graph((1, 8), (8,), axis=1), run_module=run_module)


def test_unary(run_module):
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
        relay.erf,
    ]:
        run_and_verify_func(get_graph(op), run_module=run_module)


def test_clip(run_module):
    def get_graph(x_shape=(1, 8, 3, 3)):
        x = relay.var("x", shape=(x_shape), dtype="float16")
        out = relay.clip(x, a_min=-0.2, a_max=0.4)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph(), run_module=run_module, data_type="float16")


def test_relu(run_module):
    def get_graph(x_shape=(1, 8, 3, 4)):
        x = relay.var("x", shape=(x_shape), dtype="float16")
        out = relay.nn.relu(x)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph(), run_module=run_module, data_type="float16")


def test_leaky_relu(run_module):
    def get_graph(x_shape=(1, 8, 3, 4)):
        x = relay.var("x", shape=(x_shape), dtype="float16")
        out = relay.nn.leaky_relu(x, alpha=0.1)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph(), run_module=run_module, data_type="float16")


def test_binary(run_module):
    def get_graph(op, x_shape, y_shape, y_is_const=False, d_type="float16"):
        x = relay.var("x", shape=(x_shape), dtype=d_type)
        if y_is_const:
            y = relay.const(np.ones(y_shape).astype(d_type))
            out = op(x, y)
            f = relay.Function([x], out)
            return f, {"x": x_shape}, []
        y = relay.var("y", shape=(y_shape), dtype=d_type)
        out = op(x, y)
        f = relay.Function([x, y], out)
        return f, {"x": x_shape, "y": y_shape}, []

    for op in [relay.add, relay.subtract, relay.multiply, relay.divide, relay.power]:
        for d_type in SUPPORTED_DTYPES:
            for y_is_const in [True, False]:
                run_and_verify_func(
                    get_graph(op, (1, 8, 3, 3), (1, 8, 3, 3), y_is_const, d_type),
                    run_module=run_module,
                    data_type=d_type,
                )
                run_and_verify_func(
                    get_graph(op, (1, 8, 1, 3), (1, 8, 3, 1), y_is_const, d_type),
                    run_module=run_module,
                    data_type=d_type,
                )
                run_and_verify_func(
                    get_graph(op, (1, 10), (10,), y_is_const, d_type),
                    run_module=run_module,
                    data_type=d_type,
                )
                run_and_verify_func(
                    get_graph(op, (1, 1, 1, 10), (10,), y_is_const, d_type),
                    run_module=run_module,
                    data_type=d_type,
                )
                run_and_verify_func(
                    get_graph(op, (1, 1, 1), (3,), y_is_const, d_type),
                    run_module=run_module,
                    data_type=d_type,
                )


def test_reduce(run_module):
    def get_graph(op, x_shape=(1, 2, 3, 4), axis=(2, 3), keepdims=False, d_type="float32"):
        x = relay.var("x", shape=(x_shape), dtype=d_type)
        out = op(x, axis=axis, keepdims=keepdims)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    for type in SUPPORTED_DTYPES:
        for op in [relay.sum, relay.prod, relay.max, relay.min, relay.mean]:
            for keepdims in [True, False]:
                run_and_verify_func(
                    get_graph(op, axis=(1), keepdims=keepdims, d_type=type),
                    run_module=run_module,
                    data_type=type,
                )
                run_and_verify_func(
                    get_graph(op, axis=(2, 3), keepdims=keepdims, d_type=type),
                    run_module=run_module,
                    data_type=type,
                )
                run_and_verify_func(
                    get_graph(op, axis=(1, 2), keepdims=keepdims, d_type=type),
                    run_module=run_module,
                    data_type=type,
                )
                run_and_verify_func(
                    get_graph(op, axis=(1, 2, 3), keepdims=keepdims, d_type=type),
                    run_module=run_module,
                    data_type=type,
                )


def test_strided_slice(run_module):
    def get_graph(x_shape, begin, end, strides=None, slice_mode="size"):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        if strides:
            out = relay.strided_slice(x, begin, end, strides, slice_mode=slice_mode)
        else:
            out = relay.strided_slice(x, begin, end, slice_mode=slice_mode)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    for slice_mode in ["size", "end"]:
        run_and_verify_func(
            get_graph((1, 3, 6, 7), (0, 0, 0, 0), (1, 1, 6, 7), slice_mode=slice_mode),
            run_module=run_module,
        )
        run_and_verify_func(
            get_graph((1, 3, 6, 7), [0, 1, 0, 0], [1, 2, 6, 6], slice_mode=slice_mode),
            run_module=run_module,
        )
        run_and_verify_func(
            get_graph((2, 3, 6, 7), [0, 0, 0, 0], [-1, -1, -1, -1], slice_mode=slice_mode),
            run_module=run_module,
        )
        run_and_verify_func(
            get_graph((2, 3, 6, 7), [0, 1, 0, 0], [-1, -1, -1, -1], slice_mode=slice_mode),
            run_module=run_module,
        )
        run_and_verify_func(
            get_graph((1, 6), [0, 1], [1, 3], slice_mode=slice_mode), run_module=run_module
        )


def test_adaptive_pool2d(run_module):
    def get_graph(op, x_shape=(1, 3, 32, 32), out_size=(1, 1), data_type="float16"):
        x = relay.var("x", shape=(x_shape), dtype=data_type)
        out = op(x, out_size)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    for type in SUPPORTED_DTYPES:
        run_and_verify_func(
            get_graph(relay.nn.adaptive_max_pool2d, data_type=type),
            run_module=run_module,
            data_type=type,
        )
        run_and_verify_func(
            get_graph(relay.nn.adaptive_avg_pool2d, data_type=type),
            run_module=run_module,
            data_type=type,
        )


def test_multiple_outputs(run_module):
    def get_graph(d_type="float16"):
        x = relay.var("x", shape=(1, 3), dtype=d_type)
        y = relay.var("y", shape=(1, 3), dtype=d_type)
        z = relay.add(x, y)
        w = relay.add(z, y)
        out = relay.Tuple((z, w))
        f = relay.Function([x, y], out)
        return f, {"x": (1, 3), "y": (1, 3)}, []

    for type in SUPPORTED_DTYPES:
        run_and_verify_func(get_graph(d_type=type), run_module=run_module, data_type=type)


@pytest.mark.skip(reason=("Fails assert_allclose. See https://github.com/apache/tvm/issues/11765"))
def test_conv3d(run_module):
    def get_graph(
        x_shape=(1, 24, 8, 8, 8),
        k_shape=(16, 24, 3, 3, 3),
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

    run_and_verify_func(get_graph(), run_module=run_module)
    run_and_verify_func(get_graph(padding=(0, 0, 0, 1, 1, 1)), run_module=run_module)


def test_pool3d(run_module):
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
            out = op(x, pool_size=pool_size, strides=strides, padding=padding, ceil_mode=ceil_mode)
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph(relay.nn.avg_pool3d), run_module=run_module)
    run_and_verify_func(get_graph(relay.nn.max_pool3d), run_module=run_module)
    run_and_verify_func(
        get_graph(relay.nn.max_pool3d, padding=(0, 0, 0, 1, 1, 1)), run_module=run_module
    )
    run_and_verify_func(get_graph(relay.nn.max_pool3d, strides=(1, 1, 1)), run_module=run_module)


def test_conv3d_transpose(run_module):
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

    run_and_verify_func(get_graph(), run_module=run_module)
    run_and_verify_func(get_graph(strides=(2, 2, 2)), run_module=run_module)
    run_and_verify_func(
        get_graph(strides=(2, 2, 2), output_padding=(1, 1, 1)), run_module=run_module
    )


@has_tensorrt_codegen
def test_dynamic_offload():
    """
    This test checks for proper dynamic offloading of relay graphs. An addition between
    the outputs of two conv2d's is performed, one of them having all static args whereas
    the other has a arg with dynamic shape. It is expected for the TRT partitioner to
    offload the conv2d with dynamic arg to TVM while running the other in TRT.
    """

    data_shape = (1, 32, 8, 8)
    k_shape = (1, 32, 3, 3)

    x = relay.var("x", shape=(data_shape[0], data_shape[1], Any(), Any()), dtype="float32")
    y = relay.var("y", shape=(data_shape), dtype="float32")
    kernel = relay.const(np.random.rand(*k_shape).astype("float32"))

    def get_expected():
        # Create a nested TRT function that matches the expected output
        mod = tvm.IRModule()
        outer_var = relay.var("tensorrt_0_i0", shape=(data_shape), dtype="float32")
        inner_var = relay.var("FunctionVar_0_0", shape=(data_shape), dtype="float32")
        inner_body = relay.nn.conv2d(
            inner_var, kernel, channels=k_shape[0], kernel_size=k_shape[2:4]
        )
        inner_func = relay.Function([inner_var], inner_body)
        inner_func = set_inner_func_attr(inner_func, "nn.conv2d_", "tensorrt.nn.conv2d")
        outer_body = inner_func(outer_var)
        outer_func = relay.Function([outer_var], outer_body)
        outer_func = set_outer_func_attr(outer_func, "tensorrt", "tvmgen_default_tensorrt_main_0")
        gv = GlobalVar("tvmgen_default_tensorrt_main_0")
        mod[gv] = outer_func
        mod = relay.transform.InferType()(mod)

        # Create the main function
        out1 = relay.nn.conv2d(x, kernel, channels=k_shape[0], kernel_size=k_shape[2:4])
        out = relay.add(out1, gv(y))
        f = relay.Function([x, y], out)
        mod["main"] = f
        mod = relay.transform.InferType()(mod)
        return mod

    # Create relay function that will be offloaded to TRT
    out1 = relay.nn.conv2d(x, kernel, channels=k_shape[0], kernel_size=k_shape[2:4])
    out2 = relay.nn.conv2d(y, kernel, channels=k_shape[0], kernel_size=k_shape[2:4])
    out = relay.add(out1, out2)
    f = relay.Function([x, y], out)

    # Pass the function to TRT compilation
    mod = tvm.IRModule()
    mod["main"] = f
    mod = relay.transform.InferType()(mod)
    mod_trt = tensorrt.partition_for_tensorrt(mod)

    # Get the expected relay graph and compare
    mod_exp = get_expected()
    tvm.ir.assert_structural_equal(mod_trt, mod_exp, map_free_vars=True)


def test_tensorrt_dynamic_batch(run_module):
    batches_to_test = [1, 1, 0, 2, 3, 0, 1, 3, 2]
    x_shape = (relay.Any(), 1, 8, 8)
    x_data = np.ones([max(batches_to_test)] + list(x_shape)[1:]).astype("float32")
    result_arr = [{} for _ in range(len(batches_to_test))]
    for use_trt in [True, False]:
        x = relay.var("x", shape=x_shape, dtype="float32")
        out = relay.nn.relu(x)
        f = relay.Function([x], out)
        mod = tvm.IRModule()
        mod["main"] = f
        if use_trt:
            mod = tensorrt.partition_for_tensorrt(mod)

        if run_module:
            with relay.build_config(opt_level=3):
                func = relay.create_executor(
                    "vm", mod=mod, device=tvm.cpu(0), target="llvm"
                ).evaluate()
            for i, batch_size in enumerate(batches_to_test):
                result_arr[i][use_trt] = func(x_data[:batch_size, ...])

    if run_module:
        for i in range(len(batches_to_test)):
            assert_result_dict_holds(result_arr[i])


def test_tensorrt_dynamic_batch_conv(run_module):
    batches_to_test = [1, 5, 1, 0, 2, 3, 0, 1, 3, 2]
    x_shape = (relay.Any(), 32, 8, 8)
    x_data = np.ones([max(batches_to_test)] + list(x_shape)[1:]).astype("float32")
    k_shape = (16, 32, 3, 3)
    params = {"kernel": np.random.uniform(-1, 1, k_shape).astype("float32")}
    for use_implicit_batch in [True, False]:
        result_arr = [{"cuda": {}, "llvm": {}} for _ in range(len(batches_to_test))]
        for use_trt in [True, False]:
            x = relay.var("x", shape=x_shape, dtype="float32")
            kernel = relay.var("kernel", shape=k_shape, dtype="float32")
            out = relay.nn.conv2d(x, kernel, channels=16, kernel_size=(3, 3), groups=1)
            f = relay.Function([x, kernel], out)
            mod = tvm.IRModule()
            mod["main"] = f
            trt_target = tvm.target.Target(f"tensorrt -use_implicit_batch={use_implicit_batch}")
            if use_trt:
                mod = tensorrt.partition_for_tensorrt(mod, params=params, target=trt_target)
            if run_module:
                for target in ["llvm", "cuda"]:
                    targets = [target]
                    if use_trt:
                        targets.append(trt_target)
                    with tvm.transform.PassContext(opt_level=3):
                        func = relay.create_executor(
                            "vm", mod=mod, device=tvm.device(target), target=targets
                        ).evaluate()
                    for i, batch_size in enumerate(batches_to_test):
                        result_arr[i][target][use_trt] = func(x_data[:batch_size, ...], **params)
        if run_module:
            for i in range(len(batches_to_test)):
                for target in ["llvm", "cuda"]:
                    assert_result_dict_holds(result_arr[i][target])


def test_maskrcnn_resnet50(run_module) -> None:
    """
    This function tests the working of pytorch maskrcnn with resnet50 as backbone with
    VM and VM + TRT. Since the order of compiled model outputs is a bit different from
    original pytorch model, it uses a custom logic for comparison check.
    """
    import torch
    import torchvision

    def convert_traced_model_to_vm_trt(
        traced_module: torch.jit.TopLevelTracedModule, np_sample_input: np.ndarray, target: str
    ) -> tvm.runtime.vm.Executable:
        """
        This function converts a traced pytorch model to VM + TRT.
        """
        input_shape = np_sample_input.shape
        input_name = "input0"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(traced_module, shape_list)
        trt_target = tvm.target.Target("tensorrt -remove_no_mac_subgraphs=True")
        mod = tensorrt.partition_for_tensorrt(mod, params=params, target=trt_target)
        targets = [target, trt_target]
        with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"]):
            vm_trt_exec = relay.vm.compile(mod, target=targets, params=params)

        return vm_trt_exec

    class TraceWrapper(torch.nn.Module):
        """
        This class is a wrapper over the torch module to convert the outputs into traceable form
        """

        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(
            self, inp: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            out = self.model(inp)
            return out[0]["boxes"], out[0]["scores"], out[0]["labels"], out[0]["masks"]

    def get_traced_maskrcnn_model(np_sample_input: np.ndarray) -> torch.jit.TopLevelTracedModule:
        """
        This function takes a sample input and returns the traced maskrcnn model
        """
        model_func = torchvision.models.detection.maskrcnn_resnet50_fpn
        model = TraceWrapper(model_func(pretrained=True))
        model.eval()
        inp = torch.Tensor(np.random.uniform(0.0, 250.0, size=np_sample_input.shape))

        with torch.no_grad():
            out = model(inp)
            traced_module = torch.jit.trace(model, inp)
            traced_module.eval()

        return traced_module

    def get_maskrcnn_input(in_size: int) -> np.ndarray:
        """
        This function gets a real image with multiple objects of interest and returns it.
        """
        input_shape = (1, 3, in_size, in_size)
        img_path = "test_street_small.jpg"
        img_url = "https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/detection/street_small.jpg"
        download(img_url, img_path)
        import cv2

        img = cv2.imread(img_path).astype("float32")
        img = cv2.resize(img, (in_size, in_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img / 255.0, [2, 0, 1])
        img = np.expand_dims(img, axis=0)

        return img

    in_size = 300
    np_sample_input = get_maskrcnn_input(in_size)
    traced_module = get_traced_maskrcnn_model(np_sample_input)
    vm_trt_exec = convert_traced_model_to_vm_trt(traced_module, np_sample_input, target="llvm")

    if run_module:
        dev = tvm.cpu()
        vm = tvm.runtime.vm.VirtualMachine(vm_trt_exec, dev)
        vm.set_input("main", **{"input0": np_sample_input})
        tvm_res = vm.run()

        # Descending sort by scores and get the high confidence indices. In this example 9 is chosen,
        # because this image has 9 boxes over 0.9 confidence
        num_high_confidence_boxes = 9
        tvm_indices = np.argsort(-1 * tvm_res[1].numpy())[:num_high_confidence_boxes]

        with torch.no_grad():
            out = traced_module(torch.Tensor(np_sample_input))
            # Descending sort by scores and get the high confidence indices
            pt_indices = np.argsort(-1 * out[1].numpy())[:num_high_confidence_boxes]

        # [Box Tol, Score Tol, Label Tol, Mask Tol]
        tol = [1e-1, 5e-3, 1e-5, 4e-1]
        # Because of certain ops, there are certain minor differences in TVM outputs and PT outputs,
        # This means that the tolerance can't be 1e-4 or 1e-5 throughout. The ideal way to get around
        # this is to test it on an entire dataset and compare mAP with the original model.
        # However, since that is not practically possible on CI, the following compromise is made.
        # These tolerances are chosen based on their impact or lack thereof to the mAP score, e.g:
        # 0.1 pixel difference of a box in a 300X300 image wont make any change.
        for i, tol_val in zip(range(4), tol):
            np.testing.assert_allclose(
                tvm_res[i].numpy()[tvm_indices],
                out[i].numpy()[pt_indices],
                rtol=tol_val,
                atol=tol_val,
            )


def test_empty_subgraph(run_module):
    x_shape = (1, 3, 5)
    mod = tvm.IRModule()
    # Empty tensorrt subgraph.
    var1 = relay.var("tensorrt_0_i0", shape=(x_shape), dtype="float32")
    f1 = GlobalVar("tensorrt_0")
    func = relay.Function([var1], var1)
    func = set_outer_func_attr(func, "tensorrt", "tvmgen_default_tensorrt_0")
    mod[f1] = func
    mod = relay.transform.InferType()(mod)

    # Create the main function
    x = relay.var("x", shape=x_shape, dtype="float32")
    out = f1(relay.nn.relu(x))
    f = relay.Function([x], out)
    mod["main"] = f

    x_data = np.random.uniform(-1, 1, x_shape).astype("float32")
    for mode in ["graph", "vm"]:
        with tvm.transform.PassContext(opt_level=3):
            func = relay.create_executor(
                mode, mod=mod, device=tvm.cuda(0), target="cuda"
            ).evaluate()
            if run_module:
                results = func(x_data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tvm.testing.main()
