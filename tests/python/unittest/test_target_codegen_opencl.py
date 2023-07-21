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
import tvm
from tvm import te, relay
import tvm.testing
import re
import pytest
import numpy as np

target = "opencl"


@tvm.testing.requires_gpu
@tvm.testing.requires_opencl
def test_opencl_ternary_expression():
    def check_if_then_else(dev, n, dtype):
        A = te.placeholder((n,), name="A", dtype=dtype)
        true_value = tvm.tir.const(1, dtype=dtype)
        false_value = tvm.tir.const(3, dtype=dtype)
        max_lhs = tvm.tir.const(2, dtype=dtype)
        max_rhs = tvm.tir.if_then_else(A[0] > 0, true_value, false_value)
        C = te.compute((n,), lambda i: tvm.te.max(max_lhs, max_rhs), name="C")

        func = te.create_prim_func([A, C])
        sch = tvm.tir.Schedule(func)
        (x,) = sch.get_loops(sch.get_block("C"))
        sch.bind(x, "threadIdx.x")
        fun = tvm.build(sch.mod, target=target)
        a = tvm.nd.empty((n,), A.dtype, dev)
        c = tvm.nd.empty((n,), A.dtype, dev)
        # Only need to test compiling here
        fun(a, c)

    def check_select(dev, n, dtype):
        A = te.placeholder((n,), name="A", dtype=dtype)
        true_value = tvm.tir.const(1, dtype=dtype)
        false_value = tvm.tir.const(3, dtype=dtype)
        max_lhs = tvm.tir.const(2, dtype=dtype)
        max_rhs = tvm.tir.Select(A[0] > 0, true_value, false_value)
        C = te.compute((n,), lambda i: tvm.te.max(max_lhs, max_rhs), name="C")
        func = te.create_prim_func([A, C])
        sch = tvm.tir.Schedule(func)
        (x,) = sch.get_loops(sch.get_block("C"))
        sch.bind(x, "threadIdx.x")
        fun = tvm.build(sch.mod, target=target)

        a = tvm.nd.empty((n,), A.dtype, dev)
        c = tvm.nd.empty((n,), A.dtype, dev)
        # Only need to test compiling here
        fun(a, c)

    dev = tvm.device(target, 0)

    check_if_then_else(dev, 1, "int8")
    check_if_then_else(dev, 1, "uint8")
    check_if_then_else(dev, 1, "int16")
    check_if_then_else(dev, 1, "uint16")
    check_select(dev, 1, "int8")
    check_select(dev, 1, "uint8")
    check_select(dev, 1, "int16")
    check_select(dev, 1, "uint16")


@tvm.testing.requires_gpu
@tvm.testing.requires_opencl
def test_opencl_inf_nan():
    def check_inf_nan(dev, n, value, dtype):
        A = te.placeholder((n,), name="A", dtype=dtype)
        inf_value = tvm.tir.const(value, dtype=dtype)
        C = te.compute((n,), lambda i: inf_value, name="C")
        func = te.create_prim_func([A, C])
        sch = tvm.tir.Schedule(func)
        (x,) = sch.get_loops(sch.get_block("C"))
        sch.bind(x, "threadIdx.x")
        fun = tvm.build(sch.mod, target=target)
        a = tvm.nd.empty((n,), A.dtype, dev)
        c = tvm.nd.empty((n,), A.dtype, dev)
        # Only need to test compiling here
        fun(a, c)

    dev = tvm.device(target, 0)

    check_inf_nan(dev, 1, -float("inf"), "float32")
    check_inf_nan(dev, 1, -float("inf"), "float64")
    check_inf_nan(dev, 1, float("inf"), "float32")
    check_inf_nan(dev, 1, float("inf"), "float64")
    check_inf_nan(dev, 1, float("nan"), "float32")
    check_inf_nan(dev, 1, float("nan"), "float64")


@tvm.testing.requires_gpu
@tvm.testing.requires_opencl
def test_opencl_max():
    def check_max(dev, n, dtype):
        A = te.placeholder((n,), name="A", dtype=dtype)
        max_lhs = A[0] + tvm.tir.const(1, dtype=dtype)
        max_rhs = tvm.tir.const(0, dtype=dtype)
        C = te.compute((n,), lambda i: tvm.te.max(max_lhs, max_rhs), name="C")
        func = te.create_prim_func([A, C])
        sch = tvm.tir.Schedule(func)
        (x,) = sch.get_loops(sch.get_block("C"))
        sch.bind(x, "threadIdx.x")
        fun = tvm.build(sch.mod, target=target)

        a = tvm.nd.empty((n,), A.dtype, dev)
        c = tvm.nd.empty((n,), A.dtype, dev)
        # Only need to test compiling here
        fun(a, c)

    dev = tvm.device(target, 0)

    check_max(dev, 1, "int8")
    check_max(dev, 1, "uint8")
    check_max(dev, 1, "int16")
    check_max(dev, 1, "uint16")
    check_max(dev, 1, "float32")
    check_max(dev, 1, "float64")


def test_opencl_erf():
    def check_erf(dev, n, dtype):
        A = te.placeholder((n,), name="A", dtype=dtype)
        C = te.compute(A.shape, lambda *i: te.erf(A(*i)), name="C")
        s = te.create_schedule(C.op)
        s[C].bind(s[C].op.axis[0], te.thread_axis("threadIdx.x"))
        fun = tvm.build(s, [A, C], target)
        source_str = fun.imported_modules[0].get_source()
        matches = re.findall("erf", source_str)
        error_matches = re.findall("erff", source_str)
        assert len(matches) == 1 and len(error_matches) == 0

    dev = tvm.device(target, 0)

    check_erf(dev, 1, "float32")
    check_erf(dev, 1, "float64")


@tvm.testing.requires_gpu
@tvm.testing.requires_opencl
def test_opencl_type_casting():
    def check_type_casting(ctx, n, dtype):
        block_size = 4
        C = te.compute(
            (n,),
            lambda i: tvm.tir.Select(
                tvm.tir.all(
                    *[
                        i // block_size == tvm.tir.const(3, "int32"),
                        i % 3 == tvm.tir.const(1, "int32"),
                    ]
                ),
                tvm.tir.const(1, dtype),
                tvm.tir.const(0, dtype),
            ),
            name="C",
        )
        # NOTE: test simple convert pattern
        func = te.create_prim_func([C])
        sch = tvm.tir.Schedule(func)
        (x,) = sch.get_loops(sch.get_block("C"))
        tx, vx = sch.split(x, factors=[None, block_size])
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vx)

        fun = tvm.build(sch.mod, target=target)
        c = tvm.nd.empty((n,), dtype, ctx)
        assembly = fun.imported_modules[0].get_source()
        lcond = "convert_int4(((convert_uint4(((uint4)(((convert_int(get_local_id(0))) == 3), ((convert_int(get_local_id(0))) == 3), ((convert_int(get_local_id(0))) == 3), ((convert_int(get_local_id(0))) == 3)))))"
        rcond = "(convert_uint4(((((int4)(((convert_int(get_local_id(0))))+(1*0), ((convert_int(get_local_id(0))))+(1*1), ((convert_int(get_local_id(0))))+(1*2), ((convert_int(get_local_id(0))))+(1*3))) % ((int4)(3, 3, 3, 3))) == ((int4)(1, 1, 1, 1))))))))"
        pattern_cond = "({} && {})".format(lcond, rcond)
        assert assembly.count(pattern_cond) != 0
        fun(c)

    dev = tvm.device(target, 0)

    check_type_casting(dev, 32, "float32")
    # fp16 is not yet supported in ci
    # check_type_casting(dev, 16, "float16")


@tvm.testing.requires_gpu
@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl", "opencl -device=adreno")
def test_opencl_ceil_log2(target):
    def _check(target, n, dtype):
        with tvm.target.Target(target):
            C = te.compute(
                (n,),
                lambda i: tvm.topi.ceil_log2(i),
                name="C",
            )
            func = te.create_prim_func([C])
            sch = tvm.tir.Schedule(func)
            (x,) = sch.get_loops(sch.get_block("C"))
            sch.bind(x, "threadIdx.x")

            fun = tvm.build(sch.mod, target=target)
            assembly = fun.imported_modules[0].get_source()
            if "adreno" in target:
                pattern = "convert_float"
            else:
                pattern = "convert_double"
            assert assembly.count(pattern) != 0

    _check(target, 32, "float32")


def _get_maximum_kernel_args(source):
    def get_kernel_args(source):
        import re

        p = re.compile(r"__kernel void .+\((.*)\)")
        args = p.findall(source)
        return args

    args = get_kernel_args(source)
    max_args = len(args[0].split(","))
    for arg_line in args:
        max_args = max(max_args, len(arg_line.split(",")))
    return max_args


def _validate_opencl_executors(executor_type, get_model, ref_impl):
    from tvm.contrib import graph_executor
    from tvm.runtime.vm import VirtualMachine

    input_dict, model = get_model()
    if executor_type == "ge":
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(model, target_host="llvm", target=target)
        ocl_lib = lib.get_lib()
    else:
        module = tvm.IRModule({})
        module["main"] = model
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.vm.compile(module, target=target, target_host="llvm")
        ocl_lib = lib.module.imported_modules[0]
    opencl_modules = list(filter(lambda mod: mod.type_key == "opencl", ocl_lib.imported_modules))
    assembly = opencl_modules[0].get_source()
    with tvm.target.Target(target):
        limit = tvm.target.Target.current().max_function_args
    max_num = _get_maximum_kernel_args(assembly)
    assert max_num <= limit

    dev = tvm.cl()
    if executor_type == "ge":
        module = graph_executor.GraphModule(lib["default"](dev))
        module.set_input(**input_dict)
        module.run()
        tvm_out = module.get_output(0)
    else:
        vm = VirtualMachine(lib, dev, "naive")
        data = {}
        for k, v in input_dict.items():
            data[k] = tvm.nd.array(v, dev)
        vm.set_input("main", **data)
        vm.invoke_stateful("main")
        tvm_out = vm.get_outputs()[0]

    np_result = ref_impl(list(input_dict.values()))
    np.testing.assert_allclose(tvm_out.asnumpy(), np_result, rtol=1e-2, atol=1e-2)


shape_type = tvm.testing.parameter("dynamic", "static")
executor_type = tvm.testing.parameter("ge", "vm")


@tvm.testing.requires_gpu
@tvm.testing.requires_opencl
def test_opencl_args_split(executor_type, shape_type):
    def _get_model():
        if shape_type == "dynamic":
            shape = (tvm.tir.Any(), 1, 1, 3)
        else:
            shape = (1, 1, 1, 3)
        shape_np = (1, 1, 1, 3)
        dtype = "float32"
        axis = 1
        tensors_num = 300
        inputs = []
        inputs_np = {}
        for i in range(tensors_num):
            inputs.append(relay.var("p{}".format(i), shape=shape, dtype=dtype))
            inputs_np[f"p{i}"] = np.random.uniform(size=shape_np).astype(dtype)

        inp = relay.Tuple(inputs)
        concat = relay.op.concatenate(inp, axis)
        return inputs_np, relay.Function(inputs, concat)

    def ref_impl(inputs):
        axis = 1
        return np.concatenate(tuple(inputs), axis=axis)

    if executor_type == "ge" and shape_type == "dynamic":
        pytest.skip()
    _validate_opencl_executors(executor_type, _get_model, ref_impl)


@tvm.testing.requires_gpu
@tvm.testing.requires_opencl
def test_opencl_fuse_max_args(executor_type, shape_type):
    if shape_type == "dynamic":
        shape = (tvm.tir.Any(), 20)
        ops_num = 80
    else:
        shape = (1, 20)
        ops_num = 300
    shape_np = (1, 20)
    dtype = "float32"

    def _base_func(name):
        x = relay.var(name, shape=shape)
        y = relay.add(x, relay.const(1, "float32"))
        w = relay.exp(y)
        return x, w

    def _get_model():
        inp = []
        inputs_np = {}
        out = []
        for i in range(ops_num):
            x, w = _base_func(f"x{i}")
            inputs_np[f"x{i}"] = np.random.uniform(size=shape_np).astype(dtype)
            inp.append(x)
            out.append(w)
        w = out[0]
        for i in range(len(out) - 1):
            w = relay.add(w, out[i + 1])
        return inputs_np, relay.Function(inp, w)

    def ref_impl(inputs):
        w = np.exp(inputs[0] + 1)
        for i in range(len(inputs) - 1):
            w = w + np.exp(inputs[i + 1] + 1)
        return w

    if executor_type == "ge" and shape_type == "dynamic":
        pytest.skip()
    _validate_opencl_executors(executor_type, _get_model, ref_impl)


@tvm.testing.requires_gpu
@tvm.testing.requires_opencl
def test_fuse_concat_max_num_args(executor_type, shape_type):
    """
    In this test, we have an operation with 3 inputs before concat. In the
    SplitArgs we cannot calculate these inputs as inputs to the concat layer,
    because they will be added to the concat after the fusing operation. So
    FuseOps pass should handle this case and stop fusing before the concat
    layer.

    The example:
       x     y     z                  x     y     z
       \     |     /                  \     |     /
        \    |    /                    \    |    /
           where            ...           where
             |                              |
            exp                            exp
             \                              /
              \                            /
               \----->    concat    <-----/
    """
    if shape_type == "dynamic":
        shape = (tvm.tir.Any(), 20)
        ops_num = 80
    else:
        shape = (10, 20)
        ops_num = 300
    shape_np = (10, 20)
    dtype = "float32"
    axis = 1

    def _base_func(name):
        x = relay.var(name, shape=shape)
        y = relay.var(f"y{name}", shape=shape)
        z = relay.var(f"z{name}", shape=shape)
        cond = relay.less(x, relay.const(1, "float32"))
        l = relay.add(y, relay.const(1, "float32"))
        r = relay.add(z, relay.const(5, "float32"))
        w = relay.where(cond, l, r)
        w = relay.exp(w)
        return [x, y, z], w

    def _get_model():
        inp = []
        out = []
        inputs_np = {}
        for i in range(ops_num):
            inputs, w = _base_func(f"x{i}")
            inputs_np[f"x{i}"] = np.random.uniform(size=shape_np).astype(dtype)
            inputs_np[f"yx{i}"] = np.random.uniform(size=shape_np).astype(dtype)
            inputs_np[f"zx{i}"] = np.random.uniform(size=shape_np).astype(dtype)
            inp.extend(inputs)
            out.append(w)
        t = relay.Tuple(out)
        w = relay.op.concatenate(t, axis)
        return inputs_np, relay.Function(inp, w)

    def ref_impl(inputs):
        res = []
        for i in range(0, len(inputs), 3):
            x = inputs[i]
            y = inputs[i + 1]
            z = inputs[i + 2]
            comp = np.where(x < 1, y + 1, z + 5)
            comp = np.exp(comp)
            res.append(comp)
        return np.concatenate(tuple(res), axis=axis)

    if executor_type == "ge" and shape_type == "dynamic":
        pytest.skip()
    _validate_opencl_executors(executor_type, _get_model, ref_impl)


if __name__ == "__main__":
    tvm.testing.main()
