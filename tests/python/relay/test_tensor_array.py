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
import tvm
from tvm import relay
from tvm.relay import testing
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.relay import create_executor
from tvm.relay.prelude import Prelude, StaticTensorArrayOps
from tvm.relay.testing import count as count_, make_nat_value, make_nat_expr

import numpy as np


def vmobj_to_list(mod, o, dtype="float32"):
    _, tensor_nil, _, _, _, _, _, _, _ = mod.get_type(f"tensor_{dtype}_t")
    if isinstance(o, tvm.nd.NDArray):
        return [o.numpy().tolist()]
    elif isinstance(o, tvm.runtime.container.ADT):
        if len(o) == 0:
            if tensor_nil.tag == o.tag:
                return [0]
            return []

        result = []
        for f in o:
            result.extend(vmobj_to_list(mod, f, dtype))
        return result
    elif isinstance(o, tvm.relay.backend.interpreter.ConstructorValue):
        if o.constructor.name_hint == "Cons":
            tl = vmobj_to_list(mod, o.fields[1], dtype)
            hd = vmobj_to_list(mod, o.fields[0], dtype)
            hd.extend(tl)
            return hd
        elif o.constructor.name_hint == "Nil":
            return []
        elif "tensor_nil" in o.constructor.name_hint:
            return [0]
        elif "tensor" in o.constructor.name_hint:
            return [o.fields[0].numpy()]
        else:
            raise RuntimeError("Unknown object type: %s" % o.constructor.name_hint)
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))


def check_tensor_array(ta_mod, ref_res, *args, dtype="float32", rtol=1e-5):
    for kind in ["debug", "vm"]:
        for target, dev in [("llvm", tvm.cpu(0))]:  # testing.enabled_targets():
            if kind == "debug" and dev.device_type != tvm.cpu().device_type:
                continue
            result = relay.create_executor(kind, mod=ta_mod, device=dev, target=target).evaluate()(
                *args
            )
            got = vmobj_to_list(ta_mod, result, dtype)
            tvm.testing.assert_allclose(ref_res, got, rtol=rtol, atol=rtol)


@tvm.testing.uses_gpu
def test_tensor_expand_dims():
    def run(dtype):
        x = relay.var("x")
        mod = tvm.IRModule()
        p = Prelude(mod)
        expand_dims_func = p.get_global_var("tensor_expand_dims", dtype)
        tensor1 = p.get_tensor_ctor("tensor1", dtype)
        mod["main"] = relay.Function([x], expand_dims_func(tensor1(x)))
        x_np = np.random.uniform(low=0.0, high=8.0, size=(1,)).astype(dtype)
        expected = [np.expand_dims(x_np, axis=0)]
        check_tensor_array(mod, expected, x_np)

    run("float32")
    run("int32")


@tvm.testing.uses_gpu
def test_tensor_array_constructor():
    def run(dtype):
        x = relay.var("x")
        mod = tvm.IRModule()
        p = Prelude(mod)
        tensor_array = p.get_global_var("tensor_array", dtype)
        mod["main"] = relay.Function([x], tensor_array(x))
        expected = np.array([0, 0, 0, 0, 0])
        check_tensor_array(mod, expected, 5, dtype=dtype)

    run("float32")
    run("int32")


@tvm.testing.uses_gpu
def test_tensor_array_read():
    def run(dtype):
        mod = tvm.IRModule()
        p = Prelude(mod)
        l = relay.var("l")
        i = relay.var("i")
        read_func = p.get_global_var("tensor_array_read", dtype)
        tensor_array = p.get_global_var("tensor_array", dtype)
        mod["main"] = relay.Function([l, i], read_func(tensor_array(l), i))
        expected = [0]
        check_tensor_array(mod, expected, *(1, 0), dtype=dtype)
        check_tensor_array(mod, expected, *(5, 1), dtype=dtype)

    run("float32")
    run("int32")


@tvm.testing.uses_gpu
def test_tensor_array_write():
    def run(dtype):
        mod = tvm.IRModule()
        p = Prelude(mod)
        tensor_t = p.get_type("tensor_t", dtype)
        v1 = relay.var("v1")
        v2 = relay.var("v2")
        tensor_array = p.get_global_var("tensor_array", dtype)
        init_tensor_array = tensor_array(relay.const(2))
        write_func = p.get_global_var("tensor_array_write", dtype)
        tensor1 = p.get_tensor_ctor("tensor1", dtype)
        tensor_array1 = write_func(init_tensor_array, relay.const(0), tensor1(v1))
        tensor_array2 = write_func(tensor_array1, relay.const(1), tensor1(v2))
        mod["main"] = relay.Function([v1, v2], tensor_array2)
        expected = [3, 7]
        check_tensor_array(mod, expected, *(3, 7), dtype=dtype)

    run("float32")
    run("int32")


@tvm.testing.uses_gpu
def test_tensor_array_stack():
    def run(dtype):
        mod = tvm.IRModule()
        p = Prelude(mod)
        tensor_t = p.get_type("tensor_t", dtype)
        rlist = p.mod.get_global_type_var(f"List")
        tensor_array = p.get_global_var("tensor_array", dtype)
        tensor1 = p.get_tensor_ctor("tensor1", dtype)
        write = p.get_global_var("tensor_array_write", dtype)
        stack = p.get_global_var("tensor_array_stack", dtype)
        # TODO extract test case from inference failures
        # setting this wrong causes crashes
        v = relay.var("v", shape=(1,), dtype=dtype)
        init_tensor_array = tensor_array(relay.const(3))
        tensor_array1 = write(init_tensor_array, relay.const(0), tensor1(v))
        tensor_array2 = write(tensor_array1, relay.const(1), tensor1(v))
        tensor_array3 = write(tensor_array2, relay.const(2), tensor1(v))
        tensor_array4 = stack(tensor_array3)
        mod["main"] = relay.Function([v], tensor_array4, tensor_t())
        t = np.random.uniform(low=0.0, high=8.0, size=(1,)).astype(dtype)
        expected = [np.stack([t, t, t])]
        check_tensor_array(mod, expected, t, dtype=dtype)

    run("float32")
    run("int32")


@tvm.testing.uses_gpu
def test_tensor_array_unstack():
    def run(dtype):
        mod = tvm.IRModule()
        p = Prelude(mod)
        unstack_tensor1 = p.get_global_var("tensor_array_unstack_tensor1", dtype)
        v = relay.var("v")
        mod["main"] = relay.Function([v], unstack_tensor1(v))
        t = np.random.uniform(low=0.0, high=8.0, size=(1,)).astype(dtype)
        check_tensor_array(mod, t, t, dtype=dtype)

    run("float32")
    run("int32")


@tvm.testing.uses_gpu
def test_tensor_take():
    def run(dtype):
        mod = tvm.IRModule()
        p = Prelude(mod)
        take = p.get_global_var("tensor_take", dtype)
        tensor2 = p.get_tensor_ctor("tensor2", dtype)
        v = relay.var("v")
        lower = relay.var("lower")
        upper = relay.var("upper")
        mod["main"] = relay.Function([v, lower, upper], take(tensor2(v), lower, upper))
        v_data = np.random.uniform(low=0.0, high=8.0, size=(10, 10)).astype(dtype)
        expected = [np.take(v_data, range(2, 5), axis=0)]
        check_tensor_array(mod, expected, *(v_data, 2, 5), dtype=dtype)
        expected = [np.take(v_data, range(0, 9), axis=0)]
        check_tensor_array(mod, expected, *(v_data, 0, 9), dtype=dtype)

    run("float32")
    run("int32")


@tvm.testing.uses_gpu
def test_tensor_concatenate():
    def run(dtype):
        mod = tvm.IRModule()
        p = Prelude(mod)
        concat = p.get_global_var("tensor_concatenate", dtype)
        tensor1 = p.get_tensor_ctor("tensor1", dtype)
        v1 = relay.var("v1", shape=(tvm.tir.Any(),), dtype=dtype)
        v2 = relay.var("v2", shape=(tvm.tir.Any(),), dtype=dtype)
        mod["main"] = relay.Function([v1, v2], concat(tensor1(v1), tensor1(v2)))
        v1_data = np.random.uniform(low=0.0, high=8.0, size=(5,)).astype(dtype)
        v2_data = np.random.uniform(low=0.0, high=8.0, size=(5,)).astype(dtype)
        expected = [np.concatenate((v1_data, v2_data))]
        check_tensor_array(mod, expected, *(v1_data, v2_data), dtype=dtype)

    run("float32")
    run("int32")


@tvm.testing.uses_gpu
def test_tensor_array_concat():
    def run(dtype):
        mod = tvm.IRModule()
        p = Prelude(mod)
        v1 = relay.var("v1")
        v2 = relay.var("v2")
        tensor_array = p.get_global_var("tensor_array", dtype)
        tensor_array1 = tensor_array(relay.const(2))
        write_func = p.get_global_var("tensor_array_write", dtype)
        concat_func = p.get_global_var("tensor_array_concat", dtype)
        tensor1 = p.get_tensor_ctor("tensor2", dtype)
        tensor_array1 = write_func(tensor_array1, relay.const(0), tensor1(v1))
        tensor_array1 = write_func(tensor_array1, relay.const(1), tensor1(v2))
        tensor_array_concat = concat_func(tensor_array1)
        mod["main"] = relay.Function([v1, v2], tensor_array_concat)
        v1_data = np.random.uniform(low=0.0, high=8.0, size=(2, 3)).astype(dtype)
        v2_data = np.random.uniform(low=0.0, high=8.0, size=(1, 3)).astype(dtype)
        expected = [np.concatenate((v1_data, v2_data), axis=0)]
        check_tensor_array(mod, expected, *(v1_data, v2_data), dtype=dtype)

    run("float32")
    run("int32")


@tvm.testing.uses_gpu
def test_tensor_array_scatter():
    def run(dtype):
        mod = tvm.IRModule()
        p = Prelude(mod)

        # tensor array
        v1 = relay.var("v1")
        v2 = relay.var("v2")
        v3 = relay.var("v2")
        tensor_array = p.get_global_var("tensor_array", dtype)
        tensor_array1 = tensor_array(relay.const(3))
        write_func = p.get_global_var("tensor_array_write", dtype)
        scatter_func = p.get_global_var("tensor_array_scatter", dtype)
        tensor2 = p.get_tensor_ctor("tensor2", dtype)
        tensor_array1 = write_func(tensor_array1, relay.const(0), tensor2(v1))
        tensor_array1 = write_func(tensor_array1, relay.const(1), tensor2(v2))
        tensor_array1 = write_func(tensor_array1, relay.const(2), tensor2(v3))

        # indices array
        index = relay.var("index")

        # values array
        value_0 = relay.var("value_0")
        value_1 = relay.var("value_1")
        values_array = tensor_array(relay.const(2))
        values_array = write_func(values_array, relay.const(0), tensor2(value_0))
        values_array = write_func(values_array, relay.const(1), tensor2(value_1))

        # create the scatter function
        tensor_array_scatter = scatter_func(tensor_array1, index, values_array)
        mod["main"] = relay.Function([v1, v2, v3, index, value_0, value_1], tensor_array_scatter)

        # initialize and check
        v1_data = np.random.uniform(low=0.0, high=8.0, size=(2, 3)).astype(dtype)
        v2_data = np.random.uniform(low=0.0, high=8.0, size=(2, 3)).astype(dtype)
        v3_data = np.random.uniform(low=0.0, high=8.0, size=(2, 3)).astype(dtype)
        index_data = np.array([0, 1], dtype="int32")
        val1_data = np.random.uniform(low=0.0, high=8.0, size=(2, 3)).astype(dtype)
        val2_data = np.random.uniform(low=0.0, high=8.0, size=(2, 3)).astype(dtype)
        expected = [val1_data, val2_data, v3_data]
        check_tensor_array(
            mod,
            expected,
            *(v1_data, v2_data, v3_data, index_data, val1_data, val2_data),
            dtype=dtype,
        )

    run("float32")
    run("int32")


@tvm.testing.uses_gpu
def test_tensor_array_split():
    def run(dtype):
        mod = tvm.IRModule()
        p = Prelude(mod)

        # tensor array
        v1 = relay.var("v1")
        v2 = relay.var("v2")
        v3 = relay.var("v2")
        tensor_array = p.get_global_var("tensor_array", dtype)
        tensor_array1 = tensor_array(relay.const(3))
        write_func = p.get_global_var("tensor_array_write", dtype)
        split_func = p.get_global_var("tensor_array_split", dtype)
        tensor2 = p.get_tensor_ctor("tensor2", dtype)
        tensor_array1 = write_func(tensor_array1, relay.const(0), tensor2(v1))
        tensor_array1 = write_func(tensor_array1, relay.const(1), tensor2(v2))
        tensor_array1 = write_func(tensor_array1, relay.const(2), tensor2(v3))

        # value tensor
        value = relay.var("value")

        # lengths tensor
        ta_len = relay.var("length")

        # create the scatter function
        tensor_array_split = split_func(tensor_array1, tensor2(value), ta_len)
        mod["main"] = relay.Function([v1, v2, v3, value, ta_len], tensor_array_split)

        # initialize and check
        v1_data = np.random.uniform(low=0.0, high=8.0, size=(2, 3)).astype(dtype)
        v2_data = np.random.uniform(low=0.0, high=8.0, size=(2, 3)).astype(dtype)
        v3_data = np.random.uniform(low=0.0, high=8.0, size=(2, 3)).astype(dtype)
        value_data = np.random.uniform(low=0.0, high=8.0, size=(4, 3)).astype(dtype)
        length_data = np.array([2, 2], dtype="int32")
        expected = np.concatenate([value_data, v3_data])
        expected = np.split(expected, indices_or_sections=[2, 4])
        check_tensor_array(
            mod, expected, *(v1_data, v2_data, v3_data, value_data, length_data), dtype=dtype
        )

    run("float32")
    run("int32")


@tvm.testing.uses_gpu
def test_static_tensor_take():
    def run(dtype, shape):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()

        take = p.get_global_var_static("tensor_take", dtype, shape)
        tensor_constructor = p.get_tensor_ctor_static("tensor_constructor", dtype, shape)
        v = relay.var("v")
        lower = relay.var("lower")
        upper = relay.var("upper")
        mod["main"] = relay.Function([v, lower, upper], take(tensor_constructor(v), lower, upper))
        v_data = np.random.uniform(low=0.0, high=8.0, size=shape).astype(dtype)
        expected = [np.take(v_data, range(2, 5), axis=0)]
        check_tensor_array(mod, expected, *(v_data, 2, 5), dtype=dtype)
        expected = [np.take(v_data, range(0, 9), axis=0)]
        check_tensor_array(mod, expected, *(v_data, 0, 9), dtype=dtype)

    run("float32", [10, 10])
    run("int32", [15, 11])


@tvm.testing.uses_gpu
def test_static_tensor_concatenate():
    def run(dtype, shape):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()

        concat = p.get_global_var_static("tensor_concatenate", dtype, shape)
        tensor = p.get_tensor_ctor_static("tensor_constructor", dtype, shape)
        v1 = relay.var("v1")
        v2 = relay.var("v2")
        mod["main"] = relay.Function([v1, v2], concat(tensor(v1), tensor(v2)))
        v1_data = np.random.uniform(low=0.0, high=8.0, size=shape).astype(dtype)
        v2_data = np.random.uniform(low=0.0, high=8.0, size=shape).astype(dtype)
        expected = [np.concatenate((v1_data, v2_data))]
        check_tensor_array(mod, expected, *(v1_data, v2_data), dtype=dtype)

    run(
        "float32",
        [
            5,
        ],
    )
    run("int32", [2, 3])


@tvm.testing.uses_gpu
def test_static_tensor_expand_dims():
    def run(dtype, shape):
        x = relay.var("x")
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()

        expand_dims_func = p.get_global_var_static("tensor_expand_dims", dtype, shape)
        tensor = p.get_tensor_ctor_static("tensor_constructor", dtype, shape)
        mod["main"] = relay.Function([x], expand_dims_func(tensor(x)))
        x_np = np.random.uniform(low=0.0, high=8.0, size=shape).astype(dtype)
        expected = [np.expand_dims(x_np, axis=0)]
        check_tensor_array(mod, expected, x_np)

    run("float32", [])
    run(
        "int32",
        [
            2,
        ],
    )


@tvm.testing.uses_gpu
def test_static_tensor_array_constructor():
    def run(dtype, shape):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()
        tensor_constructor = p.get_name_static("tensor_constructor", dtype, shape)
        assert tensor_constructor != None

    run("float32", [1, 1])


@tvm.testing.uses_gpu
def test_static_tensor_array_read():
    def run(dtype, shape):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()

        np_data_list = []
        ta_length = 3
        for _ in range(ta_length):
            np_data_list.append(np.random.uniform(0, 10, size=shape).astype(dtype))

        v0 = relay.var("v0")
        v1 = relay.var("v1")
        v2 = relay.var("v2")
        n = relay.var("n")
        tensor = p.get_tensor_ctor_static("tensor_constructor", dtype, shape)
        tensor_array = p.get_global_var_static("tensor_array", dtype, shape)
        init_tensor_array = tensor_array(relay.const(ta_length))
        read_func = p.get_global_var_static("tensor_array_read", dtype, shape)
        write_func = p.get_global_var_static("tensor_array_write", dtype, shape)
        tensor_array0 = write_func(init_tensor_array, relay.const(0), tensor(v0))
        tensor_array1 = write_func(tensor_array0, relay.const(1), tensor(v1))
        tensor_array2 = write_func(tensor_array1, relay.const(2), tensor(v2))

        mod["main"] = relay.Function([v0, v1, v2, n], read_func(tensor_array2, n))
        expected = [np_data_list[0]]
        check_tensor_array(mod, expected, *list(np_data_list + [0]), dtype=dtype)
        expected = [np_data_list[1]]
        check_tensor_array(mod, expected, *list(np_data_list + [1]), dtype=dtype)
        expected = [np_data_list[2]]
        check_tensor_array(mod, expected, *list(np_data_list + [2]), dtype=dtype)

    run("float32", [])
    run("int32", [2, 3])


@tvm.testing.uses_gpu
def test_static_tensor_array_write():
    def run(dtype, shape):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()

        ta_length = 2
        np_data_list = [
            np.random.uniform(0, 10, size=shape).astype(dtype) for _ in range(ta_length)
        ]

        v0 = relay.var("v0")
        v1 = relay.var("v1")
        tensor_array = p.get_global_var_static("tensor_array", dtype, shape)
        init_tensor_array = tensor_array(relay.const(ta_length))
        write_func = p.get_global_var_static("tensor_array_write", dtype, shape)
        tensor = p.get_tensor_ctor_static("tensor_constructor", dtype, shape)
        tensor_array0 = write_func(init_tensor_array, relay.const(0), tensor(v0))
        tensor_array1 = write_func(tensor_array0, relay.const(1), tensor(v1))
        mod["main"] = relay.Function([v0, v1], tensor_array1)
        expected = np_data_list
        check_tensor_array(mod, expected, *np_data_list, dtype=dtype)

    run("float32", [])
    run("int32", [2, 3])


@tvm.testing.uses_gpu
def test_static_tensor_array_unstack():
    def run(dtype, shape):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()

        unstack_tensor = p.get_global_var_static("tensor_array_unstack", dtype, shape)
        v = relay.var("v")
        mod["main"] = relay.Function([v], unstack_tensor(v))
        t = np.random.uniform(low=0, high=10, size=shape).astype(dtype)
        (*expected,) = t
        check_tensor_array(mod, expected, t, dtype=dtype)

    run("float32", [4])
    run("int32", [2, 3])


@tvm.testing.uses_gpu
def test_static_tensor_array_scatter():
    def run(dtype, shape, indices_shape=None):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()
        if indices_shape is not None:
            static_tensor_array_ops.define_tensor_array_scatter(indices_shape, True)

        # tensor array
        v1 = relay.var("v1")
        v2 = relay.var("v2")
        v3 = relay.var("v2")
        tensor_array = p.get_global_var_static("tensor_array", dtype, shape)
        tensor_array0 = tensor_array(relay.const(3))
        write_func = p.get_global_var_static("tensor_array_write", dtype, shape)
        scatter_func = p.get_global_var_static("tensor_array_scatter", dtype, shape)
        tensor = p.get_tensor_ctor_static("tensor_constructor", dtype, shape)
        tensor_array1 = write_func(tensor_array0, relay.const(0), tensor(v1))
        tensor_array1 = write_func(tensor_array1, relay.const(1), tensor(v2))
        tensor_array1 = write_func(tensor_array1, relay.const(2), tensor(v3))

        # indices array
        index = relay.var("index")

        # values array
        value_0 = relay.var("value_0")
        value_1 = relay.var("value_1")
        values_array = tensor_array(relay.const(2))
        values_array = write_func(values_array, relay.const(0), tensor(value_0))
        values_array = write_func(values_array, relay.const(1), tensor(value_1))

        # create the scatter function
        tensor_array_scatter = scatter_func(tensor_array1, index, values_array)
        mod["main"] = relay.Function([v1, v2, v3, index, value_0, value_1], tensor_array_scatter)

        # initialize and check
        v1_data = np.random.uniform(low=0.0, high=8.0, size=shape).astype(dtype)
        v2_data = np.random.uniform(low=0.0, high=8.0, size=shape).astype(dtype)
        v3_data = np.random.uniform(low=0.0, high=8.0, size=shape).astype(dtype)
        index_data = np.array([0, 1], dtype="int32")
        val1_data = np.random.uniform(low=0.0, high=8.0, size=shape).astype(dtype)
        val2_data = np.random.uniform(low=0.0, high=8.0, size=shape).astype(dtype)
        expected = [val1_data, val2_data, v3_data]
        check_tensor_array(
            mod,
            expected,
            *(v1_data, v2_data, v3_data, index_data, val1_data, val2_data),
            dtype=dtype,
        )

    run("float32", [2, 3])
    run("int32", [2, 3])
    run(
        "float32",
        [2, 3],
        [
            2,
        ],
    )


@tvm.testing.uses_gpu
def test_static_tensor_array_split():
    def run(dtype, shape, value_shape=None, lengths_shape=None):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()
        if value_shape is not None or lengths_shape is not None:
            static_tensor_array_ops.define_tensor_array_split(value_shape, lengths_shape, False)

        # tensor array
        v1 = relay.var("v1")
        v2 = relay.var("v2")
        v3 = relay.var("v2")

        adt_shape = [
            relay.Any(),
        ] + shape[1:]
        test_ops = StaticTensorArrayOps(p, dtype, adt_shape)
        test_ops.register()
        tensor_array = test_ops.get_global_var("tensor_array")

        tensor_array1 = tensor_array(relay.const(3))
        write_func = test_ops.get_global_var("tensor_array_write")
        split_ops = StaticTensorArrayOps(p, dtype, shape)
        split_ops.register()
        split_func = split_ops.get_global_var("tensor_array_split")
        tensor = p.get_tensor_ctor_static("tensor_constructor", dtype, test_ops.shape)
        tensor_array1 = write_func(tensor_array1, relay.const(0), tensor(v1))
        tensor_array1 = write_func(tensor_array1, relay.const(1), tensor(v2))
        tensor_array1 = write_func(tensor_array1, relay.const(2), tensor(v3))

        # value tensor
        value = relay.var("value")

        # lengths tensor
        ta_len = relay.var("length")

        # create the split function
        if value_shape is None:
            tensor1 = p.get_tensor_ctor_static("tensor_constructor", dtype, shape)
        else:
            static_tensor_array_ops = StaticTensorArrayOps(p, dtype, value_shape)
            static_tensor_array_ops.register()
            tensor1 = p.get_tensor_ctor_static("tensor_constructor", dtype, test_ops.shape)

        tensor_array_split = split_func(tensor_array1, tensor1(value), ta_len)
        mod["main"] = relay.Function([v1, v2, v3, value, ta_len], tensor_array_split)

        # initialize and check
        v1_data = np.random.uniform(low=0.0, high=8.0, size=[2, 3]).astype(dtype)
        v2_data = np.random.uniform(low=0.0, high=8.0, size=[2, 3]).astype(dtype)
        v3_data = np.random.uniform(low=0.0, high=8.0, size=[2, 3]).astype(dtype)
        value_data = np.random.uniform(low=0.0, high=8.0, size=value_shape or shape).astype(dtype)
        length_data = np.array([2, 2], dtype="int32")
        expected = np.concatenate([value_data, v3_data])
        expected = np.split(expected, indices_or_sections=[2, 4])
        check_tensor_array(
            mod, expected, *(v1_data, v2_data, v3_data, value_data, length_data), dtype=dtype
        )

    run("float32", [4, 3])
    run("int32", [4, 3])
    run(
        "int32",
        [relay.Any(), 3],
        [4, 3],
        [
            2,
        ],
    )


@tvm.testing.uses_gpu
def test_static_tensor_array_concat():
    def run(dtype, shape):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()

        v1 = relay.var("v1")
        v2 = relay.var("v2")
        tensor_array = p.get_global_var_static("tensor_array", dtype, shape)
        tensor_array1 = tensor_array(relay.const(2))
        write_func = p.get_global_var_static("tensor_array_write", dtype, shape)
        concat_func = p.get_global_var_static("tensor_array_concat", dtype, shape)
        tensor = p.get_tensor_ctor_static("tensor_constructor", dtype, shape)
        tensor_array1 = write_func(tensor_array1, relay.const(0), tensor(v1))
        tensor_array1 = write_func(tensor_array1, relay.const(1), tensor(v2))
        tensor_array_concat = concat_func(tensor_array1)
        mod["main"] = relay.Function([v1, v2], tensor_array_concat)
        v1_data = np.random.uniform(low=0.0, high=8.0, size=(2, 3)).astype(dtype)
        v2_data = np.random.uniform(low=0.0, high=8.0, size=(1, 3)).astype(dtype)
        expected = [np.concatenate((v1_data, v2_data), axis=0)]
        check_tensor_array(mod, expected, *(v1_data, v2_data), dtype=dtype)

    run("float32", [relay.Any(), 3])
    run("int32", [relay.Any(), 3])


@tvm.testing.uses_gpu
def test_static_tensor_array_gather():
    def run(dtype, shape):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()

        tensor_array = p.get_global_var_static("tensor_array", dtype, shape)
        tensor = p.get_tensor_ctor_static("tensor_constructor", dtype, shape)
        write = p.get_global_var_static("tensor_array_write", dtype, shape)
        gather = p.get_global_var_static("tensor_array_gather", dtype, shape)
        v = relay.var("v")
        indice = relay.var("indice")
        init_tensor_array = tensor_array(relay.const(3))
        tensor_array1 = write(init_tensor_array, relay.const(0), tensor(v))
        tensor_array2 = write(tensor_array1, relay.const(1), tensor(v))
        tensor_array3 = write(tensor_array2, relay.const(2), tensor(v))
        out = gather(tensor_array3, indice)
        mod["main"] = relay.Function([v, indice], out)
        t = np.random.uniform(low=0.0, high=8.0, size=shape).astype(dtype)
        indice_data = np.array([0, 2], dtype="int32")
        expected = [np.stack([t, t])]
        check_tensor_array(mod, expected, *(t, indice_data), dtype=dtype)

    run("float32", [])
    run("int32", [2, 3])


@tvm.testing.uses_gpu
def test_static_tensor_array_stack():
    def run(dtype, shape):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()

        tensor_array = p.get_global_var_static("tensor_array", dtype, shape)
        tensor = p.get_tensor_ctor_static("tensor_constructor", dtype, shape)
        write = p.get_global_var_static("tensor_array_write", dtype, shape)
        stack = p.get_global_var_static("tensor_array_stack", dtype, shape)
        v = relay.var("v")
        init_tensor_array = tensor_array(relay.const(3))
        tensor_array1 = write(init_tensor_array, relay.const(0), tensor(v))
        tensor_array2 = write(tensor_array1, relay.const(1), tensor(v))
        tensor_array3 = write(tensor_array2, relay.const(2), tensor(v))
        tensor_array4 = stack(tensor_array3)
        mod["main"] = relay.Function([v], tensor_array4)
        t = np.random.uniform(low=0.0, high=8.0, size=shape).astype(dtype)
        expected = [np.stack([t, t, t])]
        check_tensor_array(mod, expected, t, dtype=dtype)

    run("float32", [])
    run("int32", [2, 3])


@tvm.testing.uses_gpu
def test_static_tensor_get_data():
    def run(dtype, shape):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()

        np_data_list = []
        ta_length = 3
        for _ in range(ta_length):
            np_data_list.append(np.random.uniform(0, 10, size=shape).astype(dtype))

        v0 = relay.var("v0")
        v1 = relay.var("v1")
        v2 = relay.var("v2")
        n = relay.var("n")
        tensor = p.get_tensor_ctor_static("tensor_constructor", dtype, shape)
        tensor_array = p.get_global_var_static("tensor_array", dtype, shape)
        init_tensor_array = tensor_array(relay.const(ta_length))
        read_func = p.get_global_var_static("tensor_array_read", dtype, shape)
        write_func = p.get_global_var_static("tensor_array_write", dtype, shape)
        get_data_func = p.get_global_var_static("tensor_get_data", dtype, shape)
        tensor_array0 = write_func(init_tensor_array, relay.const(0), tensor(v0))
        tensor_array1 = write_func(tensor_array0, relay.const(1), tensor(v1))
        tensor_array2 = write_func(tensor_array1, relay.const(2), tensor(v2))

        mod["main"] = relay.Function([v0, v1, v2, n], get_data_func(read_func(tensor_array2, n)))
        expected = [np_data_list[0]]
        check_tensor_array(mod, expected, *list(np_data_list + [0]), dtype=dtype)
        expected = [np_data_list[1]]
        check_tensor_array(mod, expected, *list(np_data_list + [1]), dtype=dtype)
        expected = [np_data_list[2]]
        check_tensor_array(mod, expected, *list(np_data_list + [2]), dtype=dtype)

    run("float32", [])
    run("int32", [2, 3])


if __name__ == "__main__":
    tvm.testing.main()
