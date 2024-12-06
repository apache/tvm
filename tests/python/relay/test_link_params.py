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
import collections
import ctypes
import json
import os
import re
from contextlib import redirect_stderr
from io import StringIO

import numpy as np
import tvm
import tvm.relay
import tvm.testing
from tvm import meta_schedule as ms
from tvm import relay
from tvm.contrib import utils
from tvm.relay.backend import Executor, Runtime

INPUT_SHAPE = (1, 3, 16, 16)

KERNEL_SHAPE = (3, 3, 3, 3)


# The data types that are linkable.
linkable_dtype = tvm.testing.parameter(
    *(
        [f"uint{b}" for b in (8, 16, 32, 64)]
        + [f"int{b}" for b in (8, 16, 32, 64)]
        + ["float32", "float64"]
    )
)


def dtype_info(dtype):
    """Lookup numpy type info for the given string dtype (of linkable_dtype params above)."""
    if "int" in dtype:
        return np.iinfo(getattr(np, dtype))
    else:
        return np.finfo(getattr(np, dtype))


# Note: for debugging, set this to an integer (i.e. 1.0). Then all "random" tensors will become
# predictable
RANDOM_TENSOR_START = None


def _make_random_tensor(dtype, shape):
    """Create a random test tensor with given shape and dtype."""
    global RAND_SEED
    if RANDOM_TENSOR_START is not None:
        to_return = np.arange(
            RANDOM_TENSOR_START, RANDOM_TENSOR_START + np.prod(shape), dtype=dtype
        ).reshape(shape)
        RAND_SEED += np.prod(shape)
        return to_return

    dinfo = dtype_info(dtype)
    if "int" in dtype:
        return np.random.randint(dinfo.min, dinfo.max, shape, dtype=dtype)
    else:
        to_return = np.random.uniform(0, dinfo.max, shape).astype(dtype)
        np.reshape(to_return, np.prod(shape))[::2] *= -1
        return to_return


def _lookup_sid(graph, name):
    """Lookup the storage id of a named parameter.

    Arguments
    ---------
    graph : dict
        Parsed JSON graph.

    name : str
        Name of the tensor parameter to lookup.

    Returns
    -------
    int :
        The storage_id of the parameter.
    """
    num_outputs_seen = 0
    for i, n in enumerate(graph["nodes"]):
        if n["name"] == name:
            print("sid", name, graph["attrs"]["storage_id"][1], num_outputs_seen)
            return graph["attrs"]["storage_id"][1][num_outputs_seen]
        else:
            if "attrs" in n and "num_outputs" in n["attrs"]:
                num_outputs_seen += int(n["attrs"]["num_outputs"])
            else:
                num_outputs_seen += 1

    raise KeyError(f"no such param: {name}")


def _get_ctypes_dtype(dt):
    """Return a ctypes c_* datatype given a string data type."""
    if "int" in dt:
        return getattr(ctypes, f"c_{dt}")
    elif dt == "float32":
        return ctypes.c_float
    elif dt == "float64":
        return ctypes.c_double
    else:
        assert False, f"unknown dtype: {dt}"


def _verify_linked_param(dtype, lib, mod, graph, name):
    """Directly read memory from the linked library to verify the linked parameter is correct."""
    sid = _lookup_sid(graph, name)
    # NOTE: query_imports=True because when loading a module from disk (i.e. for C backend),
    # a GraphExecutorFactory module is created instead of the module itself.
    param_ptr = mod.get_function("_lookup_linked_param", True)(sid)
    gen_param = lib.params[name]
    arr_data = (_get_ctypes_dtype(dtype) * np.prod(gen_param.shape)).from_address(param_ptr.value)
    arr = np.ndarray(shape=gen_param.shape, dtype=gen_param.dtype, buffer=arr_data, order="C")
    if "int" in gen_param.dtype:
        np.testing.assert_equal(gen_param.numpy(), arr)
    else:
        np.testing.assert_allclose(gen_param.numpy(), arr)
    return dtype == gen_param.dtype


def _make_mod_and_params(dtype):
    """Create a Relay module and parameters to test the given datatype."""
    param_decls = collections.OrderedDict()
    param_init = {}

    def _add_decl(name, dtype):
        param_decls[name] = f"%{name} : Tensor[{KERNEL_SHAPE}, {dtype}]"
        param_init[name] = _make_random_tensor(dtype, KERNEL_SHAPE)

    # Add several parameters so that the number of parameters
    _add_decl(f"{dtype}_a", dtype)
    _add_decl(f"{dtype}_b", dtype)

    mod_lines = [
        '#[version = "0.0.5"]',
        f"def @main(%rand_input : Tensor[{INPUT_SHAPE}, {dtype}], { ', '.join(param_decls.values()) } )  {{",
        # This program ensures that GraphPlanMemory alternates between the same two storage IDs for a
        # while. In doing this, it ensures that param %{dtype}_b will be placed into the graph at an
        # index unequal to its storage_id. This ensures that GraphExecutorCodegen encodes the storage_id
        # and not the parameter index into the graph.
        (
            f'    %0 = nn.conv2d(%rand_input, %{dtype}_a, data_layout="NCHW", kernel_layout="OIHW", '
            f'kernel_size=[3, 3], out_dtype="{dtype}");'
        ),
        (
            f'    %1 = nn.conv2d(%0, %{dtype}_a, data_layout="NCHW", kernel_layout="OIHW", '
            f'kernel_size=[3, 3], out_dtype="{dtype}");'
        ),
        (
            f'    %2 = nn.conv2d(%1, %{dtype}_a, data_layout="NCHW", kernel_layout="OIHW", '
            f'kernel_size=[3, 3], out_dtype="{dtype}");'
        ),
        (
            f'    %3 = nn.conv2d(%2, %{dtype}_b, data_layout="NCHW", kernel_layout="OIHW", '
            f'kernel_size=[3, 3], out_dtype="{dtype}");'
        ),
        "    %3",
        "}",
    ]

    mod = tvm.relay.fromtext("\n".join(mod_lines))
    return mod, param_init


@tvm.testing.requires_llvm
def test_llvm_link_params(linkable_dtype):
    ir_mod, param_init = _make_mod_and_params(linkable_dtype)
    rand_input = _make_random_tensor(linkable_dtype, INPUT_SHAPE)
    main_func = ir_mod["main"]
    target = "llvm"
    runtime = Runtime("crt", {"system-lib": True})
    executor = Executor("graph", {"link-params": True})
    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.relay.build(ir_mod, target, runtime=runtime, executor=executor, params=param_init)

        # NOTE: Need to export_library() and load_library() to link all the Module(llvm, ...)
        # against one another.
        temp_dir = utils.TempDirectory()
        export_file = temp_dir / "lib.so"
        lib.lib.export_library(export_file)
        mod = tvm.runtime.load_module(export_file)
        assert len(lib.params.keys()) == 0  # NOTE: params became tir.constants
        assert mod.get_function("TVMSystemLibEntryPoint") != None

        graph = json.loads(lib.graph_json)
        for p in lib.params:
            _verify_linked_param(linkable_dtype, lib, mod, graph, p) or found_one

        # Wrap in function to explicitly deallocate the runtime.
        def _run_linked(lib, mod):
            graph_json, _, _ = lib
            graph_rt = tvm.contrib.graph_executor.create(graph_json, mod, tvm.cpu(0))
            graph_rt.set_input("rand_input", rand_input)  # NOTE: params not required.
            graph_rt.run()
            return graph_rt.get_output(0)

        linked_output = _run_linked(lib, mod)

    runtime = Runtime("cpp", {"system-lib": True})
    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.relay.build(ir_mod, "llvm", runtime=runtime, params=param_init)

        def _run_unlinked(lib):
            graph_json, mod, lowered_params = lib
            graph_rt = tvm.contrib.graph_executor.create(graph_json, mod, tvm.cpu(0))
            graph_rt.set_input("rand_input", rand_input, **lowered_params)
            graph_rt.run()
            return graph_rt.get_output(0)

        unlinked_output = _run_unlinked(lib)

    if "int" in linkable_dtype:
        np.testing.assert_equal(unlinked_output.numpy(), linked_output.numpy())
    else:
        np.testing.assert_allclose(unlinked_output.numpy(), linked_output.numpy())


def _get_c_datatype(dtype):
    """Translate LINKABLE_DTYPES element to c datatype."""
    if "int" in dtype:
        return f"{dtype}_t"
    elif dtype == "float32":
        return "float"
    elif dtype == "float64":
        return "double"
    else:
        assert False, f"unknown dtype {dtype}"


HEX_NUM_RE = re.compile(r"[+\-]?(?:(?:0x[0-9A-Fa-f.p+-]+)|(?:INFINITY)|(?:NAN))")


def test_c_link_params(linkable_dtype):
    temp_dir = utils.tempdir()
    mod, param_init = _make_mod_and_params(linkable_dtype)
    rand_input = _make_random_tensor(linkable_dtype, INPUT_SHAPE)
    main_func = mod["main"]
    target = "c"
    executor = Executor("graph", {"link-params": True})
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lib = tvm.relay.build(mod, target, executor=executor, params=param_init)
        assert len(lib.params.keys()) == 0  # NOTE: params became tir.constants

        src = lib.lib.get_source()
        lib.lib.save(temp_dir.relpath("test.c"), "c")
        c_dtype = _get_c_datatype(linkable_dtype)
        src_lines = src.split("\n")
        param = param_init[f"{linkable_dtype}_a"].reshape(np.prod(KERNEL_SHAPE))
        param_def = rf"^static const {c_dtype} __attribute__\(\(section\(\".rodata.tvm\"\), aligned\(16\)\)\) [a-zA-Z_0-9]*constant_\d+\[{np.prod(param.shape)}\] = {{$"

        for i, line in enumerate(src_lines):
            if re.match(param_def, line):
                i += 1
                break
        else:
            assert False, f'did not find parameter definition "{param_def}":\n{src}'

        cursor = 0
        width = dtype_info(linkable_dtype).bits // 4 + 2
        if linkable_dtype.startswith("int"):
            width += 1  # Account for sign

        while "};" not in src_lines[i]:
            for match in HEX_NUM_RE.finditer(src_lines[i]):
                cursor += 1
            i += 1

        assert cursor == np.prod(param.shape)

        # Need a unique name per library to avoid dlopen caching the lib load.
        lib_path = temp_dir.relpath(f"test-{linkable_dtype}-linked.so")
        lib["remove_params"]().export_library(lib_path)
        lib_mod = tvm.runtime.load_module(lib_path)

        #            lib_mod = lib_factory['default']()
        graph = json.loads(lib.graph_json)
        for p in lib.params:
            _verify_linked_param(linkable_dtype, lib, lib_mod, graph, p)

        # Wrap in function to explicitly deallocate the runtime.
        def _run_linked(lib_mod):
            graph_rt = tvm.contrib.graph_executor.GraphModule(lib_mod["default"](tvm.cpu(0)))
            graph_rt.set_input("rand_input", rand_input)  # NOTE: params not required.
            graph_rt.run()

            return graph_rt.get_output(0)

        linked_output = _run_linked(lib_mod)

    linked_params = lib.params
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lib = tvm.relay.build(mod, "c", params=param_init)
        _, _, params = lib
        # Need a unique name per library to avoid dlopen caching the lib load.
        lib_path = temp_dir.relpath(f"test-{linkable_dtype}-unlinked.so")
        lib.export_library(lib_path)
        lib_mod = tvm.runtime.load_module(lib_path)

        def _run_unlinked(lib_mod):
            graph_rt = tvm.contrib.graph_executor.GraphModule(lib_mod["default"](tvm.cpu(0)))
            graph_rt.set_input("rand_input", rand_input, **params)
            graph_rt.run()
            return graph_rt.get_output(0)

        unlinked_output = _run_unlinked(lib_mod)

    if "int" in linkable_dtype:
        np.testing.assert_equal(unlinked_output.numpy(), linked_output.numpy())
    else:
        np.testing.assert_allclose(unlinked_output.numpy(), linked_output.numpy())


def test_tir_link_params():
    def get_dense(data_shape, weight_shape):
        data = relay.var("data", shape=data_shape, dtype="float32")
        weight = relay.var("weight", shape=weight_shape, dtype="float32")
        dense = relay.nn.dense(data, weight)
        return relay.Function([data, weight], dense)

    def get_ref_dense(data_np, weight_np):
        return np.dot(data_np, np.transpose(weight_np))

    def schedule_dense(sch):
        dense = sch.get_block("T_matmul_NT")
        _y, _x, _k = sch.get_loops(dense)

    M, N, K = 128, 128, 128
    data_shape = (M, K)
    weight_shape = (N, K)
    relay_mod = tvm.IRModule.from_expr(get_dense(data_shape, weight_shape))
    relay_mod = relay.transform.InferType()(relay_mod)
    data_np = np.random.randn(*data_shape).astype("float32")
    weight_np = np.random.randn(*weight_shape).astype("float32")
    target = "llvm"
    params = {"weight": weight_np}

    def schedule_fn(sch):
        if "nn_dense" in sch.mod.attrs["task_name"]:
            schedule_dense(sch)
            return True
        return False

    with StringIO() as stderr_buf, redirect_stderr(stderr_buf):
        with ms.database.ScheduleFnDatabase(schedule_fn), tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True},
        ):
            executor = Executor("graph", {"link-params": True})
            lib = relay.build(relay_mod, target=target, executor=executor)

        # Workload look up should succeed. This does not work when the test is invoked from pytest.
        assert not "Cannot find workload" in stderr_buf.getvalue()

    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    runtime.set_input(**params)
    runtime.set_input("data", data_np)
    runtime.run()
    out = runtime.get_output(0).numpy()
    ref = get_ref_dense(data_np, weight_np)
    tvm.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    tvm.testing.main()
