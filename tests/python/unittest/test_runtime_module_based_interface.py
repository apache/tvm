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
"""Test runtime module-based interface"""

import os

import numpy as np

import tvm
import tvm.relay.testing
from tvm import relay, runtime
from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor
from tvm.contrib.cuda_graph import cuda_graph_executor


def input_shape(mod):
    return [int(x) for x in mod["main"].checked_type.arg_types[0].shape]


def verify(data):
    """verify data"""
    if not tvm.runtime.enabled("llvm"):
        print("Skip because llvm is not enabled")
        return None

    mod, params = relay.testing.synthetic.get_workload()
    with relay.build_config(opt_level=3):
        graph, lib, graph_params = relay.build_module.build(mod, "llvm", params=params)

    dev = tvm.cpu()
    module = graph_executor.create(graph, lib, dev)
    module.set_input("data", data)
    module.set_input(**graph_params)
    module.run()
    out = module.get_output(0).numpy()

    return out


@tvm.testing.requires_llvm
def test_legacy_compatibility():
    """Test legacy compatibility"""
    mod, params = relay.testing.synthetic.get_workload()
    with relay.build_config(opt_level=3):
        graph, lib, graph_params = relay.build_module.build(mod, "llvm", params=params)
    data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")
    dev = tvm.cpu()
    module = graph_executor.create(graph, lib, dev)
    module.set_input("data", data)
    module.set_input(**graph_params)
    module.run()
    out = module.get_output(0).numpy()
    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)


@tvm.testing.requires_llvm
def test_cpu():
    """Test CPU"""
    mod, params = relay.testing.synthetic.get_workload()
    with relay.build_config(opt_level=3):
        complied_graph_lib = relay.build_module.build(mod, "llvm", params=params)
    data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")
    # raw api
    dev = tvm.cpu()
    gmod = complied_graph_lib["default"](dev)
    set_input = gmod["set_input"]
    run = gmod["run"]
    get_output = gmod["get_output"]
    set_input("data", tvm.nd.array(data))
    run()
    out = get_output(0).numpy()
    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    # graph executor wrapper
    gmod = graph_executor.GraphModule(complied_graph_lib["default"](dev))
    gmod.set_input("data", data)
    gmod.run()
    out = gmod.get_output(0).numpy()
    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)


@tvm.testing.requires_llvm
def test_cpu_get_graph_json():
    """Test CPU get graph JSON"""
    mod, params = relay.testing.synthetic.get_workload()
    with relay.build_config(opt_level=3):
        complied_graph_lib = relay.build_module.build(mod, "llvm", params=params)

    temp = tvm.contrib.utils.tempdir()
    file_name = "deploy_lib.so"
    path_lib = temp.relpath(file_name)
    complied_graph_lib.export_library(path_lib)
    loaded_lib = tvm.runtime.load_module(path_lib)
    json = loaded_lib["get_graph_json"]()
    assert isinstance(json, str) is True
    assert json.find("tvmgen_default_fused_nn_softmax_add") > -1


@tvm.testing.requires_llvm
def test_cpu_get_graph_params_run():
    """Test CPU get graph parameters and run"""
    mod, params = relay.testing.synthetic.get_workload()
    with tvm.transform.PassContext(opt_level=3):
        complied_graph_lib = relay.build_module.build(mod, "llvm", params=params)
    data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")
    dev = tvm.cpu()

    temp = tvm.contrib.utils.tempdir()
    file_name = "deploy_lib.so"
    path_lib = temp.relpath(file_name)
    complied_graph_lib.export_library(path_lib)

    loaded_lib = tvm.runtime.load_module(path_lib)
    loaded_params = loaded_lib["get_graph_params"]()

    gmod = graph_executor.GraphModule(loaded_lib["default"](dev))
    gmod.set_input(key="data", value=data, **loaded_params)
    gmod.run()
    out = gmod.get_output(0).numpy()
    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)


@tvm.testing.requires_llvm
def test_cpu_get_graph_params_compare():
    """Test CPU get graph parameters and compare"""
    # Create sample net

    inp_shape = (1, 3, 24, 12)
    dtype = "float32"
    data = relay.var("data", shape=inp_shape, dtype=dtype)
    conv_shape = [inp_shape[1], inp_shape[1], 3, 3]
    conv = relay.nn.conv2d(
        data,
        relay.var("conv_weight", shape=conv_shape, dtype=dtype),
        padding=1,
        kernel_size=3,
    )
    args = relay.analysis.free_vars(conv)
    func = relay.Function(args, conv)

    mod, params = relay.testing.init.create_workload(
        func,
        initializer=relay.testing.init.Constant()
    )

    with tvm.transform.PassContext(opt_level=3):
        complied_graph_lib = relay.build_module.build(mod, "llvm", params=params)

    temp = tvm.contrib.utils.tempdir()
    file_name = "deploy_lib.so"
    path_lib = temp.relpath(file_name)
    complied_graph_lib.export_library(path_lib)

    loaded_lib = tvm.runtime.load_module(path_lib)
    loaded_params = loaded_lib["get_graph_params"]()

    tvm.testing.assert_allclose(
        params["conv_weight"].numpy(), loaded_params["p0"].numpy()[0][0], atol=1e-5
    )


@tvm.testing.requires_cuda
@tvm.testing.requires_gpu
def test_gpu():
    """Test GPU"""
    mod, params = relay.testing.synthetic.get_workload()
    with relay.build_config(opt_level=3):
        complied_graph_lib = relay.build_module.build(mod, "cuda", params=params)
    data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")
    dev = tvm.cuda()

    # raw api
    gmod = complied_graph_lib["default"](dev)
    set_input = gmod["set_input"]
    run = gmod["run"]
    get_output = gmod["get_output"]
    set_input("data", tvm.nd.array(data))
    run()
    out = get_output(0).numpy()
    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    # graph executor wrapper
    gmod = graph_executor.GraphModule(complied_graph_lib["default"](dev))
    gmod.set_input("data", data)
    gmod.run()
    out = gmod.get_output(0).numpy()
    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)


@tvm.testing.uses_gpu
def test_mod_export():
    """Test module export"""

    def verify_cpu_export(obj_format):
        mod, params = relay.testing.synthetic.get_workload()
        with relay.build_config(opt_level=3):
            complied_graph_lib = relay.build_module.build(mod, "llvm", params=params)

        temp = tvm.contrib.utils.tempdir()
        if obj_format == ".so":
            file_name = "deploy_lib.so"
        else:
            assert obj_format == ".tar"
            file_name = "deploy_lib.tar"
        path_lib = temp.relpath(file_name)
        complied_graph_lib.export_library(path_lib)

        # run the setup in a separate function, so the load_lib
        # can get destructed right away
        # test the robustness wrt to parent module destruction
        def setup_gmod():
            loaded_lib = tvm.runtime.load_module(path_lib)
            dev = tvm.cpu(0)
            return loaded_lib["default"](dev)

        gmod = setup_gmod()
        # raw api
        set_input = gmod["set_input"]
        run = gmod["run"]
        get_output = gmod["get_output"]
        data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")
        set_input("data", tvm.nd.array(data))
        run()
        out = get_output(0).numpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

        # graph executor wrapper
        gmod = graph_executor.GraphModule(setup_gmod())
        gmod.set_input("data", data)
        gmod.run()
        out = gmod.get_output(0).numpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    def verify_gpu_export(obj_format):
        if not tvm.testing.device_enabled("cuda"):
            print("Skip because cuda is not enabled")
            return
        mod, params = relay.testing.synthetic.get_workload()
        with relay.build_config(opt_level=3):
            complied_graph_lib = relay.build_module.build(mod, "cuda", params=params)

        temp = tvm.contrib.utils.tempdir()
        if obj_format == ".so":
            file_name = "deploy_lib.so"
        else:
            assert obj_format == ".tar"
            file_name = "deploy_lib.tar"
        path_lib = temp.relpath(file_name)
        complied_graph_lib.export_library(path_lib)

        data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")

        # run the setup in a separate function, so the load_lib
        # can get destructed right away
        # test the robustness wrt to parent module destruction
        def setup_gmod():
            loaded_lib = tvm.runtime.load_module(path_lib)
            dev = tvm.cuda()
            return loaded_lib["default"](dev)

        gmod = setup_gmod()
        # raw api
        set_input = gmod["set_input"]
        run = gmod["run"]
        get_output = gmod["get_output"]
        set_input("data", tvm.nd.array(data))
        run()
        out = get_output(0).numpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

        # graph executor wrapper
        gmod = graph_executor.GraphModule(setup_gmod())
        gmod.set_input("data", data)
        gmod.run()
        out = gmod.get_output(0).numpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    @tvm.testing.requires_llvm
    def verify_rpc_cpu_export(obj_format):
        mod, params = relay.testing.synthetic.get_workload()
        with relay.build_config(opt_level=3):
            complied_graph_lib = relay.build_module.build(mod, "llvm", params=params)

        temp = tvm.contrib.utils.tempdir()
        if obj_format == ".so":
            file_name = "deploy_lib.so"
        else:
            assert obj_format == ".tar"
            file_name = "deploy_lib.tar"
        path_lib = temp.relpath(file_name)
        complied_graph_lib.export_library(path_lib)

        remote = tvm.rpc.LocalSession()
        remote.upload(path_lib)
        loaded_lib = remote.load_module(path_lib)
        data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")
        dev = remote.cpu()

        # raw api
        gmod = loaded_lib["default"](dev)
        set_input = gmod["set_input"]
        run = gmod["run"]
        get_output = gmod["get_output"]
        set_input("data", tvm.nd.array(data, device=dev))
        run()
        out = get_output(0).numpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

        # graph executor wrapper
        gmod = graph_executor.GraphModule(loaded_lib["default"](dev))
        gmod.set_input("data", data)
        gmod.run()
        out = gmod.get_output(0).numpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    def verify_rpc_gpu_export(obj_format):
        if not tvm.testing.device_enabled("cuda"):
            print("Skip because cuda is not enabled")
            return
        mod, params = relay.testing.synthetic.get_workload()
        with relay.build_config(opt_level=3):
            complied_graph_lib = relay.build_module.build(mod, "cuda", params=params)

        temp = tvm.contrib.utils.tempdir()
        if obj_format == ".so":
            file_name = "deploy_lib.so"
        else:
            assert obj_format == ".tar"
            file_name = "deploy_lib.tar"
        path_lib = temp.relpath(file_name)
        complied_graph_lib.export_library(path_lib)

        def check_remote(server):
            remote = tvm.rpc.connect(server.host, server.port)
            remote.upload(path_lib)
            loaded_lib = remote.load_module(path_lib)
            data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")
            dev = remote.cuda()

            # raw api
            gmod = loaded_lib["default"](dev)
            set_input = gmod["set_input"]
            run = gmod["run"]
            get_output = gmod["get_output"]
            set_input("data", tvm.nd.array(data, device=dev))
            run()
            out = get_output(0).numpy()
            tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

            # graph executor wrapper
            gmod = graph_executor.GraphModule(loaded_lib["default"](dev))
            gmod.set_input("data", data)
            gmod.run()
            out = gmod.get_output(0).numpy()
            tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

        check_remote(rpc.Server("127.0.0.1"))

    for obj_format in [".so", ".tar"]:
        verify_cpu_export(obj_format)
        verify_gpu_export(obj_format)
        verify_rpc_cpu_export(obj_format)
        verify_rpc_gpu_export(obj_format)


@tvm.testing.requires_llvm
@tvm.testing.uses_gpu
def test_remove_package_params():
    """Test removing package parameters"""
    def verify_cpu_remove_package_params(obj_format):
        mod, params = relay.testing.synthetic.get_workload()
        with relay.build_config(opt_level=3):
            complied_graph_lib = relay.build_module.build(mod, "llvm", params=params)

        temp = tvm.contrib.utils.tempdir()
        if obj_format == ".so":
            file_name = "deploy_lib.so"
        else:
            assert obj_format == ".tar"
            file_name = "deploy_lib.tar"
        path_lib = temp.relpath(file_name)
        complied_graph_lib_no_params = complied_graph_lib["remove_params"]()
        complied_graph_lib_no_params.export_library(path_lib)
        with open(temp.relpath("deploy_param.params"), "wb") as fout:
            fout.write(runtime.save_param_dict(complied_graph_lib.get_params()))
        loaded_lib = tvm.runtime.load_module(path_lib)
        data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")
        dev = tvm.cpu(0)

        # raw api
        gmod = loaded_lib["default"](dev)
        set_input = gmod["set_input"]
        run = gmod["run"]
        get_output = gmod["get_output"]
        load_params = gmod["load_params"]
        loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
        set_input("data", tvm.nd.array(data))
        load_params(loaded_params)
        run()
        out = get_output(0).numpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

        # graph executor wrapper
        gmod = graph_executor.GraphModule(loaded_lib["default"](dev))
        loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
        gmod.set_input("data", data)
        gmod.load_params(loaded_params)
        gmod.run()
        out = gmod.get_output(0).numpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    def verify_gpu_remove_package_params(obj_format):
        if not tvm.testing.device_enabled("cuda"):
            print("Skip because cuda is not enabled")
            return
        mod, params = relay.testing.synthetic.get_workload()
        with relay.build_config(opt_level=3):
            complied_graph_lib = relay.build_module.build(mod, "cuda", params=params)

        temp = tvm.contrib.utils.tempdir()
        if obj_format == ".so":
            file_name = "deploy_lib.so"
        else:
            assert obj_format == ".tar"
            file_name = "deploy_lib.tar"
        path_lib = temp.relpath(file_name)
        complied_graph_lib_no_params = complied_graph_lib["remove_params"]()
        complied_graph_lib_no_params.export_library(path_lib)
        with open(temp.relpath("deploy_param.params"), "wb") as fout:
            fout.write(runtime.save_param_dict(complied_graph_lib.get_params()))
        loaded_lib = tvm.runtime.load_module(path_lib)
        data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")
        dev = tvm.cuda(0)

        # raw api
        gmod = loaded_lib["default"](dev)
        set_input = gmod["set_input"]
        run = gmod["run"]
        get_output = gmod["get_output"]
        load_params = gmod["load_params"]
        loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
        set_input("data", tvm.nd.array(data))
        load_params(loaded_params)
        run()
        out = get_output(0).numpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

        # graph executor wrapper
        gmod = graph_executor.GraphModule(loaded_lib["default"](dev))
        loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
        gmod.set_input("data", data)
        gmod.load_params(loaded_params)
        gmod.run()
        out = gmod.get_output(0).numpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    @tvm.testing.requires_llvm
    def verify_rpc_cpu_remove_package_params(obj_format):
        mod, params = relay.testing.synthetic.get_workload()
        with relay.build_config(opt_level=3):
            complied_graph_lib = relay.build_module.build(mod, "llvm", params=params)

        temp = tvm.contrib.utils.tempdir()
        if obj_format == ".so":
            file_name = "deploy_lib.so"
        else:
            assert obj_format == ".tar"
            file_name = "deploy_lib.tar"
        path_lib = temp.relpath(file_name)
        complied_graph_lib_no_params = complied_graph_lib["remove_params"]()
        complied_graph_lib_no_params.export_library(path_lib)
        path_params = temp.relpath("deploy_param.params")
        with open(path_params, "wb") as fout:
            fout.write(runtime.save_param_dict(complied_graph_lib.get_params()))

        remote = tvm.rpc.LocalSession()
        remote.upload(path_lib)
        loaded_lib = remote.load_module(path_lib)
        data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")
        dev = remote.cpu()

        # raw api
        gmod = loaded_lib["default"](dev)
        set_input = gmod["set_input"]
        run = gmod["run"]
        get_output = gmod["get_output"]
        load_params = gmod["load_params"]
        loaded_params = bytearray(open(path_params, "rb").read())
        set_input("data", tvm.nd.array(data, device=dev))
        load_params(loaded_params)
        run()
        out = get_output(0).numpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

        # graph executor wrapper
        gmod = graph_executor.GraphModule(loaded_lib["default"](dev))
        loaded_params = bytearray(open(path_params, "rb").read())
        gmod.set_input("data", data)
        gmod.load_params(loaded_params)
        gmod.run()
        out = gmod.get_output(0).numpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    def verify_rpc_gpu_remove_package_params(obj_format):
        if not tvm.testing.device_enabled("cuda"):
            print("Skip because cuda is not enabled")
            return
        mod, params = relay.testing.synthetic.get_workload()
        with relay.build_config(opt_level=3):
            complied_graph_lib = relay.build_module.build(mod, "cuda", params=params)

        temp = tvm.contrib.utils.tempdir()
        if obj_format == ".so":
            file_name = "deploy_lib.so"
        else:
            assert obj_format == ".tar"
            file_name = "deploy_lib.tar"
        path_lib = temp.relpath(file_name)
        complied_graph_lib_no_params = complied_graph_lib["remove_params"]()
        complied_graph_lib_no_params.export_library(path_lib)
        path_params = temp.relpath("deploy_param.params")
        with open(path_params, "wb") as fout:
            fout.write(runtime.save_param_dict(complied_graph_lib.get_params()))

        remote = tvm.rpc.LocalSession()
        remote.upload(path_lib)
        loaded_lib = remote.load_module(path_lib)
        data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")
        dev = remote.cuda()

        # raw api
        gmod = loaded_lib["default"](dev)
        set_input = gmod["set_input"]
        run = gmod["run"]
        get_output = gmod["get_output"]
        load_params = gmod["load_params"]
        loaded_params = bytearray(open(path_params, "rb").read())
        set_input("data", tvm.nd.array(data, device=dev))
        load_params(loaded_params)
        run()
        out = get_output(0).numpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

        # graph executor wrapper
        gmod = graph_executor.GraphModule(loaded_lib["default"](dev))
        loaded_params = bytearray(open(path_params, "rb").read())
        gmod.set_input("data", data)
        gmod.load_params(loaded_params)
        gmod.run()
        out = gmod.get_output(0).numpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    for obj_format in [".so", ".tar"]:
        verify_cpu_remove_package_params(obj_format)
        verify_gpu_remove_package_params(obj_format)
        verify_rpc_cpu_remove_package_params(obj_format)
        verify_rpc_gpu_remove_package_params(obj_format)


@tvm.testing.requires_llvm
def test_debug_graph_executor():
    """Test debug graph executor"""
    mod, params = relay.testing.synthetic.get_workload()
    with relay.build_config(opt_level=3):
        complied_graph_lib = relay.build_module.build(mod, "llvm", params=params)
    data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")

    # raw api
    dev = tvm.cpu()
    try:
        gmod = complied_graph_lib["debug_create"]("default", dev)
    except:  # pylint: disable=bare-except
        print("Skip because debug graph_executor not enabled")
        return
    set_input = gmod["set_input"]
    run = gmod["run"]
    get_output = gmod["get_output"]
    set_input("data", tvm.nd.array(data))
    run()
    out = get_output(0).numpy()
    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    # debug graph executor wrapper
    debug_g_mod = debug_executor.GraphModuleDebug(
        complied_graph_lib["debug_create"]("default", dev),
        [dev],
        complied_graph_lib.get_graph_json(),
        None,
    )
    debug_g_mod.set_input("data", data)
    debug_g_mod.run()
    out = debug_g_mod.get_output(0).numpy()
    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)


@tvm.testing.requires_cudagraph
def test_cuda_graph_executor():
    """Test CUDA graph executor"""
    mod, params = relay.testing.synthetic.get_workload()
    with tvm.transform.PassContext(opt_level=3):
        complied_graph_lib = relay.build_module.build(mod, "cuda", params=params)
    data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")

    dev = tvm.cuda()
    try:
        gmod = complied_graph_lib["cuda_graph_create"](dev)
    except:  # pylint: disable=bare-except
        print("Skip because cuda_graph not enabled")
        return
    set_input = gmod["set_input"]
    run = gmod["run"]
    get_output = gmod["get_output"]
    set_input("data", tvm.nd.array(data))
    run()
    out = get_output(0).numpy()
    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    # cuda graph executor wrapper
    cu_gmod = cuda_graph_executor.GraphModuleCudaGraph(gmod)
    cu_gmod.set_input("data", data)
    cu_gmod.run()
    out = cu_gmod.get_output(0).numpy()
    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)


def test_multiple_imported_modules():
    """Test multiple imported modules"""

    def make_func(symbol):
        n = tvm.te.size_var("n")
        a_b = tvm.tir.decl_buffer((n,), dtype="float32")
        i = tvm.te.var("i")
        stmt = tvm.tir.For(
            i,
            0,
            n - 1,
            tvm.tir.ForKind.SERIAL,
            tvm.tir.BufferStore(a_b, tvm.tir.BufferLoad(a_b, [i]) + 1, [i + 1]),
        )
        return tvm.tir.PrimFunc([a_b], stmt).with_attr("global_symbol", symbol)

    def make_module(mod):
        mod = tvm.IRModule(mod)
        mod = tvm.driver.build(mod, target="llvm")
        return mod

    module_main = make_module({"main": make_func("main")})
    module_a = make_module({"func_a": make_func("func_a")})
    module_b = make_module({"func_b": make_func("func_b")})
    module_main.import_module(module_a)
    module_main.import_module(module_b)
    module_main.get_function("func_a", query_imports=True)
    module_main.get_function("func_b", query_imports=True)


def test_num_threads():
    """Test number of threads"""
    reported = tvm.runtime.num_threads()
    env_threads = os.getenv("TVM_NUM_THREADS")
    omp_env_threads = os.getenv("OMP_NUM_THREADS")
    if env_threads is not None:
        assert reported == env_threads
    elif omp_env_threads is not None:
        assert reported == omp_env_threads
    else:
        hardware_threads = os.cpu_count()
        assert reported in (hardware_threads, hardware_threads // 2)


if __name__ == "__main__":
    test_legacy_compatibility()
    test_cpu()
    test_gpu()
    test_mod_export()
    test_remove_package_params()
    test_debug_graph_executor()
    test_multiple_imported_modules()
    test_cpu_get_graph_json()
    test_cpu_get_graph_params_run()
    test_cpu_get_graph_params_compare()
