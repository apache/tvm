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
from tvm import relay
from tvm.relay import testing
import tvm
from tvm.contrib import graph_runtime
from tvm.contrib.debugger import debug_runtime
import tvm.testing


def input_shape(mod):
    return [int(x) for x in mod["main"].checked_type.arg_types[0].shape]


def verify(data):
    if not tvm.runtime.enabled("llvm"):
        print("Skip because llvm is not enabled")
        return
    mod, params = relay.testing.synthetic.get_workload()
    with relay.build_config(opt_level=3):
        graph, lib, graph_params = relay.build_module.build(mod, "llvm", params=params)

    ctx = tvm.cpu()
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input("data", data)
    module.set_input(**graph_params)
    module.run()
    out = module.get_output(0).asnumpy()

    return out


def test_legacy_compatibility():
    if not tvm.testing.device_enabled("llvm"):
        print("Skip because llvm is not enabled")
        return
    mod, params = relay.testing.synthetic.get_workload()
    with relay.build_config(opt_level=3):
        graph, lib, graph_params = relay.build_module.build(mod, "llvm", params=params)
    data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")
    ctx = tvm.cpu()
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input("data", data)
    module.set_input(**graph_params)
    module.run()
    out = module.get_output(0).asnumpy()
    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)


def test_cpu():
    if not tvm.testing.device_enabled("llvm"):
        print("Skip because llvm is not enabled")
        return
    mod, params = relay.testing.synthetic.get_workload()
    with relay.build_config(opt_level=3):
        complied_graph_lib = relay.build_module.build(mod, "llvm", params=params)
    data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")
    # raw api
    ctx = tvm.cpu()
    gmod = complied_graph_lib["default"](ctx)
    set_input = gmod["set_input"]
    run = gmod["run"]
    get_output = gmod["get_output"]
    set_input("data", tvm.nd.array(data))
    run()
    out = get_output(0).asnumpy()
    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    # graph runtime wrapper
    gmod = graph_runtime.GraphModule(complied_graph_lib["default"](ctx))
    gmod.set_input("data", data)
    gmod.run()
    out = gmod.get_output(0).asnumpy()
    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)


@tvm.testing.requires_cuda
@tvm.testing.requires_gpu
def test_gpu():
    mod, params = relay.testing.synthetic.get_workload()
    with relay.build_config(opt_level=3):
        complied_graph_lib = relay.build_module.build(mod, "cuda", params=params)
    data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")
    ctx = tvm.gpu()

    # raw api
    gmod = complied_graph_lib["default"](ctx)
    set_input = gmod["set_input"]
    run = gmod["run"]
    get_output = gmod["get_output"]
    set_input("data", tvm.nd.array(data))
    run()
    out = get_output(0).asnumpy()
    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    # graph runtime wrapper
    gmod = graph_runtime.GraphModule(complied_graph_lib["default"](ctx))
    gmod.set_input("data", data)
    gmod.run()
    out = gmod.get_output(0).asnumpy()
    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)


@tvm.testing.uses_gpu
def test_mod_export():
    def verify_cpu_export(obj_format):
        if not tvm.testing.device_enabled("llvm"):
            print("Skip because llvm is not enabled")
            return
        mod, params = relay.testing.synthetic.get_workload()
        with relay.build_config(opt_level=3):
            complied_graph_lib = relay.build_module.build(mod, "llvm", params=params)

        from tvm.contrib import util

        temp = util.tempdir()
        if obj_format == ".so":
            file_name = "deploy_lib.so"
        else:
            assert obj_format == ".tar"
            file_name = "deploy_lib.tar"
        path_lib = temp.relpath(file_name)
        complied_graph_lib.export_library(path_lib)
        loaded_lib = tvm.runtime.load_module(path_lib)
        ctx = tvm.cpu(0)
        gmod = loaded_lib["default"](ctx)

        # raw api
        set_input = gmod["set_input"]
        run = gmod["run"]
        get_output = gmod["get_output"]
        data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")
        set_input("data", tvm.nd.array(data))
        run()
        out = get_output(0).asnumpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

        # graph runtime wrapper
        gmod = graph_runtime.GraphModule(loaded_lib["default"](ctx))
        gmod.set_input("data", data)
        gmod.run()
        out = gmod.get_output(0).asnumpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    def verify_gpu_export(obj_format):
        if not tvm.testing.device_enabled("cuda"):
            print("Skip because cuda is not enabled")
            return
        mod, params = relay.testing.synthetic.get_workload()
        with relay.build_config(opt_level=3):
            complied_graph_lib = relay.build_module.build(mod, "cuda", params=params)

        from tvm.contrib import util

        temp = util.tempdir()
        if obj_format == ".so":
            file_name = "deploy_lib.so"
        else:
            assert obj_format == ".tar"
            file_name = "deploy_lib.tar"
        path_lib = temp.relpath(file_name)
        complied_graph_lib.export_library(path_lib)
        loaded_lib = tvm.runtime.load_module(path_lib)
        data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")
        ctx = tvm.gpu()

        # raw api
        gmod = loaded_lib["default"](ctx)
        set_input = gmod["set_input"]
        run = gmod["run"]
        get_output = gmod["get_output"]
        set_input("data", tvm.nd.array(data))
        run()
        out = get_output(0).asnumpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

        # graph runtime wrapper
        gmod = graph_runtime.GraphModule(loaded_lib["default"](ctx))
        gmod.set_input("data", data)
        gmod.run()
        out = gmod.get_output(0).asnumpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    def verify_rpc_cpu_export(obj_format):
        if not tvm.testing.device_enabled("llvm"):
            print("Skip because llvm is not enabled")
            return
        mod, params = relay.testing.synthetic.get_workload()
        with relay.build_config(opt_level=3):
            complied_graph_lib = relay.build_module.build(mod, "llvm", params=params)

        from tvm.contrib import util

        temp = util.tempdir()
        if obj_format == ".so":
            file_name = "deploy_lib.so"
        else:
            assert obj_format == ".tar"
            file_name = "deploy_lib.tar"
        path_lib = temp.relpath(file_name)
        complied_graph_lib.export_library(path_lib)

        from tvm import rpc

        remote = rpc.LocalSession()
        remote.upload(path_lib)
        loaded_lib = remote.load_module(path_lib)
        data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")
        ctx = remote.cpu()

        # raw api
        gmod = loaded_lib["default"](ctx)
        set_input = gmod["set_input"]
        run = gmod["run"]
        get_output = gmod["get_output"]
        set_input("data", tvm.nd.array(data, ctx=ctx))
        run()
        out = get_output(0).asnumpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

        # graph runtime wrapper
        gmod = graph_runtime.GraphModule(loaded_lib["default"](ctx))
        gmod.set_input("data", data)
        gmod.run()
        out = gmod.get_output(0).asnumpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    def verify_rpc_gpu_export(obj_format):
        if not tvm.testing.device_enabled("cuda"):
            print("Skip because cuda is not enabled")
            return
        mod, params = relay.testing.synthetic.get_workload()
        with relay.build_config(opt_level=3):
            complied_graph_lib = relay.build_module.build(mod, "cuda", params=params)

        from tvm.contrib import util

        temp = util.tempdir()
        if obj_format == ".so":
            file_name = "deploy_lib.so"
        else:
            assert obj_format == ".tar"
            file_name = "deploy_lib.tar"
        path_lib = temp.relpath(file_name)
        complied_graph_lib.export_library(path_lib)

        from tvm import rpc

        server = rpc.Server("localhost", use_popen=True, port=9094)
        remote = rpc.connect(server.host, server.port)
        remote.upload(path_lib)
        loaded_lib = remote.load_module(path_lib)
        data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")
        ctx = remote.gpu()

        # raw api
        gmod = loaded_lib["default"](ctx)
        set_input = gmod["set_input"]
        run = gmod["run"]
        get_output = gmod["get_output"]
        set_input("data", tvm.nd.array(data, ctx=ctx))
        run()
        out = get_output(0).asnumpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

        # graph runtime wrapper
        gmod = graph_runtime.GraphModule(loaded_lib["default"](ctx))
        gmod.set_input("data", data)
        gmod.run()
        out = gmod.get_output(0).asnumpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    for obj_format in [".so", ".tar"]:
        verify_cpu_export(obj_format)
        verify_gpu_export(obj_format)
        verify_rpc_cpu_export(obj_format)
        verify_rpc_gpu_export(obj_format)


@tvm.testing.uses_gpu
def test_remove_package_params():
    def verify_cpu_remove_package_params(obj_format):
        if not tvm.testing.device_enabled("llvm"):
            print("Skip because llvm is not enabled")
            return
        mod, params = relay.testing.synthetic.get_workload()
        with relay.build_config(opt_level=3):
            complied_graph_lib = relay.build_module.build(mod, "llvm", params=params)

        from tvm.contrib import util

        temp = util.tempdir()
        if obj_format == ".so":
            file_name = "deploy_lib.so"
        else:
            assert obj_format == ".tar"
            file_name = "deploy_lib.tar"
        path_lib = temp.relpath(file_name)
        complied_graph_lib_no_params = complied_graph_lib["remove_params"]()
        complied_graph_lib_no_params.export_library(path_lib)
        with open(temp.relpath("deploy_param.params"), "wb") as fo:
            fo.write(relay.save_param_dict(complied_graph_lib.get_params()))
        loaded_lib = tvm.runtime.load_module(path_lib)
        data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")
        ctx = tvm.cpu(0)

        # raw api
        gmod = loaded_lib["default"](ctx)
        set_input = gmod["set_input"]
        run = gmod["run"]
        get_output = gmod["get_output"]
        load_params = gmod["load_params"]
        loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
        set_input("data", tvm.nd.array(data))
        load_params(loaded_params)
        run()
        out = get_output(0).asnumpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

        # graph runtime wrapper
        gmod = graph_runtime.GraphModule(loaded_lib["default"](ctx))
        loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
        gmod.set_input("data", data)
        gmod.load_params(loaded_params)
        gmod.run()
        out = gmod.get_output(0).asnumpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    def verify_gpu_remove_package_params(obj_format):
        if not tvm.testing.device_enabled("cuda"):
            print("Skip because cuda is not enabled")
            return
        mod, params = relay.testing.synthetic.get_workload()
        with relay.build_config(opt_level=3):
            complied_graph_lib = relay.build_module.build(mod, "cuda", params=params)

        from tvm.contrib import util

        temp = util.tempdir()
        if obj_format == ".so":
            file_name = "deploy_lib.so"
        else:
            assert obj_format == ".tar"
            file_name = "deploy_lib.tar"
        path_lib = temp.relpath(file_name)
        complied_graph_lib_no_params = complied_graph_lib["remove_params"]()
        complied_graph_lib_no_params.export_library(path_lib)
        with open(temp.relpath("deploy_param.params"), "wb") as fo:
            fo.write(relay.save_param_dict(complied_graph_lib.get_params()))
        loaded_lib = tvm.runtime.load_module(path_lib)
        data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")
        ctx = tvm.gpu(0)

        # raw api
        gmod = loaded_lib["default"](ctx)
        set_input = gmod["set_input"]
        run = gmod["run"]
        get_output = gmod["get_output"]
        load_params = gmod["load_params"]
        loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
        set_input("data", tvm.nd.array(data))
        load_params(loaded_params)
        run()
        out = get_output(0).asnumpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

        # graph runtime wrapper
        gmod = graph_runtime.GraphModule(loaded_lib["default"](ctx))
        loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
        gmod.set_input("data", data)
        gmod.load_params(loaded_params)
        gmod.run()
        out = gmod.get_output(0).asnumpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    def verify_rpc_cpu_remove_package_params(obj_format):
        if not tvm.testing.device_enabled("llvm"):
            print("Skip because llvm is not enabled")
            return
        mod, params = relay.testing.synthetic.get_workload()
        with relay.build_config(opt_level=3):
            complied_graph_lib = relay.build_module.build(mod, "llvm", params=params)

        from tvm.contrib import util

        temp = util.tempdir()
        if obj_format == ".so":
            file_name = "deploy_lib.so"
        else:
            assert obj_format == ".tar"
            file_name = "deploy_lib.tar"
        path_lib = temp.relpath(file_name)
        complied_graph_lib_no_params = complied_graph_lib["remove_params"]()
        complied_graph_lib_no_params.export_library(path_lib)
        path_params = temp.relpath("deploy_param.params")
        with open(path_params, "wb") as fo:
            fo.write(relay.save_param_dict(complied_graph_lib.get_params()))

        from tvm import rpc

        remote = rpc.LocalSession()
        remote.upload(path_lib)
        loaded_lib = remote.load_module(path_lib)
        data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")
        ctx = remote.cpu()

        # raw api
        gmod = loaded_lib["default"](ctx)
        set_input = gmod["set_input"]
        run = gmod["run"]
        get_output = gmod["get_output"]
        load_params = gmod["load_params"]
        loaded_params = bytearray(open(path_params, "rb").read())
        set_input("data", tvm.nd.array(data, ctx=ctx))
        load_params(loaded_params)
        run()
        out = get_output(0).asnumpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

        # graph runtime wrapper
        gmod = graph_runtime.GraphModule(loaded_lib["default"](ctx))
        loaded_params = bytearray(open(path_params, "rb").read())
        gmod.set_input("data", data)
        gmod.load_params(loaded_params)
        gmod.run()
        out = gmod.get_output(0).asnumpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    def verify_rpc_gpu_remove_package_params(obj_format):
        if not tvm.testing.device_enabled("cuda"):
            print("Skip because cuda is not enabled")
            return
        mod, params = relay.testing.synthetic.get_workload()
        with relay.build_config(opt_level=3):
            complied_graph_lib = relay.build_module.build(mod, "cuda", params=params)

        from tvm.contrib import util

        temp = util.tempdir()
        if obj_format == ".so":
            file_name = "deploy_lib.so"
        else:
            assert obj_format == ".tar"
            file_name = "deploy_lib.tar"
        path_lib = temp.relpath(file_name)
        complied_graph_lib_no_params = complied_graph_lib["remove_params"]()
        complied_graph_lib_no_params.export_library(path_lib)
        path_params = temp.relpath("deploy_param.params")
        with open(path_params, "wb") as fo:
            fo.write(relay.save_param_dict(complied_graph_lib.get_params()))

        from tvm import rpc

        remote = rpc.LocalSession()
        remote.upload(path_lib)
        loaded_lib = remote.load_module(path_lib)
        data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")
        ctx = remote.gpu()

        # raw api
        gmod = loaded_lib["default"](ctx)
        set_input = gmod["set_input"]
        run = gmod["run"]
        get_output = gmod["get_output"]
        load_params = gmod["load_params"]
        loaded_params = bytearray(open(path_params, "rb").read())
        set_input("data", tvm.nd.array(data, ctx=ctx))
        load_params(loaded_params)
        run()
        out = get_output(0).asnumpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

        # graph runtime wrapper
        gmod = graph_runtime.GraphModule(loaded_lib["default"](ctx))
        loaded_params = bytearray(open(path_params, "rb").read())
        gmod.set_input("data", data)
        gmod.load_params(loaded_params)
        gmod.run()
        out = gmod.get_output(0).asnumpy()
        tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    for obj_format in [".so", ".tar"]:
        verify_cpu_remove_package_params(obj_format)
        verify_gpu_remove_package_params(obj_format)
        verify_rpc_cpu_remove_package_params(obj_format)
        verify_rpc_gpu_remove_package_params(obj_format)


def test_debug_graph_runtime():
    if not tvm.testing.device_enabled("llvm"):
        print("Skip because llvm is not enabled")
        return
    mod, params = relay.testing.synthetic.get_workload()
    with relay.build_config(opt_level=3):
        complied_graph_lib = relay.build_module.build(mod, "llvm", params=params)
    data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")

    # raw api
    ctx = tvm.cpu()
    try:
        gmod = complied_graph_lib["debug_create"]("default", ctx)
    except:
        print("Skip because debug graph_runtime not enabled")
        return
    set_input = gmod["set_input"]
    run = gmod["run"]
    get_output = gmod["get_output"]
    set_input("data", tvm.nd.array(data))
    run()
    out = get_output(0).asnumpy()
    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    # debug graph runtime wrapper
    debug_g_mod = debug_runtime.GraphModuleDebug(
        complied_graph_lib["debug_create"]("default", ctx),
        [ctx],
        complied_graph_lib.get_json(),
        None,
    )
    debug_g_mod.set_input("data", data)
    debug_g_mod.run()
    out = debug_g_mod.get_output(0).asnumpy()
    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)


if __name__ == "__main__":
    test_legacy_compatibility()
    test_cpu()
    test_gpu()
    test_mod_export()
    test_remove_package_params()
    test_debug_graph_runtime()
