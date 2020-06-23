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
from tvm.runtime import graph_runtime_factory

def get_workload(num_layers=18):
    mod, params = relay.testing.resnet.get_workload(num_layers=num_layers)
    return mod, params

def verify(data, num_layers=18):
    mod, params = get_workload(num_layers)
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
    mod, params = get_workload()
    with relay.build_config(opt_level=3):
        graph, lib, graph_params = relay.build_module.build(mod, "llvm", params=params)
    data = np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype("float32")
    ctx = tvm.cpu()
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input("data", data)
    module.set_input(**graph_params)
    module.run()
    out = module.get_output(0).asnumpy()
    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

def test_cpu():
    mod, params = get_workload()
    with relay.build_config(opt_level=3):
        complied_graph_lib = relay.build_module.build(mod, "llvm", params=params)
    data = np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype("float32")
    # raw api
    ctx = tvm.cpu()
    gmod = complied_graph_lib['default'](ctx)
    set_input = gmod["set_input"]
    run = gmod["run"]
    get_output = gmod["get_output"]
    set_input("data", tvm.nd.array(data))
    run()
    out = get_output(0).asnumpy()

    # graph runtime
    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)
    gmod = graph_runtime.create4unified(complied_graph_lib['default'], ctx)
    gmod.set_input("data", data)
    gmod.run()
    out = gmod.get_output(0).asnumpy()

    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

def test_gpu():
    mod, params = get_workload()
    with relay.build_config(opt_level=3):
        complied_graph_lib = relay.build_module.build(mod, "cuda", params=params)
    data = np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype("float32")
    ctx = tvm.gpu()
    gmod = complied_graph_lib['default'](ctx)
    set_input = gmod["set_input"]
    run = gmod["run"]
    get_output = gmod["get_output"]
    set_input("data", tvm.nd.array(data))
    run()
    out = get_output(0).asnumpy()

    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

def test_multi_models():
    resnet18_mod, resnet18_params = get_workload()
    resnet50_mod, resnet50_params = get_workload(50)
    with relay.build_config(opt_level=3):
        complied_graph_lib = relay.build_module.build(
            resnet18_mod, "llvm", params=resnet18_params, mod_name='resnet18')
    with relay.build_config(opt_level=3):
        resnet50_gpu_lib = relay.build_module.build(
            resnet50_mod, "cuda", params=resnet50_params, mod_name='resnet50')
    complied_graph_lib.import_module(resnet50_gpu_lib)
    data = np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype("float32")
    # resnet18
    cpu_ctx = tvm.cpu()
    gmod = complied_graph_lib['resnet18'](cpu_ctx)
    set_input = gmod["set_input"]
    get_input = gmod["get_input"]
    run = gmod["run"]
    get_output = gmod["get_output"]
    set_input("data", tvm.nd.array(data))
    run()
    out = get_output(0).asnumpy()
    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    # resnet50
    gpu_ctx = tvm.gpu()
    gmod = complied_graph_lib['resnet50'](gpu_ctx)
    set_input = gmod["set_input"]
    run = gmod["run"]
    get_output = gmod["get_output"]
    set_input("data", tvm.nd.array(data))
    run()
    out = get_output(0).asnumpy()
    tvm.testing.assert_allclose(out, verify(data, 50), atol=1e-5)

def test_cpu_export(format=".so"):
    mod, params = get_workload()
    with relay.build_config(opt_level=3):
        complied_graph_lib = relay.build_module.build(mod, "llvm", params=params)

    from tvm.contrib import util
    temp = util.tempdir()
    if format == ".so":
        file_name = "deploy_lib.so"
    else:
        assert format == ".tar"
        file_name = "deploy_lib.tar"
    path_lib = temp.relpath(file_name)
    complied_graph_lib.export_library(path_lib)
    loaded_lib = tvm.runtime.load_module(path_lib)
    ctx = tvm.cpu(0)
    gmod = loaded_lib['default'](ctx)
    set_input = gmod["set_input"]
    run = gmod["run"]
    get_output = gmod["get_output"]
    data = np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype("float32")
    set_input("data", tvm.nd.array(data))
    run()
    out = get_output(0).asnumpy()

    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

def test_gpu_export(format=".so"):
    mod, params = get_workload()
    with relay.build_config(opt_level=3):
        complied_graph_lib = relay.build_module.build(mod, "cuda", params=params)

    from tvm.contrib import util
    temp = util.tempdir()
    if format == ".so":
        file_name = "deploy_lib.so"
    else:
        assert format == ".tar"
        file_name = "deploy_lib.tar"
    path_lib = temp.relpath(file_name)
    complied_graph_lib.export_library(path_lib)
    loaded_lib = tvm.runtime.load_module(path_lib)
    data = np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype("float32")
    ctx = tvm.gpu()
    gmod = loaded_lib['default'](ctx)
    set_input = gmod["set_input"]
    run = gmod["run"]
    get_output = gmod["get_output"]
    set_input("data", tvm.nd.array(data))
    run()
    out = get_output(0).asnumpy()

    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

def test_previous_cpu_export(format=".so"):
    mod, params = get_workload()
    with relay.build_config(opt_level=3):
        graph, lib, graph_params = relay.build_module.build(mod, "llvm", params=params)

    from tvm.contrib import util
    temp = util.tempdir()
    if format == ".so":
        file_name = "deploy_lib.so"
    else:
        assert format == ".tar"
        file_name = "deploy_lib.tar"
    path_lib = temp.relpath(file_name)
    lib.export_library(path_lib)
    with open(temp.relpath("deploy_graph.json"), "w") as fo:
        fo.write(graph)
    with open(temp.relpath("deploy_param.params"), "wb") as fo:
        fo.write(relay.save_param_dict(graph_params))
    loaded_json = open(temp.relpath("deploy_graph.json")).read()
    loaded_lib = tvm.runtime.load_module(path_lib)
    loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
    data = np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype("float32")
    ctx = tvm.cpu()
    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    module.load_params(loaded_params)
    module.set_input("data", data)
    module.run()
    out = module.get_output(0).asnumpy()

    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

def test_previous_gpu_export(format=".so"):
    mod, params = get_workload()
    with relay.build_config(opt_level=3):
        graph, lib, graph_params = relay.build_module.build(mod, "cuda", params=params)

    from tvm.contrib import util
    temp = util.tempdir()
    if format == ".so":
        file_name = "deploy_lib.so"
    else:
        assert format == ".tar"
        file_name = "deploy_lib.tar"
    path_lib = temp.relpath(file_name)
    lib.export_library(path_lib)
    with open(temp.relpath("deploy_graph.json"), "w") as fo:
        fo.write(graph)
    with open(temp.relpath("deploy_param.params"), "wb") as fo:
        fo.write(relay.save_param_dict(graph_params))
    loaded_json = open(temp.relpath("deploy_graph.json")).read()
    loaded_lib = tvm.runtime.load_module(path_lib)
    loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
    data = np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype("float32")
    ctx = tvm.gpu()
    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    module.load_params(loaded_params)
    module.set_input("data", data)
    module.run()
    out = module.get_output(0).asnumpy()

    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

def test_rpc_export(format=".so"):
    mod, params = get_workload()
    with relay.build_config(opt_level=3):
        complied_graph_lib = relay.build_module.build(mod, "llvm", params=params)

    from tvm.contrib import util
    temp = util.tempdir()
    if format == ".so":
        file_name = "deploy_lib.so"
    else:
        assert format == ".tar"
        file_name = "deploy_lib.tar"
    path_lib = temp.relpath(file_name)
    complied_graph_lib.export_library(path_lib)

    from tvm import rpc
    server = rpc.Server("localhost", use_popen=True)
    remote = rpc.connect(server.host, server.port)
    remote.upload(path_lib)
    loaded_lib = remote.load_module(path_lib)
    data = np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype("float32")
    ctx = remote.cpu()
    gmod = loaded_lib['default'](ctx)
    set_input = gmod["set_input"]
    run = gmod["run"]
    get_output = gmod["get_output"]
    set_input("data", tvm.nd.array(data, ctx=ctx))
    run()
    out = get_output(0).asnumpy()
    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    gmod = graph_runtime.create4unified(loaded_lib['default'], ctx)
    gmod.set_input("data", data)
    gmod.run()
    out = gmod.get_output(0).asnumpy()

    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

def test_previous_rpc_export(format=".so"):
    mod, params = get_workload()
    with relay.build_config(opt_level=3):
        graph, lib, graph_params = relay.build_module.build(mod, "llvm", params=params)

    from tvm.contrib import util
    temp = util.tempdir()
    if format == ".so":
        file_name = "deploy_lib.so"
    else:
        assert format == ".tar"
        file_name = "deploy_lib.tar"
    path_lib = temp.relpath(file_name)
    lib.export_library(path_lib)
    with open(temp.relpath("deploy_graph.json"), "w") as fo:
        fo.write(graph)
    with open(temp.relpath("deploy_param.params"), "wb") as fo:
        fo.write(relay.save_param_dict(graph_params))

    from tvm import rpc
    server = rpc.Server("localhost", use_popen=True)
    remote = rpc.connect(server.host, server.port)
    remote.upload(path_lib)
    loaded_json = open(temp.relpath("deploy_graph.json")).read()
    loaded_lib = remote.load_module(path_lib)
    loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
    data = np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype("float32")
    ctx = remote.cpu()
    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    module.load_params(loaded_params)
    module.set_input("data", data)
    module.run()
    out = module.get_output(0).asnumpy()

    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

def test_package_params(format=".so"):
    mod, params = get_workload()
    with relay.build_config(opt_level=3):
        complied_graph_lib = relay.build_module.build(mod, "llvm", params=params)

    from tvm.contrib import util
    temp = util.tempdir()
    if format == ".so":
        file_name = "deploy_lib.so"
    else:
        assert format == ".tar"
        file_name = "deploy_lib.tar"
    path_lib = temp.relpath(file_name)
    complied_graph_lib.export_library(path_lib, package_params=False)
    loaded_lib = tvm.runtime.load_module(path_lib)
    ctx = tvm.cpu(0)
    gmod = loaded_lib['default'](ctx)
    set_input = gmod["set_input"]
    run = gmod["run"]
    get_output = gmod["get_output"]
    load_params = gmod["load_params"]
    data = np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype("float32")
    loaded_params = bytearray(open(temp.relpath("deploy_0.params"), "rb").read())
    set_input("data", tvm.nd.array(data))
    load_params(loaded_params)
    run()
    out = get_output(0).asnumpy()

    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

def test_multi_models_package_params(format=".so"):
    resnet18_mod, resnet18_params = get_workload()
    resnet50_mod, resnet50_params = get_workload(50)
    with relay.build_config(opt_level=3):
        complied_graph_lib = relay.build_module.build(
            resnet18_mod, "llvm", params=resnet18_params, mod_name='resnet18')
    with relay.build_config(opt_level=3):
        resnet50_gpu_lib = relay.build_module.build(
            resnet50_mod, "cuda", params=resnet50_params, mod_name='resnet50')
    complied_graph_lib.import_module(resnet50_gpu_lib)

    from tvm.contrib import util
    temp = util.tempdir()
    if format == ".so":
        file_name = "deploy_lib.so"
    else:
        assert format == ".tar"
        file_name = "deploy_lib.tar"
    path_lib = temp.relpath(file_name)
    complied_graph_lib.export_library(path_lib, package_params=False)
    loaded_lib = tvm.runtime.load_module(path_lib)

    # resnet18
    ctx = tvm.cpu(0)
    gmod = loaded_lib['resnet18'](ctx)
    set_input = gmod["set_input"]
    run = gmod["run"]
    get_output = gmod["get_output"]
    load_params = gmod["load_params"]
    data = np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype("float32")
    loaded_params = bytearray(open(temp.relpath("deploy_0.params"), "rb").read())
    set_input("data", tvm.nd.array(data))
    load_params(loaded_params)
    run()
    out = get_output(0).asnumpy()
    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    # resnet50
    ctx = tvm.gpu()
    gmod = loaded_lib['resnet50'](ctx)
    set_input = gmod["set_input"]
    run = gmod["run"]
    get_output = gmod["get_output"]
    load_params = gmod["load_params"]
    data = np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype("float32")
    loaded_params = bytearray(open(temp.relpath("deploy_1.params"), "rb").read())
    set_input("data", tvm.nd.array(data))
    load_params(loaded_params)
    run()
    out = get_output(0).asnumpy()
    tvm.testing.assert_allclose(out, verify(data, num_layers=50), atol=1e-5)

if __name__ == "__main__":
    test_legacy_compatibility()
    test_cpu()
    test_gpu()
    test_multi_models()
    test_cpu_export(".so")
    test_cpu_export(".tar")
    test_gpu_export(".so")
    test_gpu_export(".tar")
    test_rpc_export(".so")
    test_rpc_export(".tar")
    test_previous_cpu_export(".so")
    test_previous_cpu_export(".tar")
    test_previous_gpu_export(".so")
    test_previous_gpu_export(".tar")
    test_previous_rpc_export(".so")
    test_previous_rpc_export(".tar")
    test_package_params(".so")
    test_multi_models_package_params(".so")