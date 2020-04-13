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
from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te

from tvm.contrib import util
header_file_dir_path = util.tempdir()


def gen_engine_header():
    code = r'''
        #ifndef _ENGINE_H_
        #define _ENGINE_H_
        #include <cstdint>
        #include <string>
        #include <sstream>
        #include <vector>
        class Engine {
        };

        #endif
        '''
    header_file = header_file_dir_path.relpath("gcc_engine.h")
    with open(header_file, 'w') as f:
        f.write(code)


def generate_engine_module():
    code = r'''
        #include <tvm/runtime/c_runtime_api.h>
        #include <dlpack/dlpack.h>
        #include "gcc_engine.h"

        extern "C" void gcc_1_(float* gcc_input4, float* gcc_input5,
                float* gcc_input6, float* gcc_input7, float* out) {
            Engine engine;
        }
        '''
    import tvm.runtime._ffi_api
    gen_engine_header()
    csource_module = tvm.runtime._ffi_api.CSourceModuleCreate(code, "cc")
    return csource_module


def test_mod_export():
    def verify_gpu_mod_export(obj_format):
        for device in ["llvm", "cuda"]:
            if not tvm.runtime.enabled(device):
                print("skip because %s is not enabled..." % device)
                return

        resnet18_mod, resnet18_params = relay.testing.resnet.get_workload(num_layers=18)
        resnet50_mod, resnet50_params = relay.testing.resnet.get_workload(num_layers=50)
        with relay.build_config(opt_level=3):
            _, resnet18_gpu_lib, _ = relay.build_module.build(resnet18_mod, "cuda", params=resnet18_params)
            _, resnet50_cpu_lib, _ = relay.build_module.build(resnet50_mod, "llvm", params=resnet50_params)

        from tvm.contrib import util
        temp = util.tempdir()
        if obj_format == ".so":
            file_name = "deploy_lib.so"
        else:
            assert obj_format == ".tar"
            file_name = "deploy_lib.tar"
        path_lib = temp.relpath(file_name)
        resnet18_gpu_lib.imported_modules[0].import_module(resnet50_cpu_lib)
        resnet18_gpu_lib.export_library(path_lib)
        loaded_lib = tvm.runtime.load_module(path_lib)
        assert loaded_lib.type_key == "library"
        assert loaded_lib.imported_modules[0].type_key == "cuda"
        assert loaded_lib.imported_modules[0].imported_modules[0].type_key == "library"

    def verify_multi_dso_mod_export(obj_format):
        for device in ["llvm"]:
            if not tvm.runtime.enabled(device):
                print("skip because %s is not enabled..." % device)
                return

        resnet18_mod, resnet18_params = relay.testing.resnet.get_workload(num_layers=18)
        with relay.build_config(opt_level=3):
            _, resnet18_cpu_lib, _ = relay.build_module.build(resnet18_mod, "llvm", params=resnet18_params)

        A = te.placeholder((1024,), name='A')
        B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
        s = te.create_schedule(B.op)
        f = tvm.build(s, [A, B], "llvm", name="myadd")
        from tvm.contrib import util
        temp = util.tempdir()
        if obj_format == ".so":
            file_name = "deploy_lib.so"
        else:
            assert obj_format == ".tar"
            file_name = "deploy_lib.tar"
        path_lib = temp.relpath(file_name)
        resnet18_cpu_lib.import_module(f)
        resnet18_cpu_lib.export_library(path_lib)
        loaded_lib = tvm.runtime.load_module(path_lib)
        assert loaded_lib.type_key == "library"
        assert loaded_lib.imported_modules[0].type_key == "library"

    def verify_json_import_dso(obj_format):
        for device in ["llvm"]:
            if not tvm.runtime.enabled(device):
                print("skip because %s is not enabled..." % device)
                return

        # Get subgraph Json.
        subgraph_json = ("json_rt_0\n" +
                         "input 0 10 10\n" +
                         "input 1 10 10\n" +
                         "input 2 10 10\n" +
                         "input 3 10 10\n" +
                         "add 4 inputs: 0 1 shape: 10 10\n" +
                         "sub 5 inputs: 4 2 shape: 10 10\n" +
                         "mul 6 inputs: 5 3 shape: 10 10\n" +
                         "json_rt_1\n" +
                         "input 0 10 10\n" +
                         "input 1 10 10\n" +
                         "input 2 10 10\n" +
                         "input 3 10 10\n" +
                         "add 4 inputs: 0 1 shape: 10 10\n" +
                         "sub 5 inputs: 4 2 shape: 10 10\n" +
                         "mul 6 inputs: 5 3 shape: 10 10")

        from tvm.contrib import util
        temp = util.tempdir()
        subgraph_path = temp.relpath('subgraph.examplejson')
        with open(subgraph_path, 'w') as f:
            f.write(subgraph_json)

        # Get Json and module.
        A = te.placeholder((1024,), name='A')
        B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
        s = te.create_schedule(B.op)
        f = tvm.build(s, [A, B], "llvm", name="myadd")
        try:
            ext_lib = tvm.runtime.load_module(subgraph_path, "examplejson")
        except:
            print("skip because Loader of examplejson is not presented")
            return
        ext_lib.import_module(f)
        if obj_format == ".so":
            file_name = "deploy_lib.so"
        else:
            assert obj_format == ".tar"
            file_name = "deploy_lib.tar"
        path_lib = temp.relpath(file_name)
        ext_lib.export_library(path_lib)
        lib = tvm.runtime.load_module(path_lib)
        assert lib.type_key == "examplejson"
        assert lib.imported_modules[0].type_key == "library"

    def verify_multi_c_mod_export():
        from shutil import which
        if which("gcc") is None:
            print("Skip test because gcc is not available.")

        for device in ["llvm"]:
            if not tvm.runtime.enabled(device):
                print("skip because %s is not enabled..." % device)
                return

        resnet18_mod, resnet18_params = relay.testing.resnet.get_workload(num_layers=18)
        with relay.build_config(opt_level=3):
            _, resnet18_cpu_lib, _ = relay.build_module.build(resnet18_mod, "llvm", params=resnet18_params)

        A = te.placeholder((1024,), name='A')
        B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
        s = te.create_schedule(B.op)
        f = tvm.build(s, [A, B], "c", name="myadd")
        engine_module = generate_engine_module()
        from tvm.contrib import util
        temp = util.tempdir()
        file_name = "deploy_lib.so"
        path_lib = temp.relpath(file_name)
        resnet18_cpu_lib.import_module(f)
        resnet18_cpu_lib.import_module(engine_module)
        kwargs = {"options": ["-O2", "-std=c++14", "-I" + header_file_dir_path.relpath("")]}
        resnet18_cpu_lib.export_library(path_lib, fcompile=False, **kwargs)
        loaded_lib = tvm.runtime.load_module(path_lib)
        assert loaded_lib.type_key == "library"
        assert loaded_lib.imported_modules[0].type_key == "library"
        assert loaded_lib.imported_modules[1].type_key == "library"

    for obj_format in [".so", ".tar"]:
        verify_gpu_mod_export(obj_format)
        verify_multi_dso_mod_export(obj_format)
        verify_json_import_dso(obj_format)

    verify_multi_c_mod_export()


if __name__ == "__main__":
    test_mod_export()
