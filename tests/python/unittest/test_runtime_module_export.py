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

import shutil
import sys

import pytest

import tvm
import tvm.testing
from tvm import relay, te
from tvm.contrib import utils
from tvm.relay import testing

header_file_dir_path = utils.tempdir()


def gen_engine_header():
    code = r"""
        #ifndef _ENGINE_H_
        #define _ENGINE_H_
        #include <cstdint>
        #include <string>
        #include <sstream>
        #include <vector>
        class Engine {
        };

        #endif
        """
    header_file = header_file_dir_path.relpath("gcc_engine.h")
    with open(header_file, "w") as f:
        f.write(code)


def generate_engine_module():
    code = r"""
        #include <tvm/runtime/c_runtime_api.h>
        #include <dlpack/dlpack.h>
        #include "gcc_engine.h"

        extern "C" void gcc_1_(float* gcc_input4, float* gcc_input5,
                float* gcc_input6, float* gcc_input7, float* out) {
            Engine engine;
        }
        """
    import tvm.runtime._ffi_api

    gen_engine_header()
    csource_module = tvm.runtime._ffi_api.CSourceModuleCreate(code, "cc", [], None)
    return csource_module


class TestModExport:
    obj_format = tvm.testing.parameter(".so", ".tar")

    # The Cuda and LLVM libraries contain identically named functions,
    # and cannot be linked into the same .so file.  This function
    # currently doesn't run on CI, as no environments have both LLVM
    # and CUDA enabled.
    @pytest.mark.xfail(reason="Known failing case")
    @tvm.testing.requires_llvm
    @tvm.testing.requires_cuda
    def test_gpu_mod_export(self, obj_format):
        synthetic_mod, synthetic_params = relay.testing.synthetic.get_workload()
        synthetic_llvm_mod, synthetic_llvm_params = relay.testing.synthetic.get_workload()
        with tvm.transform.PassContext(opt_level=3):
            _, synthetic_gpu_lib, _ = relay.build_module.build(
                synthetic_mod, "cuda", params=synthetic_params
            )
            _, synthetic_llvm_cpu_lib, _ = relay.build_module.build(
                synthetic_llvm_mod, "llvm", params=synthetic_llvm_params
            )

        temp = utils.tempdir()

        assert obj_format in [".so", ".tar"]
        file_name = f"deploy_lib{obj_format}"

        path_lib = temp.relpath(file_name)
        synthetic_gpu_lib.import_module(synthetic_llvm_cpu_lib)
        synthetic_gpu_lib.export_library(path_lib)
        loaded_lib = tvm.runtime.load_module(path_lib)
        assert loaded_lib.type_key == "library"
        assert loaded_lib.imported_modules[0].type_key == "cuda"
        #  dso modules are merged together
        assert len(loaded_lib.imported_modules) == 1

    @tvm.testing.requires_llvm
    def test_multi_dso_mod_export(self, obj_format):
        A = te.placeholder((1024,), name="A")
        B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
        s = te.create_schedule(B.op)
        mod0 = tvm.build(s, [A, B], "llvm", name="myadd0")
        mod1 = tvm.build(s, [A, B], "llvm", name="myadd1")

        temp = utils.tempdir()
        assert obj_format in [".so", ".tar"]
        file_name = f"deploy_lib{obj_format}"

        path_lib = temp.relpath(file_name)
        mod0.import_module(mod1)
        mod0.export_library(path_lib)
        loaded_lib = tvm.runtime.load_module(path_lib)
        assert loaded_lib.type_key == "library"
        # dso modules are merged
        assert len(loaded_lib.imported_modules) == 0

    @tvm.testing.requires_llvm
    def test_json_import_dso(self, obj_format):
        # Get subgraph Json.
        subgraph_json = (
            "json_rt_0\n"
            + "input 0 10 10\n"
            + "input 1 10 10\n"
            + "input 2 10 10\n"
            + "input 3 10 10\n"
            + "add 4 inputs: 0 1 shape: 10 10\n"
            + "sub 5 inputs: 4 2 shape: 10 10\n"
            + "mul 6 inputs: 5 3 shape: 10 10\n"
            + "json_rt_1\n"
            + "input 0 10 10\n"
            + "input 1 10 10\n"
            + "input 2 10 10\n"
            + "input 3 10 10\n"
            + "add 4 inputs: 0 1 shape: 10 10\n"
            + "sub 5 inputs: 4 2 shape: 10 10\n"
            + "mul 6 inputs: 5 3 shape: 10 10"
        )

        temp = utils.tempdir()
        subgraph_path = temp.relpath("subgraph.examplejson")
        with open(subgraph_path, "w") as f:
            f.write(subgraph_json)

        # Get Json and module.
        A = te.placeholder((1024,), name="A")
        B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
        s = te.create_schedule(B.op)
        f = tvm.build(s, [A, B], "llvm", name="myadd")

        try:
            ext_lib = tvm.runtime.load_module(subgraph_path, "examplejson")
        except tvm.TVMError:
            pytest.skip("Loader for examplejson is not present")

        ext_lib = tvm.runtime.load_module(subgraph_path, "examplejson")

        ext_lib.import_module(f)

        assert obj_format in [".so", ".tar"]
        file_name = f"deploy_lib{obj_format}"

        path_lib = temp.relpath(file_name)
        ext_lib.export_library(path_lib)
        lib = tvm.runtime.load_module(path_lib)
        assert lib.type_key == "examplejson"
        assert lib.imported_modules[0].type_key == "library"

    @tvm.testing.requires_llvm
    @pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc is unavailable")
    def test_multi_c_mod_export(self):

        synthetic_mod, synthetic_params = relay.testing.synthetic.get_workload()
        with tvm.transform.PassContext(opt_level=3):
            _, synthetic_cpu_lib, _ = relay.build_module.build(
                synthetic_mod, "llvm", params=synthetic_params
            )

        A = te.placeholder((1024,), name="A")
        B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
        s = te.create_schedule(B.op)
        f = tvm.build(s, [A, B], "c", name="myadd")
        engine_module = generate_engine_module()

        temp = utils.tempdir()
        file_name = "deploy_lib.so"
        path_lib = temp.relpath(file_name)
        synthetic_cpu_lib.import_module(f)
        synthetic_cpu_lib.import_module(engine_module)
        kwargs = {"options": ["-O2", "-std=c++14", "-I" + header_file_dir_path.relpath("")]}
        synthetic_cpu_lib.export_library(path_lib, fcompile=False, **kwargs)
        loaded_lib = tvm.runtime.load_module(path_lib)
        assert loaded_lib.type_key == "library"
        # dso modules are merged
        assert len(loaded_lib.imported_modules) == 0


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
