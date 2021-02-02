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
"""
This module provides infrastructure to verify the correctness of
the command stream produced.
Currently it will invoke vela to generate a vela-optimized tflite
in which the command stream is contained as a custom operator.
This class include methods to parse the custom operator to extract
the command stream and perform an equivalency check for single operator
test cases.
"""
import tflite
import os
import io
import struct
import numpy as np
import pathlib
import shutil
import subprocess
import tempfile
import tarfile


import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.op.contrib import get_pattern_table
from tvm.contrib import utils
from tvm.relay.backend import compile_engine
from tvm.contrib import utils
from tvm.contrib import graph_runtime
from tvm.micro import export_model_library_format


def subprocess_with_stdout_and_log(cmd, cwd, logfile, stdout):
    """
    This method runs a process and logs the output to both a log file and stdout
    """
    with subprocess.Popen(
        cmd, cwd=cwd, shell=True, bufsize=0, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ) as proc, open(logfile, "a") as f:
        while True:
            data = proc.stdout.readline()
            result = proc.poll()
            # process is done if there is no data and the result is valid
            if data == b"" and result is not None:
                return int(result)
            if data:
                text = data.decode("ascii", errors="backslashreplace")
                f.write(text)
                if stdout:
                    print(text, end="")


def create_main(test_name, input_list, output_list, output_path):
    file_path = pathlib.Path(f"{output_path}/" + test_name).resolve()
    # create header file
    raw_path = file_path.with_suffix(".c").resolve()
    with open(raw_path, "w") as main_file:
        main_file.write("#include <stdio.h>\n")
        main_file.write("#include <tvm_executor.h>\n")
        main_file.write("#define WORKSPACE_SIZE (16384*1024)\n")
        main_file.write("static uint8_t g_aot_memory[WORKSPACE_SIZE];\n")

        for i in range(0, len(input_list)):
            main_file.write('#include "input_data%i.h"\n' % i)
        for i in range(0, len(output_list)):
            main_file.write('#include "expected_output_data%i.h"\n' % i)
            main_file.write('#include "output_data%i.h"\n' % i)

        main_file.write("extern tvm_model_t network;\n")
        main_file.write("extern tvm_workspace_t *tvm_runtime_workspace;\n")
        main_file.write("int main(){\n")
        main_file.write("void* inputs[%i] = { " % (len(input_list)))

        for i in range(0, len(input_list)):
            main_file.write("input_data%i, " % i)
        main_file.write("};\n")

        main_file.write("void* outputs[%i]  = { " % (len(output_list)))
        for i in range(0, len(output_list)):
            main_file.write("output_data%i, " % i)
        main_file.write("};\n")

        main_file.write("")
        main_file.write(
            "tvm_workspace_t app_workspace = {.next_alloc=g_aot_memory, .workspace=g_aot_memory, .workspace_size=WORKSPACE_SIZE};\n"
        )
        main_file.write("tvm_runtime_workspace = &app_workspace;\n")
        main_file.write("tvm_runtime_run(&network, inputs, outputs, NULL);")

        for i in range(0, len(output_list)):
            main_file.write("for (int i = 0; i<output_data%i_len; i++){\n" % i)
            main_file.write(
                'if (output_data%s[i]!=expected_output_data%s[i]){printf("ko\\n");return -1;}\n'
                % (i, i)
            )
            main_file.write("}\n")

        main_file.write('printf("ok\\n");')
        main_file.write("return 0;")
        main_file.write("}\n")


def create_header_file(tensor_name, npy_data, output_path):
    """
    This method generates a header file containing the data contained in the numpy array provided.
    It is used to capture the tensor data (for both inputs and expected outputs) to be bundled into the standalone ethosu_test_runner.
    """
    file_path = pathlib.Path(f"{output_path}/" + tensor_name).resolve()
    # create header file
    raw_path = file_path.with_suffix(".h").resolve()
    with open(raw_path, "w") as header_file:
        header_file.write("#include <stddef.h>\n")
        header_file.write("#include <stdint.h>\n")
        header_file.write("#include <dlpack/dlpack.h>\n")
        header_file.write(f"const size_t {tensor_name}_len = {npy_data.size};\n")

        if npy_data.dtype == "int8":
            header_file.write(f"int8_t {tensor_name}[] =")
        elif npy_data.dtype == "int32":
            header_file.write(f"int32_t {tensor_name}[] = ")
        elif npy_data.dtype == "uint8":
            header_file.write(f"uint8_t {tensor_name}[] = ")
        elif npy_data.dtype == "float32":
            header_file.write(f"float {tensor_name}[] = ")

        header_file.write("{")
        for i in np.ndindex(npy_data.shape):
            header_file.write(f"{npy_data[i]}, ")
        header_file.write("};\n\n")


def verify_source(mod, input_list, output_list, params=None):
    """
    This method verifies the generated source
    """
    target = "c -runtime=c --link-params --executor=aot"

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lib = tvm.relay.build(mod, target, target_host=target, params=params)

    tmp_path = utils.tempdir()
    tmp_dir = tmp_path.temp_dir

    base_path = os.path.join(tmp_dir, "test")
    build_path = os.path.join(base_path, "build")
    os.makedirs(build_path, exist_ok=True)

    tar_file = os.path.join(base_path, "test.tar")
    export_model_library_format(lib, tar_file)
    t = tarfile.open(tar_file)
    t.extractall(base_path)

    for i in range(len(input_list)):
        create_header_file((f"input_data{i}"), input_list[i], build_path)

    for i in range(len(output_list)):
        create_header_file(
            (f"output_data{i}"),
            np.zeros(output_list[i].shape, output_list[i].dtype),
            build_path,
        )
        create_header_file((f"expected_output_data{i}"), output_list[i], build_path)

    create_main("test.c", input_list, output_list, build_path)

    # Verify that compiles fine
    file_dir = os.path.dirname(os.path.abspath(__file__))
    makefile = os.path.join(file_dir, "aot_test.mk")
    make_cmd = f"make -f {makefile} build_dir=" + build_path + f" TVM_ROOT={file_dir}/../../../.."

    compile_log_path = os.path.join(build_path, "test_compile.log")
    ret = subprocess_with_stdout_and_log(make_cmd, ".", compile_log_path, False)
    assert ret == 0

    # Verify that runs fine
    run_log_path = os.path.join(build_path, "test_run.log")
    ret = subprocess_with_stdout_and_log("./aot_test_runner", build_path, run_log_path, False)
    assert ret == 0


def generate_ref_data(mod, input_data, params=None, target="llvm"):
    """Generate reference data through executing the relay module"""
    compile_engine.get().clear()
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    lib_name = "mod.so"
    temp = utils.tempdir()
    lib_path = temp.relpath(lib_name)
    lib.export_library(lib_path)
    lib = tvm.runtime.load_module(lib_path)
    grt_mod = graph_runtime.GraphModule(lib["default"](tvm.cpu()))
    grt_mod.set_input(**input_data)
    grt_mod.run()
    output_count = grt_mod.get_num_outputs()
    out = [grt_mod.get_output(i).asnumpy() for i in range(output_count)]
    return out
