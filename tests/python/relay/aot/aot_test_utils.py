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

import os
import io
import struct
import numpy as np
import pathlib
import shutil
import subprocess
import tempfile
import tarfile
import json


import tvm
from tvm import relay
from tvm.relay import transform
from tvm.contrib import utils, graph_executor
from tvm.relay.backend import compile_engine
from tvm.relay.backend.utils import mangle_module_name
from tvm.contrib import utils
from tvm.micro import export_model_library_format


def mangle_name(mod_name, name):
    mod_name = mangle_module_name(mod_name)
    return mod_name + "_" + name


def convert_to_relay(
    tflite_model_buf,
    input_data,
    input_node,
):
    """Convert a tflite model buffer in a Relay module"""

    def convert_to_list(x):
        if not isinstance(x, list):
            x = [x]
        return x

    # TFLite.Model.Model has changed to TFLite.Model from 1.14 to 2.1
    try:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except ImportError:
        raise ImportError("The tflite package must be installed")

    input_data = convert_to_list(input_data)
    input_node = convert_to_list(input_node)

    shape_dict = {}
    dtype_dict = {}
    for i, e in enumerate(input_node):
        shape_dict[e] = input_data[i].shape
        dtype_dict[e] = input_data[i].dtype.name

    mod, params = relay.frontend.from_tflite(
        tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict
    )
    mod["main"] = relay.build_module.bind_params_by_name(mod["main"], params)
    return mod, params


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


def emit_main_network_definition(main_file, mod_name):
    main_file.write(f'extern tvm_model_t {mangle_name(mod_name,"network")};\n')


def emit_main_prologue(main_file, workspace_bytes):
    main_file.write(f"#define WORKSPACE_SIZE ({workspace_bytes})\n")
    main_file.write("static uint8_t g_aot_memory[WORKSPACE_SIZE];\n")
    main_file.write("tvm_workspace_t app_workspace;\n")
    main_file.write(
        """
tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
    return StackMemoryManager_Allocate(&app_workspace, num_bytes, out_ptr);
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
    return StackMemoryManager_Free(&app_workspace,ptr);
}

void  TVMPlatformAbort(tvm_crt_error_t code) { }

void TVMLogf(const char* msg, ...) { }

TVM_DLL int TVMFuncRegisterGlobal(const char* name, TVMFunctionHandle f, int override) {}
int main(){\n
     
        """
    )


def emit_main_data(main_file, input_list, output_list, mod_name):
    for i in range(0, len(input_list)):
        main_file.write(f'#include "{mangle_name(mod_name,"input_data")}{i}.h"\n')

    for i in range(0, len(output_list)):
        main_file.write(f'#include "{mangle_name(mod_name,"expected_output_data")}{i}.h"\n')
        main_file.write(f'#include "{mangle_name(mod_name,"output_data")}{i}.h"\n')


def emit_main_run(main_file, input_list, output_list, mod_name):
    num_outputs = len(output_list)
    num_inputs = len(input_list)

    main_file.write(f'void* {mangle_name(mod_name,"inputs")}[{num_inputs}] = {{ ')

    for i in range(0, len(input_list)):
        main_file.write(f'{mangle_name(mod_name,"input_data")}{i}, ')
    main_file.write("};\n")

    main_file.write(f'void* {mangle_name(mod_name,"outputs")}[{num_outputs}]  = {{ ')
    for i in range(0, len(output_list)):
        main_file.write(f'{mangle_name(mod_name,"output_data")}{i}, ')
    main_file.write("};\n")
    main_file.write(
        f'tvm_runtime_run(&{mangle_name(mod_name,"network")}, {mangle_name(mod_name,"inputs")}, {mangle_name(mod_name,"outputs")});'
    )


def emit_main_compare(main_file, output_list, mod_name):
    for i in range(0, len(output_list)):
        is_float_dtype = output_list[i].dtype == "float32"
        main_file.write(f'for (int i = 0; i<{mangle_name(mod_name,"output_data")}{i}_len; i++){{\n')
        if is_float_dtype:
            main_file.write(
                f'if (fabs({mangle_name(mod_name,"output_data")}{i}[i]-{mangle_name(mod_name,"expected_output_data")}{i}[i]) > 0.001f){{printf("ko\\n");return -1;}}\n'
            )
        else:
            main_file.write(
                f'if ({mangle_name(mod_name,"output_data")}{i}[i]!={mangle_name(mod_name, "expected_output_data")}{i}[i]){{printf("ko\\n");return -1;}}\n'
            )
        main_file.write("}\n")


def emit_main_init_memory_manager(main_file):
    main_file.write("StackMemoryManager_Init(&app_workspace, g_aot_memory, WORKSPACE_SIZE);")


def emit_main_epilogue(main_file):
    main_file.write('printf("ok\\n");')
    main_file.write("return 0;")
    main_file.write("}\n")


def emit_main_common_includes(main_file):
    main_file.write("#include <stdio.h>\n")
    main_file.write("#include <math.h>\n")
    main_file.write('#include "tvm/runtime/crt/internal/aot_executor/aot_executor.h"\n')
    main_file.write('#include "tvm/runtime/crt/stack_allocator.h"\n')


def create_main(test_name, input_list_map, output_list_map, output_path, workspace_bytes):
    file_path = pathlib.Path(f"{output_path}/" + test_name).resolve()
    # create header file
    raw_path = file_path.with_suffix(".c").resolve()
    with open(raw_path, "w") as main_file:
        emit_main_common_includes(main_file)

        for k in input_list_map:
            emit_main_network_definition(main_file, k)

        emit_main_prologue(main_file, workspace_bytes)

        for k in input_list_map:
            emit_main_data(main_file, input_list_map[k], output_list_map[k], k)

        emit_main_init_memory_manager(main_file)

        for k in input_list_map:
            emit_main_run(main_file, input_list_map[k], output_list_map[k], k)

        for k in input_list_map:
            emit_main_compare(main_file, output_list_map[k], k)

        emit_main_epilogue(main_file)


def create_header_file(tensor_name, npy_data, output_path):
    """
    This method generates a header file containing the data contained in the numpy array provided.
    It is used to capture the tensor data (for both inputs and expected outputs) to be bundled into the standalone application.
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


def extract_main_workspace_sizebytes(extract_dir):
    with open(os.path.join(extract_dir, "metadata.json")) as json_f:
        metadata = json.load(json_f)
        return metadata["memory"]["functions"]["main"][0]["workspace_size_bytes"]


def compile_and_run(
    mod,
    input_list,
    output_list,
    target_options,
    use_calculated_workspaces,
    params=None,
    workspace_byte_alignment=8,
    mod_name=None,
    enable_op_fusion=True,
):
    """
    This method verifies the generated source
    """
    target = f"c -runtime=c --link-params --executor=aot --workspace-byte-alignment={workspace_byte_alignment} {target_options}"
    cflags = f"-DTVM_RUNTIME_ALLOC_ALIGNMENT_BYTES={workspace_byte_alignment} "

    # The calculated workspaces will not account for stack allocator tags used for debugging
    if not use_calculated_workspaces:
        cflags += "-DTVM_CRT_STACK_ALLOCATOR_ENABLE_LIFO_CHECK "

    config = {"tir.disable_vectorize": True}
    if not enable_op_fusion:
        config["relay.FuseOps.max_depth"] = 1

    with tvm.transform.PassContext(opt_level=3, config=config):
        lib = tvm.relay.build(mod, target, target_host=target, params=params, mod_name=mod_name)

    tmp_path = utils.tempdir()
    tmp_dir = tmp_path.temp_dir

    base_path = os.path.join(tmp_dir, "test")
    build_path = os.path.join(base_path, "build")
    os.makedirs(build_path, exist_ok=True)

    tar_file = os.path.join(base_path, "test.tar")
    export_model_library_format(lib, tar_file)
    t = tarfile.open(tar_file)
    t.extractall(base_path)
    if use_calculated_workspaces:
        workspace_bytes = extract_main_workspace_sizebytes(base_path)
    else:
        workspace_bytes = 16384 * 1024

    for i in range(len(input_list)):
        create_header_file((f'{mangle_name(mod_name, "input_data")}{i}'), input_list[i], build_path)

    for i in range(len(output_list)):
        create_header_file(
            (f'{mangle_name(mod_name,"output_data")}{i}'),
            np.zeros(output_list[i].shape, output_list[i].dtype),
            build_path,
        )
        create_header_file(
            (f'{mangle_name(mod_name, "expected_output_data")}{i}'), output_list[i], build_path
        )

    create_main(
        "test.c", {mod_name: input_list}, {mod_name: output_list}, build_path, workspace_bytes
    )

    # Verify that compiles fine
    file_dir = os.path.dirname(os.path.abspath(__file__))
    makefile = os.path.join(file_dir, "aot_test.mk")
    make_cmd = (
        f"make CFLAGS='{cflags}' -f {makefile} build_dir="
        + build_path
        + f" TVM_ROOT={file_dir}/../../../.."
    )

    compile_log_path = os.path.join(build_path, "test_compile.log")
    ret = subprocess_with_stdout_and_log(make_cmd, ".", compile_log_path, False)
    assert ret == 0

    # Verify that runs fine
    run_log_path = os.path.join(build_path, "test_run.log")
    ret = subprocess_with_stdout_and_log("./aot_test_runner", build_path, run_log_path, False)
    assert ret == 0


def compile_and_run_multiple_models(
    mod_map, input_list_map, output_list_map, target_options, param_map
):
    """
    This method verifies the generated source
    """
    target = f"c -runtime=c --link-params --executor=aot {target_options}"
    tmp_path = utils.tempdir()
    tmp_dir = tmp_path.temp_dir

    base_path = os.path.join(tmp_dir, "test")
    build_path = os.path.join(base_path, "build")
    os.makedirs(build_path, exist_ok=True)
    for mod_name, mod in mod_map.items():

        with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
            lib = tvm.relay.build(
                mod, target, target_host=target, params=param_map[mod_name], mod_name=mod_name
            )

        tar_file = os.path.join(base_path, "test.tar")
        export_model_library_format(lib, tar_file)
        t = tarfile.open(tar_file)
        t.extractall(base_path)

        input_list = input_list_map[mod_name]
        output_list = output_list_map[mod_name]

        for i in range(len(input_list_map[mod_name])):
            create_header_file(
                (f'{mangle_name(mod_name,"input_data")}{i}'), input_list[i], build_path
            )

        for i in range(len(output_list_map[mod_name])):
            create_header_file(
                (f'{mangle_name(mod_name,"output_data")}{i}'),
                np.zeros(output_list[i].shape, output_list[i].dtype),
                build_path,
            )
            create_header_file(
                (f'{mangle_name(mod_name,"expected_output_data")}{i}'), output_list[i], build_path
            )

    create_main("test.c", input_list_map, output_list_map, build_path, workspace_bytes=16384 * 1024)

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
    grt_mod = graph_executor.GraphModule(lib["default"](tvm.cpu()))
    grt_mod.set_input(**input_data)
    grt_mod.run()
    output_count = grt_mod.get_num_outputs()
    out = [grt_mod.get_output(i).numpy() for i in range(output_count)]
    return out
