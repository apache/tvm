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

import datetime
import itertools
import json
import logging
import os
import pathlib
import platform
import re
import shutil
import subprocess
import tarfile
from typing import Any, NamedTuple, Union, Optional, List, Dict

import pytest
import numpy as np

import tvm
from tvm import relay
from tvm.contrib import utils, graph_executor
from tvm.relay.backend import compile_engine
from tvm.relay.backend.utils import mangle_module_name
from tvm.micro import export_model_library_format


_LOG = logging.getLogger(__name__)

AOT_SUCCESS_TOKEN = "AOT_TEST_SUCCESS"
AOT_FAILURE_TOKEN = "AOT_TEST_FAILURE"


class AOTTestModel(NamedTuple):
    """Class to describe a model under test

    Parameters
    ----------
    module: tvm.IRModule
        IRModule to generate AOT executor for
    inputs: Dict[str, np.array]
        Dict of input names to value arrays
    outputs: List[np.array]
        Ordered list of output value arrays
    output_tolerance: Optional[Union[int, float]]
        Allowed tolerance of the output
    name: str
        Name to use for this model
    params: Optional[Dict[str, np.array]]
        Dict of parameter names to value arrays
    extra_memory_in_bytes: int
        Extra memory to allocate after planned memory
    """

    module: tvm.IRModule
    inputs: Dict[str, np.array]
    outputs: List[np.array]
    output_tolerance: Optional[Union[int, float]] = None
    name: str = "default"
    params: Optional[Dict[str, np.array]] = None
    extra_memory_in_bytes: int = 0


class AOTCompiledTestModel(NamedTuple):
    """A compiled AOTTestModel with associated module

    Parameters
    ----------
    model: AOTTestModel
        Input model to be compiled
    module: tvm.runtime.Module
        The compiled Module for the associated AOTTestModel
    """

    model: AOTTestModel
    executor_factory: tvm.relay.backend.executor_factory.AOTExecutorFactoryModule


class AOTDataLinkage(NamedTuple):
    """A compiled AOTTestModel with associated module

    Parameters
    ----------
    section: str
        Named section to place data into
    alignment: int
        Section alignment
    """

    section: str
    alignment: int


class AOTTestRunner(NamedTuple):
    """Class to describe a test runner for AOT code

    Parameters
    ----------
    makefile: str
        Premade Makefile to use from the AOT test folder
    prologue: str
        Code to prepend to the main function
    includes: List[str]
        Additional includes required to run the AOT test runner
    parameters: Dict[str, str]
        Additional parameters to pass to the make command
    pass_config: Dict[str, Any]
        Additional pass configuration when building the model
    """

    makefile: str = "default"
    prologue: str = ""
    includes: List[str] = []
    parameters: Dict[str, str] = {}
    pass_config: Dict[str, Any] = {}


AOT_DEFAULT_RUNNER = AOTTestRunner()

# AOT Test Runner using the Arm® Corstone™-300 Reference Systems
# see: https://developer.arm.com/ip-products/subsystem/corstone/corstone-300
AOT_CORSTONE300_RUNNER = AOTTestRunner(
    makefile="corstone300",
    prologue="""
    uart_init();
    """,
    includes=["uart.h"],
    parameters={"NPU_VARIANT": "256"},
)


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


def parametrize_aot_options(test):
    """Parametrize over valid option combinations"""

    skip_i386 = pytest.mark.skipif(
        platform.machine() == "i686", reason="Reference system unavailable in i386 container"
    )
    requires_arm_eabi = pytest.mark.skipif(
        shutil.which("arm-none-eabi-gcc") is None, reason="ARM embedded toolchain unavailable"
    )

    interface_api = ["packed", "c"]
    use_unpacked_api = [True, False]
    test_runner = [AOT_DEFAULT_RUNNER, AOT_CORSTONE300_RUNNER]

    all_combinations = itertools.product(interface_api, use_unpacked_api, test_runner)

    # Filter out packed operators with c interface
    valid_combinations = filter(
        lambda parameters: not (parameters[0] == "c" and not parameters[1]),
        all_combinations,
    )

    # Only use reference system for C interface and unpacked API calls
    valid_combinations = filter(
        lambda parameters: not (
            parameters[2] == AOT_CORSTONE300_RUNNER
            and (parameters[0] == "packed" or not parameters[1])
        ),
        valid_combinations,
    )

    # Skip reference system tests if running in i386 container
    marked_combinations = map(
        lambda parameters: pytest.param(*parameters, marks=[skip_i386, requires_arm_eabi])
        if parameters[2] == AOT_CORSTONE300_RUNNER
        else parameters,
        valid_combinations,
    )

    return pytest.mark.parametrize(
        ["interface_api", "use_unpacked_api", "test_runner"],
        marked_combinations,
    )(test)


def subprocess_log_output(cmd, cwd, logfile):
    """
    This method runs a process and logs the output to both a log file and stdout
    """
    _LOG.info("Execute (%s): %s", cwd, cmd)
    cmd_base = cmd[0] if isinstance(cmd, (list, tuple)) else cmd.split(" ", 1)[0]
    proc = subprocess.Popen(
        cmd, cwd=cwd, shell=True, bufsize=0, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    with open(logfile, "ab") as f:
        f.write(
            bytes(
                "\n"
                + "-" * 80
                + f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Execute ({cwd}): {cmd}\n"
                + "-" * 80,
                "utf-8",
            )
        )
        while True:
            data = proc.stdout.readline()
            _LOG.debug("%s: %s", cmd_base, str(data, "utf-8", "replace").rstrip("\n"))
            f.write(data)

            # process is done if there is no data and the result is valid
            if not data:  # EOF
                break

    return proc.wait()


# TODO: Move to linker script with list of symbols rather than coding into source
def emit_data_linkage(output_file, data_linkage):
    if data_linkage is not None:
        output_file.write(
            f'__attribute__((section("{data_linkage.section}"), aligned({data_linkage.alignment}))) '
        )


def emit_main_prologue(main_file, custom_prologue, workspace_bytes, data_linkage):
    # Add TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES because of memory alignment.
    main_file.write(
        f"#define WORKSPACE_SIZE ({workspace_bytes} + TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES)\n"
    )
    emit_data_linkage(main_file, data_linkage)
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

void TVMPlatformAbort(tvm_crt_error_t code) { exit(-1); }

void TVMLogf(const char* msg, ...) {
  va_list args;
  va_start(args, msg);
  vfprintf(stdout, msg, args);
  va_end(args);
}

TVM_DLL int TVMFuncRegisterGlobal(const char* name, TVMFunctionHandle f, int override) {}
int main(){\n
"""
    )
    main_file.write(custom_prologue)


def emit_main_data(main_file, input_map, output_list, mod_name):
    for key in input_map:
        sanitized_tensor_name = re.sub(r"\W", "_", key)
        main_file.write(
            f'#include "{mangle_name(mod_name,"input_data")}_{sanitized_tensor_name}.h"\n'
        )

    for i in range(0, len(output_list)):
        main_file.write(f'#include "{mangle_name(mod_name,"expected_output_data")}{i}.h"\n')
        main_file.write(f'#include "{mangle_name(mod_name,"output_data")}{i}.h"\n')


def emit_main_data_structs(main_file, input_map, output_list, mod_name):
    main_file.write(
        f"struct {mangle_name(mod_name, 'inputs')} {mangle_name(mod_name, 'inputs')} = {{"
    )
    for key in input_map:
        sanitized_tensor_name = re.sub(r"\W", "_", key)
        main_file.write(
            f"\t.{sanitized_tensor_name} = {mangle_name(mod_name, 'input_data')}_{sanitized_tensor_name},\n"
        )
    main_file.write("};\n")

    main_file.write(
        f"struct {mangle_name(mod_name, 'outputs')} {mangle_name(mod_name, 'outputs')} = {{"
    )
    num_outputs = len(output_list)
    if num_outputs == 1:
        main_file.write(f"\t.output = {mangle_name(mod_name, 'output_data')}0,\n")
    else:
        for i in range(0, num_outputs):
            main_file.write(f"\t.output{i} = {mangle_name(mod_name, 'output_data')}{i},\n")
    main_file.write("};\n")


def emit_main_data_setup(main_file, input_map, output_list, mod_name):
    num_outputs = len(output_list)
    num_inputs = len(input_map)

    main_file.write(f'void* {mangle_name(mod_name,"inputs")}[{num_inputs}] = {{ ')
    for key in input_map:
        sanitized_tensor_name = re.sub(r"\W", "_", key)
        main_file.write(f'{mangle_name(mod_name,"input_data")}_{sanitized_tensor_name}, ')
    main_file.write("};\n")

    main_file.write(f'void* {mangle_name(mod_name,"outputs")}[{num_outputs}]  = {{ ')
    for i in range(0, num_outputs):
        main_file.write(f'{mangle_name(mod_name,"output_data")}{i}, ')
    main_file.write("};\n")


def emit_main_c_interface_call(main_file, mod_name):
    main_file.write(
        f'{mangle_name(mod_name,"run")}(&{mangle_name(mod_name,"inputs")}, &{mangle_name(mod_name,"outputs")});\n'
    )


def emit_main_fake_packed_values(main_file):
    main_file.write(
        """
    static DLDevice fake_device = {kDLCPU, 0};
    static int64_t fake_dims = 0;
    static int64_t fake_shape = {0};
    """
    )


def emit_main_packed_call(main_file, input_map, output_list, mod_name):
    tensors_name = mangle_name(mod_name, "tensors")
    values_name = mangle_name(mod_name, "values")
    typeids_name = mangle_name(mod_name, "typeids")

    def fake_tensor(source, source_index, packed_index):
        main_file.write(
            f"""
        {tensors_name}[{packed_index}].device = fake_device;
        {tensors_name}[{packed_index}].data = {source}[{source_index}];
        {tensors_name}[{packed_index}].shape = &fake_shape;
        {tensors_name}[{packed_index}].ndim = fake_dims;
        {tensors_name}[{packed_index}].byte_offset = 0;
        {tensors_name}[{packed_index}].strides = NULL;
        {values_name}[{packed_index}].v_handle = &{tensors_name}[{packed_index}];
        """
        )

    num_outputs = len(output_list)
    num_inputs = len(input_map)
    num_tensors = num_inputs + num_outputs
    main_file.write(
        f"""
    DLTensor {tensors_name}[{num_tensors}];
    TVMValue {values_name}[{num_tensors}];
    int32_t {typeids_name}[{num_tensors}];
    """
    )

    for i in range(0, num_inputs):
        fake_tensor(mangle_name(mod_name, "inputs"), i, i)
    for i in range(0, num_outputs):
        fake_tensor(mangle_name(mod_name, "outputs"), i, i + num_inputs)

    main_file.write(
        f'{mangle_name(mod_name, "run")}({values_name}, {typeids_name}, 0, NULL, 0, NULL);\n'
    )
    main_file.write("\n")


def emit_main_compare(main_file, output_list, output_tolerance, mod_name):
    num_outputs = len(output_list)
    actual_data_name = mangle_name(mod_name, "output_data")
    expected_data_name = mangle_name(mod_name, "expected_output_data")

    for i in range(0, num_outputs):
        is_float_dtype = output_list[i].dtype == "float32"

        comparison_function = "abs"
        tolerance = output_tolerance or 0
        if is_float_dtype:
            comparison_function = "fabs"
            tolerance = output_tolerance or 0.001

        main_file.write(
            f"""
            for (int i = 0; i<{actual_data_name}{i}_len; i++) {{
                if ({comparison_function}({actual_data_name}{i}[i]-{expected_data_name}{i}[i]) > {tolerance}) {{
                    printf("{AOT_FAILURE_TOKEN}\\n");
                    return -1;
                }}
            }}
            """
        )


def emit_main_init_memory_manager(main_file):
    main_file.write("StackMemoryManager_Init(&app_workspace, g_aot_memory, WORKSPACE_SIZE);")
    main_file.write("\n")


def emit_main_epilogue(main_file):
    main_file.write(f'printf("{AOT_SUCCESS_TOKEN}\\n");')
    main_file.write("return 0;")
    main_file.write("}\n")


def emit_main_common_includes(main_file, custom_includes):
    main_file.write("#include <stdio.h>\n")
    main_file.write("#include <stdarg.h>\n")
    main_file.write("#include <stdlib.h>\n")
    main_file.write("#include <math.h>\n")
    main_file.write('#include "tvm/runtime/c_runtime_api.h"\n')
    main_file.write('#include "tvm/runtime/crt/stack_allocator.h"\n')
    for include in custom_includes:
        main_file.write(f'#include "{include}"\n')


def emit_main_micro_include(main_file, mod_name):
    main_file.write(f"#include <{mangle_module_name(mod_name)}.h>\n")


def create_main(
    test_name,
    models,
    output_path,
    custom_includes,
    custom_prologue,
    data_linkage,
    interface_api,
    workspace_bytes,
):
    file_path = pathlib.Path(f"{output_path}/" + test_name).resolve()
    # create header file
    raw_path = file_path.with_suffix(".c").resolve()
    with open(raw_path, "w") as main_file:
        emit_main_common_includes(main_file, custom_includes)

        if interface_api == "c":
            for model in models:
                emit_main_micro_include(main_file, model.name)
        for model in models:
            emit_main_data(main_file, model.inputs, model.outputs, model.name)

        emit_main_prologue(main_file, custom_prologue, workspace_bytes, data_linkage)
        emit_main_init_memory_manager(main_file)

        if interface_api == "c":
            for model in models:
                emit_main_data_structs(main_file, model.inputs, model.outputs, model.name)
                emit_main_c_interface_call(main_file, model.name)
        else:
            emit_main_fake_packed_values(main_file)
            for model in models:
                emit_main_data_setup(main_file, model.inputs, model.outputs, model.name)
                emit_main_packed_call(main_file, model.inputs, model.outputs, model.name)

        for model in models:
            emit_main_compare(main_file, model.outputs, model.output_tolerance, model.name)
        emit_main_epilogue(main_file)


def create_header_file(tensor_name, npy_data, output_path, data_linkage):
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

        emit_data_linkage(header_file, data_linkage)

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


def extract_main_workspace_size_bytes(extract_dir):
    with open(os.path.join(extract_dir, "metadata.json")) as json_f:
        metadata = json.load(json_f)
        return metadata["memory"]["functions"]["main"][0]["workspace_size_bytes"]


def compile_models(
    models: Union[List[AOTTestModel], AOTTestModel],
    interface_api: str,
    use_unpacked_api: bool,
    workspace_byte_alignment: int = 8,
    enable_op_fusion: bool = True,
    pass_config: Dict[str, Any] = None,
    target_opts: Dict = None,
) -> List[AOTCompiledTestModel]:
    """
    This method generates runtime.Modules for the tests
    """
    if not isinstance(models, list):
        models = [models]

    base_target = "c -runtime=c --link-params --executor=aot"
    extra_target = f"--workspace-byte-alignment={workspace_byte_alignment} --interface-api={interface_api} --unpacked-api={int(use_unpacked_api)}"
    for key, val in target_opts.items():
        extra_target += f" {key}={val}"
    target = f"{base_target} {extra_target}"

    config = {"tir.disable_vectorize": True}
    if pass_config:
        config = {**config, **pass_config}
    if not enable_op_fusion:
        config["relay.FuseOps.max_depth"] = 1

    compiled_mods = list()
    for model in models:
        with tvm.transform.PassContext(opt_level=3, config=config):
            executor_factory = tvm.relay.build(
                model.module,
                target,
                target_host=target,
                params=model.params,
                mod_name=model.name,
            )
            compiled_mods.append(
                AOTCompiledTestModel(model=model, executor_factory=executor_factory)
            )
    return compiled_mods


def run_and_check(
    models: List[AOTCompiledTestModel],
    runner: AOTTestRunner,
    interface_api: str,
    debug_calculated_workspaces=False,
    workspace_byte_alignment=8,
    data_linkage: AOTDataLinkage = None,
):
    """
    This method uses the original test data and compiled runtime.Modules
    to run in the test runner to verify the results.
    """

    tmp_path = utils.tempdir()
    tmp_dir = tmp_path.temp_dir

    cflags = f"-DTVM_RUNTIME_ALLOC_ALIGNMENT_BYTES={workspace_byte_alignment} "
    # The calculated workspaces will not account for stack allocator tags used for debugging
    if debug_calculated_workspaces:
        cflags += "-DTVM_CRT_STACK_ALLOCATOR_ENABLE_LIFO_CHECK "

    base_path = os.path.join(tmp_dir, "test")
    build_path = os.path.join(base_path, "build")
    os.makedirs(build_path, exist_ok=True)

    include_path = os.path.join(base_path, "include")
    os.mkdir(include_path)
    crt_root = tvm.micro.get_standalone_crt_dir()
    shutil.copy2(
        os.path.join(crt_root, "template", "crt_config-template.h"),
        os.path.join(include_path, "crt_config.h"),
    )

    workspace_bytes = 0
    for compiled_model in models:
        model = compiled_model.model
        tar_file = os.path.join(base_path, f"{model.name}.tar")
        export_model_library_format(compiled_model.executor_factory, tar_file)
        t = tarfile.open(tar_file)
        t.extractall(base_path)

        workspace_bytes += model.extra_memory_in_bytes
        workspace_bytes += extract_main_workspace_size_bytes(base_path)

        for key in model.inputs:
            sanitized_tensor_name = re.sub(r"\W", "_", key)
            create_header_file(
                f'{mangle_name(model.name, "input_data")}_{sanitized_tensor_name}',
                model.inputs[key],
                include_path,
                data_linkage,
            )

        for i in range(len(model.outputs)):
            create_header_file(
                (f'{mangle_name(model.name,"output_data")}{i}'),
                np.zeros(model.outputs[i].shape, model.outputs[i].dtype),
                include_path,
                data_linkage,
            )
            create_header_file(
                (f'{mangle_name(model.name, "expected_output_data")}{i}'),
                model.outputs[i],
                include_path,
                data_linkage,
            )

    create_main(
        "test.c",
        [compiled_model.model for compiled_model in models],
        build_path,
        runner.includes,
        runner.prologue,
        data_linkage,
        interface_api,
        workspace_bytes,
    )

    # Verify that compiles fine
    file_dir = os.path.dirname(os.path.abspath(__file__))
    codegen_path = os.path.join(base_path, "codegen")
    makefile = os.path.join(file_dir, f"{runner.makefile}.mk")
    custom_params = " ".join([f" {param}='{value}'" for param, value in runner.parameters.items()])
    make_command = (
        f"make -f {makefile} build_dir={build_path}"
        + f" CFLAGS='{cflags}'"
        + f" TVM_ROOT={file_dir}/../../../.."
        + f" AOT_TEST_ROOT={file_dir}"
        + f" CODEGEN_ROOT={codegen_path}"
        + f" STANDALONE_CRT_DIR={tvm.micro.get_standalone_crt_dir()}"
        + custom_params
    )

    compile_log_path = os.path.join(build_path, "test_compile.log")
    compile_command = f"{make_command} aot_test_runner"
    ret = subprocess_log_output(compile_command, ".", compile_log_path)
    assert ret == 0

    # Verify that runs fine
    run_log_path = os.path.join(build_path, "test_run.log")
    run_command = f"{make_command} run"
    ret = subprocess_log_output(run_command, build_path, run_log_path)
    assert ret == 0

    with open(run_log_path) as run_log:
        assert AOT_SUCCESS_TOKEN in run_log.read()


def compile_and_run(
    models: Union[List[AOTTestModel], AOTTestModel],
    runner: AOTTestRunner,
    interface_api: str,
    use_unpacked_api: bool,
    debug_calculated_workspaces: bool = False,
    workspace_byte_alignment: int = 8,
    enable_op_fusion: bool = True,
    data_linkage: AOTDataLinkage = None,
    target_opts: Dict = None,
):
    """This is a wrapper API to compile and run models as test for AoT"""
    compiled_test_mods = compile_models(
        models=models,
        interface_api=interface_api,
        use_unpacked_api=use_unpacked_api,
        workspace_byte_alignment=workspace_byte_alignment,
        enable_op_fusion=enable_op_fusion,
        pass_config=runner.pass_config,
        target_opts=target_opts,
    )
    run_and_check(
        models=compiled_test_mods,
        runner=runner,
        interface_api=interface_api,
        debug_calculated_workspaces=debug_calculated_workspaces,
        workspace_byte_alignment=workspace_byte_alignment,
        data_linkage=data_linkage,
    )


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
