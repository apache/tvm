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
# pylint: disable=use-list-literal, consider-using-with, f-string-without-interpolation
"""Common functions for AOT test cases"""
import contextlib
import datetime
import os
import pathlib
import re
import subprocess
import tarfile
import logging
from typing import Any, NamedTuple, Union, Tuple, Optional, List, Dict, Callable
import numpy as np

import tvm
from tvm import relay
from tvm import autotvm
from tvm.contrib import utils, graph_executor
from tvm.relay.backend import Executor, Runtime
from tvm.relay.backend.utils import mangle_module_name
from tvm.micro import export_model_library_format
from tvm.micro.testing.utils import mlf_extract_workspace_size_bytes

_LOG = logging.getLogger(__name__)

NP_TYPE_TO_C = {
    "int8": "int8_t",
    "uint8": "uint8_t",
    "int16": "int16_t",
    "uint16": "uint16_t",
    "int32": "int32_t",
    "uint32": "uint32_t",
    "float32": "float",
}

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
        Dict of output names to value arrays
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
    outputs: Dict[str, np.array]
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
    epilogue: str
        Code to append to the main function
    includes: List[str]
        Additional includes required to run the AOT test runner
    parameters: Dict[str, str]
        Additional parameters to pass to the make command
    pass_config: Dict[str, Any]
        Additional pass configuration when building the model
    """

    makefile: str = "default"
    prologue: str = ""
    epilogue: str = ""
    includes: List[str] = []
    parameters: Dict[str, str] = {}
    pass_config: Dict[str, Any] = {}


def _subprocess_check_log_output(cmd, cwd, logfile):
    """
    This method runs a process and logs the output to both a log file and stdout
    """
    _LOG.info("Execute (%s): %s", cwd, cmd)
    cmd_base = cmd[0] if isinstance(cmd, (list, tuple)) else cmd.split(" ", 1)[0]
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        shell=True,
        bufsize=0,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
    )
    stdout = ""
    with open(logfile, "a") as f:
        msg = (
            "\n"
            + "-" * 80
            + f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Execute ({cwd}): {cmd}\n"
            + "-" * 80
        )
        f.write(msg)
        stdout += msg + "\n"
        while True:
            data = proc.stdout.readline()
            stdout += data
            _LOG.debug("%s: %s", cmd_base, data.rstrip("\n"))
            f.write(data)

            # process is done if there is no data and the result is valid
            if not data:  # EOF
                break

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Subprocess failed: {cmd}\nstdout:\n{stdout}")


def _mangle_name(mod_name, name):
    mod_name = mangle_module_name(mod_name)
    return mod_name + "_" + name


# TODO: Move to linker script with list of symbols rather than coding into source
def _emit_data_linkage(output_file, data_linkage):
    if data_linkage is not None:
        output_file.write(
            f'__attribute__((section("{data_linkage.section}"), '
            f"aligned({data_linkage.alignment}))) "
        )


def _emit_main_prologue(
    main_file,
    custom_prologue,
    workspace_bytes,
    data_linkage,
    compiled_models,
    interface_api,
    use_stack_allocator=True,
    debug_last_error=False,
):
    if use_stack_allocator:
        workspace_define = f"#define WORKSPACE_SIZE ({workspace_bytes}"
        if interface_api == "c":
            for compiled_model in compiled_models:
                model = compiled_model.model
                workspace_define += f" + TVMGEN_{model.name.upper()}_WORKSPACE_SIZE"
        # Add TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES because of memory alignment.
        workspace_define += " + TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES)\n"
        main_file.write(workspace_define)
        _emit_data_linkage(main_file, data_linkage)
        main_file.write("static uint8_t g_aot_memory[WORKSPACE_SIZE];\n")
        main_file.write("tvm_workspace_t app_workspace;\n")
        main_file.write(
            """\n
tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
    return StackMemoryManager_Allocate(&app_workspace, num_bytes, out_ptr);
}
tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
    return StackMemoryManager_Free(&app_workspace,ptr);
}
        """
        )
    else:
        # An implementation is not needed for these if the stack allocator is not used
        main_file.write(
            """\n
tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
    return kTvmErrorFunctionCallNotImplemented;
}
tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
    return kTvmErrorFunctionCallNotImplemented;
}
            """
        )
    main_file.write(
        """\n
void TVMPlatformAbort(tvm_crt_error_t code) { exit(-1); }
void TVMLogf(const char* msg, ...) {
  va_list args;
  va_start(args, msg);
  vfprintf(stdout, msg, args);
  va_end(args);
}
    """
    )
    if debug_last_error:
        main_file.write(
            """\n
tvm_crt_error_t TVMPlatformTimerStart() {
  return kTvmErrorFunctionCallNotImplemented;
}
tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
  return kTvmErrorFunctionCallNotImplemented;
}
const TVMModule* TVMSystemLibEntryPoint(void) { return NULL; }
"""
        )
    else:
        main_file.write(
            """\n
TVM_DLL int TVMFuncRegisterGlobal(const char* name, TVMFunctionHandle f, int override) {}
"""
        )
    main_file.write("\nint main(){\n")
    main_file.write(custom_prologue)


def _emit_main_data(main_file, input_map, output_map, mod_name):
    for key in input_map:
        sanitized_tensor_name = re.sub(r"\W", "_", key)
        main_file.write(
            f'#include "{_mangle_name(mod_name,"input_data")}_{sanitized_tensor_name}.h"\n'
        )

    for key in output_map:
        sanitized_tensor_name = re.sub(r"\W", "_", key)
        main_file.write(
            f'#include "{_mangle_name(mod_name,"expected_output_data")}_'
            f'{sanitized_tensor_name}.h"\n'
            f'#include "{_mangle_name(mod_name,"output_data")}_'
            f'{sanitized_tensor_name}.h"\n'
        )


def _emit_main_device_structs(main_file, devices, mod_name):
    if devices:
        main_file.write(
            f"struct {_mangle_name(mod_name, 'devices')} {_mangle_name(mod_name, 'devices')} = {{"
        )
        for device in devices:
            main_file.write(f"\t.{device} = {device},\n")
        main_file.write("};\n")


def _emit_main_workspace_pool_structs(main_file, workspace_pool_names, mod_name):
    if workspace_pool_names and len(workspace_pool_names) > 0:
        main_file.write(
            f"struct {_mangle_name(mod_name, 'workspace_pools')} "
            f"{_mangle_name(mod_name, 'workspace_pools')} = {{"
        )
        for workspace_pool_name in workspace_pool_names.keys():
            main_file.write(
                f"\t.{workspace_pool_name} = {workspace_pool_names[workspace_pool_name]}"
                f"{workspace_pool_name},\n"
            )
        main_file.write("};\n")


def _emit_main_data_structs(main_file, input_map, output_map, mod_name):
    main_file.write(
        f"struct {_mangle_name(mod_name, 'inputs')} {_mangle_name(mod_name, 'inputs')} = {{"
    )
    for key in input_map:
        sanitized_tensor_name = re.sub(r"\W", "_", key)
        main_file.write(
            f"\t.{sanitized_tensor_name} = "
            f"{_mangle_name(mod_name, 'input_data')}_{sanitized_tensor_name},\n"
        )
    main_file.write("};\n")

    main_file.write(
        f"struct {_mangle_name(mod_name, 'outputs')} {_mangle_name(mod_name, 'outputs')} = {{"
    )
    for key in output_map:
        sanitized_tensor_name = re.sub(r"\W", "_", key)
        main_file.write(
            f"\t.{sanitized_tensor_name} = {_mangle_name(mod_name, 'output_data')}_"
            f"{sanitized_tensor_name},\n"
        )
    main_file.write("};\n")


def _emit_main_data_setup(main_file, input_map, output_map, mod_name):
    num_outputs = len(output_map)
    num_inputs = len(input_map)
    main_file.write(f'void* {_mangle_name(mod_name,"inputs")}[{num_inputs}] = {{ ')
    for key in input_map:
        sanitized_tensor_name = re.sub(r"\W", "_", key)
        main_file.write(f'{_mangle_name(mod_name,"input_data")}_{sanitized_tensor_name}, ')
    main_file.write("};\n")
    main_file.write(f'void* {_mangle_name(mod_name,"outputs")}[{num_outputs}]  = {{ ')
    for key in output_map:
        sanitized_tensor_name = re.sub(r"\W", "_", key)
        main_file.write(f'{_mangle_name(mod_name, "output_data")}_{sanitized_tensor_name}, ')
    main_file.write("};\n")


def _emit_main_c_interface_call(
    main_file, devices, workspace_pool_names, mod_name, use_workspace_io, debug_last_error
):
    sub_strings = list()
    sub_strings.append(f'if ({_mangle_name(mod_name,"run")}(')
    if not use_workspace_io:
        sub_strings.append(f'&{_mangle_name(mod_name,"inputs")}, ')
        sub_strings.append(f'&{_mangle_name(mod_name,"outputs")}, ')
    if workspace_pool_names:
        sub_strings.append(f'&{_mangle_name(mod_name,"workspace_pools")}, ')
    if devices:
        sub_strings.append(f'&{_mangle_name(mod_name,"devices")}, ')
    # Removing the last two characters that is a comma and a space
    sub_strings[-1] = sub_strings[-1][:-2]
    # Adding brackets and newline instead
    sub_strings[-1] = sub_strings[-1] + ") == -1) {\n"
    main_file_string = "".join(sub_strings)
    main_file.write(main_file_string)
    if debug_last_error:
        main_file.write(f'\tprintf("ERROR: %s\\n", TVMGetLastError());\n')
    main_file.write(f'\tprintf("{AOT_FAILURE_TOKEN}\\n");\n')
    main_file.write("\treturn -1;\n")
    main_file.write("}\n")


def _emit_main_fake_packed_values(main_file):
    main_file.write(
        """
    static DLDevice fake_device = {kDLCPU, 0};
    static int64_t fake_dims = 0;
    static int64_t fake_shape = {0};
    """
    )


def _emit_main_packed_call(main_file, input_map, output_list, mod_name):
    tensors_name = _mangle_name(mod_name, "tensors")
    values_name = _mangle_name(mod_name, "values")
    typeids_name = _mangle_name(mod_name, "typeids")

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
        fake_tensor(_mangle_name(mod_name, "inputs"), i, i)
    for i in range(0, num_outputs):
        fake_tensor(_mangle_name(mod_name, "outputs"), i, i + num_inputs)

    main_file.write(
        f'{_mangle_name(mod_name, "run")}({values_name}, {typeids_name}, 0, NULL, 0, NULL);\n'
    )
    main_file.write("\n")


def _emit_main_compare(main_file, outputs, output_tolerance, mod_name, use_interface_c=False):
    for key in outputs:
        sanitized_tensor_name = re.sub(r"\W", "_", key)
        expected_data_name = _mangle_name(mod_name, f"expected_output_data_{sanitized_tensor_name}")
        is_float_dtype = outputs[key].dtype == "float32"

        comparison_function = "abs"
        tolerance = output_tolerance or 0
        if is_float_dtype:
            comparison_function = "fabs"
            tolerance = output_tolerance or 0.001

        data_length_var_name = (
            _mangle_name(mod_name, f"output_data_{sanitized_tensor_name}") + "_len"
        )
        if use_interface_c:
            c_type = NP_TYPE_TO_C[str(outputs[key].dtype)]
            actual_data_name = f"(({c_type}*)" + _mangle_name(
                mod_name, f"outputs.{sanitized_tensor_name})"
            )
        else:
            actual_data_name = _mangle_name(mod_name, f"output_data_{sanitized_tensor_name}")
        main_file.write(
            f"for (int i = 0; i<{data_length_var_name}; i++) {{\n"
            f"\tif ({comparison_function}({actual_data_name}[i]-"
            f"{expected_data_name}[i]) > {tolerance}) {{\n"
            f'\t\tprintf("{AOT_FAILURE_TOKEN}\\n");\n'
            f"\t\treturn -1;\n"
            f"\t}}\n"
            f"}}"
        )


def _emit_main_init_memory_manager(main_file):
    main_file.write("StackMemoryManager_Init(&app_workspace, g_aot_memory, WORKSPACE_SIZE);")
    main_file.write("\n")


def _emit_main_epilogue(main_file, custom_epilogue):
    main_file.write(custom_epilogue)
    main_file.write(f'printf("{AOT_SUCCESS_TOKEN}\\n");')
    main_file.write("return 0;")
    main_file.write("}\n")


def _emit_main_common_includes(main_file, custom_includes, debug_last_error):
    main_file.write("#include <stdio.h>\n")
    main_file.write("#include <stdarg.h>\n")
    main_file.write("#include <stdlib.h>\n")
    main_file.write("#include <math.h>\n")
    main_file.write('#include "tvm/runtime/c_runtime_api.h"\n')
    main_file.write('#include "tvm/runtime/crt/stack_allocator.h"\n')
    if debug_last_error:
        main_file.write('#include "tvm/runtime/crt/module.h"\n')
    for include in custom_includes:
        main_file.write(f'#include "{include}"\n')


def _emit_main_micro_include(main_file, mod_name):
    main_file.write(f"#include <{mangle_module_name(mod_name)}.h>\n")


def _create_main(
    test_name,
    compiled_models,
    output_path,
    custom_includes,
    custom_prologue,
    custom_epilogue,
    data_linkage,
    interface_api,
    workspace_bytes,
    use_stack_allocator=True,
    use_workspace_io=False,
    debug_last_error=False,
):
    file_path = pathlib.Path(f"{output_path}/" + test_name).resolve()
    # create header file
    raw_path = file_path.with_suffix(".c").resolve()
    with open(raw_path, "w") as main_file:
        _emit_main_common_includes(main_file, custom_includes, debug_last_error)

        if interface_api == "c":
            for compiled_model in compiled_models:
                model = compiled_model.model
                _emit_main_micro_include(main_file, model.name)
        for compiled_model in compiled_models:
            model = compiled_model.model
            _emit_main_data(main_file, model.inputs, model.outputs, model.name)

        _emit_main_prologue(
            main_file,
            custom_prologue,
            workspace_bytes,
            data_linkage,
            compiled_models,
            interface_api,
            use_stack_allocator,
            debug_last_error,
        )
        if use_stack_allocator:
            _emit_main_init_memory_manager(main_file)

        if interface_api == "c":
            for compiled_model in compiled_models:
                model = compiled_model.model
                executor_codegen_metadata = (
                    compiled_model.executor_factory.executor_codegen_metadata
                )
                devices = compiled_model.executor_factory.get_devices()
                workspace_pool_names = {}
                if executor_codegen_metadata.pool_inputs:
                    workspace_pool_names = {
                        allocated_pool.pool_info.pool_name: "&"
                        if isinstance(
                            allocated_pool.pool_info, tvm.ir.memory_pools.ConstantPoolInfo
                        )
                        else ""
                        for allocated_pool in dict(executor_codegen_metadata.pool_inputs).values()
                        if not allocated_pool.pool_info.is_internal
                    }
                _emit_main_device_structs(main_file, devices, model.name)
                if not use_workspace_io:
                    _emit_main_workspace_pool_structs(main_file, workspace_pool_names, model.name)
                    _emit_main_data_structs(main_file, model.inputs, model.outputs, model.name)
                _emit_main_c_interface_call(
                    main_file,
                    devices,
                    list(workspace_pool_names.keys()),
                    model.name,
                    use_workspace_io,
                    debug_last_error,
                )
        else:
            _emit_main_fake_packed_values(main_file)
            for compiled_model in compiled_models:
                model = compiled_model.model
                _emit_main_data_setup(main_file, model.inputs, model.outputs, model.name)
                _emit_main_packed_call(main_file, model.inputs, model.outputs, model.name)

        for compiled_model in compiled_models:
            model = compiled_model.model
            _emit_main_compare(
                main_file, model.outputs, model.output_tolerance, model.name, interface_api == "c"
            )
        _emit_main_epilogue(main_file, custom_epilogue)


def _create_header_file(tensor_name, npy_data, output_path, data_linkage):
    """
    This method generates a header file containing the data contained in the numpy array provided.
    It is used to capture the tensor data (for both inputs and expected outputs)
    to be bundled into the standalone application.
    """
    file_path = pathlib.Path(f"{output_path}/" + tensor_name).resolve()
    # create header file
    raw_path = file_path.with_suffix(".h").resolve()
    with open(raw_path, "w") as header_file:
        header_file.write("#include <stddef.h>\n")
        header_file.write("#include <stdint.h>\n")
        header_file.write("#include <dlpack/dlpack.h>\n")
        header_file.write(f"const size_t {tensor_name}_len = {npy_data.size};\n")

        _emit_data_linkage(header_file, data_linkage)

        header_file.write(f"{NP_TYPE_TO_C[str(npy_data.dtype)]} {tensor_name}[] =")

        header_file.write("{")
        for i in np.ndindex(npy_data.shape):
            header_file.write(f"{npy_data[i]}, ")
        header_file.write("};\n\n")


def convert_to_relay(tflite_model_buf, bind_params_by_name=True):
    """Convert a tflite model buffer in a Relay module"""
    # TFLite.Model.Model has changed to TFLite.Model from 1.14 to 2.1
    try:
        import tflite.Model  # pylint: disable=import-outside-toplevel

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite  # pylint: disable=import-outside-toplevel

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except ImportError:
        raise ImportError("The tflite package must be installed")

    mod, params = relay.frontend.from_tflite(tflite_model)
    if bind_params_by_name:
        mod["main"] = relay.build_module.bind_params_by_name(mod["main"], params)
    return mod, params


def compile_models(
    models: Union[List[AOTTestModel], AOTTestModel],
    interface_api: str,
    use_unpacked_api: bool,
    workspace_byte_alignment: int = 8,
    constant_byte_alignment: int = 8,
    enable_op_fusion: bool = True,
    pass_config: Dict[str, Any] = None,
    use_runtime_executor: bool = True,
    target: tvm.target.Target = tvm.target.Target("c"),
    workspace_memory_pools=None,
    constant_memory_pools=None,
    schedule_name: str = None,
) -> List[AOTCompiledTestModel]:
    """
    This method generates runtime.Modules for the tests
    """
    if not isinstance(models, list):
        models = [models]

    runtime = Runtime("crt")
    executor = Executor(
        "aot",
        {
            "workspace-byte-alignment": workspace_byte_alignment,
            "constant-byte-alignment": constant_byte_alignment,
            "interface-api": interface_api,
            "unpacked-api": use_unpacked_api,
        },
    )

    config = {"tir.disable_vectorize": True}
    if pass_config:
        config = {**config, **pass_config}
    if not enable_op_fusion:
        config["relay.FuseOps.max_depth"] = 1

    compiled_mods = list()
    for model in models:
        with contextlib.ExitStack() as context_stack:
            if schedule_name:
                # Testing with deterministic schedule
                task_list = autotvm.task.extract_from_program(
                    model.module, target=target, params=model.params
                )
                context_stack.enter_context(
                    tvm.autotvm.apply_fixed_config(task_list, schedule_name)
                )

            context_stack.enter_context(tvm.transform.PassContext(opt_level=3, config=config))

            build_kwargs = dict(
                ir_mod=model.module,
                params=model.params,
                mod_name=model.name,
            )

            # TODO(Mousius) - Remove once executor/runtime are fully removed from Target
            if use_runtime_executor:
                build_kwargs.update(
                    dict(
                        target=target,
                        executor=executor,
                        runtime=runtime,
                        workspace_memory_pools=workspace_memory_pools,
                        constant_memory_pools=constant_memory_pools,
                    )
                )
            else:
                build_kwargs.update(dict(target=tvm.target.Target(target, host=target)))

            executor_factory = tvm.relay.build(**build_kwargs)
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
    constant_byte_alignment=8,
    data_linkage: AOTDataLinkage = None,
    test_dir: str = None,
    verbose: bool = False,
    use_workspace_io: bool = False,
    debug_last_error: bool = False,
    checker: Optional[Callable[[str], bool]] = None,
):
    """
    This method uses the original test data and compiled runtime.Modules
    to run in the test runner to verify the results.
    """

    def run_and_check_body(base_path):
        cflags = (
            f"-DTVM_RUNTIME_ALLOC_ALIGNMENT_BYTES={workspace_byte_alignment} "
            f" -DTVM_RUNTIME_CONST_ALLOC_ALIGNMENT_BYTES={constant_byte_alignment} "
        )
        # The calculated workspaces will not account for stack allocator tags used for debugging
        if debug_calculated_workspaces:
            cflags += "-DTVM_CRT_STACK_ALLOCATOR_ENABLE_LIFO_CHECK "

        base_path = os.path.abspath(base_path)
        build_path = os.path.join(base_path, "build")
        os.makedirs(build_path, exist_ok=True)

        include_path = os.path.join(base_path, "include")
        os.mkdir(include_path)
        tvm.micro.copy_crt_config_header("crt", include_path)

        workspace_bytes = 0
        for compiled_model in models:
            model = compiled_model.model
            tar_file = os.path.join(base_path, f"{model.name}.tar")
            export_model_library_format(compiled_model.executor_factory, tar_file)
            t = tarfile.open(tar_file)
            t.extractall(base_path)

            # Interface C APIs does not need compiler generated
            # workspace to generate the test application, because
            # workspace size is codegen'd as a macro to
            # tvmgen_<model_name>.h.
            if interface_api != "c":
                workspace_bytes += mlf_extract_workspace_size_bytes(tar_file)

            workspace_bytes += model.extra_memory_in_bytes
            for key in model.inputs:
                sanitized_tensor_name = re.sub(r"\W", "_", key)
                _create_header_file(
                    f'{_mangle_name(model.name, "input_data")}_{sanitized_tensor_name}',
                    model.inputs[key],
                    include_path,
                    data_linkage,
                )

            for key in model.outputs:
                sanitized_tensor_name = re.sub(r"\W", "_", key)
                _create_header_file(
                    f'{_mangle_name(model.name, "output_data")}_{sanitized_tensor_name}',
                    np.zeros(model.outputs[key].shape, model.outputs[key].dtype),
                    include_path,
                    data_linkage,
                )
                _create_header_file(
                    f'{_mangle_name(model.name, "expected_output_data")}_{sanitized_tensor_name}',
                    model.outputs[key],
                    include_path,
                    data_linkage,
                )

        use_usmp = runner.pass_config.get("tir.usmp.enable", False)
        # We only need the stack allocator if USMP is not used
        use_stack_allocator = not use_usmp

        _create_main(
            "test.c",
            models,
            build_path,
            runner.includes,
            runner.prologue,
            runner.epilogue,
            data_linkage,
            interface_api,
            workspace_bytes,
            use_stack_allocator,
            use_workspace_io,
            debug_last_error,
        )

        if checker and (not checker(base_path)):
            return False

        # Verify that compiles fine
        file_dir = os.path.dirname(os.path.abspath(__file__))
        makefile_dir = os.path.join(file_dir, "../../../tests/python/relay/aot")
        codegen_path = os.path.join(base_path, "codegen")
        makefile = os.path.join(makefile_dir, f"{runner.makefile}.mk")
        fvp_dir = "/opt/arm/FVP_Corstone_SSE-300/models/Linux64_GCC-6.4/"
        # TODO(@grant-arm): Remove once ci_cpu docker image has been updated to FVP_Corstone_SSE
        if not os.path.isdir(fvp_dir):
            fvp_dir = "/opt/arm/FVP_Corstone_SSE-300_Ethos-U55/models/Linux64_GCC-6.4/"
        custom_params = " ".join(
            [f" {param}='{value}'" for param, value in runner.parameters.items()]
        )
        make_command = (
            f"make -f {makefile} build_dir={build_path}"
            + f" CFLAGS='{cflags}'"
            + f" TVM_ROOT={file_dir}/../../.."
            + f" AOT_TEST_ROOT={makefile_dir}"
            + f" CODEGEN_ROOT={codegen_path}"
            + f" STANDALONE_CRT_DIR={tvm.micro.get_standalone_crt_dir()}"
            + f" FVP_DIR={fvp_dir}"
            + custom_params
        )

        compile_log_path = os.path.join(build_path, "test_compile.log")
        compile_command = f"{make_command} aot_test_runner"
        if verbose:
            print("Compile command:\n", compile_command)
        _subprocess_check_log_output(compile_command, ".", compile_log_path)

        # Verify that runs fine
        run_log_path = os.path.join(build_path, "test_run.log")
        run_command = f"{make_command} run"
        if verbose:
            print("Run command:\n", run_command)

        _subprocess_check_log_output(run_command, build_path, run_log_path)

        with open(run_log_path) as run_log:
            assert AOT_SUCCESS_TOKEN in run_log.read()

        return True

    if test_dir is None:
        tmpdir = utils.tempdir()
        return run_and_check_body(os.path.join(tmpdir.path, "test"))
    else:
        return run_and_check_body(test_dir)


def compile_and_run(
    models: Union[List[AOTTestModel], AOTTestModel],
    runner: AOTTestRunner,
    interface_api: str,
    use_unpacked_api: bool,
    debug_calculated_workspaces: bool = False,
    workspace_byte_alignment: int = 8,
    constant_byte_alignment: int = 8,
    enable_op_fusion: bool = True,
    data_linkage: AOTDataLinkage = None,
    use_runtime_executor: bool = True,
    target: Union[str, tvm.target.Target, List[tvm.target.Target]] = "c",
    target_opts: Dict = None,
    test_dir: str = None,
    verbose: bool = False,
    schedule_name: str = None,
    debug_last_error: bool = False,
    checker: Optional[Callable[[str], bool]] = None,
) -> bool:
    """This is a wrapper API to compile and run models as test for AoT

    Parameters
    ----------
    test_dir : str
        This path will contain build, codegen, include directories
    verbose: bool
        Prints commands to build and run AOT test runner
    """

    if target_opts:
        for key, val in target_opts.items():
            target += f" {key}={val}"

    if isinstance(target, str):
        target = tvm.target.Target(target)

    compiled_test_mods = compile_models(
        models=models,
        interface_api=interface_api,
        use_unpacked_api=use_unpacked_api,
        workspace_byte_alignment=workspace_byte_alignment,
        constant_byte_alignment=constant_byte_alignment,
        enable_op_fusion=enable_op_fusion,
        pass_config=runner.pass_config,
        use_runtime_executor=use_runtime_executor,
        target=target,
        schedule_name=schedule_name,
    )

    return run_and_check(
        models=compiled_test_mods,
        runner=runner,
        interface_api=interface_api,
        debug_calculated_workspaces=debug_calculated_workspaces,
        workspace_byte_alignment=workspace_byte_alignment,
        constant_byte_alignment=constant_byte_alignment,
        data_linkage=data_linkage,
        test_dir=test_dir,
        verbose=verbose,
        debug_last_error=debug_last_error,
        checker=checker,
    )


def get_dtype_range(dtype: str) -> Tuple[int, int]:
    """
    Produces the min,max for a give data type.

    Parameters
    ----------
    dtype : str
        a type string (e.g., int8, float64)

    Returns
    -------
    type_info.min : int
        the minimum of the range
    type_info.max : int
        the maximum of the range
    """
    type_info = None
    np_dtype = np.dtype(dtype)
    kind = np_dtype.kind

    if kind == "f":
        type_info = np.finfo(np_dtype)
    elif kind in ["i", "u"]:
        type_info = np.iinfo(np_dtype)
    else:
        raise TypeError(f"dtype ({dtype}) must indicate some floating-point or integral data type.")
    return type_info.min, type_info.max


def generate_ref_data(mod, input_data, params=None, target="llvm"):
    """Generate reference data through executing the relay module"""
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
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
    if isinstance(mod, tvm.relay.Function):
        main = mod
    else:
        main = mod["main"]
    if main.attrs is None or main.attrs["output_tensor_names"] is None:
        output_tensor_names = (
            ["output"] if output_count == 1 else [f"output{i}" for i in range(output_count)]
        )
    else:
        output_tensor_names = main.attrs["output_tensor_names"]

    return dict(zip(output_tensor_names, out))


def create_relay_module_and_inputs_from_tflite_file(tflite_model_file, bind_params_by_name=True):
    """A helper function to create a Relay IRModule with inputs
    and params from a tflite file"""
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()
    mod, params = convert_to_relay(tflite_model_buf, bind_params_by_name)

    inputs = dict()
    for param in mod["main"].params:
        name = str(param.name_hint)
        data_shape = [int(i) for i in param.type_annotation.shape]
        dtype = str(param.type_annotation.dtype)
        if np.issubdtype(dtype, np.floating):
            # Since np.random.uniform only allows the ranges of float32,
            # at first float16 is used and scaled afterwards, if necessary.
            in_min, in_max = (np.finfo("float16").min, np.finfo("float16").max)
            data = np.random.uniform(low=in_min, high=in_max, size=data_shape).astype(dtype)
            scale = np.finfo(dtype).min / np.finfo("float16").min
            data *= scale
        elif np.issubdtype(dtype, np.integer):
            in_min, in_max = (np.iinfo(dtype).min, np.iinfo(dtype).max)
            data = np.random.randint(in_min, high=in_max, size=data_shape, dtype=dtype)
        else:
            raise TypeError(f"Type {dtype} not supported")
        inputs[name] = data

    return mod, inputs, params
