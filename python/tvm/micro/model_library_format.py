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
# pylint: disable=cell-var-from-loop, use-list-literal

"""Defines functions for exporting to Model Library Format."""

import datetime
import json
import os
import pathlib
import re
import tarfile
import typing

import tvm
from tvm.micro import get_standalone_crt_dir, get_microtvm_template_projects

from .._ffi import get_global_func
from ..contrib import utils
from ..driver import build_module
from ..relay import param_dict
from ..relay.backend import executor_factory
from ..relay.backend.name_transforms import prefix_generated_name, to_c_variable_style
from ..tir import expr

# This should be kept identical to runtime::symbol::tvm_module_main
MAIN_FUNC_NAME_STR = "__tvm_main__"
STANDALONE_CRT_URL = "./runtime"
CRT_TEMPLATE_FILES_URL = "./templates"
METADATA_FILE = "metadata.json"


class UnsupportedInModelLibraryFormatError(Exception):
    """Raised when export_model_library_format does not support the given Module tree."""


def generate_c_interface_header(
    module_name,
    inputs,
    outputs,
    pools,
    io_pool_allocations,
    devices,
    workspace_size,
    include_path,
    input_sizes,
    output_sizes,
):
    """Generate C Interface header to be included in MLF"""
    mangled_name = to_c_variable_style(prefix_generated_name(module_name))
    metadata_header = os.path.join(include_path, f"{mangled_name}.h")

    interface_c_create = tvm._ffi.get_global_func("runtime.InterfaceCCreate")
    interface_c_module = interface_c_create(
        module_name,
        inputs,
        outputs,
        pools,
        io_pool_allocations,
        devices,
        workspace_size,
        input_sizes,
        output_sizes,
    )

    with open(metadata_header, "w") as header_file:
        header_file.write(interface_c_module.get_source())

    return metadata_header


# List of type_key for modules which are ephemeral and do not need to be exported.
EPHEMERAL_MODULE_TYPE_KEYS = ("metadata_module",)


def _populate_codegen_dir(
    mods: typing.Union[
        typing.List[executor_factory.ExecutorFactoryModule],
        typing.List[tvm.runtime.Module],
    ],
    codegen_dir: str,
):
    """Populate the codegen sub-directory as part of a Model Library Format export.

    Parameters
    ----------
    mods : List[tvm.relay.backend.executor_factory.ExecutorFactoryModule], List[tvm.runtime.Module]
        A list of the return value of tvm.relay.build, which
        will be exported into Model Library Format.
    codegen_dir : str
        Path to the codegen directory on disk.
    module_name: Optional[str]
        Name used to prefix the generated source files

    """
    dso_modules = []
    for mod in mods:
        if isinstance(mod, executor_factory.ExecutorFactoryModule):
            lib = mod.lib
        elif isinstance(mod, tvm.runtime.Module):
            lib = mod
        else:
            raise RuntimeError(f"Not supported module type: {type(mod)}")

        dso_modules = lib._collect_dso_modules()
        non_dso_modules = lib._collect_from_import_tree(lambda m: m not in dso_modules)

        # Filter ephemeral modules which cannot be exported.
        dso_modules = [m for m in dso_modules if m.type_key not in EPHEMERAL_MODULE_TYPE_KEYS]
        non_dso_modules = [
            m for m in non_dso_modules if m.type_key not in EPHEMERAL_MODULE_TYPE_KEYS
        ]

        if non_dso_modules:
            raise UnsupportedInModelLibraryFormatError(
                f"Don't know how to export non-c or non-llvm modules; found: {non_dso_modules!r}"
            )

        mod_indices = {"lib": 0, "src": 0}
        host_codegen_dir = os.path.join(codegen_dir, "host")
        lib_name = (
            f"{mod.libmod_name}_lib"
            if isinstance(mod, executor_factory.ExecutorFactoryModule)
            else "lib"
        )

        for dso_mod in dso_modules:
            if dso_mod.type_key == "c":
                assert dso_mod.format in ["c", "cc", "cpp"]
                ext = dso_mod.format
                index = mod_indices["src"]
                mod_indices["src"] += 1
                parent_dir = os.path.join(host_codegen_dir, "src")
                file_name = os.path.join(parent_dir, f"{lib_name}{index}.{ext}")
            elif dso_mod.type_key == "llvm":
                index = mod_indices["lib"]
                mod_indices["lib"] += 1
                parent_dir = os.path.join(host_codegen_dir, "lib")
                file_name = os.path.join(parent_dir, f"{lib_name}{index}.o")
            else:
                assert (
                    False
                ), f"do not expect module with type_key={lib.type_key} from _collect_dso_modules"

            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            dso_mod.save(file_name)


def _build_memory_map(mod):
    ret = dict()
    if isinstance(mod, executor_factory.GraphExecutorFactoryModule):
        ret["sids"] = _build_sid_map(mod.graph_json)
    ret["functions"] = _build_function_memory_map(mod.function_metadata)
    return ret


def _build_sid_map(graph_json):
    """Build a simpler storage id info map from graph JSON.

    Parameters
    ----------
    graph_json : str
        String representation of the graph_json created from tvm.relay.build().

    Returns
    -------
    list :
        A list with one entry per storage id describing that memory.
    """
    graph = json.loads(graph_json)
    seen_storage_ids = set()
    memory_map = []
    for node_id, storage_id in enumerate(graph["attrs"]["storage_id"][1]):
        if storage_id in seen_storage_ids:
            continue

        seen_storage_ids.add(storage_id)
        num_elements = 1
        for dim in graph["attrs"]["shape"][1][storage_id]:
            num_elements *= dim

        dltype = graph["attrs"]["dltype"][1][storage_id]
        m = re.match(r"^[a-zA-Z]+([0-9]+)$", dltype)
        assert m, f"Exported graph contains unknown dltype {dltype}"

        elem_bits = int(m.group(1))

        map_entry = {
            "storage_id": storage_id,
            "size_bytes": (num_elements * elem_bits + 7) // 8,
        }
        if node_id in graph["arg_nodes"]:
            map_entry["input_binding"] = graph["nodes"][node_id]["name"]

        memory_map.append(map_entry)

    return memory_map


def _create_type_metadata(input_type):
    return {
        "size": int(_shape_to_size(input_type.shape, input_type.dtype)),
        "dtype": str(input_type.dtype),
    }


def _flatten_tuple_outputs(ret_type, predefined_names, offset=0):
    if isinstance(ret_type, tvm.ir.tensor_type.TensorType):
        name = predefined_names[offset] if predefined_names else f"output{offset}"
        return {name: ret_type}

    added_fields = len(ret_type.fields)
    outputs = {}
    for output_index in range(added_fields):
        next_output = offset + len(outputs)
        outputs.update(
            _flatten_tuple_outputs(ret_type.fields[output_index], predefined_names, next_output)
        )

    return outputs


def _get_outputs_from_ret_type(ret_type, predefined_names):
    if isinstance(ret_type, tvm.ir.tensor_type.TensorType):
        name = predefined_names[0] if predefined_names else "output"
        return {name: ret_type}
    return _flatten_tuple_outputs(ret_type, predefined_names)


def _build_function_memory_map(function_metadata):
    """Build a simple map that shows how much workspace is required to execute
    each primitive function. The main_func describes how much memory is required
    to execute the main control code.

    Parameters
    ----------
    function_metadata : Map<String, FunctionInfo>
        This contains all the compiled metadata on a function basis

    Returns
    -------
    dict :
        This will have two entries:
        1.) A list with one entry per function describing local memory it is using.
        2.) A global memory requirement if all functions are executed sequentially
    """
    device_max_workspace = dict()
    main_func_metadata = function_metadata[MAIN_FUNC_NAME_STR]
    func_entries = []
    target_local_entries = dict()

    for func_name, finfo in function_metadata.items():
        # Skip a few unsupported cases:
        # 1. The main function metadata is exported elsewhere.
        # 2. BYOC operator implementations do not currently export useful FunctionInfo.
        if func_name == MAIN_FUNC_NAME_STR or not finfo.tir_primfuncs:
            continue
        if func_name not in target_local_entries.keys():
            target_local_entries[func_name] = list()
        for target in dict(finfo.workspace_sizes).keys():
            workspace_size = finfo.workspace_sizes[target]
            target_entry = {
                "device": int(target.get_target_device_type()),
                "workspace_size_bytes": int(workspace_size),
            }
            target_local_entries[func_name].append(target_entry)
            if workspace_size >= device_max_workspace.get(int(target.get_target_device_type()), 0):
                device_max_workspace[int(target.get_target_device_type())] = workspace_size

    for func_name, target_entries_ in target_local_entries.items():
        func_entry = {
            "function_name": str(func_name),
            "workspace": target_entries_,
        }
        func_entries.append(func_entry)

    target_main_entries = dict()

    def _create_empty_entry(target_device_type):
        return {
            "device": int(target_device_type),
            "workspace_size_bytes": 0,
            "constants_size_bytes": 0,
            "io_size_bytes": 0,
        }

    for target in dict(main_func_metadata.workspace_sizes).keys():
        main_func_local_workspace = main_func_metadata.workspace_sizes[target]
        target_main_entries[int(target.get_target_device_type())] = _create_empty_entry(
            int(target.get_target_device_type())
        )
        target_main_entries[int(target.get_target_device_type())]["workspace_size_bytes"] = int(
            device_max_workspace.get(int(target.get_target_device_type()), 0)
        ) + int(main_func_local_workspace)

    for target in dict(main_func_metadata.constant_sizes).keys():
        if int(target.get_target_device_type()) not in target_main_entries.keys():
            target_main_entries[int(target.get_target_device_type())] = _create_empty_entry(
                int(target.get_target_device_type())
            )
        target_main_entries[int(target.get_target_device_type())]["constants_size_bytes"] = int(
            main_func_metadata.constant_sizes[target]
        )

    for target in dict(main_func_metadata.io_sizes).keys():
        if int(target.get_target_device_type()) not in target_main_entries.keys():
            target_main_entries[int(target.get_target_device_type())] = _create_empty_entry(
                int(target.get_target_device_type())
            )
        target_main_on_device = target_main_entries[int(target.get_target_device_type())]
        target_main_on_device["io_size_bytes"] = int(main_func_metadata.io_sizes[target])

        main_relay_func = main_func_metadata.relay_primfuncs[target]
        target_main_on_device["inputs"] = {
            input_param.name_hint: _create_type_metadata(input_param.checked_type)
            for input_param in main_relay_func.params
        }
        predefined_names = (
            main_relay_func.attrs["output_tensor_names"]
            if "output_tensor_names" in main_relay_func.attrs
            else None
        )
        target_main_on_device["outputs"] = {
            name: _create_type_metadata(output_type)
            for name, output_type in _get_outputs_from_ret_type(
                main_relay_func.ret_type, predefined_names
            ).items()
        }

    ret = {
        "operator_functions": func_entries,
        "main": list(target_main_entries.values()),
    }
    return ret


def _get_pools_from_module(mod):
    return list(dict(mod.executor_codegen_metadata.pool_inputs).values())


def _get_io_pool_allocation_from_module(mod):
    return dict(mod.executor_codegen_metadata.io_pool_allocations)


def _should_generate_interface_header(mod):
    return "interface-api" in mod.executor and mod.executor["interface-api"] == "c"


def _make_tar(source_dir, tar_file_path, modules):
    """Build a tar file from source_dir."""
    with tarfile.open(tar_file_path, "w") as tar_f:

        def reset(tarinfo):
            tarinfo.uid = tarinfo.gid = 0
            tarinfo.uname = tarinfo.gname = "root"
            return tarinfo

        tar_f.add(str(source_dir), arcname=".", filter=reset)

        for mod in modules:
            is_aot = isinstance(mod, executor_factory.AOTExecutorFactoryModule)
            if is_aot and str(mod.runtime) == "crt":
                crt_template_path = pathlib.Path(get_microtvm_template_projects("crt"))
                tar_f.add(get_standalone_crt_dir(), arcname=STANDALONE_CRT_URL)

                # Add template files from CRT template project
                for file in [
                    "templates/crt_config.h.template",
                    "templates/platform.c.template",
                ]:
                    tar_f.add(
                        crt_template_path / pathlib.Path(file),
                        arcname=f"{CRT_TEMPLATE_FILES_URL}/{pathlib.Path(file).name}",
                    )
                break


_GENERATED_VERSION = 7


def _is_module_names_unique(mods: typing.List[executor_factory.ExecutorFactoryModule]):
    """Check if built modules have unique names.

    Parameters
    ----------
    mods : List[tvm.relay.backend.executor_factory.ExecutorFactoryModule]
        A list of the return value of tvm.relay.build,
        which will be exported into Model Library Format.
    """
    all_names = []
    for mod in mods:
        all_names.append(mod.libmod_name)

    return len(set(all_names)) == len(all_names)


def _export_graph_model_library_format(
    mods: typing.List[executor_factory.ExecutorFactoryModule], tempdir: pathlib.Path
):
    """Export a tvm.relay.build artifact in Model Library Format.

    Parameters
    ----------
    mods : List[tvm.relay.backend.executor_factory.ExecutorFactoryModule]
        A list of the return value of tvm.relay.build,
        which will be exported into Model Library Format.
    tempdir : pathlib.Path
        Temporary directory to populate with Model Library Format contents.
    """

    assert _is_module_names_unique(mods), "Multiple modules should have unique names."

    metadata = {
        "version": _GENERATED_VERSION,
    }
    metadata["modules"] = {}
    for mod in mods:
        is_aot = isinstance(mod, executor_factory.AOTExecutorFactoryModule)
        executor = ["aot"] if is_aot else ["graph"]
        module_name = mod.libmod_name
        metadata["modules"][module_name] = {
            "model_name": module_name,
            "export_datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%SZ"),
            "memory": _build_memory_map(mod),
            "target": [str(t) for t in mod.target],
            "executors": executor,
            "style": "full-model",
        }

        if is_aot and (str(mod.runtime) == "crt"):
            standalone_crt = {
                "short_name": "tvm_standalone_crt",
                "url": f"{STANDALONE_CRT_URL}",
                "url_type": "mlf_path",
                "version_spec": f"{tvm.__version__}",
            }
            external_dependencies = [standalone_crt]
            metadata["modules"][module_name]["external_dependencies"] = external_dependencies

    with open(tempdir / METADATA_FILE, "w") as json_f:
        json.dump(metadata, json_f, indent=2, sort_keys=True)

    codegen_dir = tempdir / "codegen"
    codegen_dir.mkdir()
    _populate_codegen_dir(mods, codegen_dir)

    parameters_dir = tempdir / "parameters"
    parameters_dir.mkdir()
    src_dir = tempdir / "src"
    src_dir.mkdir()
    graph_config_dir = tempdir / "executor-config" / "graph"
    for mod in mods:
        if _should_generate_interface_header(mod):
            include_path = codegen_dir / "host" / "include"
            if not include_path.exists():
                include_path.mkdir()

            devices = mod.get_devices()
            pools = _get_pools_from_module(mod)
            io_pool_allocations = _get_io_pool_allocation_from_module(mod)
            main_func = metadata["modules"][mod.libmod_name]["memory"]["functions"]["main"][0]
            workspace_size = int(main_func["workspace_size_bytes"])
            inputs = main_func["inputs"]
            outputs = main_func["outputs"]
            inputs_sizes = {name: property_map["size"] for name, property_map in inputs.items()}
            output_sizes = {name: property_map["size"] for name, property_map in outputs.items()}
            input_names = list(inputs.keys())
            output_names = list(outputs.keys())

            generate_c_interface_header(
                mod.libmod_name,
                input_names,
                output_names,
                pools,
                io_pool_allocations,
                devices,
                workspace_size,
                include_path,
                inputs_sizes,
                output_sizes,
            )

        is_aot = isinstance(mod, executor_factory.AOTExecutorFactoryModule)
        param_filename = parameters_dir / f"{mod.libmod_name}.params"
        with open(param_filename, "wb") as f:
            f.write(param_dict.save_param_dict(mod.params))

        with open(src_dir / f"{mod.libmod_name}.relay", "w") as f:
            f.write(str(mod.ir_mod))

        if not is_aot:
            if not graph_config_dir.exists():
                graph_config_dir.mkdir(parents=True)
            with open(graph_config_dir / f"{mod.libmod_name}.graph", "w") as f:
                f.write(mod.get_executor_config())


class NonStaticShapeError(Exception):
    """Raised when a shape has elements other than IntImm."""


def _shape_to_size(shape, dtype):
    bits_per_item = int(
        re.match(r"((float)|(int)|(uint))(?P<width_bits>[0-9]+)", dtype).group("width_bits")
    )
    assert bits_per_item is not None, f"don't know how to compute size of type {dtype}"
    total_bits = bits_per_item
    for s in shape:
        total_bits *= s

    return (total_bits + 7) // 8


def _write_tir_and_build_operator_memory_map(src_dir, targets, ir_module_by_target):
    def _eval_shape(param_name, buffer_shape):
        shape = []
        for x in buffer_shape:
            if not isinstance(x, expr.IntImm):
                raise NonStaticShapeError(
                    f"Parameter {param_name} has shape with non-IntImm elements: {buffer_shape}"
                )
            shape.append(x.value)
        return shape

    memory_map = {}
    for target in targets:
        # TODO(mbs): The device type is not unique, better would be to use target.kind.name
        target_device_type = target.get_target_device_type()
        ir_mod = ir_module_by_target[target]
        printer = get_global_func("relay.ir.ModelLibraryFormatPrinter")(False, None, False)
        with open(src_dir / f"tir-{target_device_type}.txt", "w") as f:
            f.write(printer["print"](ir_mod))

        for v in ir_mod.get_global_vars():
            map_entry = []
            for p, b in ir_mod[v.name_hint].buffer_map.items():
                shape = _eval_shape(p.name, b.shape)
                buffer_size_bytes = _shape_to_size(shape, str(b.dtype))
                # NOTE: cannot tell what is an input or output at this point.
                map_entry.append(
                    {
                        "size_bytes": buffer_size_bytes,
                        "shape": [int(x) for x in b.shape],
                        "dtype": b.dtype,
                        "input_binding": printer["get_var_name"](p),
                    }
                )
            memory_map[v.name_hint] = map_entry

    return memory_map


def _export_operator_model_library_format(mod: build_module.OperatorModule, tempdir):
    """Export the result of tvm.build() in Model Library Format.
    Parameters
    ----------
    mod : runtime.Module
        The Module returned from tvm.build().
    tempdir : str
        Path to the .tar archive to generate.
    """
    targets = []
    for target in mod.ir_module_by_target.keys():
        if str(target.kind) not in ("llvm", "c"):
            raise UnsupportedInModelLibraryFormatError(
                f"Operator has non-DSO-exportable target {target!s}, which is not yet supported in "
                "Model Library Format"
            )

        targets.append(target)

    src_dir = tempdir / "src"
    src_dir.mkdir()
    memory_map = _write_tir_and_build_operator_memory_map(src_dir, targets, mod.ir_module_by_target)

    metadata = {
        "version": _GENERATED_VERSION,
        "model_name": mod.name,
        "export_datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%SZ"),
        "memory": memory_map,
        "target": [str(t) for t in targets],
        "executors": [],
        "style": "operator",
    }
    with open(tempdir / METADATA_FILE, "w") as metadata_f:
        json.dump(metadata, metadata_f)

    codegen_dir = tempdir / "codegen"
    codegen_dir.mkdir()
    _populate_codegen_dir(list([mod]), codegen_dir)


ExportableModule = typing.Union[
    build_module.OperatorModule,
    executor_factory.AOTExecutorFactoryModule,
    executor_factory.GraphExecutorFactoryModule,
]


def export_model_library_format(
    mods: typing.Union[ExportableModule, typing.List[ExportableModule]],
    file_name: typing.Union[str, pathlib.Path],
):
    """Export the build artifact in Model Library Format.

    This function creates a .tar archive containing the build artifacts in a standardized
    layout. It's intended to allow downstream automation to build TVM artifacts against the C
    runtime.

    Parameters
    ----------
    mod : ExportableModule, List[ExportableModule]
        The return value of tvm.build or tvm.relay.build.
    file_name : str
        Path to the .tar archive to generate.

    Returns
    -------
    file_name : str
        The path to the generated .tar archive.
    """
    modules = mods
    if not isinstance(mods, list):
        modules = list([mods])

    operator_module_type = all(isinstance(mod, build_module.OperatorModule) for mod in modules)
    graph_module_type = all(
        isinstance(
            mod,
            (
                executor_factory.AOTExecutorFactoryModule,
                executor_factory.GraphExecutorFactoryModule,
            ),
        )
        for mod in modules
    )

    file_name = pathlib.Path(file_name)
    tempdir = utils.tempdir()

    if operator_module_type:
        if len(modules) != 1:
            raise RuntimeError("Multiple operator is not supported.")
        _export_operator_model_library_format(modules[0], tempdir.path)
    elif graph_module_type:
        _export_graph_model_library_format(modules, tempdir.path)
    else:
        raise NotImplementedError(
            f"Don't know how to export module of type {modules[0].__class__!r}"
        )

    _make_tar(tempdir.path, file_name, modules)

    return file_name
