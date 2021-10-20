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

"""Defines functions for exporting to Model Library Format."""

import datetime
import json
import os
import pathlib
import re
import tarfile
import typing

import tvm
from tvm.ir.type import TupleType
from .._ffi import get_global_func
from ..contrib import utils
from ..driver import build_module
from ..runtime import ndarray as _nd
from ..relay.backend import executor_factory
from ..relay.backend.name_transforms import to_c_variable_style, prefix_generated_name
from ..relay import param_dict
from ..tir import expr

# This should be kept identical to runtime::symbol::tvm_module_main
MAIN_FUNC_NAME_STR = "__tvm_main__"


class UnsupportedInModelLibraryFormatError(Exception):
    """Raised when export_model_library_format does not support the given Module tree."""


def generate_c_interface_header(module_name, inputs, outputs, include_path):
    """Generate C Interface header to be included in MLF"""
    mangled_name = to_c_variable_style(prefix_generated_name(module_name))
    metadata_header = os.path.join(include_path, f"{mangled_name}.h")

    interface_c_create = tvm._ffi.get_global_func("runtime.InterfaceCCreate")
    interface_c_module = interface_c_create(module_name, inputs, outputs)

    with open(metadata_header, "w") as header_file:
        header_file.write(interface_c_module.get_source())

    return metadata_header


def _populate_codegen_dir(mod, codegen_dir: str, module_name: str = None):
    """Populate the codegen sub-directory as part of a Model Library Format export.

    Parameters
    ----------
    mod : tvm.runtime.Module
        Module which should be written to codegen_dir.
    codegen_dir : str
        Path to the codegen directory on disk.
    module_name: Optional[str]
        Name used to prefix the generated source files

    """
    dso_modules = mod._collect_dso_modules()
    non_dso_modules = mod._collect_from_import_tree(lambda m: m not in dso_modules)
    if non_dso_modules:
        raise UnsupportedInModelLibraryFormatError(
            f"Don't know how to export non-c or non-llvm modules; found: {non_dso_modules!r}"
        )

    mod_indices = {"lib": 0, "src": 0}
    host_codegen_dir = os.path.join(codegen_dir, "host")
    lib_name = f"{module_name}_lib" if module_name else "lib"

    for dso_mod in dso_modules:
        if dso_mod.type_key == "c":
            index = mod_indices["src"]
            mod_indices["src"] += 1
            parent_dir = os.path.join(host_codegen_dir, "src")
            file_name = os.path.join(parent_dir, f"{lib_name}{index}.c")
        elif dso_mod.type_key == "llvm":
            index = mod_indices["lib"]
            mod_indices["lib"] += 1
            parent_dir = os.path.join(host_codegen_dir, "lib")
            file_name = os.path.join(parent_dir, f"{lib_name}{index}.o")
        else:
            assert (
                False
            ), f"do not expect module with type_key={mod.type_key} from _collect_dso_modules"

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
    num_targets = len(main_func_metadata.workspace_sizes.items())
    func_entries = []
    target_local_entries = dict()
    for i in range(num_targets):
        target = main_func_metadata.workspace_sizes.items()[i][0]
        device_max_workspace[target] = 0
        for func_name, finfo in function_metadata.items():
            if func_name == MAIN_FUNC_NAME_STR:
                continue
            target_local_entries[func_name] = list()

        for func_name, finfo in function_metadata.items():
            # Skip a few unsupported cases:
            # 1. The main function metadata is exported elsewhere.
            # 2. BYOC operator implementations do not currently export useful FunctionInfo.
            if func_name == MAIN_FUNC_NAME_STR or not finfo.tir_primfuncs:
                continue
            assert (
                len(finfo.constant_sizes.items()) == num_targets
            ), f"{func_name}: found {finfo.constant_sizes!r} vs {num_targets}"
            assert len(finfo.io_sizes.items()) == num_targets
            target = finfo.workspace_sizes.items()[i][0]
            workspace_size = finfo.workspace_sizes.items()[i][1]
            target_entry = {
                "device": int(target.kind.device_type),
                "workspace_size_bytes": int(workspace_size),
            }
            target_local_entries[func_name].append(target_entry)
            if workspace_size > device_max_workspace[target]:
                device_max_workspace[target] = workspace_size

    for func_name, target_entries_ in target_local_entries.items():
        func_entry = {
            "function_name": str(func_name),
            "workspace": target_entries_,
        }
        func_entries.append(func_entry)

    target_main_entries = list()
    for i in range(num_targets):
        target = main_func_metadata.workspace_sizes.items()[i][0]
        main_func_local_workspace = main_func_metadata.workspace_sizes.items()[i][1]
        main_func_constants = main_func_metadata.constant_sizes.items()[i][1]
        main_func_io = main_func_metadata.io_sizes.items()[i][1]
        target_main_entries.append(
            {
                "device": int(target.kind.device_type),
                "workspace_size_bytes": int(device_max_workspace[target])
                + int(main_func_local_workspace),
                "constants_size_bytes": int(main_func_constants),
                "io_size_bytes": int(main_func_io),
            }
        )

    ret = {
        "operator_functions": func_entries,
        "main": target_main_entries,
    }
    return ret


def _get_main_relay_func(mod: executor_factory.ExecutorFactoryModule):
    main_func = mod.function_metadata[MAIN_FUNC_NAME_STR]
    target = list(main_func.relay_primfuncs.keys())[0]
    return main_func.relay_primfuncs[target]


def _convert_tuple_to_outputs(ret_type, offset=0):
    outputs = []
    added_fields = len(ret_type.fields)
    for output_index in range(added_fields):
        next_output = offset + len(outputs)
        if isinstance(ret_type.fields[output_index], TupleType):
            outputs.extend(_convert_tuple_to_outputs(ret_type.fields[output_index], next_output))
        else:
            outputs.append(f"output{next_output}")
    return outputs


def _get_inputs_and_outputs_from_module(mod):
    main_func = _get_main_relay_func(mod)
    inputs = [argument.name_hint for argument in main_func.params]

    outputs = ["output"]
    if isinstance(main_func.ret_type, TupleType):
        outputs = _convert_tuple_to_outputs(main_func.ret_type)

    return inputs, outputs


def _should_generate_interface_header(mod):
    return any(target.attrs.get("interface-api") == "c" for target in mod.target.values())


def _make_tar(source_dir, tar_file_path):
    """Build a tar file from source_dir."""
    with tarfile.open(tar_file_path, "w") as tar_f:

        def reset(tarinfo):
            tarinfo.uid = tarinfo.gid = 0
            tarinfo.uname = tarinfo.gname = "root"
            return tarinfo

        tar_f.add(str(source_dir), arcname=".", filter=reset)


_GENERATED_VERSION = 5


def _export_graph_model_library_format(
    mod: executor_factory.ExecutorFactoryModule, tempdir: pathlib.Path
):
    """Export a tvm.relay.build artifact in Model Library Format.

    Parameters
    ----------
    mod : tvm.relay.backend.executor_factory.ExecutorFactoryModule
        The return value of tvm.relay.build, which will be exported into Model Library Format.
    tempdir : pathlib.Path
        Temporary directory to populate with Model Library Format contents.
    """
    is_aot = isinstance(mod, executor_factory.AOTExecutorFactoryModule)
    executor = ["aot"] if is_aot else ["graph"]

    metadata = {
        "version": _GENERATED_VERSION,
        "model_name": mod.libmod_name,
        "export_datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%SZ"),
        "memory": _build_memory_map(mod),
        "target": {int(k): str(v) for k, v in mod.target.items()},
        "executors": executor,
        "style": "full-model",
    }

    with open(tempdir / "metadata.json", "w") as json_f:
        json.dump(metadata, json_f, indent=2, sort_keys=True)

    codegen_dir = tempdir / "codegen"
    codegen_dir.mkdir()
    _populate_codegen_dir(mod.lib, codegen_dir, mod.libmod_name)

    if _should_generate_interface_header(mod):
        include_path = codegen_dir / "host" / "include"
        include_path.mkdir()
        inputs, outputs = _get_inputs_and_outputs_from_module(mod)
        generate_c_interface_header(mod.libmod_name, inputs, outputs, include_path)

    parameters_dir = tempdir / "parameters"
    parameters_dir.mkdir()
    param_filename = parameters_dir / f"{mod.libmod_name}.params"
    with open(param_filename, "wb") as f:
        f.write(param_dict.save_param_dict(mod.params))

    src_dir = tempdir / "src"
    src_dir.mkdir()
    with open(src_dir / "relay.txt", "w") as f:
        f.write(str(mod.ir_mod))

    if not is_aot:
        graph_config_dir = tempdir / "executor-config" / "graph"
        graph_config_dir.mkdir(parents=True)
        with open(graph_config_dir / "graph.json", "w") as f:
            f.write(mod.get_executor_config())


class NonStaticShapeError(Exception):
    """Raised when a shape has elements other than IntImm."""


def _shape_to_size(shape, dtype):
    bits_per_item = int(
        re.match(r"((float)|(int))(?P<width_bits>[0-9]+)", dtype).group("width_bits")
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
    for target_device_type, target in targets.items():
        ir_mod = ir_module_by_target[target]
        printer = get_global_func("tir.ModelLibraryFormatPrinter")(False, None, False)
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
    args : list of Buffer or Tensor or Var, optional
        The args supplied to tvm.build().
    file_name : str
        Path to the .tar archive to generate.
    """
    targets = {}
    for target in mod.ir_module_by_target.keys():
        if str(target.kind) not in ("llvm", "c"):
            raise UnsupportedInModelLibraryFormatError(
                f"Operator has non-DSO-exportable target {target!s}, which is not yet supported in "
                "Model Library Format"
            )

        targets[int(_nd.device(str(target)).device_type)] = target

    src_dir = tempdir / "src"
    src_dir.mkdir()
    memory_map = _write_tir_and_build_operator_memory_map(src_dir, targets, mod.ir_module_by_target)

    metadata = {
        "version": _GENERATED_VERSION,
        "model_name": mod.name,
        "export_datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%SZ"),
        "memory": memory_map,
        "target": {k: str(v) for k, v in targets.items()},
        "executors": [],
        "style": "operator",
    }
    with open(tempdir / "metadata.json", "w") as metadata_f:
        json.dump(metadata, metadata_f)

    codegen_dir = tempdir / "codegen"
    codegen_dir.mkdir()
    _populate_codegen_dir(mod, codegen_dir)


ExportableModule = typing.Union[
    build_module.OperatorModule,
    executor_factory.AOTExecutorFactoryModule,
    executor_factory.GraphExecutorFactoryModule,
]


def export_model_library_format(mod: ExportableModule, file_name: typing.Union[str, pathlib.Path]):
    """Export the build artifact in Model Library Format.

    This function creates a .tar archive containing the build artifacts in a standardized
    layout. It's intended to allow downstream automation to build TVM artifacts against the C
    runtime.

    Parameters
    ----------
    mod : ExportableModule
        The return value of tvm.build or tvm.relay.build.
    file_name : str
        Path to the .tar archive to generate.

    Returns
    -------
    file_name : str
        The path to the generated .tar archive.
    """
    file_name = pathlib.Path(file_name)

    tempdir = utils.tempdir()

    if isinstance(mod, build_module.OperatorModule):
        _export_operator_model_library_format(mod, tempdir.path)
    elif isinstance(
        mod,
        (executor_factory.AOTExecutorFactoryModule, executor_factory.GraphExecutorFactoryModule),
    ):
        _export_graph_model_library_format(mod, tempdir.path)
    else:
        raise NotImplementedError(f"Don't know how to export module of type {mod.__class__!r}")

    _make_tar(tempdir.path, file_name)

    return file_name
