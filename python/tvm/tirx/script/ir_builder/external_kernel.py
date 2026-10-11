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
# ruff: noqa: E722
"""External kernel integration fro TIR"""

import json
import logging
from pathlib import Path
from typing import Any

import tvm_ffi

from tvm import __version__ as tvm_version
from tvm import libinfo, tirx
from tvm.backend.config import backend_config_json, merge_backend_configs
from tvm.ir import Expr, PointerType, const, is_prim_expr
from tvm.runtime import Module


class BaseKernel:  # pylint: disable=too-few-public-methods
    """Base class for external kernels."""

    def compile_to_device_module(
        self, launch_args, *args, **kwargs
    ) -> tuple[str, Module, list[Any]]:
        """Compile the kernel to a device module."""
        raise NotImplementedError()

    def _format_tvm_module_metadata(self, kernel_name, arg_types, launch_param_tags):
        """Format the TVM module metadata."""
        tvm_metadata = """{{
            "tvm_version": "{version}",
            "func_info": {{
                "{kernel_name}": {{
                    "name": "",
                    "arg_types": {arg_types},
                    "launch_param_tags": {launch_param_tags}
                }}
            }}
        }}""".format_map(
            {
                "version": tvm_version,
                "kernel_name": kernel_name,
                "arg_types": json.dumps(arg_types),
                "launch_param_tags": json.dumps(launch_param_tags),
            }
        )
        return tvm_metadata

    def _create_cuda_module(
        self,
        binary_data,
        kernel_arg_types,
        launch_param_tags,
        kernel_name,
        fmt="ptx",
        backend_config=None,
    ):
        """
        Create a CUDA module from compiled binary (PTX or cubin) and metadata.

        Parameters
        ----------
        binary_data : str or bytes
            The compiled binary data (PTX as str, cubin as bytes).

        kernel_arg_types : List[str]
            The types of the kernel arguments.

        launch_param_tags : List[str]
            The tags of the launch parameters.

        kernel_name : str
            The name of the kernel.

        fmt : str
            The format of the binary data: "ptx" or "cubin".

        Returns
        -------
        kernel_module : Module
            The CUDA module.
        """
        tvm_metadata = self._format_tvm_module_metadata(
            kernel_name, kernel_arg_types, launch_param_tags
        )
        # Build the FunctionInfo map in-memory from the JSON metadata, then
        # construct the CUDA module via the FFI registry without going to
        # disk.  Avoids the load_from_file dispatch path entirely.
        if isinstance(binary_data, str):
            binary_bytes = binary_data.encode("utf-8")
        else:
            binary_bytes = bytes(binary_data)
        load_meta = tvm_ffi.get_global_func("runtime.LoadMetaDataFromJSON")
        fmap = load_meta(tvm_metadata)
        create_cuda = tvm_ffi.get_global_func("ffi.Module.create.cuda")
        source = {"backend_config": backend_config_json(backend_config)} if backend_config else {}
        kernel_module = create_cuda(binary_bytes, fmt, fmap, source)
        return kernel_module


class SourceKernel(BaseKernel):  # pylint: disable=too-few-public-methods
    """A kernel from source code."""

    def __init__(self, source_code: str):
        self.source_code = source_code

    def compile_to_device_module(  # pylint: disable=arguments-differ
        self,
        grid: list[list[int | tirx.Expr]],
        *args: list[Any],
        **kwargs: dict[str, Any],
    ) -> tuple[str, Module, list[Any]]:
        """Compile the kernel to a device module."""
        kernel_name = kwargs["kernel_name"]
        assert len(grid) == 2, (
            "grid should be two list of integers, representing the dimension of "
            "['blockIdx.x', 'blockIdx.y', 'blockIdx.z'] and "
            "['threadIdx.x', 'threadIdx.y', 'threadIdx.z']"
        )
        assert isinstance(grid[0], list | tuple) and isinstance(grid[1], list | tuple)
        launch_param_tags = ["blockIdx.x", "blockIdx.y", "blockIdx.z"][: len(grid[0])] + [
            "threadIdx.x",
            "threadIdx.y",
            "threadIdx.z",
        ][: len(grid[1])]
        runtime_args = [arg if isinstance(arg, Expr) else const(arg) for arg in args]
        kernel_arg_types = []
        for arg in runtime_args:
            if isinstance(arg.ty, PointerType):
                kernel_arg_types.append("handle")
            else:
                assert is_prim_expr(arg)
                kernel_arg_types.append(str(arg.ty.dtype))
        runtime_args = runtime_args + list(grid[0]) + list(grid[1])

        include_paths = [
            Path(libinfo.find_include_path()),
            Path(tvm_ffi.libinfo.find_include_path()),
            Path(tvm_ffi.libinfo.find_dlpack_include_path()),
        ]
        for include_path in dict.fromkeys(include_paths):
            if not include_path.is_dir():
                raise ValueError(f"Not an include directory: {include_path}")
        source_code = self.source_code
        try:
            source_path = Path(source_code)
            if source_path.is_file():
                with open(source_path) as f:
                    source_code = f.read()
        except:  # pylint: disable=bare-except
            pass

        from tvm.backend.cuda.compiler import compile_source

        flags = ["--use_fast_math", "--std=c++17"]
        config = merge_backend_configs(
            {"cuda": {"nvcc": flags, "nvrtc": flags}}, kwargs.get("backend_config")
        )
        # These headers are part of the external-kernel adapter, independent of
        # whether the caller replaces the native toolchain argument lists.
        for compiler in ("nvcc", "nvrtc"):
            config["cuda"][compiler].extend(f"-I{p}" for p in include_paths)
        result = compile_source(source_code, config)
        kernel_module = self._create_cuda_module(
            result.binary,
            kernel_arg_types,
            launch_param_tags,
            kernel_name,
            fmt=result.target_format,
            backend_config=result.config,
        )

        return kernel_name, kernel_module, runtime_args


def call_kernel(
    kernel,
    launch_args: list[int | tirx.Expr | list[int | tirx.Expr]],
    *args: list[Any],
    **kwargs: dict[str, Any],
):
    """
    Call an external kernel.

    Parameters
    ----------
    kernel : Any
        The external kernel to call.

    launch_args : List[Union[int, tirx.Expr, List[Union[int, tirx.Expr]]]]
        The launch arguments. A list of integers for grid size, block size, and shared memory size.
        The actual requirements depend on the kernel.

    args : List[tirx.Expr]
        The arguments to pass to the kernel.

    kwargs : Dict[str, Any]
        Additional keyword arguments to pass to the kernel or compilation.
    """
    from tvm.script.ir_builder.ir import (  # pylint: disable=import-outside-toplevel
        module_get_attr,
        module_set_attr,
    )

    from .op import call_packed  # pylint: disable=import-outside-toplevel

    kernel_type = f"{type(kernel).__module__}.{type(kernel).__qualname__}"
    if kernel_type == "triton.runtime.jit.JITFunction":
        from .triton import TritonKernel  # pylint: disable=import-outside-toplevel

        kernel = TritonKernel(kernel)
    elif kernel_type == "builtins.str":
        kernel = SourceKernel(kernel)
    else:
        raise ValueError(f"Unsupported kernel type {kernel_type}")

    kernel_name, kernel_module, runtime_args = kernel.compile_to_device_module(
        launch_args, *args, **kwargs
    )

    # Attach the kernel module to the current IRModule
    external_mods: list[Module] = module_get_attr("external_mods") or []
    kernel_exists = any([mod.implements_function(kernel_name) for mod in external_mods])
    if kernel_exists:
        logging.debug("Kernel %s already exists in the IRModule", kernel_name)
    else:
        external_mods.append(kernel_module)
        module_set_attr("external_mods", external_mods, True)
    return call_packed(kernel_name, *runtime_args)
