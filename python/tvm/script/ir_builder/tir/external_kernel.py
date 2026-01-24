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
"""External kernel integration fro TIR"""
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from tvm import __version__ as tvm_version
from tvm import tir
from tvm.contrib import nvcc
from tvm.runtime import Module, const, load_module


class BaseKernel:  # pylint: disable=too-few-public-methods
    """Base class for external kernels."""

    def compile_to_device_module(
        self, launch_args, *args, **kwargs
    ) -> Tuple[str, Module, List[Any]]:
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
        self, binary_data, kernel_arg_types, launch_param_tags, kernel_name, fmt="ptx"
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
        with tempfile.TemporaryDirectory() as temp_dir:
            binary_path = f"{temp_dir}/{kernel_name}.{fmt}"
            if fmt == "ptx":
                with open(binary_path, "w") as f:
                    f.write(binary_data)
            else:
                with open(binary_path, "wb") as f:
                    f.write(binary_data)
            with open(f"{temp_dir}/{kernel_name}.tvm_meta.json", "w") as f:
                f.write(tvm_metadata)
            kernel_module = load_module(binary_path)
        return kernel_module


class SourceKernel(BaseKernel):  # pylint: disable=too-few-public-methods
    """A kernel from source code."""

    def __init__(self, source_code: str):
        self.source_code = source_code

    def compile_to_device_module(  # pylint: disable=arguments-differ
        self,
        grid: List[List[Union[int, tir.PrimExpr]]],
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> Tuple[str, Module, List[Any]]:
        """Compile the kernel to a device module."""
        from tvm.relax.frontend.nn import (  # pylint: disable=import-outside-toplevel
            SourceModule,
        )

        kernel_name = kwargs["kernel_name"]
        assert len(grid) == 2, (
            "grid should be two list of integers, representing the dimension of "
            "['blockIdx.x', 'blockIdx.y', 'blockIdx.z'] and "
            "['threadIdx.x', 'threadIdx.y', 'threadIdx.z']"
        )
        assert isinstance(grid[0], (list, tuple)) and isinstance(grid[1], (list, tuple))
        launch_param_tags = ["blockIdx.x", "blockIdx.y", "blockIdx.z"][: len(grid[0])] + [
            "threadIdx.x",
            "threadIdx.y",
            "threadIdx.z",
        ][: len(grid[1])]
        runtime_args = [arg if hasattr(arg, "dtype") else const(arg) for arg in args]
        kernel_arg_types = [arg.dtype for arg in runtime_args]
        runtime_args = runtime_args + list(grid[0]) + list(grid[1])

        # Reuse compilation path from SourceModule
        compile_options = SourceModule.get_compile_options("cu")
        source_code = self.source_code
        try:
            source_path = Path(source_code)
            if source_path.is_file():
                with open(source_path, "r") as f:
                    source_code = f.read()
        except:  # pylint: disable=bare-except
            pass

        with tempfile.TemporaryDirectory() as temp_dir:
            # Check if NVSHMEM is used - requires cubin output for device library linking
            use_nvshmem = (
                "#include <nvshmem.h>" in source_code or "#include <nvshmemx.h>" in source_code
            )
            target_format = "cubin" if use_nvshmem else "ptx"
            output_path = f"{temp_dir}/{kernel_name}.{target_format}"

            compiler = os.environ.get("TVM_CUDA_COMPILE_MODE", "nvcc")
            nvcc.compile_cuda(
                source_code,
                target_format=target_format,
                options=compile_options,
                path_target=output_path,
                compiler=compiler,
            )

            if target_format == "ptx":
                with open(output_path, "r") as f:
                    binary_data = f.read()
            else:
                with open(output_path, "rb") as f:
                    binary_data = f.read()

            kernel_module = self._create_cuda_module(
                binary_data, kernel_arg_types, launch_param_tags, kernel_name, fmt=target_format
            )

        return kernel_name, kernel_module, runtime_args


def call_kernel(
    kernel,
    launch_args: List[Union[int, tir.PrimExpr, List[Union[int, tir.PrimExpr]]]],
    *args: List[Any],
    **kwargs: Dict[str, Any],
):
    """
    Call an external kernel.

    Parameters
    ----------
    kernel : Any
        The external kernel to call.

    launch_args : List[Union[int, tir.PrimExpr, List[Union[int, tir.PrimExpr]]]]
        The launch arguments. A list of integers for grid size, block size, and shared memory size.
        The actual requirements depend on the kernel.

    args : List[tir.PrimExpr]
        The arguments to pass to the kernel.

    kwargs : Dict[str, Any]
        Additional keyword arguments to pass to the kernel or compilation.
    """
    from ..ir import (  # pylint: disable=import-outside-toplevel
        module_get_attr,
        module_set_attr,
    )
    from .ir import call_packed  # pylint: disable=import-outside-toplevel

    kernel_type = f"{type(kernel).__module__}.{type(kernel).__qualname__}"
    if kernel_type == "triton.runtime.jit.JITFunction":
        from .triton import TritonKernel  # pylint: disable=import-outside-toplevel

        kernel = TritonKernel(kernel)
    elif kernel_type == "builtins.str":
        kernel = SourceKernel(kernel)
    else:
        raise ValueError("Unsupported kernel type {}".format(kernel_type))

    kernel_name, kernel_module, runtime_args = kernel.compile_to_device_module(
        launch_args, *args, **kwargs
    )

    # Attach the kernel module to the current IRModule
    external_mods: List[Module] = module_get_attr("external_mods") or []
    kernel_exists = any([mod.implements_function(kernel_name) for mod in external_mods])
    if kernel_exists:
        logging.debug("Kernel %s already exists in the IRModule", kernel_name)
    else:
        external_mods.append(kernel_module)
        module_set_attr("external_mods", external_mods, True)
    return call_packed(kernel_name, *runtime_args)
