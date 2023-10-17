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
# pylint: disable=invalid-name, dangerous-default-value, arguments-differ
"""Driver for partitioning and building a Relay module for ComposableKernel offload."""
import os
import logging

import tvm

from tvm import relax
from tvm._ffi.registry import register_func
from tvm.topi.utils import get_const_tuple

from .library import LayoutType
from .gemm_generator import GemmKernelGenerator
from .compile_utils import get_composable_kernel_path


logger = logging.getLogger("composable_kernel")


@tvm._ffi.register_func("contrib.composable_kernel.generate_kernel")
def generate_kernel(call, func):
    """Return ComposableKernel host code based on a template and the provided annotations.

    Parameters
    ----------
    func_name: str
        A string to identify the type of the kernel (dense/matmul, batched_matmul, or conv2d).

    annotations: container.Map
        Key and value pairs annotated during kernel selection.

    func_args: list
        Names of the function arguments.

    Returns
    -------
    codegen_result : CodegenResult
        Generated ComposableKernel host code and required header-file names.
    """
    func_name = func.attrs["Composite"]
    if "matmul" in func_name:
        generator = GemmKernelGenerator(call, func)
    return generator.run()


def _get_composable_kernel_compile_options():
    composable_kernel_root = get_composable_kernel_path()
    composable_kernel_include = os.path.join(composable_kernel_root, "include")

    kwargs = {}
    kwargs["cc"] = "hipcc"
    kwargs["options"] = ["-c", "-std=c++17", f"-I{composable_kernel_include}"]
    return kwargs


@register_func("contrib.composable_kernel.compile")
def compile_composable_kernel_module(c_source_module, options):
    """Compile all ComposableKerel kernels in the given C-source module.

    Parameters
    ----------
    c_source_module: runtime.Module
        A C-source module containing ComposableKerel kernels.

    options: dict
        Compilation options. Currently recognizes
          "sm": The target architecture (compute capability), for example 75 or 80 (default: 80)
          "threads": The number of threads to use in NVCC parallel compilation (default:
          use all logical cores)
          "use_fast_math": Whether or not to use faster but approximate arithmetic in some
          ComposableKerel epilogues (default: False)

    Returns
    -------
    rt_mod : runtime.Module
        A runtime module where all ComposableKerel kernels have been compiled.
    """
    tmp_dir = options.get("tmp_dir", "./tmp")

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    function_names = c_source_module.get_function("get_func_names")()
    compile_options = _get_composable_kernel_compile_options()
    lib_path = os.path.join(tmp_dir, "composable_kernel.o")
    logger.info("Compiling generated ComposableKerel code")
    c_source_module.export_library(lib_path, workspace_dir=tmp_dir, **compile_options)

    # Recover static library
    return tvm.runtime.load_static_library(lib_path, function_names)
