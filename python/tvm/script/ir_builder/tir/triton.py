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
"""Triton kernel integration with TIR"""

from typing import Tuple, List, Union, Any, Dict

import triton
from triton.runtime.jit import type_canonicalisation_dict
from tvm import tir
from tvm.topi.utils import get_const_int
from tvm.runtime import Module
from .external_kernel import BaseKernel


class TritonKernel(BaseKernel):
    """A kernel from Triton JIT function.

    This class bridges the Triton kernel with TVM runtime. The compilation includes the following
    steps:
        - Deduce the kernel signature and generate the Triton kernel
        - Embed the compiled kernel into the current IRModule as an external module
        - Generate a call to the Triton kernel following its calling convention via call_packed.
    """

    def __init__(self, func):
        self.func = func

    def compile_to_device_module(
        self,
        launch_args: List[Union[int, tir.PrimExpr]],
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> Tuple[str, Module, List[Any]]:
        """Compile the kernel to a device module.

        Parameters
        ----------
        launch_args : List[int]
            The grid size of the kernel. A list of one to three expressions, representing the number
            of
            "blockIdx.x", "blockIdx.y", and "blockIdx.z" respectively.

        args : List[Any]
            Arguments to the kernel function.

        kwargs : Dict[str, Any]
            Additional options for the kernel compilation.
        """
        triton_kernel, kernel_args = self._generate_triton_kernel(self.func, *args, **kwargs)
        kernel_metadata = triton_kernel.metadata
        ptx = triton_kernel.asm["ptx"]
        assert kernel_metadata.num_ctas == 1, "Cluster is not supported"
        num_warps = kernel_metadata.num_warps
        grid = launch_args
        launch_param_tags = ["threadIdx.x"] + ["blockIdx.x", "blockIdx.y", "blockIdx.z"][
            : len(grid)
        ]
        launch_args = [num_warps * 32] + list(grid)
        kernel_arg_types = [arg.dtype for arg in kernel_args]
        if triton_kernel.metadata.shared > 0:
            # Add shared memory size to the launch arguments
            launch_param_tags.append("tir.use_dyn_shared_memory")
            launch_args.append(triton_kernel.metadata.shared)

        kernel_module = self._create_cuda_module(
            ptx, kernel_arg_types, launch_param_tags, triton_kernel.name
        )

        return triton_kernel.name, kernel_module, kernel_args + launch_args

    def _generate_triton_kernel(
        self, func, *args, **kwargs
    ) -> Tuple["triton.compiler.CompiledKernel", List[tir.PrimExpr]]:
        """Deduce the kernel signature and generate the Triton kernel"""

        kernel_params = func.params
        assert len(kernel_params) == len(
            args
        ), f"Number of arguments does not match, expected {len(kernel_params)}, got {len(args)}"

        signature = {}
        constants = {}
        kernel_args = []  # Arguments to invoke the kernel
        for i, arg in enumerate(args):
            if kernel_params[i].is_constexpr:
                constants[kernel_params[i].name] = get_const_int(arg)
                continue
            if arg.dtype == "handle":
                assert isinstance(arg, tir.Var)
                elem_type = arg.type_annotation.element_type.dtype
                pointer_type = "*" + type_canonicalisation_dict[elem_type]
                signature[kernel_params[i].name] = pointer_type
            else:
                signature[kernel_params[i].name] = type_canonicalisation_dict[arg.dtype]
            kernel_args.append(arg)

        # TODO: Support default argument in the kernel
        # TODO: Add specialization for aligned buffer pointers
        source = triton.compiler.ASTSource(fn=func, constants=constants, signature=signature)
        compiled = triton.compiler.compile(source, options=kwargs)
        return compiled, kernel_args
