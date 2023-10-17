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
# pylint: disable=invalid-name, unused-wildcard-import, wildcard-import, pointless-exception-statement
"""Generator for ComposableKernel GEMM kernels."""
import os
import subprocess
import operator
from functools import reduce
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import library
from .generator_utils import *
from .gemm_template import *
from .kernel_factory.gemm import get_instances
from .compile_utils import get_composable_kernel_path


class GemmKernelGenerator:
    def __init__(self, call, func, options={}):
        self.call = call
        self.func = func
        self.options = options

        # Gemm attributes and argument
        self.attributes = None
        self.argument = None
        self.lhs_arg = None
        self.rhs_arg = None
        self.init_gemm_attrs()

    def init_gemm_attrs(self):
        op_type = self.func.attrs["Composite"]

        sig = extract_relax_function_signature(self.func)
        arg_idx = extract_arg_idx(op_type, self.func)

        lhs_arg_idx = arg_idx["lhs"]
        rhs_arg_idx = arg_idx["rhs"]

        lhs_arg = f"arg{lhs_arg_idx}"
        rhs_arg = f"arg{rhs_arg_idx}"

        lhs_shape = sig[f"{lhs_arg}_shape"]
        rhs_shape = sig[f"{rhs_arg}_shape"]
        out_shape = sig["ret_shape"]

        y_transposed = "transposed" in op_type

        self.attributes = GemmOpAttributes(
            a_data_type=dtype_map[sig[f"{lhs_arg}_dtype"]],
            b_data_type=dtype_map[sig[f"{rhs_arg}_dtype"]],
            c_data_type=dtype_map[sig["ret_dtype"]],
            a_layout=library.LayoutType.RowMajor,
            b_layout=library.LayoutType.ColumnMajor
            if y_transposed
            else library.LayoutType.RowMajor,
            c_layout=library.LayoutType.RowMajor,
            a_element_op=library.TensorOperation.PassThrough,
            b_element_op=library.TensorOperation.PassThrough,
            c_element_op=library.TensorOperation.PassThrough,
        )

        M = lhs_shape[-2]
        K = lhs_shape[-1]
        N = rhs_shape[-2] if y_transposed else rhs_shape[-1]

        lhs_batches = reduce(operator.mul, lhs_shape[:-2], 1)
        rhs_batches = reduce(operator.mul, rhs_shape[:-2], 1)
        if lhs_batches == 1 and rhs_batches == 1:
            B = 1
        else:
            B = lhs_batches if rhs_batches == 1 else rhs_batches

        self.argument = GemmOpArgument(
            M=M,
            N=N,
            K=K,
            B=B,
            a_stride=K,
            b_stride=K if y_transposed else N,
            c_stride=N,
            a_batch_stride=M * K,
            b_batch_stride=N * K,
            c_batch_stride=M * N,
            batched=B > 1,
        )

        self.lhs_arg = self.call.args[lhs_arg_idx]
        self.rhs_arg = self.call.args[rhs_arg_idx]

    def tune(self):
        instances = get_instances(
            self.attributes.a_data_type,
            self.attributes.b_data_type,
            self.attributes.c_data_type,
            self.attributes.a_layout,
            self.attributes.b_layout,
            self.attributes.c_layout,
            self.argument.B > 1,
        )

        assert len(instances) > 0

        # Prepare compiling environment
        tmp_dir = self.options.get("tmp_dir", "./tmp")
        instance_dir = os.path.join(tmp_dir, "instnaces")
        os.makedirs(instance_dir, exist_ok=True)
        composable_kernel_include_path = os.path.join(get_composable_kernel_path(), "include")

        # 1. Emit C++ source for gemm instance and profiler
        emitter = GemmInstanceEmitter()

        # emit header file for gemm instances
        instance_header_path = os.path.join(instance_dir, "GemmInstances.h")
        with open(instance_header_path, "w") as output_file:
            src = emitter.emit_header(self.attributes, instances)
            output_file.write(src)

        # emit .cc file for each gemm instance
        instance_cc_paths = []
        for instance in instances:
            instance_cc_path = os.path.join(instance_dir, str(instance) + ".cc")
            instance_cc_paths.append(instance_cc_path)
            with open(instance_cc_path, "w") as output_file:
                src = emitter.emit_cc(
                    instance,
                    str(Path(instance_header_path).relative_to(instance_dir)),
                )
                output_file.write(src)

        # emit .cc file for gemm profiler
        profiler_name = "ckGemmProfiler"
        profiler_cc_path = os.path.join(tmp_dir, f"{profiler_name}.cc")
        with open(profiler_cc_path, "w") as output_file:
            emitter = GemmProfilerEmitter()
            src = emitter.emit(
                self.argument,
                str(Path(instance_header_path).relative_to(tmp_dir)),
                instances,
            )
            output_file.write(src)

        # 2. Compile object files for gemm instances and profiler
        object_paths = []
        with ThreadPoolExecutor(max_workers=64) as executor:

            def compile_object(path):
                object_path = str(Path(path).with_suffix(".o"))
                object_paths.append(object_path)
                # TODO(tiandi)
                if (path != profiler_cc_path) and os.path.isfile(object_path):
                    return
                subprocess.check_call(
                    f"hipcc {path} -c -std=c++17 -L/usr/local/lib/ -I{composable_kernel_include_path} -o {object_path}",
                    shell=True,
                )

            compile_results = executor.map(compile_object, instance_cc_paths + [profiler_cc_path])
            for result in compile_results:
                pass

        # 3. Link object files of gemm instances and profiler
        profiler_bin_path = os.path.join(tmp_dir, f"{profiler_name}")
        subprocess.check_call(f"hipcc {' '.join(object_paths)} -o {profiler_bin_path}", shell=True)

        # 4. Run profiler to get the best instance
        output = subprocess.check_output(profiler_bin_path, shell=True)
        best_inst_idx = int(output.splitlines()[-1])
        assert best_inst_idx >= 0
        return instances[best_inst_idx]

    def generate(self, instance: gemm.GemmOperation) -> CodegenResult:
        emitter = GemmOpEmitter()
        code = emitter.emit(
            self.lhs_arg,
            self.rhs_arg,
            self.attributes,
            self.argument,
            instance,
        )
        headers = []
        headers.append("iostream")
        headers.append("ck/ck.hpp")
        headers.append(instance.header())

        return CodegenResult(code, headers)

    def run(self) -> CodegenResult:
        instance = self.tune()
        return self.generate(instance)
