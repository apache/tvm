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
# pylint: disable=invalid-name
"""Driver for partitioning and building a Relay module for CUTLASS offload."""
import os
import tvm
from tvm import runtime, relay
from .gen_gemm import CutlassGemmProfiler


def _get_cutlass_path():
    tvm_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../../")
    cutlass_path = os.path.join(tvm_root, "3rdparty/cutlass")
    assert os.path.exists(cutlass_path), "The CUTLASS root directory not found in {}".format(
        cutlass_path
    )
    return cutlass_path


class GemmAnnotator(tvm.relay.ExprVisitor):
    """Annotates partitioned functions with shape and dtype information."""

    def __init__(self):
        super().__init__()
        self.signature = {}

    def visit_call(self, call):
        op = call.op
        if isinstance(op, relay.Function) and "PartitionedFromPattern" in op.attrs:
            self.signature["op_type"] = op.attrs["Composite"]
            for i, arg in enumerate(op.params):
                self.signature["arg%d_shape" % i] = arg.checked_type.shape
                self.signature["arg%d_dtype" % i] = arg.checked_type.dtype
            self.signature["ret_shape"] = op.ret_type.shape
            self.signature["ret_dtype"] = op.ret_type.dtype


def tune_cutlass_kernels(mod, sm, profile_all=True, use_multiprocessing=False, tmp_dir="./tmp"):
    """Given a module partitioned for CUTLASS offloading, profile each workload to select which
    kernels to emit.

    Parameters
    ----------
    mod : IRModule
        The Relay module with cutlass partitions.

    sm : int
        An integer specifying the compute capability. For example, 75 for Turing and
        80 or 86 for Ampere.

    profile_all : bool
        Whether or not profile all candidate kernels, or stop profiling after
        the first applicable kernel is found.

    use_multiprocessing : bool
        Whether or not compile profiler executables for different kernels in parallel.

    tmp_dir : string, optional
        A temporary directory where intermediate compiled artifacts will be stored.

    Returns
    -------
    mod : IRModule
        The updated module annotated with cutlass profiling information.

    num_cutlass_partition : int
        The number of partitioned functions created for CUTLASS.
    """
    cutlass_profiler = CutlassGemmProfiler(sm, _get_cutlass_path(), tmp_dir)
    num_cutlass_partition = 0
    for var in mod.get_global_vars():
        fun_name = var.name_hint
        func = mod[fun_name]
        annotator = GemmAnnotator()
        if "cutlass" in fun_name:
            num_cutlass_partition += 1
            annotator.visit(func)
            # call cutlass profiler to find best settings, update attr
            new_attrs = {}
            new_attrs.update(annotator.signature)
            for key in func.attrs.keys():
                new_attrs[key] = func.attrs[key]
            # call profiler
            arg0_shape = new_attrs["arg0_shape"]
            arg1_shape = new_attrs["arg1_shape"]
            MM = arg0_shape[0]
            KK = arg0_shape[1]
            NN = arg1_shape[0]
            out = cutlass_profiler.profile(
                MM, NN, KK, annotator.signature["ret_dtype"], profile_all, use_multiprocessing
            )
            if new_attrs["op_type"] == "cutlass.dense":
                new_attrs["cutlass_op_def"] = out["opdef"]
            elif new_attrs["op_type"] == "cutlass.dense_bias":
                new_attrs["cutlass_op_def"] = out["opdef_bias"]
            elif new_attrs["op_type"] == "cutlass.dense_bias_relu":
                new_attrs["cutlass_op_def"] = out["opdef_bias_relu"]
            elif "cutlass.dense_bias_gelu" in new_attrs["op_type"]:
                new_attrs["cutlass_op_def"] = out["opdef_bias_gelu"]
            else:
                raise ValueError("%s pattern is not implemented." % new_attrs["op_type"])
            new_attrs["cutlass_op_name"] = out["name"]

            print("The best kernel is " + new_attrs["cutlass_op_name"])
            if new_attrs["cutlass_op_name"].find("_tn_align") > 0:
                new_attrs["lda"] = "K"
                new_attrs["ldb"] = "K"
                new_attrs["ldc"] = "N"
            elif new_attrs["cutlass_op_name"].find("_nt_align") > 0:
                new_attrs["lda"] = "M"
                new_attrs["ldb"] = "N"
                new_attrs["ldc"] = "N"
            else:
                raise ValueError("%s unsupported operation" % new_attrs["cutlass_op_name"])
            new_attrs = tvm.ir.make_node("DictAttrs", **new_attrs)
            new_func = relay.Function(
                func.params,
                func.body,
                ret_type=func.ret_type,
                type_params=func.type_params,
                attrs=new_attrs,
            )
            mod.update_func(var, new_func)

    return mod, num_cutlass_partition


def build_cutlass_kernels(lib, sm, tmp_dir="./tmp", lib_path="compile.so"):
    """Compile CUTLASS kernels in lib and return the runtime module ready to run.

    Parameters
    ----------
    lib : GraphExecutorFactoryModule
        The output from relay.build containing compiled host code and non-cutlass kernels.

    sm : int
        An integer specifying the compute capability. For example, 75 for Turing and
        80 or 86 for Ampere.

    tmp_dir : string, optional
        A temporary directory where intermediate compiled artifacts will be stored.

    lib_path : string, optional
        The path to a shared library which will be generated as the result of the build  process

    Returns
    -------
    updated_lib : runtime.Module
        The updated module with compiled cutlass kernels.
    """
    cutlass_root = _get_cutlass_path()
    cutlass_include = os.path.join(cutlass_root, "include")
    cutlass_util_include = os.path.join(cutlass_root, "tools/util/include")

    kwargs = {}
    kwargs["cc"] = "nvcc"
    kwargs["options"] = [
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-gencode=arch=compute_%d,code=[sm_%d,compute_%d]" % (sm, sm, sm),
        "-Xcompiler=-fPIC",
        "-Xcompiler=-Wconversion",
        "-Xcompiler=-fno-strict-aliasing",
        "-O3",
        "-std=c++14",
        "-I" + cutlass_include,
        "-I" + cutlass_util_include,
    ]
    lib.export_library(lib_path, workspace_dir=tmp_dir, **kwargs)
    return runtime.load_module(lib_path)
