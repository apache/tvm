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
import os
import tvm
from tvm import runtime, relay
from tvm.relay.op.contrib.cutlass import partition_for_cutlass
from . import gen_gemm


class GemmCollector(tvm.relay.ExprVisitor):
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


def profile_and_build(mod, params, sm):
    cutlass_profiler = gen_gemm.CutlassGemmProfiler(sm, "../../../3rdparty/cutlass", "./temp")
    mod = partition_for_cutlass(mod)
    for var in mod.get_global_vars():
        fun_name = var.name_hint
        func = mod[fun_name]
        collector = GemmCollector()
        if "cutlass" in fun_name:
            collector.visit(func)
            # call cutlass profiler to find best settings, update attr
            new_attrs = {}
            new_attrs.update(collector.signature)
            for key in func.attrs.keys():
                new_attrs[key] = func.attrs[key]
            # call profiler
            arg0_shape = new_attrs["arg0_shape"]
            arg1_shape = new_attrs["arg1_shape"]
            MM = arg0_shape[0]
            KK = arg0_shape[1]
            NN = arg1_shape[0]
            out = cutlass_profiler.profile(
                "GenerateSM80_TensorOp_16816", new_attrs["arg0_dtype"], MM, NN, KK
            )
            if new_attrs["op_type"] == "cutlass.dense":
                new_attrs["cutlass_op_def"] = out["opdef"]
            elif new_attrs["op_type"] == "cutlass.dense_bias":
                new_attrs["cutlass_op_def"] = out["opdef_bias"]
            elif new_attrs["op_type"] == "cutlass.dense_bias_relu":
                new_attrs["cutlass_op_def"] = out["opdef_bias_relu"]
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
                is_nt = True
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

    with tvm.transform.PassContext(opt_level=3):
        # json, lib, param = relay.build(mod, target=target, params=params)
        # print("====================")
        # print(lib.imported_modules[1].get_source())
        lib = relay.build(mod, target="cuda", params=params)

    lib_path = "compiled.so"
    cutlass_path = "../../../3rdparty/cutlass/include"
    cutlass_util_path = "../../../3rdparty/cutlass/tools/util/include"
    workdir = "tmp"

    os.makedirs(workdir, exist_ok=True)

    kwargs = {}
    kwargs["cc"] = "nvcc"
    kwargs["options"] = [
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-gencode=arch=compute_%s,code=[sm_%s,compute_%s]" % (sm, sm, sm),
        "-Xcompiler=-fPIC",
        "-Xcompiler=-Wconversion",
        "-Xcompiler=-fno-strict-aliasing",
        "-O3",
        "-std=c++14",
        "-I" + cutlass_path,
        "-I" + cutlass_util_path,
    ]
    lib.export_library(lib_path, workspace_dir=workdir, **kwargs)
    lib = runtime.load_module(lib_path)
    ctx = tvm.gpu()
    rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](ctx))
    return rt_mod, ctx
