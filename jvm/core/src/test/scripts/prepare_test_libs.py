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
# Prepare test library for standalone wasm runtime test.

import sys
import os
import tvm
from tvm import te
from tvm import relax
from tvm.script import relax as R


def prepare_relax_lib(base_path):
    pipeline = relax.get_pipeline()

    @tvm.script.ir_module
    class Mod:
        @R.function
        def main(x: R.Tensor(["n"], "float32"), y: R.Tensor(["n"], "float32")):
            lv0 = R.add(x, y)
            return lv0

    target = tvm.target.Target("llvm")

    mod = pipeline(Mod)
    ex = relax.build(mod, target)
    relax_path = os.path.join(base_path, "add_relax.so")
    ex.export_library(relax_path)


def prepare_cpu_lib(base_path):
    target = "llvm"
    if not tvm.runtime.enabled(target):
        raise RuntimeError("Target %s is not enbaled" % target)
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    mod = tvm.IRModule.from_expr(te.create_prim_func([A, B, C]).with_attr("global_symbol", "myadd"))
    fadd = tvm.build(mod, target)
    lib_path = os.path.join(base_path, "add_cpu.so")
    fadd.export_library(lib_path)


def prepare_gpu_lib(base_path):
    if not tvm.cuda().exist:
        print("CUDA is not enabled, skip the generation")
        return
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    mod = tvm.IRModule.from_expr(te.create_prim_func([A, B, C]).with_attr("global_symbol", "myadd"))
    sch = tvm.tir.Schedule(mod)
    sch.work_on("myadd")
    (i,) = sch.get_loops(block=sch.get_block("C"))
    i0, i1 = sch.split(i, [None, 32])
    sch.bind(i0, "blockIdx.x")
    sch.bind(i1, "threadIdx.x")
    fadd = tvm.build(sch.mod, "cuda")
    lib_path = os.path.join(base_path, "add_cuda.so")
    fadd.export_library(lib_path)


if __name__ == "__main__":
    base_path = sys.argv[1]
    prepare_cpu_lib(base_path)
    prepare_gpu_lib(base_path)
    prepare_relax_lib(base_path)
