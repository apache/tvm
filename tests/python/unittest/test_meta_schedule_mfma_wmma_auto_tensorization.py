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
"""Tests for WMMA Auto Tensorization on AMD GPU"""

import tempfile
import numpy as np

import tvm
from tvm import te
from tvm import meta_schedule as ms
from tvm._ffi import register_func
from tvm.meta_schedule.testing.space_generation import (
    check_sketches,
    generate_design_space,
)
from tvm.meta_schedule.builder import LocalBuilder
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir import Schedule
from tvm.tir.schedule import Trace

# get tensor intrin
from tvm.tir.tensor_intrin import rocm  # pylint: disable=unused-import
from tvm.tir.tensor_intrin.rocm import get_rocwmma_intrin_group
import tvm.testing

def matmul_fp16(M: int, N: int, K: int, out_dtype: str):
    x = te.placeholder((M, K), name="X", dtype="float16")
    y = te.placeholder((K, N), name="Y", dtype="float16")
    k = te.reduce_axis((0, K), name="k")
    c = te.compute(  # pylint: disable=invalid-name
        (M, N),
        lambda i, j: te.sum(x[i][k].astype(out_dtype) * y[k][j].astype(out_dtype), axis=[k]),
        name="C",
    )
    return (x, y, c)


def initializer():
    @register_func("meta_schedule.builder.async_build")
    def async_build(mod, target, _params):  # pylint: disable=unused-variable, unused-argument
        # pylint: disable=import-outside-toplevel
        from tvm.driver import build as tvm_build
        from tvm.tir.transform import RemoveWeightLayoutRewriteBlock

        # re-import here for local builder to register index_map_m16n8k8_matrixC
        # pylint: disable=import-outside-toplevel, unused-import
        from tvm.tir.tensor_intrin import rocm

        mod = RemoveWeightLayoutRewriteBlock(skip_ndarray_rewrite=True)(mod)
        with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
            rt_mod = tvm_build(mod, target=target)
        return rt_mod

@tvm.testing.requires_matrixcore
def test_wmma_tune():
    M, N, K = 1024, 1024, 1024
    def tune(out_dtype):
        target = Target("hip")
        func = te.create_prim_func(matmul_fp16(M=M, N=N, K=K, out_dtype=out_dtype)).with_attr(
            {"global_symbol": "main"}
        )

        space=ms.space_generator.PostOrderApply(
          sch_rules="rocm-matrixcore",
          postprocs="rocm-matrixcore",
          mutator_probs="rocm-matrixcore",
        )

        mod = tvm.IRModule({"main": func})
        work_dir = "hip_auto_schedule.log"
        db = ms.tir_integration.tune_tir(
            mod=mod,
            target=target,
            work_dir=work_dir,
            max_trials_global=32,
            builder=LocalBuilder(
                f_build="meta_schedule.builder.async_build", initializer=initializer
            ),
            space = space,
            

        )
        sch = db.query_schedule(mod, target=target, workload_name="main")
        with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
            rt_mod = tvm.build(sch.mod, target=target)

        a_np = np.random.uniform(0, 1, size=(M, K)).astype("float16")
        b_np = np.random.uniform(0, 1, size=(K, N)).astype("float16")
        
        dev = tvm.rocm(0)
        a_tvm = tvm.nd.array(a_np, device=tvm.rocm(0))
        b_tvm = tvm.nd.array(b_np, device=tvm.rocm(0))
        c_tvm = tvm.nd.array(np.empty((M, N)).astype(out_dtype), device=tvm.rocm(0))
        rt_mod(a_tvm, b_tvm, c_tvm)
        if M < 256 and N < 256 and K < 256:
            golden = np.matmul(a_np.astype("float16"), b_np.astype("float16")) 
            tvm.testing.assert_allclose(golden, c_tvm.numpy(), atol=1e-3, rtol=1e-3)

    tune("float32")


if __name__ == "__main__":
    test_wmma_tune()
