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

import tempfile

import tvm
import tvm.testing
import tvm.meta_schedule as ms
from tvm import relax
from tvm.ir import transform
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax.transform.tuning_api import Trace
from tvm.script import relax as R
from tvm.script import tir as T

target = tvm.target.Target("llvm --num-cores=16")


@tvm.script.ir_module
class InputModule:
    @T.prim_func
    def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
        T.func_attr({"global_symbol": "tir_matmul"})
        k = T.int32()
        A = T.match_buffer(x, (32, 32))
        B = T.match_buffer(y, (32, 32))
        C = T.match_buffer(z, (32, 32))

        for (i0, j0, k0) in T.grid(32, 32, 32):
            with T.block():
                i, j, k = T.axis.remap("SSR", [i0, j0, k0])
                with T.init():
                    C[i, j] = 0.0
                C[i, j] += A[i, k] * B[j, k]

    @T.prim_func
    def tir_relu(x: T.handle, y: T.handle):
        T.func_attr({"global_symbol": "tir_relu"})
        A = T.match_buffer(x, (32, 32))
        B = T.match_buffer(y, (32, 32))
        for (i, j) in T.grid(32, 32):
            with T.block():
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = T.max(A[vi, vj], 0.0)

    @R.function
    def main(x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")) -> R.Tensor:
        cls = InputModule
        with R.dataflow():
            lv0 = R.call_tir(cls.tir_matmul, (x, w), R.Tensor((32, 32), dtype="float32"))
            lv1 = R.call_tir(cls.tir_relu, (lv0), R.Tensor((32, 32), dtype="float32"))
            R.output(lv1)
        return lv1


# TODO(@sunggg): determine how to pass MS database object across different passes.
# PassContext might be an option, but we already have TuningAPI database.
# (MS database and TuningAPI database will be unified in the future)
# For now, we only support default JSON database config.
def test_ms_tuning_irmodule():

    mod = InputModule
    assert isinstance(mod, IRModule)

    with tempfile.TemporaryDirectory() as work_dir:
        with target, transform.PassContext(trace=Trace(mod), opt_level=0):
            tuning_pass = relax.transform.MetaScheduleTuneIRMod(
                params={}, work_dir=work_dir, max_trials_global=4
            )
            out_mod = tuning_pass(mod)
            assert PassContext.current().get_trace_stack_size() == 1
            assert PassContext.current().get_current_trace().size == 1
            tvm.ir.assert_structural_equal(mod, out_mod)

            application_pass = relax.transform.MetaScheduleApplyDatabase(work_dir)

            out_mod = application_pass(mod)
            assert not tvm.ir.structural_equal(mod, out_mod)


def test_ms_tuning_primfunc():
    mod = InputModule
    assert isinstance(mod, IRModule)
    with tempfile.TemporaryDirectory() as work_dir:
        with target, transform.PassContext(trace=Trace(mod), opt_level=0):
            tuning_pass = relax.transform.MetaScheduleTuneTIR(
                work_dir=work_dir, max_trials_global=4
            )
            out_mod = tuning_pass(mod)
            assert PassContext.current().get_trace_stack_size() == 1
            # TODO (@sunggg): Need to determine how to track subgraph-level tuning traces.
            # Currently, we don't track this so the trace size. Revisit this later.
            tvm.ir.assert_structural_equal(mod, out_mod)

            application_pass = relax.transform.MetaScheduleApplyDatabase(work_dir)
            out_mod = application_pass(mod)
            assert not tvm.ir.structural_equal(mod, out_mod)


if __name__ == "__main__":
    tvm.testing.main()
