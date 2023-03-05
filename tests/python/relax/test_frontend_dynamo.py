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
import pytest

pytest.importorskip("torch._dynamo")


import tvm
from tvm import relax, meta_schedule as ms, tir
import tvm.testing
import torch
import torch._dynamo as dynamo
from tvm.relax.frontend.torch import relax_dynamo
from tvm.script.parser import relax as R, tir as T


def test_relax_dynamo():
    class Input1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(100, 10)

        def forward(self, x):
            return torch.nn.functional.relu(self.lin(x))

    model = Input1()
    ### construct the database
    @tvm.script.ir_module
    class Input1_ir:
        @T.prim_func
        def main(
            inp_0: T.Buffer((T.int64(10), T.int64(100)), "float32"),
            param_0: T.Buffer((T.int64(100), T.int64(10)), "float32"),
            param_1: T.Buffer(T.int64(10), "float32"),
            compute: T.Buffer((T.int64(10), T.int64(10)), "float32"),
        ):
            # function attr dict
            T.func_attr({"tir.noalias": True, "global_symbol": "main"})
            # body
            # with T.block("root")
            matmul = T.alloc_buffer([T.int64(10), T.int64(10)], dtype="float32")
            T_add = T.alloc_buffer([T.int64(10), T.int64(10)], dtype="float32")
            for i0, i1, k in T.grid(T.int64(10), T.int64(10), T.int64(100)):
                with T.block("matmul"):
                    v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                    T.reads(inp_0[v_i0, v_k], param_0[v_k, v_i1])
                    T.writes(matmul[v_i0, v_i1])
                    with T.init():
                        matmul[v_i0, v_i1] = T.float32(0)
                    matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + inp_0[v_i0, v_k] * param_0[v_k, v_i1]
            for ax0, ax1 in T.grid(T.int64(10), T.int64(10)):
                with T.block("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(matmul[v_ax0, v_ax1], param_1[v_ax1])
                    T.writes(T_add[v_ax0, v_ax1])
                    T_add[v_ax0, v_ax1] = matmul[v_ax0, v_ax1] + param_1[v_ax1]
            for i0, i1 in T.grid(T.int64(10), T.int64(10)):
                with T.block("compute"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(T_add[v_i0, v_i1])
                    T.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.max(T_add[v_i0, v_i1], T.float32(0))

    db = ms.Database.create("memory")
    workload = db.commit_workload(Input1_ir)

    sch = tir.Schedule(Input1_ir, debug_mask="all")
    b0 = sch.get_block(name="matmul", func_name="main")
    b1 = sch.get_block(name="T_add", func_name="main")
    b2 = sch.get_block(name="root", func_name="main")
    sch.compute_inline(block=b1)
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")
    l3, l4, l5 = sch.get_loops(block=b0)
    v6, v7, v8, v9 = sch.sample_perfect_tile(
        loop=l3, n=4, max_innermost_factor=64, decision=[1, 2, 5, 1]
    )
    l10, l11, l12, l13 = sch.split(loop=l3, factors=[v6, v7, v8, v9], preserve_unit_iters=True)
    v14, v15, v16, v17 = sch.sample_perfect_tile(
        loop=l4, n=4, max_innermost_factor=64, decision=[1, 1, 10, 1]
    )
    l18, l19, l20, l21 = sch.split(loop=l4, factors=[v14, v15, v16, v17], preserve_unit_iters=True)
    v22, v23 = sch.sample_perfect_tile(loop=l5, n=2, max_innermost_factor=64, decision=[100, 1])
    l24, l25 = sch.split(loop=l5, factors=[v22, v23], preserve_unit_iters=True)
    sch.reorder(l10, l18, l11, l19, l24, l12, l20, l25, l13, l21)
    (b26,) = sch.get_consumers(block=b0)
    sch.reverse_compute_at(block=b26, loop=l18, preserve_unit_loops=True, index=-1)
    sch.annotate(block_or_loop=b2, ann_key="meta_schedule.parallel", ann_val=96)
    sch.annotate(block_or_loop=b2, ann_key="meta_schedule.vectorize", ann_val=64)
    v27 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b2, ann_key="meta_schedule.unroll_explicit", ann_val=v27)

    tuning_record = ms.database.TuningRecord(sch.trace, workload, run_secs=[0.0])
    db.commit_tuning_record(tuning_record)
    ### Optimize the model with tuned-log
    with db:
        opt_model = torch.compile(model, backend=relax_dynamo())
    inp = torch.randn(10, 100)
    tvm.testing.assert_allclose(
        opt_model(inp).detach().numpy(), model(inp).detach().numpy(), rtol=1e-5, atol=1e-5
    )


def test_subgraph_capture():
    import torch
    from tvm.relax.frontend.torch.dynamo import dynamo_capture_subgraphs

    class Input1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(100, 10)

        def forward(self, x):
            return torch.nn.functional.relu(self.lin(x))

    @tvm.script.ir_module
    class Expected1:
        @R.function
        def subgraph_0(
            inp_0: R.Tensor((10, 100), dtype="float32"),
            w0: R.Tensor((10, 100), dtype="float32"),
            w1: R.Tensor((10,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((100, 10), dtype="float32") = R.permute_dims(w0, axes=None)
                lv1: R.Tensor((10, 10), dtype="float32") = R.matmul(inp_0, lv, out_dtype="float32")
                lv2: R.Tensor((10, 10), dtype="float32") = R.add(lv1, w1)
                lv3: R.Tensor((10, 10), dtype="float32") = R.nn.relu(lv2)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    model = Input1()
    mod = dynamo_capture_subgraphs(model, torch.randn(10, 100)).mod
    binding = {"w0": model.lin.weight.detach().numpy(), "w1": model.lin.bias.detach().numpy()}
    binding = {k: tvm.nd.array(v) for k, v in binding.items()}
    expected = relax.transform.BindParams("subgraph_0", binding)(Expected1)
    tvm.ir.assert_structural_equal(mod, expected)

    def Input2(a, b):
        x = a / (torch.sin(a) + 1)
        if torch.sum(b) < 1:
            b = b * -1
        return x * b

    @tvm.script.ir_module
    class Expected2:
        @R.function
        def subgraph_0(
            inp_0: R.Tensor((10,), dtype="float32"), inp_1: R.Tensor((10,), dtype="float32")
        ) -> R.Tuple(R.Tensor((10,), dtype="float32"), R.Tensor((), dtype="bool")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10,), dtype="float32") = R.sin(inp_0)
                lv1: R.Tensor((10,), dtype="float32") = R.add(lv, R.const(1, "float32"))
                lv2: R.Tensor((10,), dtype="float32") = R.divide(inp_0, lv1)
                lv3: R.Tensor((), dtype="float32") = R.sum(inp_1, axis=None, keepdims=False)
                lv4: R.Tensor((), dtype="bool") = R.less(lv3, R.const(1, "float32"))
                gv: R.Tuple(R.Tensor((10,), dtype="float32"), R.Tensor((), dtype="bool")) = (
                    lv2,
                    lv4,
                )
                R.output(gv)
            return gv

        @R.function
        def subgraph_1(
            inp_01: R.Tensor((10,), dtype="float32"), inp_11: R.Tensor((10,), dtype="float32")
        ) -> R.Tuple(R.Tensor((10,), dtype="float32")):
            # block 0
            with R.dataflow():
                lv5: R.Tensor((10,), dtype="float32") = R.multiply(inp_11, inp_01)
                gv1: R.Tuple(R.Tensor((10,), dtype="float32")) = (lv5,)
                R.output(gv1)
            return gv1

    mod = dynamo_capture_subgraphs(Input2, torch.randn(10), torch.ones(10)).mod
    tvm.ir.assert_structural_equal(mod, Expected2)


if __name__ == "__main__":
    tvm.testing.main()
