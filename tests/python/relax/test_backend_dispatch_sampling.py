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
# pylint: disable=missing-docstring

import tvm
import tvm.script
import tvm.testing
from tvm.ir.base import assert_structural_equal
from tvm.relax.backend import DispatchSampling
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


@I.ir_module
class MultiFromUniformModule:
    @R.function
    def foo(
        prob: R.Tensor((3, 5), "float32"),
        uniform_sample: R.Tensor((6, 1), "float32"),
        sample_indices: R.Tensor((6, 1), "int64"),
    ):
        with R.dataflow():
            gv = R.multinomial_from_uniform(prob, uniform_sample, sample_indices, dtype="int64")
            R.output(gv)
        return gv


def test_dispatch_multinomial_from_uniform_generic():
    # fmt: off
    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def get_sample_index(A: T.handle, B: T.handle, C: T.handle, D: T.handle):
            batch, vocab_size = T.int64(), T.int64()
            prob = T.match_buffer(A, (batch, vocab_size))
            out_batch = T.int64()
            usample = T.match_buffer(B, (out_batch, 1))
            sample_indices = T.match_buffer(C, (out_batch, 1), "int64")
            output_index = T.match_buffer(D, (out_batch, 1), "int64")
            # with T.block("root"):
            for ax0, ax1 in T.grid(out_batch, vocab_size):
                with T.block("T_get_sample_index"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    if usample[v_ax0, T.int64(0)] < prob[sample_indices[v_ax0, T.int64(0)], v_ax1] or v_ax1 + T.int64(1) == vocab_size:
                        if v_ax1 == T.int64(0):
                            output_index[v_ax0, 0] = T.int64(0)
                        else:
                            if usample[v_ax0, T.int64(0)] >= prob[sample_indices[v_ax0, T.int64(0)], v_ax1 - T.int64(1)]:
                                output_index[v_ax0, 0] = v_ax1

        @R.function
        def foo(prob: R.Tensor((3, 5), dtype="float32"), uniform_sample: R.Tensor((6, 1), dtype="float32"), sample_indices: R.Tensor((6, 1), dtype="int64")) -> R.Tensor((6, 1), dtype="int64"):
            cls = Expected
            with R.dataflow():
                lv: R.Tensor((3, 5), dtype="float32") = R.cumsum(prob, axis=1, dtype="float32", exclusive=0)
                gv = R.call_tir(cls.get_sample_index, (lv, uniform_sample, sample_indices), out_sinfo=R.Tensor((6, 1), dtype="int64"))
                R.output(gv)
            return gv
    # fmt: on

    with tvm.target.Target("llvm"):
        mod = DispatchSampling()(MultiFromUniformModule)

    assert_structural_equal(mod, Expected)


def test_dispatch_multinomial_from_uniform_gpu():
    # fmt: off
    @I.ir_module
    class Expected:
        @T.prim_func
        def parallel_sampling_from_prob(var_prob: T.handle, var_uniform_samples: T.handle, var_row_indices: T.handle, var_sampled_token_ids: T.handle):
            T.func_attr({"tir.is_scheduled": 1})
            n, vocab_size = T.int64(), T.int64()
            prob = T.match_buffer(var_prob, (n, vocab_size))
            batch_size = T.int64()
            uniform_samples = T.match_buffer(var_uniform_samples, (batch_size, 1))
            row_indices = T.match_buffer(var_row_indices, (batch_size, 1), "int64")
            token_ids = T.match_buffer(var_sampled_token_ids, (batch_size, 1), "int64")
            # with T.block("root"):
            aggregate = T.alloc_buffer((), scope="local")
            sample_id_local = T.alloc_buffer((), "int64", scope="local")
            step_iter = T.alloc_buffer((), "int32", scope="local")
            for bx in T.thread_binding(batch_size, thread="blockIdx.x"):
                row_idx: T.int64 = row_indices[bx, 0]
                for ty in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                    for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                        u: T.float32 = uniform_samples[bx, 0]
                        aggregate[()] = T.Cast("float32", 0)
                        step_iter[()] = 0
                        while T.tvm_thread_invariant((step_iter[()] == 0 or aggregate[()] < u - T.float32(9.9999999999999995e-07)) and T.Cast("int64", step_iter[()]) < (vocab_size + T.int64(512) - T.int64(1)) // T.int64(512)):
                            with T.block(""):
                                T.reads(step_iter[()], prob[row_idx, T.Cast("int64", step_iter[()]) * T.int64(512) + ty * T.int64(128) + tx * T.int64(4):T.Cast("int64", step_iter[()]) * T.int64(512) + ty * T.int64(128) + tx * T.int64(4) + T.int64(4)], aggregate[()])
                                T.writes(sample_id_local[()], aggregate[()])
                                prob_gt_threshold = T.alloc_buffer((T.int64(4),), scope="local")
                                cumsum = T.alloc_buffer((T.int64(512),), scope="shared")
                                greater_than_u = T.alloc_buffer((T.int64(4),), "bool", scope="local")
                                mask = T.alloc_buffer((T.int64(4),), "bool", scope="local")
                                valid = T.alloc_buffer((T.int64(4),), "bool", scope="local")
                                indices = T.alloc_buffer((T.int64(4),), "int64", scope="local")
                                step_aggregate = T.alloc_buffer((), scope="local")
                                for v in T.unroll(T.int64(4)):
                                    idx: T.int64 = T.Cast("int64", step_iter[()]) * T.int64(512) + ty * T.int64(128) + tx * T.int64(4) + v
                                    prob_local: T.float32 = T.if_then_else(idx < vocab_size, prob[row_idx, idx], T.Cast("float32", 0))
                                    prob_gt_threshold[v] = T.if_then_else(prob_local > T.float32(0), prob_local, T.Cast("float32", 0))
                                    valid[v] = prob_local > T.float32(0) and idx < vocab_size
                                with T.block(""):
                                    T.reads(prob_gt_threshold[T.int64(0):T.int64(4)])
                                    T.writes(step_aggregate[()])
                                    local_sum = T.alloc_buffer((), scope="local")
                                    shared_buf = T.alloc_buffer((T.int64(128),), scope="shared")
                                    idx: T.int64 = ty * T.int64(32) + tx
                                    local_sum[()] = T.Cast("float32", 0)
                                    for i in T.unroll(T.int64(4)):
                                        local_sum[()] = local_sum[()] + prob_gt_threshold[i]
                                    shared_buf[idx] = local_sum[()]
                                    for i in T.unroll(T.int64(7)):
                                        if idx % T.shift_left(T.int64(1), i + T.int64(1)) == T.int64(0):
                                            shared_buf[idx] = shared_buf[idx] + shared_buf[idx + T.shift_left(T.int64(1), i)]
                                    step_aggregate[()] = shared_buf[0]
                                if T.tvm_thread_invariant(aggregate[()] + step_aggregate[()] >= u - T.float32(9.9999999999999995e-07)):
                                    for i in T.unroll(T.int64(1), T.int64(4)):
                                        prob_gt_threshold[i] = prob_gt_threshold[i] + prob_gt_threshold[i - T.int64(1)]
                                    for i in T.vectorized(T.int64(4)):
                                        cumsum[ty * T.int64(128) + tx * T.int64(4) + i] = prob_gt_threshold[i]
                                    for i in T.unroll(T.int64(5)):
                                        for j in T.vectorized(T.int64(4)):
                                            idx: T.int64 = ty * T.int64(128) + tx * T.int64(4)
                                            if tx >= T.shift_left(T.int64(1), i):
                                                cumsum[idx + j] = cumsum[idx + j] + cumsum[idx - T.shift_left(T.int64(1), i) * T.int64(4) + T.int64(4) - T.int64(1)]
                                    for i in T.unroll(T.int64(1), T.int64(4)):
                                        for j in T.vectorized(T.int64(4)):
                                            if ty == T.int64(0):
                                                idx: T.int64 = i * T.int64(128) + tx * T.int64(4)
                                                cumsum[idx + j] = cumsum[idx + j] + cumsum[i * T.int64(128) - T.int64(1)]
                                    for v in T.unroll(T.int64(4)):
                                        greater_than_u[v] = cumsum[ty * T.int64(128) + tx * T.int64(4) + v] + aggregate[()] >= u - T.float32(9.9999999999999995e-07)
                                    with T.block(""):
                                        T.reads(greater_than_u[T.int64(0):T.int64(4)])
                                        T.writes(mask[T.int64(0):T.int64(4)])
                                        shared_buf = T.alloc_buffer((T.int64(128),), "bool", scope="shared")
                                        tx_idx: T.int64 = ty * T.int64(32) + tx
                                        shared_buf[tx_idx] = greater_than_u[T.int64(3)]
                                        mask[0] = T.if_then_else(tx_idx != T.int64(0), T.Cast("int8", greater_than_u[0]) != T.Cast("int8", shared_buf[tx_idx - T.int64(1)]), greater_than_u[0])
                                        for i in T.unroll(T.int64(1), T.int64(4)):
                                            mask[i] = T.Cast("int8", greater_than_u[i]) != T.Cast("int8", greater_than_u[i - T.int64(1)])
                                    for v in T.unroll(T.int64(4)):
                                        mask[v] = mask[v] and valid[v]
                                        indices[v] = T.Cast("int64", step_iter[()]) * T.int64(512) + ty * T.int64(128) + tx * T.int64(4) + v
                                    with T.block(""):
                                        T.reads(mask[T.int64(0):T.int64(4)], indices[T.int64(0):T.int64(4)])
                                        T.writes(sample_id_local[()])
                                        local_sum = T.alloc_buffer((), "int64", scope="local")
                                        shared_buf = T.alloc_buffer((T.int64(128),), "int64", scope="shared")
                                        idx: T.int64 = ty * T.int64(32) + tx
                                        local_sum[()] = T.Cast("int64", vocab_size - T.int64(1))
                                        for i in T.unroll(T.int64(4)):
                                            if mask[i]:
                                                local_sum[()] = T.min(local_sum[()], indices[i])
                                        shared_buf[idx] = local_sum[()]
                                        for i in T.unroll(T.int64(7)):
                                            if idx % T.shift_left(T.int64(1), i + T.int64(1)) == T.int64(0):
                                                shared_buf[idx] = T.min(shared_buf[idx], shared_buf[idx + T.shift_left(T.int64(1), i)])
                                        sample_id_local[()] = shared_buf[0]
                                aggregate[()] = aggregate[()] + step_aggregate[()]
                            step_iter[()] = step_iter[()] + 1
                        if tx == T.int64(0) and ty == T.int64(0):
                            token_ids[bx, 0] = sample_id_local[()]

        @R.function
        def foo(prob: R.Tensor((3, 5), dtype="float32"), uniform_sample: R.Tensor((6, 1), dtype="float32"), sample_indices: R.Tensor((6, 1), dtype="int64")) -> R.Tensor((6, 1), dtype="int64"):
            cls = Expected
            with R.dataflow():
                gv = R.call_tir(cls.parallel_sampling_from_prob, (prob, uniform_sample, sample_indices), out_sinfo=R.Tensor((6, 1), dtype="int64"))
                R.output(gv)
            return gv
    # fmt: on

    with tvm.target.Target("cuda"):
        mod = DispatchSampling()(MultiFromUniformModule)

    assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm.testing.main()
