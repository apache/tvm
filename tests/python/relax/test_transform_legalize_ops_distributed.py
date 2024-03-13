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

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.transform import LegalizeOps
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


def test_redistribute_replica_to_shard():
    # fmt: off
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((10, 10), "float32"))  -> R.Tensor((10, 5), "float32"):
            gv0 = R.dist.redistribute_replica_to_shard(x, num_workers=2, axis=1)
            return gv0

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def strided_slice(A: T.Buffer((T.int64(10), T.int64(10)), "float32"), redistribute_replica_to_shard: T.Buffer((T.int64(10), T.int64(5)), "float32"), worker_id: T.int64):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1 in T.grid(T.int64(10), T.int64(5)):
                with T.block("redistribute_replica_to_shard"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(A[v_i0, worker_id * T.int64(5) + v_i1])
                    T.writes(redistribute_replica_to_shard[v_i0, v_i1])
                    redistribute_replica_to_shard[v_i0, v_i1] = A[v_i0, worker_id * T.int64(5) + v_i1]

        @R.function
        def main(x: R.Tensor((10, 10), dtype="float32")) -> R.Tensor((10, 5), dtype="float32"):
            worker_id = T.int64()
            cls = Expected
            gv: R.Shape(ndim=-1) = R.call_pure_packed("runtime.disco.worker_id", sinfo_args=(R.Shape(ndim=-1),))
            gv1: R.Shape([worker_id]) = R.match_cast(gv, R.Shape([worker_id]))
            gv0 = R.call_tir(cls.strided_slice, (x,), out_sinfo=R.Tensor((10, 5), dtype="float32"), tir_vars=R.shape([worker_id]))
            return gv0
    # fmt: on

    mod = LegalizeOps()(Before)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm.testing.main()
