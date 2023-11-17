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

from typing import Union

import tvm
import tvm.script
import tvm.testing
from tvm import IRModule, relax
from tvm.relax import Function
from tvm.script import relax as R, tir as T, ir as I

from tvm.relax.transform import optimize_layout_transform, DeadCodeElimination, LegalizeOps
from tvm.relax.backend import DispatchOps


def test_cumsum():
    # fmt: off
    @I.ir_module
    class Cumsum:
        @R.function
        def main(x: R.Tensor((3, 2, 3), "float32")):
            gv = R.cumsum(x, axis=1, dtype="int32")
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def cumsum(var_rxplaceholder: T.handle, out_buf: T.Buffer((T.int64(3), T.int64(2), T.int64(3)), "int32")):
            T.func_attr({"tir.noalias": True})
            rxplaceholder = T.match_buffer(var_rxplaceholder, (T.int64(3), T.int64(2), T.int64(3)), offset_factor=1)
            with T.block("cumsum_generic"):
                T.reads(rxplaceholder[T.int64(0):T.int64(3), T.int64(0):T.int64(2), T.int64(0):T.int64(3)])
                T.writes(out_buf[T.int64(0):T.int64(3), T.int64(0):T.int64(2), T.int64(0):T.int64(3)])
                for fused in T.parallel(T.int64(9)):
                    out_buf[(fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3)) // T.int64(3) // T.int64(2), (fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3)) // T.int64(3) % T.int64(2), (fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3)) % T.int64(3)] = T.Cast("int32", rxplaceholder[(fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3)) // T.int64(3) // T.int64(2), (fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3)) // T.int64(3) % T.int64(2), (fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3)) % T.int64(3)])
                    for _k in range(T.int64(1)):
                        out_buf[(fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3) + (_k + T.int64(1)) * T.int64(3)) // T.int64(3) // T.int64(2), (fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3) + (_k + T.int64(1)) * T.int64(3)) // T.int64(3) % T.int64(2), (fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3) + (_k + T.int64(1)) * T.int64(3)) % T.int64(3)] = out_buf[(fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3) + (_k + T.int64(1) - T.int64(1)) * T.int64(3)) // T.int64(3) // T.int64(2), (fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3) + (_k + T.int64(1) - T.int64(1)) * T.int64(3)) // T.int64(3) % T.int64(2), (fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3) + (_k + T.int64(1) - T.int64(1)) * T.int64(3)) % T.int64(3)] + T.Cast("int32", rxplaceholder[(fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3) + (_k + T.int64(1)) * T.int64(3)) // T.int64(3) // T.int64(2), (fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3) + (_k + T.int64(1)) * T.int64(3)) // T.int64(3) % T.int64(2), (fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3) + (_k + T.int64(1)) * T.int64(3)) % T.int64(3)])

        @R.function
        def main(x: R.Tensor((3, 2, 3), dtype="float32")) -> R.Tensor((3, 2, 3), dtype="int32"):
            cls = Expected
            gv = R.call_tir(cls.cumsum, (x,), out_sinfo=R.Tensor((3, 2, 3), dtype="int32"))
            return gv
    # fmt: on

    mod = LegalizeOps()(Cumsum)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_dispatch_cumsum():
    # fmt: off
    @I.ir_module
    class Cumsum:
        I.module_global_infos({"vdevice": [I.vdevice("cuda", 0)]})
        @R.function
        def main(x: R.Tensor((3, 2, 3), "float32", "cuda")) -> R.Tensor((3, 2, 3), "float32", "cuda"):
            with R.dataflow():
                lv: R.Tensor((3, 2, 3), "float32", "cuda") = R.reshape(x, R.shape([3, 2, 3]))
                lv1: R.Tensor((3, 2, 3), "float32", "cuda") = R.reshape(lv, R.shape([3, 2, 3]))
                gv = R.cumsum(lv1)# R.cumsum(lv1, axis=1, dtype="int32")
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        I.module_global_infos({"vdevice": [I.vdevice("cuda", 0)]})
        @R.function
        def main(x: R.Tensor((3, 2, 3), "float32", "cuda")) -> R.Tensor((3, 2, 3), "int32", "cuda"):
            gv = R.call_dps_packed("cumsum", (x,), out_sinfo=R.Tensor((3, 2, 3), "int32", "cuda"))
            return gv
    # fmt: on

    mod = DispatchOps()(Cumsum)
    mod = DeadCodeElimination()(mod)
    mod.show()
    # tvm.ir.assert_structural_equal(mod, Expected)


def test_dispatch_sort():
    pass


if __name__ == "__main__":
    # tvm.testing.main()
    test_dispatch_cumsum()
