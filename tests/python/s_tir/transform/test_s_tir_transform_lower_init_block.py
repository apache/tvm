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
# ruff: noqa: F401, F821
import tvm
from tvm import s_tir
from tvm.script import s_tir as Ts
from tvm.script import tirx as T

# pylint: disable=no-self-argument


@tvm.script.ir_module
class WithInit:
    @Ts.prim_func
    def main(A: T.Buffer([64, 64, 64]), B: T.Buffer([64])) -> None:
        for i0, j0 in T.grid(64, 64):
            for k0 in T.serial(32, 64):
                with Ts.sblock():
                    i, j, k = Ts.axis.remap("SRR", [i0, j0, k0])
                    with Ts.init():
                        B[i] = T.float32(0)
                    B[i] += A[i, j, k]


@tvm.script.ir_module
class WithBranch:
    @Ts.prim_func
    def main(A: T.Buffer([64, 64, 64]), B: T.Buffer([64])) -> None:
        for i0, j0 in T.grid(64, 64):
            for k0 in T.serial(32, 64):
                with Ts.sblock():
                    i, j, k = Ts.axis.remap("SRR", [i0, j0, k0])
                    Ts.reads(A[i, j, k])
                    Ts.writes(B[i])
                    if (j == 0) and (k == 32):
                        B[i] = T.float32(0)
                    B[i] += A[i, j, k]


@tvm.script.ir_module
class InitWithMatchBuffer:
    @Ts.prim_func
    def main(A: T.Buffer([64, 64, 64]), B: T.Buffer([64])) -> None:
        for i0, j0 in T.grid(64, 64):
            for k0 in T.serial(32, 64):
                with Ts.sblock():
                    i, j, k = Ts.axis.remap("SRR", [i0, j0, k0])
                    BB = Ts.match_buffer(B[i], ())
                    AA = Ts.match_buffer(A[i, 0:64, 0:64], (64, 64))
                    with Ts.init():
                        BB[()] = T.float32(0)
                    BB[()] += AA[j, k]


@tvm.script.ir_module
class BranchWithMatchBuffer:
    @Ts.prim_func
    def main(A: T.Buffer([64, 64, 64]), B: T.Buffer([64])) -> None:
        for i0, j0 in T.grid(64, 64):
            for k0 in T.serial(32, 64):
                with Ts.sblock():
                    i, j, k = Ts.axis.remap("SRR", [i0, j0, k0])
                    Ts.reads(A[i, j, k])
                    Ts.writes(B[i])
                    BB = Ts.match_buffer(B[i], ())
                    AA = Ts.match_buffer(A[i, 0:64, 0:64], (64, 64))
                    if (j == 0) and (k == 32):
                        BB[()] = T.float32(0)
                    BB[()] += AA[j, k]


def test_lower_reduction():
    origin_mod = WithInit
    mod = tvm.s_tir.transform.LowerInitBlock()(origin_mod)
    tvm.ir.assert_structural_equal(mod, WithBranch, True)


def test_lower_match_buffer():
    origin_mod = InitWithMatchBuffer
    mod = tvm.s_tir.transform.LowerInitBlock()(origin_mod)
    tvm.ir.assert_structural_equal(mod, BranchWithMatchBuffer, True)


if __name__ == "__main__":
    test_lower_reduction()
    test_lower_match_buffer()
    test_lower_te()
