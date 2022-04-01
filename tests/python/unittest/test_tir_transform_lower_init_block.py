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
from tvm import te
from tvm.script import tir as T

# pylint: disable=no-self-argument


@tvm.script.ir_module
class WithInit:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, [64, 64, 64])
        B = T.match_buffer(b, [64])

        for i0, j0 in T.grid(64, 64):
            for k0 in T.serial(32, 64):
                with T.block():
                    i, j, k = T.axis.remap("SRR", [i0, j0, k0])
                    with T.init():
                        B[i] = T.float32(0)
                    B[i] += A[i, j, k]


@tvm.script.ir_module
class WithBranch:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, [64, 64, 64])
        B = T.match_buffer(b, [64])

        for i0, j0 in T.grid(64, 64):
            for k0 in T.serial(32, 64):
                with T.block():
                    i, j, k = T.axis.remap("SRR", [i0, j0, k0])
                    T.reads(A[i, j, k])
                    T.writes(B[i])
                    if (j == 0) and (k == 32):
                        B[i] = T.float32(0)
                    B[i] += A[i, j, k]


@tvm.script.ir_module
class InitWithMatchBuffer:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, [64, 64, 64])
        B = T.match_buffer(b, [64])

        for i0, j0 in T.grid(64, 64):
            for k0 in T.serial(32, 64):
                with T.block():
                    i, j, k = T.axis.remap("SRR", [i0, j0, k0])
                    BB = T.match_buffer(B[i], ())
                    AA = T.match_buffer(A[i, 0:64, 0:64], (64, 64))
                    with T.init():
                        BB[()] = T.float32(0)
                    BB[()] += AA[j, k]


@tvm.script.ir_module
class BranchWithMatchBuffer:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, [64, 64, 64])
        B = T.match_buffer(b, [64])

        for i0, j0 in T.grid(64, 64):
            for k0 in T.serial(32, 64):
                with T.block():
                    i, j, k = T.axis.remap("SRR", [i0, j0, k0])
                    T.reads(A[i, j, k])
                    T.writes(B[i])
                    BB = T.match_buffer(B[i], ())
                    AA = T.match_buffer(A[i, 0:64, 0:64], (64, 64))
                    if (j == 0) and (k == 32):
                        BB[()] = T.float32(0)
                    BB[()] += AA[j, k]


def test_lower_reduction():
    origin_mod = WithInit
    mod = tvm.tir.transform.LowerInitBlock()(origin_mod)
    tvm.ir.assert_structural_equal(mod, WithBranch, True)


def test_lower_match_buffer():
    origin_mod = InitWithMatchBuffer
    mod = tvm.tir.transform.LowerInitBlock()(origin_mod)
    tvm.ir.assert_structural_equal(mod, BranchWithMatchBuffer, True)


def test_lower_te():
    x = te.placeholder((1,))
    y = te.compute((1,), lambda i: x[i] + 2)
    s = te.create_schedule(y.op)
    orig_mod = tvm.driver.build_module.schedule_to_module(s, [x, y])
    mod = tvm.tir.transform.LowerInitBlock()(orig_mod)
    tvm.ir.assert_structural_equal(mod, orig_mod)  # LowerInitBlock should do nothing on TE


if __name__ == "__main__":
    test_lower_reduction()
    test_lower_match_buffer()
    test_lower_te()
