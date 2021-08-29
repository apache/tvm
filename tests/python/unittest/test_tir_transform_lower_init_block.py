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
from tvm import tir
from tvm.script import ty


@tvm.script.tir
class WithInit:
    def main(a: ty.handle, b: ty.handle) -> None:
        A = tir.match_buffer(a, [64, 64, 64])
        B = tir.match_buffer(b, [64])

        with tir.block([64, tir.reduce_axis(0, 64), tir.reduce_axis(32, 64)]) as [i, j, k]:
            with tir.init():
                B[i] = tir.float32(0)
            B[i] += A[i, j, k]


@tvm.script.tir
class WithBranch:
    def main(a: ty.handle, b: ty.handle) -> None:
        A = tir.match_buffer(a, [64, 64, 64])
        B = tir.match_buffer(b, [64])

        with tir.block([64, tir.reduce_axis(0, 64), tir.reduce_axis(32, 64)]) as [i, j, k]:
            if (j == 0) and (k == 32):
                B[i] = tir.float32(0)
            B[i] += A[i, j, k]


def test_lower_reduction():
    origin_mod = WithInit()
    mod = tvm.tir.transform.LowerInitBlock()(origin_mod)
    tvm.ir.assert_structural_equal(mod, WithBranch(), True)


if __name__ == "__main__":
    test_lower_reduction()
