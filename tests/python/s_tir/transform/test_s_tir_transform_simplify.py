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
from tvm.script import tirx as T


def test_s_tir_block_iterator_constraints():
    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer((4,), "int32")):
        for i in range(4):
            with T.sblock("write"):
                vi = T.axis.spatial(4, i)
                if vi < 4:
                    A[vi] = vi

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer((4,), "int32")):
        for i in range(4):
            with T.sblock("write"):
                vi = T.axis.spatial(4, i)
                A[vi] = vi

    result = tvm.s_tir.transform.StmtSimplify()(tvm.IRModule.from_expr(before))["main"]
    tvm.ir.assert_structural_equal(result, expected)
    assert tvm.s_tir.analysis.verify_well_formed(result)


if __name__ == "__main__":
    tvm.testing.main()
