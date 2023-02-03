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
from tvm import tir


def test_simplify_reshape_flattened_index():
    ana = tvm.arith.Analyzer()

    i0 = tir.Var("i0", "int64")
    i1 = tir.Var("i1", "int64")
    ana.bind(i0, tvm.ir.Range(0, 8))
    ana.bind(i1, tvm.ir.Range(0, 3))

    i_flattened = i0 * 3 + i1
    assert tvm.ir.structural_equal(
        ana.simplify((i_flattened) // 12 * 12 + (i_flattened) % 12 // 4 * 4 + (i_flattened) % 4),
        i_flattened,
    )


if __name__ == "__main__":
    tvm.testing.main()
