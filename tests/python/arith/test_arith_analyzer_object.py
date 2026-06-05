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
from tvm import tirx
from tvm.runtime import Object


def test_analyzer_is_ffi_object_with_persistent_state():
    analyzer = tvm.arith.Analyzer()
    x = tirx.Var("x", "int64")

    assert isinstance(analyzer, Object)

    analyzer.bind(x, tvm.ir.Range(0, 8))
    assert analyzer.const_int_bound_is_bound(x)
    assert analyzer.can_prove(x < 8)
    assert not analyzer.can_prove(x < 4)

    bound = analyzer.const_int_bound(x + 1)
    assert bound.min_value == 1
    assert bound.max_value == 8


def test_analyzer_object_constraint_scope_and_override_bind():
    analyzer = tvm.arith.Analyzer()
    x = tirx.Var("x", "int64")

    with analyzer.constraint_scope(x % 3 == 0):
        assert analyzer.modular_set(x).coeff == 3

    assert analyzer.modular_set(x).coeff != 3

    analyzer = tvm.arith.Analyzer()
    y = tirx.Var("y", "int64")
    analyzer.bind(y, tirx.const(4, "int64"))
    tvm.ir.assert_structural_equal(analyzer.simplify(y + 1), tirx.const(5, "int64"))

    analyzer.bind(y, tirx.const(8, "int64"), allow_override=True)
    tvm.ir.assert_structural_equal(analyzer.simplify(y + 1), tirx.const(9, "int64"))


if __name__ == "__main__":
    tvm.testing.main()
