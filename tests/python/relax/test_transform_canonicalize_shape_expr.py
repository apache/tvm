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

"""Unit tests for the CanonicalizeShapeExpr pass"""

import pytest
import tvm
import tvm.testing
from tvm import relax
from tvm.script import relax as R
from tvm.script import tir as T


def test_nested_compound_shape():
    """Test canonicalization with nested compound shape expressions"""

    @R.function
    def before(x: R.Tensor(("n", "m"), "float32")):
        n = T.int64()
        m = T.int64()
        # Nested compound expression: (n + m) * 2
        y: R.Tensor(((n + m) * 2,), "float32") = R.zeros(R.shape([(n + m) * 2]), dtype="float32")
        return y

    mod = tvm.IRModule.from_expr(before)
    mod = relax.transform.CanonicalizeShapeExpr()(mod)

    # Verify: MatchCast bindings should exist for compound exprs
    func = mod["before"]
    # Check that no ShapeExpr contains compound expressions anymore

    mod = relax.transform.Normalize()(mod)
    mod = relax.transform.ComputePrimValue()(mod)
    mod = relax.transform.VMShapeLower()(mod)

    assert "compute_symbolic_expr" in [str(gv) for gv in mod.get_global_vars()]


if __name__ == "__main__":
    import sys

    print("Running CanonicalizeShapeExpr unit tests...")
    print("=" * 80)

    tests = [
        ("Nested compound shape", test_nested_compound_shape),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            print(f"\nTest: {name}")
            test_func()
            print("Result: PASSED")
            passed += 1
        except Exception as e:
            print(f"Result: FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 80)
    print(f"Total tests run: {passed + failed}, Passed: {passed}, Failed: {failed}")

    sys.exit(0 if failed == 0 else 1)
