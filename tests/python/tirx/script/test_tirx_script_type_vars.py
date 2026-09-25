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

import sys

import tvm
import tvm.testing


def test_type_vars_roundtrip():
    func = tvm.script.from_source(
        """
M = I.dynamic("M")
UNUSED = I.dynamic("UNUSED")

@T.prim_func(private=True)
def func(A: T.Buffer((M, M * 2), "float32")):
    A[0, 0] = T.float32(1)
""",
        extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx},
    )

    script = func.script()
    if sys.version_info >= (3, 12):
        assert script.startswith("from __future__ import annotations\n\n")
        assert "def main[M](" in script
        assert 'T.Buffer((M, M * T.int64(2)), "float32")' in script
        assert "M = T.int64()" not in script
        typed = tvm.script.from_source(
            """
@T.prim_func(private=True)
def func[M: int](A: T.Buffer((M, M * 2), "float32")):
    A[0, 0] = T.float32(1)
""",
            extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx},
        )
        tvm.ir.assert_structural_equal(func, typed)
    else:
        assert "from __future__ import annotations" not in script
        assert 'M = I.dynamic("M", dtype="int64")' in script
        assert "M = T.int64()" not in script

    portable = func.script(extra_config={"script.use_pep695": False})
    assert "from __future__ import annotations" not in portable
    assert 'M = I.dynamic("M", dtype="int64")' in portable
    assert 'T.Buffer((M, M * T.int64(2)), "float32")' in portable
    assert "UNUSED" not in script
    assert "M = T.int64()" not in portable
    assert len(func.params) == 1
    assert not hasattr(func, "type_params")
    assert func.attrs.get("tirx.type_vars") is None
    tvm.ir.assert_structural_equal(
        func, tvm.script.from_source(script, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx})
    )
    tvm.ir.assert_structural_equal(
        func,
        tvm.script.from_source(portable, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx}),
    )


def test_dynamic_int32_roundtrip():
    func = tvm.script.from_source(
        """
n = I.dynamic("n", "int32")
@T.prim_func(private=True)
def func(A: T.Buffer((n,), "float32")):
    A[0] = T.float32(1)
""",
        extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx},
    )
    source = func.script()
    if sys.version_info >= (3, 12):
        assert "def main[n: T.int32](" in source
    else:
        assert 'n = I.dynamic("n", dtype="int32")' in source
    portable = func.script(extra_config={"script.use_pep695": False})
    assert 'n = I.dynamic("n", dtype="int32")' in portable
    tvm.ir.assert_structural_equal(
        func, tvm.script.from_source(source, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx})
    )
    tvm.ir.assert_structural_equal(
        func,
        tvm.script.from_source(portable, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx}),
    )


def test_dynamic_module_body_identity():
    mod = tvm.script.from_source(
        """
n = I.dynamic("n", "int32")
m = I.dynamic("n", "int32")
@I.ir_module
class Module:
    @T.prim_func(private=True)
    def first():
        T.evaluate(n)
    @T.prim_func(private=True)
    def second():
        T.evaluate(n + m)
""",
        check_well_formed=False,
        extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx},
    )
    source = mod.script()
    assert source.count('I.dynamic("n", dtype="int32")') == 2
    restored = tvm.script.from_source(
        source, check_well_formed=False, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx}
    )
    shared = restored["first"].body.value
    summed = restored["second"].body.value
    assert shared.same_as(summed.a)
    assert not shared.same_as(summed.b)
    tvm.ir.assert_structural_equal(mod, restored, map_free_vars=True)


if __name__ == "__main__":
    tvm.testing.main()
