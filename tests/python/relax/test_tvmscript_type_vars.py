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
from tvm.script import ir as I
from tvm.script import relax as R

M = I.dynamic("M")
UNUSED_GENERIC = I.dynamic("UNUSED_GENERIC")


def test_type_vars_roundtrip():
    @R.function(private=True)
    def func(
        x: R.Tensor((M, "M * 2"), "float32"),
    ) -> R.Tensor((M, "M * 2"), "float32"):
        return x

    script = func.script()
    if sys.version_info >= (3, 12):
        assert script.startswith("from __future__ import annotations\n\n")
        assert "def main[M](" in script
        assert 'R.Tensor((M, M * 2), dtype="float32")' in script
        assert "M = T.int64()" not in script
        typed = tvm.script.from_source(
            """
@R.function(private=True)
def func[M: int](x: R.Tensor((M, M * 2), "float32")):
    return x
"""
        )
        tvm.ir.assert_structural_equal(func, typed)
    else:
        assert "from __future__ import annotations" not in script
        assert 'M = I.dynamic("M", dtype="int64")' in script
        assert "M = T.int64()" not in script
        assert 'R.Tensor((M, M * 2), dtype="float32")' in script

    portable = func.script(extra_config={"relax.use_pep695": False})
    assert "from __future__ import annotations" not in portable
    assert 'M = I.dynamic("M", dtype="int64")' in portable
    assert 'R.Tensor((M, M * 2), dtype="float32")' in portable
    assert "M = T.int64()" not in portable
    assert "UNUSED_GENERIC" not in script
    assert [param.name for param in func.params] == ["x"]
    assert not hasattr(func, "type_params")
    assert func.attrs.get("relax.type_vars") is None
    tvm.ir.assert_structural_equal(func, tvm.script.from_source(script))
    tvm.ir.assert_structural_equal(func, tvm.script.from_source(portable))


def test_dynamic_module_symbol_identity():
    shared = I.dynamic("n")
    independent = I.dynamic("n")

    @R.function(private=True)
    def first(x: R.Tensor((shared,), "float32")):
        return x

    @R.function(private=True)
    def second(x: R.Tensor((shared, independent), "float32")):
        return x

    mod = tvm.IRModule({"first": first, "second": second})
    source = mod.script()
    assert source.count('I.dynamic("n", dtype="int64")') == 2
    restored = tvm.script.from_source(source, check_well_formed=False)
    first_n = restored["first"].params[0].ty.shape.values[0]
    second_shape = restored["second"].params[0].ty.shape.values
    assert first_n.same_as(second_shape[0])
    assert not first_n.same_as(second_shape[1])
    assert str(second_shape[1].ty.dtype) == "int64"
    tvm.ir.assert_structural_equal(mod, restored)


if __name__ == "__main__":
    tvm.testing.main()
