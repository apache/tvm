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
M = TypeVar("M")
UNUSED = TypeVar("UNUSED")

@T.prim_func(private=True)
def func(A: T.Buffer((M, M * 2), "float32")):
    A[0, 0] = T.float32(1)
"""
    )

    script = func.script()
    if sys.version_info >= (3, 12):
        assert script.startswith("from __future__ import annotations\n\n")
        assert "def main[M](" in script
        assert 'T.Buffer((M, M * T.int64(2)), "float32")' in script
        typed = tvm.script.from_source(
            """
@T.prim_func(private=True)
def func[M: int](A: T.Buffer((M, M * 2), "float32")):
    A[0, 0] = T.float32(1)
"""
        )
        tvm.ir.assert_structural_equal(func, typed)
    else:
        assert "from __future__ import annotations" not in script
        assert 'M = TypeVar("M")' in script

    portable = func.script(extra_config={"script.use_pep695": False})
    assert "from __future__ import annotations" not in portable
    assert 'M = TypeVar("M")' in portable
    assert 'T.Buffer((M, "M * T.int64(2)"), "float32")' in portable
    assert "UNUSED" not in script
    assert "M = T.int64()" not in script
    assert len(func.params) == 1
    assert not hasattr(func, "type_params")
    assert func.attrs.get("tirx.type_vars") is None
    tvm.ir.assert_structural_equal(func, tvm.script.from_source(script))
    tvm.ir.assert_structural_equal(func, tvm.script.from_source(portable))


if __name__ == "__main__":
    tvm.testing.main()
