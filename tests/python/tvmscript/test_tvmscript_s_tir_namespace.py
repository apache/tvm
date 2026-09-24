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
"""Independent S-TIR entry points and shared construction operations."""

import os
import subprocess
import sys

import pytest

import tvm
from tvm.script import ir as I
from tvm.script import s_tir as Ts
from tvm.script import tirx as T


def test_shared_operations_and_aliases():
    from tvm.script import s_tir as S

    @S.prim_func
    def shared(A: S.Buffer(("n",), "float32")):
        for i in T.serial(A.shape[0]):
            with S.sblock("copy"):
                v = T.axis.spatial(A.shape[0], i)
                A[v] = S.float32(1)

    assert shared.attrs["s_tir"]
    assert shared.params[0].ty.layout is None
    assert Ts.Buffer is T.Buffer
    assert Ts.serial is T.serial
    assert Ts.bind is T.bind
    assert Ts.tile is T.tile
    tvm.ir.assert_structural_equal(shared, tvm.script.from_source(shared.script()))


def test_mixed_module_roundtrip():
    @I.ir_module
    class Mixed:
        @Ts.prim_func
        def scheduled(A: Ts.Buffer((4,), "float32")):
            for i in Ts.serial(4):
                with Ts.sblock("copy"):
                    v = Ts.axis.spatial(4, i)
                    A[v] = 1.0

        @T.prim_func
        def direct(A: T.Buffer((4,), "float32")):
            for i in T.serial(4):
                A[i] = 2.0

    assert Mixed["scheduled"].attrs["s_tir"]
    assert not Mixed["direct"].attrs.get("s_tir", False)
    script = Mixed.script()
    assert "@Ts.prim_func" in script
    assert "@T.prim_func" in script
    assert "s_tir=True" not in script
    tvm.ir.assert_structural_equal(Mixed, tvm.script.from_source(script))
    direct_script = Mixed["direct"].script()
    assert "Ts" not in direct_script
    assert "import s_tir" not in direct_script


@pytest.mark.parametrize("option", ["s_tir=True", "is_stir=True"])
def test_tirx_rejects_legacy_mode(option):
    with pytest.raises(ValueError, match="Ts.prim_func"):
        tvm.script.from_source(
            f"@T.prim_func({option}, check_well_formed=False)\ndef main():\n    T.evaluate(0)\n"
        )


@pytest.mark.parametrize("namespace", ["T", "Ts"])
def test_tirx_rejects_s_tir_blocks(namespace):
    with pytest.raises(ValueError, match="Ts.prim_func"):
        tvm.script.from_source(
            "@T.prim_func(check_well_formed=False)\n"
            f"def main():\n    with {namespace}.sblock('bad'):\n        T.evaluate(0)\n"
        )


@pytest.mark.parametrize("value", [True, False])
def test_tirx_cannot_change_dialect_with_attribute(value):
    with pytest.raises(ValueError, match="Ts.prim_func"):
        tvm.script.from_source(
            "@T.prim_func(check_well_formed=False)\n"
            f"def main():\n    T.func_attr({{'s_tir': {value}}})\n    T.evaluate(0)\n"
        )


def test_s_tir_options_and_helpers():
    @Ts.inline
    def fill(A):
        for i in Ts.serial(4):
            A[i] = Ts.float32(1)

    @Ts.prim_func(private=True)
    def scheduled(A: Ts.Buffer((4,), "float32")):
        fill(A)

    assert scheduled.attrs["s_tir"]
    assert "global_symbol" not in scheduled.attrs
    tvm.ir.assert_structural_equal(scheduled, tvm.script.from_source(scheduled.script()))


@pytest.mark.parametrize("first", ["tvm.s_tir.script", "tvm.tirx.script", "tvm.script.parser"])
def test_import_order(first):
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"import {first}\n"
            "import tvm.s_tir.script as direct\n"
            "from tvm.script import s_tir as Ts, tirx as T\n"
            "from tvm.script.parser import s_tir as parser\n"
            "assert Ts is direct is parser\n"
            "assert Ts.Buffer is T.Buffer\n"
            "from tvm.script.parser import _NAMESPACES\n"
            "assert _NAMESPACES['Ts'] is Ts\n"
            "assert _NAMESPACES['T'] is T\n",
        ],
        check=True,
        env=os.environ.copy(),
    )
