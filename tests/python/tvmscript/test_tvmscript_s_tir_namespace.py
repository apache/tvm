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

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import s_tir as Ts
from tvm.script import tirx as T


def test_shared_operations_and_aliases():
    from tvm.script import s_tir as S

    n = S.dynamic("n")

    @S.prim_func
    def shared(A: S.Buffer((n,), "float32")):
        for i in T.serial(A.shape[0]):
            with S.sblock("copy"):
                v = Ts.axis.spatial(A.shape[0], i)
                A[v] = S.float32(1)

    assert shared.attrs["s_tir"]
    assert shared.params[0].ty.layout is None
    assert Ts.Buffer is T.Buffer
    assert Ts.serial is T.serial
    assert Ts.bind is T.bind
    assert Ts.tile is T.tile
    tvm.ir.assert_structural_equal(
        shared,
        tvm.script.from_source(
            shared.script(),
            extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx, "Ts": tvm.script.s_tir},
        ),
    )


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
    tvm.ir.assert_structural_equal(
        Mixed,
        tvm.script.from_source(
            script, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx, "Ts": tvm.script.s_tir}
        ),
    )
    direct_script = Mixed["direct"].script()
    assert "Ts" not in direct_script
    assert "import s_tir" not in direct_script


@pytest.mark.skipif(not tvm.runtime.enabled("llvm"), reason="LLVM is not enabled")
def test_tirx_construction_roundtrip_and_execution_are_independent(monkeypatch):
    def reject_s_tir_analysis(*args, **kwargs):
        pytest.fail("TIRx construction and compilation must not invoke S-TIR verification")

    monkeypatch.setattr(tvm.s_tir.analysis, "verify_well_formed", reject_s_tir_analysis)

    @I.ir_module
    class Direct:
        @T.prim_func
        def main(A: T.Buffer((4,), "int32")):
            for i in T.serial(4):
                A[i] = A[i] + 3

    restored = tvm.script.from_source(
        Direct.script(),
        extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx, "Ts": tvm.script.s_tir},
    )
    tvm.ir.assert_structural_equal(Direct, restored)
    assert "s_tir" not in restored["main"].attrs
    executable = tvm.compile(restored, target="llvm")
    data = tvm.runtime.tensor(np.arange(4, dtype="int32"))
    executable["main"](data)
    np.testing.assert_array_equal(data.numpy(), np.arange(4, dtype="int32") + 3)


@pytest.mark.parametrize("option", ["s_tir", "is_stir"])
def test_raw_tirx_builder_rejects_legacy_mode(option):
    from tvm.tirx.script import ir_builder as builder

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        builder.prim_func(**{option: True})


def test_jit_rejects_legacy_mode():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        T.jit(**{"is_stir": True})


@pytest.mark.parametrize("namespace", ["T", "Ts"])
@pytest.mark.parametrize("option", ["s_tir", "is_stir"])
def test_legacy_mode_is_not_a_function_option(namespace, option):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        tvm.script.from_source(
            f"@{namespace}.prim_func({option}=True, check_well_formed=False)\n"
            "def main():\n    T.evaluate(0)\n",
            extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx, "Ts": tvm.script.s_tir},
        )


@pytest.mark.parametrize(
    "operation",
    [
        "sblock",
        "init",
        "where",
        "reads",
        "writes",
        "sblock_attr",
        "sblock_alloc_buffer",
        "axis",
        "block_name_suffix_context",
    ],
)
def test_s_tir_operations_have_an_independent_namespace(operation):
    from tvm.s_tir.script import ir_builder as s_tir_builder
    from tvm.tirx.script import ir_builder as tirx_builder

    assert hasattr(Ts, operation)
    assert hasattr(s_tir_builder, operation)
    assert not hasattr(T, operation)
    assert not hasattr(tirx_builder, operation)


def test_tirx_rejects_s_tir_blocks():
    with pytest.raises(ValueError, match="Ts.prim_func"):
        tvm.script.from_source(
            "@T.prim_func(check_well_formed=False)\n"
            "def main():\n    with Ts.sblock('bad'):\n        T.evaluate(0)\n",
            extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx, "Ts": tvm.script.s_tir},
        )


@pytest.mark.parametrize("value", [True, False])
def test_tirx_cannot_change_dialect_with_attribute(value):
    with pytest.raises(ValueError, match="Ts.prim_func"):
        tvm.script.from_source(
            "@T.prim_func(check_well_formed=False)\n"
            f"def main():\n    T.func_attr({{'s_tir': {value}}})\n    T.evaluate(0)\n",
            extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx, "Ts": tvm.script.s_tir},
        )


@pytest.mark.parametrize("value", [True, False])
def test_tirx_cannot_change_dialect_from_exit_callback(value):
    from tvm.script.ir_builder import IRBuilder
    from tvm.tirx.script import ir_builder as builder

    with pytest.raises(ValueError, match="Ts.prim_func"):
        with IRBuilder():
            with builder.prim_func() as frame:
                frame.add_callback(lambda: builder.func_attr({"s_tir": value}))
                builder.evaluate(0)


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
    tvm.ir.assert_structural_equal(
        scheduled,
        tvm.script.from_source(
            scheduled.script(),
            extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx, "Ts": tvm.script.s_tir},
        ),
    )


@pytest.mark.parametrize(
    "first",
    [
        "tvm.s_tir.script",
        "tvm.tirx.script",
        "tvm.script.parser",
        "tvm.s_tir.script.ir_builder.frame",
        "tvm.script.ir_builder.s_tir.frame",
    ],
)
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
            "from tvm.s_tir.script import ir_builder as owned\n"
            "from tvm.script.ir_builder import s_tir as alias\n"
            "import tvm.s_tir.script.ir_builder.frame as owned_frame\n"
            "import tvm.script.ir_builder.s_tir.frame as alias_frame\n"
            "assert Ts.ir_builder is owned is alias\n"
            "assert owned.__name__ == 'tvm.s_tir.script.ir_builder'\n"
            "assert owned_frame is alias_frame\n"
            "assert Ts.ir_builder is not T.ir_builder\n"
            "from tvm.script.parser import _NAMESPACES\n"
            "assert _NAMESPACES['s_tir'] is Ts\n"
            "assert _NAMESPACES['tirx'] is T\n",
        ],
        check=True,
        env=os.environ.copy(),
    )
