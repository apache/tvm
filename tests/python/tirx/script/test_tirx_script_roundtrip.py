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


def nested_seqstmt():
    """Nested SeqStmt should be normalized to flat SeqStmt

    Nested SeqStmt are representable in the TIR structures, but are
    flattened when converted to TVMScript.  Previously, this could
    cause failures to round-trip through TVMScript, including
    erroneous use of TVMScript's concise-scoping rules.  This was
    resolved by normalizing nested SeqStmt in TIR, such that the use
    of `tirx.SeqStmt` below results in a single flat `tirx.SeqStmt`
    containing the three `tirx.Evaluate` calls.
    """
    func = tvm.tirx.PrimFunc(
        params=[],
        body=tvm.tirx.SeqStmt(
            [
                tvm.tirx.SeqStmt([tvm.tirx.Evaluate(0), tvm.tirx.Evaluate(1)]),
                tvm.tirx.Evaluate(2),
            ]
        ),
    )

    return func


ir_generator = tvm.testing.parameter(nested_seqstmt)


_NOT_ROUNDTRIP_STABLE: set[str] = set()


def test_roundtrip(ir_generator):
    if getattr(ir_generator, "__name__", "") in _NOT_ROUNDTRIP_STABLE:
        import pytest

        pytest.skip(f"{ir_generator.__name__}: not round-trip stable here")
    original = ir_generator()
    after_roundtrip = tvm.script.from_source(
        original.script(show_meta=True),
        check_well_formed=False,
        extra_vars={
            "I": tvm.script.ir,
            "T": tvm.script.tirx,
        },
    )
    tvm.ir.assert_structural_equal(original, after_roundtrip, True)


if __name__ == "__main__":
    tvm.testing.main()
