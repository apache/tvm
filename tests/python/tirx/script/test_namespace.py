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
"""TIRx script dialect selection."""

import pytest

import tvm_tirx
from tvm_tirx.script import tirx as T


@pytest.mark.parametrize("option", ["s_tir", "is_stir"])
def test_raw_tirx_builder_rejects_legacy_mode(option):
    from tvm_tirx.tirx.script import ir_builder as builder

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        builder.function(**{option: True})


def test_jit_rejects_legacy_mode():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        T.jit(**{"is_stir": True})


@pytest.mark.parametrize("option", ["s_tir", "is_stir"])
def test_legacy_mode_is_not_a_function_option(option):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        tvm_tirx.script.from_source(
            f"@T.function({option}=True, check_well_formed=False)\n"
            "def main():\n    T.evaluate(0)\n",
            extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx},
        )
