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

from typing import Dict, List

import tvm
import tvm.testing
from tvm.ir import GlobalVar

from tvm.script import ir as I, tir as T, relax as R

from tvm.ir.analysis import collect_call_map


def _build_str_map(call_map: Dict[GlobalVar, List[GlobalVar]]) -> Dict[str, List[str]]:
    return {
        caller.name_hint: [callee.name_hint for callee in callees]
        for caller, callees in call_map.items()
    }


def test_collect_relax_to_relax():
    @I.ir_module
    class Module:
        @R.function
        def main():
            return Module.subroutine()

        @R.function
        def subroutine():
            return R.tuple()

    call_map = collect_call_map(Module)
    str_map = _build_str_map(call_map)
    expected = {
        "main": ["subroutine"],
        "subroutine": [],
    }
    assert str_map == expected


def test_collect_relax_to_tir():
    @I.ir_module
    class Module:
        @R.function
        def main() -> R.Prim("int32"):
            return Module.subroutine(R.prim_value(T.int32(42)))

        @T.prim_func
        def subroutine(i: T.int32) -> T.int32:
            return i + 1

    call_map = collect_call_map(Module)
    str_map = _build_str_map(call_map)
    expected = {
        "main": ["subroutine"],
        "subroutine": [],
    }
    assert str_map == expected


def test_collect_tir_to_tir():
    @I.ir_module
    class Module:
        @T.prim_func
        def main() -> T.int32:
            return Module.subroutine(42)

        @T.prim_func
        def subroutine(i: T.int32) -> T.int32:
            return i + 1

    call_map = collect_call_map(Module)
    str_map = _build_str_map(call_map)
    expected = {
        "main": ["subroutine"],
        "subroutine": [],
    }
    assert str_map == expected


if __name__ == "__main__":
    tvm.testing.main()
