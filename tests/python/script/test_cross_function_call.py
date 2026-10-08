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
"""Parsed functions call forward, recursive and aliased module references.

Calls retain the declared callee and the caller parameter identity.
"""

from __future__ import annotations

# Script-local bindings and failing decorated definitions are intentionally observable in IR.
from tvm.script import ir as I


def test_module_forward_reference(language):
    # Forward calls must use the later member signature and the original caller parameter.
    M = language.M

    @I.ir_module
    class Module:
        @M.function
        def first(x: M.Tensor((4,))):
            second(x)

        @M.function
        def second(y: M.Tensor((4,))):
            M.record(y)

    assert set(Module) == {"first", "second"}
    first, second = Module["first"], Module["second"]
    call = first.body[0][1]
    assert call.op == "call"
    assert call.args[0] is language.references["second"]
    assert call.args[1] is first.params[0]
    assert second.body == [("emit", second.params[0])]


def test_recursive_source_function_reuses_its_declaration(language):
    # A recursive source call must reference its own single declared function and original
    # parameter.
    M = language.M

    @M.function
    def main(x: M.Tensor((4,))):
        main(x)

    call = main.body[0][1]
    assert call.op == "call" and call.args[0] is language.references["main"]
    assert call.args[1] is main.params[0]


def test_module_alias_keeps_frame_identity_and_caller_dialect(language):
    # A module alias must preserve the original callee reference rather than becoming a dialect
    # binding.
    M = language.M

    @I.ir_module
    class Module:
        @M.function
        def caller(x: M.Tensor((4,))):
            cls = Module
            cls.callee(x)

        @M.function
        def callee(y: M.Tensor((4,))):
            M.record(y)

    call = Module["caller"].body[0][1]
    assert call.op == "call"
    assert call.args == (language.references["callee"], Module["caller"].params[0])
