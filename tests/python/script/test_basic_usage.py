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
"""Basic function and module construction with collision-free source names."""

from __future__ import annotations

# Script-local bindings and failing decorated definitions are intentionally observable in IR.
from tvm.script import ir as I


def test_function(language):
    M = language.M

    @M.function
    def identity(x: M.Tensor((4,))) -> M.Tensor((4,)):
        return x

    assert identity.name == "identity"
    assert identity.params[0].args[0].args[0] == (4,)
    assert identity.ret_type.args[0] == (4,)
    assert identity.body == [("return", identity.params[0])]


def test_module(language):
    M = language.M

    # An empty source class must still produce a module without inventing functions.
    @I.ir_module
    class Empty:
        pass

    assert Empty == {}

    @I.ir_module
    class Module:
        @M.function
        def first(x: M.Tensor((4,))):
            return x

        @M.function
        def second(y: M.Tensor((4,))):
            return Module.first(y)

    assert set(Module) == {"first", "second"}
    assert Module["first"].body == [("return", Module["first"].params[0])]
    returned = Module["second"].body[0]
    assert returned[0] == "return"
    call = returned[1]
    assert call.op == "call" and call.args[0].args == ("first",)
    assert call.args[1] is Module["second"].params[0]
    assert Module["first"].params[0].args[0].args[0] == (4,)
    assert Module["second"].params[0].args[0].args[0] == (4,)


def test_name_collision(language):
    # Generated helpers must not steal identifiers already bound in source.
    M = language.M

    @M.function
    def main(_builder0: M.Tensor((4,))):
        _fn0 = 5
        _build0 = 6
        _X1 = 7
        M.record(_builder0)
        M.record(_fn0 + _build0 + _X1)

    assert main.body[0][1] is main.params[0]
    assert main.body[1][1] == 18
