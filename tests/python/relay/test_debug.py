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
from tvm.relay import var, const, create_executor
from tvm.relay.op import debug


_test_debug_hit = False


def test_debug():
    global _test_debug_hit
    x = var("x", shape=(), dtype="int32")
    _test_debug_hit = False

    def did_exec(x):
        global _test_debug_hit
        _test_debug_hit = True

    prog = debug(x, debug_func=did_exec)
    result = create_executor().evaluate(prog, {x: const(1, "int32")})
    assert _test_debug_hit
    assert result.numpy() == 1


def test_debug_with_expr():
    global _test_debug_hit
    _test_debug_hit = False
    x = var("x", shape=(), dtype="int32")
    _test_debug_hit = False

    def did_exec(x):
        global _test_debug_hit
        _test_debug_hit = True

    prog = debug(x + x * x, debug_func=did_exec)
    result = create_executor().evaluate(prog, {x: const(2, "int32")})
    assert _test_debug_hit
    assert result.numpy() == 6
