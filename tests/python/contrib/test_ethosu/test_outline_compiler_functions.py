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

"""
Test the outline compiler functions pass.
"""

import pytest

pytest.importorskip("ethosu.vela")

import tvm
from tvm import relay
from tvm.relay.backend.contrib.ethosu.codegen import OutlineCompilerFunctions


def test_outline_compiler_functions():
    compiler_name = "my-compiler"
    wrong_compiler_name = "wrong-compiler"

    def before():
        inp = relay.var("input")

        # Inlined functions for "my-compiler"
        x = relay.var("x", shape=(1, 2, 2, 4))
        x = relay.reshape(x, newshape=(1, 4, 4))
        x = relay.Function(relay.analysis.free_vars(x), x)
        x = x.with_attr("Compiler", compiler_name)
        x = x.with_attr("global_symbol", "ext_func")

        # Inlined function for "wrong-compiler"
        y = relay.var("y", shape=(1, 4, 4))
        y = relay.reshape(y, newshape=(1, 16))
        y = relay.Function(relay.analysis.free_vars(y), y)
        y = y.with_attr("Compiler", wrong_compiler_name)
        y = y.with_attr("global_symbol", "ext_func_2")

        out = relay.Call(x, [inp])
        out = relay.Call(y, [out])
        out = relay.Function([inp], out)
        return tvm.ir.IRModule.from_expr(out)

    def expected():
        mod = tvm.ir.IRModule()

        inp = relay.var("input")

        x = relay.var("x", shape=(1, 2, 2, 4))
        x = relay.reshape(x, newshape=(1, 4, 4))
        x = relay.Function(relay.analysis.free_vars(x), x)
        x = x.with_attr("Compiler", compiler_name)
        x = x.with_attr("global_symbol", "ext_func")
        mod["ext_func"] = x

        y = relay.var("y", shape=(1, 4, 4))
        y = relay.reshape(y, newshape=(1, 16))
        y = relay.Function(relay.analysis.free_vars(y), y)
        y = y.with_attr("Compiler", wrong_compiler_name)
        y = y.with_attr("global_symbol", "ext_func_2")

        out = relay.Call(mod.get_global_var("ext_func"), [inp])
        out = relay.Call(y, [out])
        mod["main"] = relay.Function([inp], out)
        return mod

    after = OutlineCompilerFunctions(compiler_name)(before())
    exp = expected()

    global_vars = [str(gv) for gv in after.get_global_vars()]
    assert 'I.GlobalVar("ext_func")' in global_vars
    assert 'I.GlobalVar("ext_func_2")' not in global_vars
    assert tvm.ir.structural_equal(after["ext_func"], exp["ext_func"])
