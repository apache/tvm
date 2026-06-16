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
from tvm import relax as rx
from tvm.ir.supply import UniqueNameSupply


def _empty_relax_func():
    return rx.Function([], rx.Tuple([]))


def test_fresh_name_empty_string():
    """Empty name should produce a valid variable name, not an empty string."""
    ns = UniqueNameSupply("")
    name = ns.fresh_name("", add_prefix=False)
    assert name == "v"
    name2 = ns.fresh_name("", add_prefix=False)
    assert name2 == "v_1"


def test_fresh_name_empty_string_with_prefix():
    """Empty name with prefix should produce a valid variable name."""
    ns = UniqueNameSupply("prefix")
    name = ns.fresh_name("", add_prefix=True)
    assert name == "prefix_v"
    name2 = ns.fresh_name("", add_prefix=True)
    assert name2 == "prefix_v_1"


def test_ir_module_from_expr_freshens_main_collision():
    main_gv = tvm.ir.GlobalVar("main")
    mod = tvm.IRModule.from_expr(_empty_relax_func(), {main_gv: _empty_relax_func()})

    assert sorted(gvar.name_hint for gvar in mod.get_global_vars()) == ["main", "main_1"]


def test_ir_module_from_expr_reuses_existing_global_symbol():
    foo_gv = tvm.ir.GlobalVar("foo")
    func = _empty_relax_func().with_attr("global_symbol", "foo")
    mod = tvm.IRModule.from_expr(func, {foo_gv: _empty_relax_func()})

    assert mod.get_global_var("foo").same_as(foo_gv)
    assert [gvar.name_hint for gvar in mod.get_global_vars()] == ["foo"]


if __name__ == "__main__":
    tvm.testing.main()
