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

from tvm import relay
from tvm.ir import GlobalVar, structural_equal
from tvm.ir.supply import NameSupply
from tvm.ir.supply import GlobalVarSupply


def test_name_supply():
    name_supply = NameSupply("prefix")
    name_supply.reserve_name("test")

    assert name_supply.contains_name("test")
    assert name_supply.fresh_name("test") == "prefix_test_1"
    assert name_supply.contains_name("test_1")
    assert not name_supply.contains_name("test_1", False)
    assert not name_supply.contains_name("test_2")


def test_global_var_supply_from_none():
    var_supply = GlobalVarSupply()
    global_var = GlobalVar("test")
    var_supply.reserve_global(global_var)

    assert structural_equal(var_supply.unique_global_for("test"), global_var)
    assert not structural_equal(var_supply.fresh_global("test"), global_var)


def test_global_var_supply_from_name_supply():
    name_supply = NameSupply("prefix")
    var_supply = GlobalVarSupply(name_supply)
    global_var = GlobalVar("test")
    var_supply.reserve_global(global_var)

    assert structural_equal(var_supply.unique_global_for("test", False), global_var)
    assert not structural_equal(var_supply.unique_global_for("test"), global_var)


def test_global_var_supply_from_ir_mod():
    x = relay.var("x")
    y = relay.var("y")
    mod = tvm.IRModule()
    global_var = GlobalVar("test")
    mod[global_var] = relay.Function([x, y], relay.add(x, y))
    var_supply = GlobalVarSupply(mod)

    second_global_var = var_supply.fresh_global("test", False)

    assert structural_equal(var_supply.unique_global_for("test", False), global_var)
    assert not structural_equal(var_supply.unique_global_for("test"), global_var)
    assert not structural_equal(second_global_var, global_var)


if __name__ == "__main__":
    tvm.testing.main()
