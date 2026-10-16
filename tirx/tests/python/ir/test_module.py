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
import pytest

import tvm_tirx
from tvm_tirx.ir.transform.instrument import PrintAfterAll, PrintBeforeAll
from tvm_tirx.ir.unique_name_supply import UniqueNameSupply


@pytest.mark.parametrize(
    "prefix,add_prefix,expected", [("", False, "v"), ("prefix", True, "prefix_v")]
)
def test_fresh_name_empty_string(prefix, add_prefix, expected):
    names = UniqueNameSupply(prefix)
    assert names.fresh_name("", add_prefix=add_prefix) == expected
    assert names.fresh_name("", add_prefix=add_prefix) == expected + "_1"


def test_module_pass_print_instruments(capsys):
    @tvm_tirx.ir.transform.module_pass(opt_level=0, name="IdentityModule")
    def identity(mod, ctx):
        return mod

    mod = tvm_tirx.IRModule()
    with tvm_tirx.ir.transform.PassContext(instruments=[PrintBeforeAll(), PrintAfterAll()]):
        result = identity(mod)
    assert result.same_as(mod)
    output = capsys.readouterr().out
    assert "Before Running Pass:" in output
    assert "After Running Pass:" in output
    assert 'name="IdentityModule"' in output
