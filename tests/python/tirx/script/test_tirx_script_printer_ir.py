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
# pylint: disable=missing-docstring

import pytest
from tvm_ffi import get_global_func

from tvm import tirx
from tvm.runtime.script_printer import _script
from tvm.script import ir_builder as I
from tvm.script.ir_builder import IRBuilder
from tvm.tirx.script import ir_builder as T


def test_failed_invalid_prefix():
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module():
            with T.prim_func():
                T.func_name_("foo")
    mod = ib.get()

    with pytest.raises(RuntimeError):
        mod.script(ir_prefix="2I")


def test_config_extension_passthrough():
    make_config = get_global_func("node.PrinterConfig")
    cfg = make_config(
        {
            "extension.option": 7,
            "custom_key": "value",
            "syntax_sugar": False,
            "render_invisible_path_info": True,
            "tirx.prefix": "invalid-prefix",
            "extra_config": {
                "extension.option": 9,
                "render_invisible_path_info": False,
                "tirx.prefix": "Custom",
            },
        }
    )
    assert cfg.extra_config["extension.option"] == 9
    assert cfg.extra_config["custom_key"] == "value"
    assert "syntax_sugar" not in cfg.extra_config
    assert "extra_config" not in cfg.extra_config
    assert not cfg.syntax_sugar
    assert not cfg.render_invisible_path_info
    assert make_config({}).syntax_sugar
    assert _script(tirx.Var("Custom", "int32"), cfg) == "Custom_1"


@pytest.mark.parametrize("key", ["tirx.prefix", "relax.prefix", "s_tir.prefix"])
@pytest.mark.parametrize("value", ["2prefix", 17])
@pytest.mark.parametrize("nested", [False, True])
def test_config_validates_dialect_prefixes(key, value, nested):
    config = {key: value}
    if nested:
        config = {"extra_config": config}
    with pytest.raises((RuntimeError, TypeError)):
        get_global_func("node.PrinterConfig")(config)


@pytest.mark.parametrize(
    "prefixes",
    [{}, {"tirx.prefix": "CustomT", "relax.prefix": "CustomR", "s_tir.prefix": "CustomTs"}],
)
def test_config_reserves_dialect_prefixes_before_variable_definition(prefixes):
    tir_prefix = prefixes.get("tirx.prefix", "T")
    relax_prefix = prefixes.get("relax.prefix", "R")
    for name in [tir_prefix, relax_prefix, prefixes.get("s_tir.prefix", "Ts")]:
        var = tirx.Var(name, "int32")
        assert var.script(verbose_expr=True, extra_config=prefixes).strip() == (
            f'{name}_1 = I.dynamic("{name}", dtype="int32")\n{name}_1'
        )
