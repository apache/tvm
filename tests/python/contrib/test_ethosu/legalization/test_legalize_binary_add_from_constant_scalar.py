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
# pylint: disable=invalid-name, unused-argument

import pytest

pytest.importorskip("ethosu.vela")

import numpy as np

import tvm
from tvm.relay.backend.contrib.ethosu import legalize
from tvm import relay
from tvm.relay import dataflow_pattern
from tvm.relay.op.contrib import ethosu
from tests.python.contrib.test_ethosu.legalization import legalize_infra


def test_binary_add_from_constant_scalar():
    dtype = "uint8"
    ifm_shape = (1, 4, 4, 8)

    def create_graph():
        inp = relay.var("input", shape=ifm_shape, dtype=dtype)
        scalar = relay.const(np.ones((1, 1, 1, 1), dtype=dtype), dtype=dtype)
        add = relay.qnn.op.add(
            inp,
            scalar,
            relay.const(1.0, dtype="float32"),
            relay.const(0, dtype="int32"),
            relay.const(1.0, dtype="float32"),
            relay.const(0, dtype="int32"),
            relay.const(1.0, dtype="float32"),
            relay.const(0, dtype="int32"),
        )
        func = relay.Function(relay.analysis.free_vars(add), add)
        return tvm.IRModule.from_expr(func)

    def verify(ext_func):
        op = ext_func.body
        assert list(op.args[0].checked_type.shape) == [1, 4, 4, 8]
        assert list(op.args[1].checked_type.shape) == [1, 1, 1, 1]
        assert op.args[0].checked_type.dtype == "uint8"
        assert list(op.checked_type.shape) == [1, 4, 4, 8]
        assert op.checked_type.dtype == "uint8"
        assert op.attrs.operator_type == "ADD"

    rewriter = legalize.AddRewriter()
    pattern_table = [
        (
            ethosu.AddParams.composite_name,
            ethosu.qnn_add_pattern(),
            lambda pat: ethosu.AddParams(pat).is_valid(),
        ),
    ]

    mod = create_graph()
    mod = legalize_infra.partition_ethosu_by_table(mod, pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        rewriter, mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


if __name__ == "__main__":
    pytest.main([__file__])
