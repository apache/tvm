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

import tvm
from tvm.relay.backend.contrib.ethosu import legalize
from tvm import relay
from tvm.relay import dataflow_pattern
from tvm.relay.op.contrib import ethosu
from tests.python.contrib.test_ethosu.legalization import legalize_infra


@pytest.mark.parametrize(
    "ifm_shape, ifm2_shape, reversed_operands",
    [
        ([1, 2, 3, 4], [1, 2, 3, 4], False),
        ([1, 2, 3, 4], [1, 1, 3, 1], False),
        ([1, 1, 3, 1], [1, 2, 3, 4], True),
    ],
)
def test_ethosu_left_shift_binary_elemwise_legalize(ifm_shape, ifm2_shape, reversed_operands):
    dtype = "int32"
    operator_type = "SHL"

    def create_graph():
        input1 = relay.var("x1", shape=ifm_shape, dtype=dtype)
        input2 = relay.var("x2", shape=ifm2_shape, dtype=dtype)
        c1 = relay.left_shift(input1, input2)
        f = relay.Function([input1, input2], c1)
        mod = tvm.IRModule()
        mod["main"] = f
        return mod

    def verify(ext_func):
        out_shape = ifm2_shape if reversed_operands else ifm_shape
        shapes = [ifm_shape, ifm2_shape]
        ifm_index, ifm2_index = (1, 0) if reversed_operands else (0, 1)
        op = ext_func.body
        assert list(op.args[0].checked_type.shape) == shapes[ifm_index]
        assert list(op.args[1].checked_type.shape) == shapes[ifm2_index]
        assert op.args[0].checked_type.dtype == dtype
        assert list(op.checked_type.shape) == out_shape
        assert op.checked_type.dtype == dtype
        assert op.attrs.operator_type == operator_type
        assert op.attrs.reversed_operands == reversed_operands
        assert str(op.attrs.activation) == "NONE"

    rewriter = legalize.ShlRewriter()
    pattern_table = [
        (
            ethosu.ShlParams.composite_name,
            ethosu.shl_pattern(),
            lambda pat: ethosu.ShlParams(pat).is_valid(),
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
