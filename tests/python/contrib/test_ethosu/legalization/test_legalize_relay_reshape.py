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
    "ifm_shape, new_shape",
    [
        ((1, 4, 1, 2), (4, 2)),
        ((1, 5, 1, 20), (100,)),
        ((12, 20), (1, 6, 4, 10)),
        ((30,), (10, 1, 3)),
    ],
)
def test_relay_reshape_legalize(ifm_shape, new_shape):

    ifm = relay.var("ifm", shape=ifm_shape, dtype="int8")
    reshape = relay.op.reshape(ifm, new_shape)
    func = relay.Function([ifm], reshape)
    mod = tvm.IRModule()
    mod["main"] = func
    mod = relay.transform.InferType()(mod)

    reshape_pattern_table = [
        (
            ethosu.ReshapeParams.composite_name,
            ethosu.reshape_pattern(),
            lambda pat: ethosu.ReshapeParams(pat).is_valid(),
        ),
    ]

    mod = legalize_infra.partition_ethosu_by_table(mod, reshape_pattern_table)
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.ReshapeRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.NoOpRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    mod = relay.transform.InferType()(mod)

    ext_func = mod["tvmgen_default_ethos_u_main_0"]

    identity = ext_func.body
    assert identity.op.name == "contrib.ethosu.identity"

    # check that the reshape is still there
    reshape = identity.args[0]
    assert reshape.op.name == "reshape"

    # check that identity's output shape matches reshape's output shape
    assert tuple(identity.checked_type.shape) == new_shape


if __name__ == "__main__":
    pytest.main([__file__])
