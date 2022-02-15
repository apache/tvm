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
from tvm.relay.op.contrib import ethosu


def test_multiple_requantize_offload():
    """
    Testing requantize offload in the case one requantize operation is part of
    an existing pattern (in this case Mean: cast->mean->requantize) and the
    other is a stand-alone requantize.
    """

    def create_model():
        ifm = relay.var("input", shape=(1, 3, 3, 4), dtype="int8")
        cast = relay.cast(ifm, dtype="int32")
        mean = relay.mean(cast, axis=1, keepdims=True)
        requantize = relay.qnn.op.requantize(
            mean,
            input_scale=relay.const(1.0, dtype="float32"),
            input_zero_point=relay.const(0, dtype="int32"),
            output_scale=relay.const(1.0, dtype="float32"),
            output_zero_point=relay.const(0, dtype="int32"),
        )
        requantize = relay.qnn.op.requantize(
            requantize,
            input_scale=relay.const(1.0, dtype="float32"),
            input_zero_point=relay.const(0, dtype="int32"),
            output_scale=relay.const(1.0, dtype="float32"),
            output_zero_point=relay.const(0, dtype="int32"),
        )
        return tvm.IRModule.from_expr(relay.Function([ifm], requantize))

    def verify(ext_func):
        # If mean operation and separate requantize were offloaded correctly,
        # there should only be a pooling operation followed by an identity
        # operation leagalized.
        op = ext_func.body
        assert op.op.name == "contrib.ethosu.identity"
        op = op.args[0]
        assert ext_func.body.args[0].op.name == "contrib.ethosu.pooling"
        op = op.args[0]
        assert isinstance(op, relay.Var)

    mod = create_model()
    mod = ethosu.partition_for_ethosu(mod)
    mod = legalize.LegalizeEthosU()(mod)
    verify(mod["tvmgen_default_ethos_u_main_0"])


if __name__ == "__main__":
    pytest.main([__file__])
