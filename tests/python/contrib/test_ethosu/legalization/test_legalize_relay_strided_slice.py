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
    "ifm_shape, begin, end",
    [
        ([1, 10, 50, 4], [0, 5, 11, 2], [1, 10, 22, 3]),
        ([1, 101, 35, 27], [0, 5, 11, 2], [1, 10, 22, 3]),
        ([15, 17, 3], [3, 0, 0], [11, 17, 1]),
        ([1, 6043], [0, 704], [1, 800]),
    ],
)
def test_relay_strided_slice_legalize(ifm_shape, begin, end):

    ifm = relay.var("ifm", shape=ifm_shape, dtype="int8")
    strided_slice = relay.op.strided_slice(ifm, begin, end)
    func = relay.Function([ifm], strided_slice)
    mod = tvm.IRModule()
    mod["main"] = func
    mod = relay.transform.InferType()(mod)

    strided_slice_pattern_table = [
        (
            ethosu.StridedSliceParams.composite_name,
            ethosu.strided_slice_pattern(),
            lambda pat: ethosu.StridedSliceParams(pat).is_valid(),
        ),
    ]

    mod = legalize_infra.partition_ethosu_by_table(mod, strided_slice_pattern_table)
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.StridedSliceRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.NoOpRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    mod = relay.transform.InferType()(mod)

    ext_func = mod["tvmgen_default_ethos_u_main_0"]

    identity = ext_func.body
    assert identity.op.name == "contrib.ethosu.identity"

    # check that the strided_slice is still there
    strided_slice = identity.args[0]
    assert strided_slice.op.name == "strided_slice"

    # check that identity's output shape matches strided slice's output shape
    slice_shape = [a - b for a, b in zip(end, begin)]
    assert list(identity.checked_type.shape) == slice_shape


if __name__ == "__main__":
    pytest.main([__file__])
