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

pytest.importorskip("ethosu.vela")
import tvm
from tvm import relay
from tvm.relay.backend.contrib.ethosu.tir.compiler import _lower_to_tir
from . import infra


def _create_single_conv2d():
    ifm = relay.var("x", shape=(1, 8, 8, 4), dtype="int8")
    conv1 = infra.make_ethosu_conv2d(ifm, 4, 4, (3, 3), (1, 1), (1, 1), (1, 1))
    func = relay.Function(relay.analysis.free_vars(conv1), conv1)
    return func


def _create_double_conv2d():
    ifm = relay.var("x", shape=(1, 8, 8, 4), dtype="int8")
    conv1 = infra.make_ethosu_conv2d(ifm, 4, 4, (3, 3), (1, 1), (1, 1), (1, 1))
    conv2 = infra.make_ethosu_conv2d(conv1, 4, 7, (2, 2), (1, 1), (1, 1), (1, 1))
    func = relay.Function(relay.analysis.free_vars(conv2), conv2)
    return func


def _create_non_linear_conv2d():
    shape = (1, 8, 8, 4)
    ifm1 = relay.var("x", shape=shape, dtype="int8")
    ifm2 = relay.var("y", shape=shape, dtype="int8")
    conv1 = infra.make_ethosu_conv2d(ifm1, 4, 4, (3, 3), (1, 1), (1, 1), (1, 1))
    conv2 = infra.make_ethosu_conv2d(ifm2, 4, 4, (3, 3), (1, 1), (1, 1), (1, 1))
    add = infra.make_ethosu_binary_elementwise(conv1, conv2, shape[3], shape[3], "ADD", "int8")
    func = relay.Function(relay.analysis.free_vars(add), add)
    return func


@pytest.mark.parametrize(
    "relay_function, arg_count",
    [(_create_single_conv2d, 2), (_create_double_conv2d, 2), (_create_non_linear_conv2d, 3)],
)
def test_lower_to_tir_arg_count(relay_function, arg_count):
    mod = tvm.IRModule()
    mod["main"] = relay_function()
    mod = relay.transform.InferType()(mod)
    tir_mod = _lower_to_tir(mod["main"])[0]
    primfunc = tir_mod["main"]
    assert len(primfunc.params) == arg_count


if __name__ == "__main__":
    tvm.testing.main()
