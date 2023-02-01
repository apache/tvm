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
from tvm.relay.testing import run_opt_pass
from tvm.relay.backend.contrib.ethosu.tir.compiler import extract_constants

import numpy as np


def test_extract_constants_single():
    def _get_func():
        var_input = relay.var("data", shape=(10, 10), dtype="uint8")
        const_data = np.random.uniform(0, 255, (10, 10)).astype("uint8")
        const_input = relay.const(const_data, dtype="uint8")
        out = relay.add(var_input, const_input)
        func = relay.Function(relay.analysis.free_vars(out), out)
        func = run_opt_pass(func, relay.transform.InferType())
        return func, const_input

    def _expected():
        var_input1 = relay.var("data", shape=(10, 10), dtype="uint8")
        var_input2 = relay.var("p1", shape=(10, 10), dtype="uint8")
        out = relay.add(var_input1, var_input2)
        func = relay.Function(relay.analysis.free_vars(out), out)
        func = run_opt_pass(func, relay.transform.InferType())
        return func

    func, const = _get_func()
    new_func, const_dict = extract_constants(func)
    assert tvm.ir.structural_equal(new_func, _expected())
    assert 1 in const_dict
    assert (const_dict[1] == const.data.asnumpy()).all()


def test_extract_constants_multi():
    def _get_func():
        var_input1 = relay.var("data1", shape=(10, 10), dtype="uint8")
        var_input2 = relay.var("data2", shape=(10, 10), dtype="uint8")
        const_data_1 = np.random.uniform(0, 255, (10, 10)).astype("uint8")
        const_data_2 = np.random.uniform(0, 255, (10, 10)).astype("uint8")
        const_data_3 = np.random.uniform(0, 255, (10, 10)).astype("uint8")
        const_data_4 = np.random.uniform(0, 255, (10, 10)).astype("uint8")
        const_input_1 = relay.const(const_data_1, dtype="uint8")
        const_input_2 = relay.const(const_data_2, dtype="uint8")
        const_input_3 = relay.const(const_data_3, dtype="uint8")
        const_input_4 = relay.const(const_data_4, dtype="uint8")
        out = relay.add(var_input1, var_input2)
        out = relay.add(out, const_input_1)
        out = relay.add(out, const_input_2)
        out = relay.add(out, const_input_3)
        out = relay.add(out, const_input_4)
        func = relay.Function(relay.analysis.free_vars(out), out)
        func = run_opt_pass(func, relay.transform.InferType())
        return func, [const_input_1, const_input_2, const_input_3, const_input_4]

    def _expected():
        var_input1 = relay.var("data1", shape=(10, 10), dtype="uint8")
        var_input2 = relay.var("data2", shape=(10, 10), dtype="uint8")
        var_input3 = relay.var("p1", shape=(10, 10), dtype="uint8")
        var_input4 = relay.var("p2", shape=(10, 10), dtype="uint8")
        var_input5 = relay.var("p3", shape=(10, 10), dtype="uint8")
        var_input6 = relay.var("p4", shape=(10, 10), dtype="uint8")
        out = relay.add(var_input1, var_input2)
        out = relay.add(out, var_input3)
        out = relay.add(out, var_input4)
        out = relay.add(out, var_input5)
        out = relay.add(out, var_input6)
        func = relay.Function(relay.analysis.free_vars(out), out)
        func = run_opt_pass(func, relay.transform.InferType())
        return func

    func, consts = _get_func()
    new_func, const_dict = extract_constants(func)
    assert tvm.ir.structural_equal(new_func, _expected())
    for i, const in enumerate(consts):
        assert i + 2 in const_dict
        assert (const_dict[i + 2] == consts[i].data.asnumpy()).all()


if __name__ == "__main__":
    tvm.testing.main()
