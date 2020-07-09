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
"""ACL Integration reshape tests."""

import numpy as np

import tvm
from tvm import relay

from .infrastructure import skip_runtime_test, skip_codegen_test, build_and_run, \
    verify, verify_codegen
from .infrastructure import Device


def _get_model(input_shape, output_shape, var_names):
    """Return a model and any parameters it may have."""
    a = relay.var(next(var_names), shape=input_shape, dtype="float32")
    reshape = relay.reshape(a, output_shape)
    return reshape


def _get_expected_codegen(input_shape, output_shape):
    codegen = {
        "name": "reshape",
        "inputs": [],
        "outputs": [],
        "attrs": {}
    }

    inputs = [{"type": "var", "shape": list(input_shape)}]
    outputs = [{"type": "var", "shape": list(output_shape)}]

    codegen["inputs"] = inputs
    codegen["outputs"] = outputs
    codegen["attrs"]["num_inputs"] = ["Size_t", len(inputs)]
    codegen["attrs"]["num_outputs"] = ["Size_t", len(outputs)]

    return codegen


def test_reshape():
    if skip_runtime_test():
        return

    device = Device()

    inputs = {
        "a": tvm.nd.array(
            np.random.uniform(-128, 127, (1, 1, 1, 1000)).astype("float32"))
    }

    for shape in [(1, 1000), (10, 10, 10)]:
        outputs = []
        func = _get_model(inputs["a"].shape, shape, iter(inputs))
        for acl in [False, True]:
            outputs.append(build_and_run(func, inputs, 1, None, device,
                                         enable_acl=acl))
        verify(outputs, atol=1e-7, rtol=1e-7)


def test_codegen_reshape():
    if skip_codegen_test():
        return

    shape = (1, 1, 1, 1000)
    inputs = {"a"}

    for new_shape in [(1, 1000), (10, 10, 10)]:
        args = (shape, new_shape)
        func = _get_model(*args, iter(inputs))
        exp_codegen = _get_expected_codegen(*args)
        verify_codegen(func, exp_codegen, 1)


if __name__ == "__main__":
    test_reshape()
    test_codegen_reshape()
