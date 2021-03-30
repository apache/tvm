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
"""BNNS integration normalization tests."""

import numpy as np
import math
import pytest
import tvm
from tvm import relay
from tvm import testing
from .infrastructure import (
    Device,
    skip_runtime_test,
    skip_codegen_test,
    verify_codegen,
    build_and_run,
    verify,
    generate_trials,
)


def _get_model(
    shape, b_shape, s_shape, dtype, var_names, axis=1, epsilon=1e-5, center=True, scale=True
):
    """Return a model and any parameters it may have"""
    src = relay.var(next(var_names), shape=shape, dtype=dtype)
    params = {}
    b = tvm.nd.array(np.random.uniform(-128, 127, b_shape).astype(dtype))
    params["b"] = b
    b = relay.const(b, dtype)
    s = tvm.nd.array(np.random.uniform(-128, 127, b_shape).astype(dtype))
    params["b"] = s
    s = relay.const(s, dtype)
    out = relay.nn.instance_norm(src, s, b, axis, epsilon, center, scale)

    return out, params


def _get_expected_codegen(shape, axis, center, scale, dtype, offload_on_bnns):
    output_shape = shape
    name = "nn.instance_norm"

    node = {
        "op": "kernel",
        "name": name,
        "inputs": [],
        "attrs": {
            "num_outputs": "1",
            "axis": [[str(axis)]],
            "center": [[str(int(center))]],
            "scale": [[str(int(scale))]],
            "shape": [[list(output_shape)]],
            "dtype": [[dtype]],
            "epsilon": [["1.0000000000000001e-05"]],
        },
    }

    inputs = [
        {"op": "input", "name": "", "attrs": {"shape": [[list(shape)]], "dtype": [[str(dtype)]]}},
        {
            "op": "const",
            "name": "",
            "attrs": {"shape": [[[shape[axis]]]], "dtype": [[str(dtype)]]},
        },
        {
            "op": "const",
            "name": "",
            "attrs": {"shape": [[[shape[axis]]]], "dtype": [[str(dtype)]]},
        },
    ]

    input_idx = 0
    for _ in range(len(inputs)):
        node["inputs"].append([input_idx, 0, 0])
        input_idx += 1
    node["attrs"]["num_inputs"] = str(len(inputs))
    inputs.append(node)
    return inputs


@pytest.mark.skipif(skip_runtime_test(), reason="Skip because BNNS codegen is not available")
def test_normalization():
    device = Device()
    np.random.seed(0)
    dtype = "float32"

    shapes_config = [
        [1, 2, 3, 4],
        [3, 2, 3, 4],
        [2, 2, 3],
        [16, 32, 32],
        [5, 3],
    ]
    axes = [-1, 0, 1, 2]

    for shape in shapes_config:
        for axis in axes:
            if len(shape) == 2 and axis != 0:
                continue
            for center in [False, True]:
                for scale in [False, True]:
                    outputs = []
                    inputs = {
                        "src": tvm.nd.array(np.random.uniform(-128, 127, shape).astype(dtype)),
                    }
                    func, params = _get_model(
                        shape,
                        [shape[axis]],
                        [shape[axis]],
                        dtype,
                        var_names=iter(inputs),
                        axis=axis,
                        center=center,
                        scale=scale,
                    )
                    for enable_bnns in [False, True]:
                        outputs.append(
                            build_and_run(
                                func,
                                inputs,
                                1,
                                params,
                                device,
                                enable_bnns=enable_bnns,
                            )[0]
                        )

                    config = {
                        "dtype": dtype,
                    }
                    verify(outputs, atol=0.001, rtol=0.01, config=config)


@pytest.mark.skipif(skip_codegen_test(), reason="Skip because BNNS codegen is not available")
def test_codegen_normalization():
    np.random.seed(0)

    dtype = "float32"
    shapes_config = [
        [1, 2, 3, 4],
        [3, 2, 3, 4],
        [2, 2, 3],
        [16, 32, 32],
        [5, 3],
    ]
    axes = [-1, 0, 1, 2]

    def check_normalization(rank, axis):
        if rank < 3 or rank > 4:
            return False
        if axis == 0 and rank == 3 or axis == 1 and rank == 4:
            return True
        return False

    for shape in shapes_config:
        for axis in axes:
            if len(shape) == 2 and axis != 0:
                continue
            for center in [False, True]:
                for scale in [False, True]:
                    inputs = {"src"}

                    args = (shape, axis, center, scale, dtype)

                    func, params = _get_model(
                        shape,
                        [shape[axis]],
                        [shape[axis]],
                        dtype,
                        var_names=iter(inputs),
                        axis=axis,
                        center=center,
                        scale=scale,
                    )

                    offload_on_bnns = check_normalization(len(shape), axis)
                    if offload_on_bnns is True:
                        bnns_blocks = 1
                    else:
                        bnns_blocks = 0
                    exp_codegen = _get_expected_codegen(*args, offload_on_bnns)
                    verify_codegen(func, exp_codegen, bnns_blocks)


if __name__ == "__main__":
    test_normalization()
    test_codegen_normalization()
