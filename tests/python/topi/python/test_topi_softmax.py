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
"""Test code for softmax"""
import logging
import os
import sys

import numpy as np
import pytest

import tvm
import tvm.testing
import tvm.topi.testing
from tvm import te, topi
from tvm.topi.utils import get_const_tuple


_softmax_schedule = {
    "generic": topi.generic.schedule_softmax,
    "cpu": topi.x86.schedule_softmax,
    "gpu": topi.cuda.schedule_softmax,
    "hls": topi.hls.schedule_softmax,
}


dtype = tvm.testing.parameter("float32", "float64")


configs = {
    "softmax": {
        "topi": topi.nn.softmax,
        "ref": tvm.topi.testing.softmax_python,
        "dimensions": [1, 2, 4],
    },
    "log_softmax": {
        "topi": topi.nn.log_softmax,
        "ref": tvm.topi.testing.log_softmax_python,
        "dimensions": [2],
    },
}
shapes = [(32, 10), (3, 4), (1, 16, 256, 256), (32,)]
softmax_operation, shape = tvm.testing.parameters(
    *[
        (name, shape)
        for name, config in configs.items()
        for shape in shapes
        if len(shape) in config["dimensions"]
    ]
)


@tvm.testing.fixture(cache_return_value=True)
def ref_data(shape, dtype, softmax_operation):
    ref_func = configs[softmax_operation]["ref"]

    a_np = np.random.uniform(size=shape).astype(dtype)

    if len(shape) == 1:
        a_np_2d = a_np[None, :]
        b_np_2d = tvm.topi.testing.softmax_python(a_np_2d)
        b_np = b_np_2d[0]
    elif len(shape) == 2:
        b_np = ref_func(a_np)
    elif len(shape) == 4:
        _, c, h, w = a_np.shape
        a_np_2d = a_np.transpose(0, 2, 3, 1).reshape(h * w, c)
        b_np_2d = tvm.topi.testing.softmax_python(a_np_2d)
        b_np = b_np_2d.reshape(1, h, w, c).transpose(0, 3, 1, 2)
    else:
        raise NotImplementedError(f"{len(shape)}-D shape not supported")

    return a_np, b_np


def test_softmax(target, dev, shape, dtype, ref_data, softmax_operation):
    target = tvm.target.Target(target)
    if target.kind.name == "vulkan" and dtype == "float64":
        # https://www.khronos.org/registry/SPIR-V/specs/1.0/GLSL.std.450.html
        pytest.xfail("Vulkan GLSL.std.450 does not support 64-bit floats")

    A = te.placeholder(shape, dtype=dtype, name="A")

    topi_op = configs[softmax_operation]["topi"]
    B = topi_op(A, axis=min(len(shape) - 1, 1))

    with tvm.target.Target(target):
        fschedule = tvm.topi.testing.dispatch(target, _softmax_schedule)
        s = fschedule(B)

    a_np, b_np = ref_data

    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)
    f = tvm.build(s, [A, B], target)
    f(a, b)
    tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()
