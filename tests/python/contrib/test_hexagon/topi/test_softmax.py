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
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import topi
from tvm import te
from tvm.contrib.hexagon.session import Session
import tvm.topi.testing
from tvm.topi.utils import get_const_tuple

dtype = tvm.testing.parameter(
    "float16",
    "float32",
)

# TODO(mehrdadh): add log_softmax to config
configs = {
    "softmax": {
        "topi": topi.nn.softmax,
        "ref": tvm.topi.testing.softmax_python,
        "dimensions": [2, 4],
    },
}

# TODO(mehrdadh): larger size like (1, 16, 256, 256) would fail due to TVM_HEXAGON_RPC_BUFF_SIZE_BYTES
shapes = [(32, 10), (3, 4), (1, 16, 32, 32)]
softmax_operation, shape = tvm.testing.parameters(
    *[
        (name, shape)
        for name, config in configs.items()
        for shape in shapes
        if len(shape) in config["dimensions"]
    ]
)


@tvm.testing.requires_hexagon
def test_softmax(hexagon_session: Session, shape, dtype, softmax_operation):
    if dtype == "float16":
        pytest.xfail("float16 is not supported.")
    A = te.placeholder(shape, dtype=dtype, name="A")

    topi_op = configs[softmax_operation]["topi"]
    B = topi_op(A, axis=1)

    def get_ref_data(shape):
        ref_func = tvm.topi.testing.softmax_python
        a_np = np.random.uniform(size=shape).astype(dtype)

        if len(shape) == 2:
            b_np = ref_func(a_np)
        elif len(shape) == 4:
            _, c, h, w = a_np.shape
            a_np_2d = a_np.transpose(0, 2, 3, 1).reshape(h * w, c)
            b_np_2d = tvm.topi.testing.softmax_python(a_np_2d)
            b_np = b_np_2d.reshape(1, h, w, c).transpose(0, 3, 1, 2)

        return a_np, b_np

    # get the test data
    a_np, b_np = get_ref_data(shape)

    target_hexagon = tvm.target.hexagon("v68")
    with tvm.target.Target(target_hexagon):
        fschedule = topi.hexagon.schedule_softmax
        s = fschedule(B)

    func = tvm.build(
        s, [A, B], tvm.target.Target(target_hexagon, host=target_hexagon), name="softmax"
    )
    mod = hexagon_session.load_module(func)

    dev = hexagon_session.device
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)
    mod["softmax"](a, b)

    tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()
