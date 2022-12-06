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

from ..infrastructure import get_hexagon_target

# TODO(mehrdadh): add log_softmax to config
OPERATOR_CONFIGS = {
    "softmax": {
        "topi": topi.nn.softmax,
        "ref": tvm.topi.testing.softmax_python,
        "dimensions": [2, 4],
    },
}


class TestSoftmax:
    """Softmax test class."""

    dtype = tvm.testing.parameter(
        "float16",
        "float32",
    )

    # TODO(mehrdadh): larger size like (1, 16, 256, 256)
    # would fail due to TVM_HEXAGON_RPC_BUFF_SIZE_BYTES
    shape = tvm.testing.parameter((32, 10), (3, 4), (1, 16, 32, 32))

    @tvm.testing.fixture
    def softmax_operation(self, shape) -> tuple:
        """Returns the operation name and shape."""
        for name, config in OPERATOR_CONFIGS.items():
            if len(shape) in config["dimensions"]:
                return name
            else:
                raise ValueError(f"Shape {shape} is not supported.")

    @tvm.testing.requires_hexagon
    def test_softmax(self, hexagon_session: Session, dtype, shape, softmax_operation):
        """Test softmax."""
        if dtype == "float16":
            pytest.xfail("float16 is not supported.")

        a_tensor = te.placeholder(shape, dtype=dtype, name="a_tensor")

        topi_op = OPERATOR_CONFIGS[softmax_operation]["topi"]
        b_tensor = topi_op(a_tensor, axis=1)

        def get_ref_data(shape):
            ref_func = tvm.topi.testing.softmax_python
            a_np = np.random.uniform(size=shape).astype(dtype)

            if len(shape) == 2:
                b_np = ref_func(a_np)
            elif len(shape) == 4:
                _, c, height, width = a_np.shape
                a_np_2d = a_np.transpose(0, 2, 3, 1).reshape(height * width, c)
                b_np_2d = tvm.topi.testing.softmax_python(a_np_2d)
                b_np = b_np_2d.reshape(1, height, width, c).transpose(0, 3, 1, 2)

            return a_np, b_np

        # get the test data
        a_np, b_np = get_ref_data(shape)

        with tvm.target.Target(get_hexagon_target("v68")):
            fschedule = topi.hexagon.schedule_softmax
            s = fschedule(b_tensor)

        func = tvm.build(s, [a_tensor, b_tensor], get_hexagon_target("v68"), name="softmax")
        mod = hexagon_session.load_module(func)

        dev = hexagon_session.device
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(np.zeros(get_const_tuple(b_tensor.shape), dtype=b_tensor.dtype), dev)
        mod["softmax"](a, b)

        tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()
