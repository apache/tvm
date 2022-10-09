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
"""Test code for reduce"""
import numpy as np

import tvm
from tvm import te, topi
from tvm.contrib.hexagon.session import Session
from tvm.topi.utils import get_const_tuple

from ..infrastructure import get_hexagon_target


@tvm.testing.requires_hexagon
def test_nn_pad(hexagon_session: Session):
    dtype = "uint8"
    in_shape = (1, 56, 56, 32)

    data_in = np.ones(in_shape).astype(dtype)

    A = te.placeholder(shape=in_shape, name="A", dtype=dtype)

    C = topi.nn.pad(A, [0, 1, 1, 0], [0, 1, 1, 0], pad_value=0)

    with tvm.target.Target(get_hexagon_target("v68")):
        fschedule = topi.hexagon.schedule_pad
        s = fschedule(C)

    func = tvm.build(s, [A, C], get_hexagon_target("v68"), name="pad")
    mod = hexagon_session.load_module(func)

    dev = hexagon_session.device
    a = tvm.nd.array(data_in, dev)
    b = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), dev)
    mod["pad"](a, b)

    # Reference numpy pad output
    ref_out = np.pad(data_in, pad_width=((0, 0), (1, 1), (1, 1), (0, 0)))

    tvm.testing.assert_allclose(b.numpy(), ref_out)


if __name__ == "__main__":
    tvm.testing.main()
