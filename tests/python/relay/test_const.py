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

import tvm
import numpy as np
from tvm import relay
from tvm.relay.frontend.common import infer_type
from tvm.relay import op as _op


def test_const_dtype():
    strides = (1, 1)
    np_array = np.array(strides).astype("int32")
    strides = _op.const(np_array, dtype="int64")

    # strides needs to be autoconverted to int64 on Windows
    assert infer_type(strides).checked_type.dtype == np.dtype(np.int64)

    a = tvm.nd.array(np.random.randint(0, high=255, size=(2, 3), dtype="uint8"))
    a = _op.const(a, dtype="uint8")
    aa = a.data.numpy()
    assert aa.dtype == np.dtype(np.uint8)

    b = _op.const(1, dtype="int8")
    bb = b.data.numpy()
    assert bb.dtype == np.dtype(np.int8)

    kshape = (3, 10, 3, 3)
    w = relay.const(np.zeros(kshape, dtype="float32"))
    assert w.data.numpy().dtype == np.dtype(np.float32)
