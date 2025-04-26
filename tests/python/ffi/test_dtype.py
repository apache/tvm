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

import numpy as np
from tvm import ffi as tvm_ffi


def test_dtype():
    float32 = tvm_ffi.dtype("float32")
    assert float32.__repr__() == "dtype('float32')"
    assert type(float32) == tvm_ffi.dtype
    x = np.array([1, 2, 3], dtype=float32)
    assert x.dtype == float32
