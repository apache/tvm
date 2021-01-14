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

from tvm import relay
from tvm.relay.frontend.common import infer_type
from tvm.relay import op as _op
import numpy as np

def test_const_dtype():
    strides = (1, 1)
    np_array = np.array(strides)
    strides = _op.const(np_array, dtype="int64")

    # strides needs to be autoconverted to int64 on Windows
    assert infer_type(strides).checked_type.dtype == np.dtype(np.int64)
