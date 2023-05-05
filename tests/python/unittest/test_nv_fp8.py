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
import tvm
import ml_dtypes


# Currently only float8_e4m3 is supported.
def test_create_nv_fp8_nd_array():
    x = np.random.rand(128, 128).astype(ml_dtypes.float8_e4m3fn)
    x_nd = tvm.nd.array(x)
    assert x_nd.dtype == "float8"


if __name__ == "__main__":
    test_create_nv_fp8_nd_array()