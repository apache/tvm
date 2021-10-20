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

# brief Example code on load and run TVM module.s
# file python_deploy.py

import tvm
from tvm import te
import numpy as np


def verify(mod, fname):
    # Get the function from the module
    f = mod.get_function(fname)
    # Use tvm.nd.array to convert numpy ndarray to tvm
    # NDArray type, so that function can be invoked normally
    N = 10
    x = tvm.nd.array(np.arange(N, dtype=np.float32))
    y = tvm.nd.array(np.zeros(N, dtype=np.float32))
    # Invoke the function
    f(x, y)
    np_x = x.numpy()
    np_y = y.numpy()
    # Verify correctness of function
    assert np.all([xi + 1 == yi for xi, yi in zip(np_x, np_y)])
    print("Finish verification...")


if __name__ == "__main__":
    # The normal dynamic loading method for deployment
    mod_dylib = tvm.runtime.load_module("lib/test_addone_dll.so")
    print("Verify dynamic loading from test_addone_dll.so")
    verify(mod_dylib, "addone")
    # There might be methods to use the system lib way in
    # python, but dynamic loading is good enough for now.
