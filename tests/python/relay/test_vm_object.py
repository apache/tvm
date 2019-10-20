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
from tvm.relay import vm

def test_tensor():
    arr = tvm.nd.array([1,2,3])
    x = vm.Tensor(arr)
    assert isinstance(x, vm.Tensor)
    assert x.asnumpy()[0] == 1
    assert x.asnumpy()[-1] == 3
    assert isinstance(x.data, tvm.nd.NDArray)


def test_adt():
    arr = tvm.nd.array([1,2,3])
    x = vm.Tensor(arr)
    y = vm.ADT(0, [x, x])

    assert len(y) == 2
    assert isinstance(y, vm.ADT)
    y[0:1][-1].data == x.data
    assert y.tag == 0
    assert isinstance(x.data, tvm.nd.NDArray)



if __name__ == "__main__":
    test_tensor()
    test_adt()
