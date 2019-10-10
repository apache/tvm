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
import os
import tvm
import numpy as np

import pytest
from tvm import relay
from tvm.relay.testing import resnet

def test_basic():
    mod, params = resnet.get_workload()
    target = 'llvm'
    ctx = tvm.cpu()
    vm = relay.profiler_vm.compile(mod, target)
    vm.init(ctx)
    vm.load_params(params)

    data = np.random.rand(1, 3, 224, 224).astype('float32')
    res = vm.invoke("main", [data])
    print("\n{}".format(vm.get_stat()))

if __name__ == "__main__":
    test_basic()
