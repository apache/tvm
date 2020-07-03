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
from tvm import te

@tvm.target.generic_func
def mygeneric(data):
    # default generic function
    return data + 1

@mygeneric.register(["cuda", "gpu"])
def cuda_func(data):
    return data + 2

@mygeneric.register("rocm")
def rocm_func(data):
    return data + 3

@mygeneric.register("cpu")
def rocm_func(data):
    return data + 10


def test_target_dispatch():
    with tvm.target.cuda():
        assert mygeneric(1) == 3

    with tvm.target.rocm():
        assert mygeneric(1) == 4

    with tvm.target.create("cuda"):
        assert mygeneric(1) == 3

    with tvm.target.arm_cpu():
        assert mygeneric(1) == 11

    with tvm.target.create("metal"):
        assert mygeneric(1) == 3

    assert tvm.target.Target.current() is None


def test_target_string_parse():
    target = tvm.target.create("cuda -model=unknown -libs=cublas,cudnn")

    assert target.id.name == "cuda"
    assert target.model == "unknown"
    assert target.keys == ['cuda', 'gpu']
    assert target.libs == ['cublas', 'cudnn']
    assert str(target) == str(tvm.target.cuda(options="-libs=cublas,cudnn"))

    assert tvm.target.intel_graphics().device_name == "intel_graphics"
    assert tvm.target.mali().device_name == "mali"
    assert tvm.target.arm_cpu().device_name == "arm_cpu"

if __name__ == "__main__":
    test_target_dispatch()
    test_target_string_parse()
