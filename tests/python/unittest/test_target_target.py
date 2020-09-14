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
import json
import tvm
from tvm import te
from tvm.target import cuda, rocm, mali, intel_graphics, arm_cpu, vta, bifrost, hexagon


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

    with tvm.target.Target("cuda"):
        assert mygeneric(1) == 3

    with tvm.target.arm_cpu():
        assert mygeneric(1) == 11

    with tvm.target.Target("metal"):
        assert mygeneric(1) == 3

    assert tvm.target.Target.current() is None


def test_target_string_parse():
    target = tvm.target.Target("cuda -model=unknown -libs=cublas,cudnn")

    assert target.kind.name == "cuda"
    assert target.model == "unknown"
    assert set(target.keys) == set(["cuda", "gpu"])
    assert set(target.libs) == set(["cublas", "cudnn"])
    assert str(target) == str(tvm.target.cuda(options="-libs=cublas,cudnn"))

    assert tvm.target.intel_graphics().device_name == "intel_graphics"
    assert tvm.target.mali().device_name == "mali"
    assert tvm.target.arm_cpu().device_name == "arm_cpu"


def test_target_create():
    targets = [cuda(), rocm(), mali(), intel_graphics(), arm_cpu("rk3399"), vta(), bifrost()]
    for tgt in targets:
        assert tgt is not None


def test_target_config():
    """
    Test that constructing a target from a dictionary works.
    """
    target_config = {
        "kind": "llvm",
        "keys": ["arm_cpu", "cpu"],
        "device": "arm_cpu",
        "libs": ["cblas"],
        "system-lib": True,
        "mfloat-abi": "hard",
        "mattr": ["+neon", "-avx512f"],
    }
    # Convert config dictionary to json string.
    target_config_str = json.dumps(target_config)
    # Test both dictionary input and json string.
    for config in [target_config, target_config_str]:
        target = tvm.target.Target(config)
        assert target.kind.name == "llvm"
        assert all([key in target.keys for key in ["arm_cpu", "cpu"]])
        assert target.device_name == "arm_cpu"
        assert target.libs == ["cblas"]
        assert "system-lib" in str(target)
        assert target.attrs["mfloat-abi"] == "hard"
        assert all([attr in target.attrs["mattr"] for attr in ["+neon", "-avx512f"]])


def test_config_map():
    """
    Confirm that constructing a target with invalid
    attributes fails as expected.
    """
    target_config = {"kind": "llvm", "libs": {"a": "b", "c": "d"}}
    failed = False
    try:
        tvm.target.Target(target_config)
    except ValueError:
        failed = True
    assert failed


def test_composite_target():
    tgt = tvm.target.Target("composite --target_host=llvm --devices=cuda,opencl")
    assert tgt.kind.name == "composite"
    assert tgt.attrs["target_host"].kind.name == "llvm"
    assert len(tgt.attrs["devices"]) == 2
    cuda_device, opencl_device = tgt.attrs["devices"]
    assert cuda_device.kind.name == "cuda"
    assert opencl_device.kind.name == "opencl"


if __name__ == "__main__":
    test_target_dispatch()
    test_target_string_parse()
    test_target_create()
    test_target_config()
    test_config_map()
    test_composite_target()
