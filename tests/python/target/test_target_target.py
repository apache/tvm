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

import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm.target import Target
from tvm.testing import env


def test_all_targets_device_type_verify():
    """Consistency verification for all targets' device type"""
    for target_kind in tvm.target.Target.list_kinds():
        target = Target(target_kind)
        device = tvm.device_from_target(target)

        assert device.dlpack_device_type() == target.get_target_device_type()


@pytest.mark.parametrize("target", ["llvm", {"kind": "llvm"}, Target("llvm")])
def test_device_from_target_input_forms(target):
    device = tvm.device_from_target(target)

    assert device == tvm.cpu()
    assert isinstance(device, tvm.runtime.Device)
    assert tvm.runtime.device_from_target(target) == tvm.cpu()


def test_device_from_target_compiler_only_kind():
    assert tvm.device_from_target("composite") == tvm.cpu()


def test_device_from_target_index():
    assert tvm.device_from_target("llvm").index == 0
    assert tvm.device_from_target("llvm", None).index == 0
    assert tvm.device_from_target("llvm", 3).index == 3


def test_device_from_target_override():
    target = Target(
        {
            "kind": "llvm",
            "target_device_type": int(tvm_ffi.DLDeviceType.kDLCUDA),
        }
    )

    assert tvm.device_from_target(target).dlpack_device_type() == tvm_ffi.DLDeviceType.kDLCUDA


def test_target_string_parse():
    target = tvm.target.Target({"kind": "cuda", "model": "unknown", "libs": ["cublas", "cudnn"]})

    assert target.kind.name == "cuda"
    assert target.attrs["model"] == "unknown"
    assert set(target.keys) == set(["cuda", "gpu"])
    assert set(target.attrs["libs"]) == set(["cublas", "cudnn"])

    assert (
        Target({"kind": "opencl", "device": "intel_graphics"}).attrs.get("device", "")
        == "intel_graphics"
    )
    assert Target({"kind": "opencl", "device": "mali"}).attrs.get("device", "") == "mali"
    assert Target({"kind": "llvm", "device": "arm_cpu"}).attrs.get("device", "") == "arm_cpu"


def test_target_string_with_spaces():
    target = tvm.target.Target(
        {"kind": "vulkan", "device_name": "Name of GPU with spaces", "device_type": "discrete"}
    )
    assert target.attrs["device_name"] == "Name of GPU with spaces"
    assert target.attrs["device_type"] == "discrete"

    target = tvm.target.Target(str(target))

    assert target.attrs["device_name"] == "Name of GPU with spaces"
    assert target.attrs["device_type"] == "discrete"


def test_target_llvm_options():
    target = tvm.target.Target(
        {"kind": "llvm", "cl-opt": ["-unroll-threshold:uint=100", "-unroll-count:uint=3"]}
    )
    assert sorted(target.attrs["cl-opt"]) == sorted(
        ["-unroll-threshold:uint=100", "-unroll-count:uint=3"]
    )


def test_target_llvm_jit_options():
    target = tvm.target.Target({"kind": "llvm", "jit": "mcjit"})
    assert target.attrs["jit"] == "mcjit"
    target = tvm.target.Target({"kind": "llvm", "jit": "orcjit"})
    assert target.attrs["jit"] == "orcjit"


def test_target_llvm_vector_width():
    target = tvm.target.Target({"kind": "llvm", "vector-width": 256})
    assert target.attrs["vector-width"] == 256
    target = tvm.target.Target({"kind": "llvm", "vector-width": 1024})
    assert target.attrs["vector-width"] == 1024


def test_target_config():
    """
    Test that constructing a target from a dictionary works.
    """
    target_config = {
        "kind": "llvm",
        "keys": ["arm_cpu", "cpu"],
        "device": "arm_cpu",
        "libs": ["cblas"],
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
        assert target.attrs.get("device", "") == "arm_cpu"
        assert list(target.attrs.get("libs", [])) == ["cblas"]
        assert target.attrs["mfloat-abi"] == "hard"
        assert all([attr in target.attrs["mattr"] for attr in ["+neon", "-avx512f"]])


def test_config_map():
    """
    Confirm that constructing a target with invalid
    attributes fails as expected.
    """
    target_config = {"kind": "llvm", "libs": {"a": "b", "c": "d"}}
    with pytest.raises(ValueError):
        tvm.target.Target(target_config)


def test_composite_target():
    tgt = tvm.target.Target(
        {"kind": "composite", "host": {"kind": "llvm"}, "devices": ["cuda", "opencl"]}
    )
    assert tgt.kind.name == "composite"
    assert tgt.host.kind.name == "llvm"
    assert len(tgt.attrs["devices"]) == 2
    cuda_device, opencl_device = tgt.attrs["devices"]
    assert cuda_device.kind.name == "cuda"
    assert opencl_device.kind.name == "opencl"


def test_target_tag_0():
    tgt = tvm.target.Target("nvidia/geforce-rtx-2080-ti")
    assert tgt.kind.name == "cuda"
    assert tgt.attrs["arch"] == "sm_75"
    assert tgt.attrs["max_shared_memory_per_block"] == 49152
    assert tgt.attrs["max_threads_per_block"] == 1024
    assert tgt.attrs["thread_warp_size"] == 32
    assert tgt.attrs["registers_per_block"] == 65536


def test_target_tag_1():
    tgt = tvm.target.Target("nvidia/jetson-nano")
    assert tgt.kind.name == "cuda"
    assert tgt.attrs["arch"] == "sm_53"
    assert tgt.attrs["max_shared_memory_per_block"] == 49152
    assert tgt.attrs["max_threads_per_block"] == 1024
    assert tgt.attrs["thread_warp_size"] == 32
    assert tgt.attrs["registers_per_block"] == 32768


def test_target_tag_override():
    """Test creating a target from a tag with attribute overrides."""
    tgt = tvm.target.Target({"tag": "nvidia/nvidia-a100", "l2_cache_size_bytes": 12345})
    assert tgt.kind.name == "cuda"
    assert tgt.attrs["arch"] == "sm_80"
    # Override should take effect
    assert int(tgt.attrs["l2_cache_size_bytes"]) == 12345
    # Base tag fields should be preserved
    assert tgt.attrs["max_shared_memory_per_block"] == 49152
    assert tgt.attrs["thread_warp_size"] == 32
    # Tag name should be recorded
    assert tgt.tag == "nvidia/nvidia-a100"


def test_list_kinds():
    targets = tvm.target.Target.list_kinds()
    assert len(targets) != 0
    assert "llvm" in targets
    assert all(isinstance(target_name, str) for target_name in targets)


def test_target_host_tags():
    tgt = tvm.target.Target("nvidia/jetson-nano", "nvidia/geforce-rtx-2080-ti")
    assert tgt.kind.name == "cuda"
    assert tgt.attrs["arch"] == "sm_53"
    assert tgt.attrs["max_shared_memory_per_block"] == 49152
    assert tgt.attrs["max_threads_per_block"] == 1024
    assert tgt.attrs["thread_warp_size"] == 32
    assert tgt.attrs["registers_per_block"] == 32768
    assert tgt.host.kind.name == "cuda"
    assert tgt.host.attrs["arch"] == "sm_75"
    assert tgt.host.attrs["max_shared_memory_per_block"] == 49152
    assert tgt.host.attrs["max_threads_per_block"] == 1024
    assert tgt.host.attrs["thread_warp_size"] == 32
    assert tgt.host.attrs["registers_per_block"] == 65536


def test_target_host_tag_dict():
    tgt = tvm.target.Target("nvidia/jetson-nano", {"kind": "llvm"})
    assert tgt.kind.name == "cuda"
    assert tgt.attrs["arch"] == "sm_53"
    assert tgt.attrs["max_shared_memory_per_block"] == 49152
    assert tgt.attrs["max_threads_per_block"] == 1024
    assert tgt.attrs["thread_warp_size"] == 32
    assert tgt.attrs["registers_per_block"] == 32768
    assert tgt.host.kind.name == "llvm"


def test_target_host_single_dict():
    tgt = tvm.target.Target({"kind": "llvm", "host": "nvidia/jetson-nano"})
    assert tgt.kind.name == "llvm"
    assert tgt.host.kind.name == "cuda"
    assert tgt.host.attrs["arch"] == "sm_53"
    assert tgt.host.attrs["max_shared_memory_per_block"] == 49152
    assert tgt.host.attrs["max_threads_per_block"] == 1024
    assert tgt.host.attrs["thread_warp_size"] == 32
    assert tgt.host.attrs["registers_per_block"] == 32768


def test_target_host_single_string():
    tgt = tvm.target.Target({"kind": "cuda", "host": {"kind": "llvm"}})
    assert tgt.kind.name == "cuda"
    assert tgt.host.kind.name == "llvm"


def test_target_host_single_string_with_tag():
    tgt = tvm.target.Target({"kind": "cuda", "host": "nvidia/jetson-nano"})
    assert tgt.kind.name == "cuda"
    assert tgt.host.kind.name == "cuda"
    assert tgt.host.attrs["arch"] == "sm_53"
    assert tgt.host.attrs["max_shared_memory_per_block"] == 49152
    assert tgt.host.attrs["max_threads_per_block"] == 1024
    assert tgt.host.attrs["thread_warp_size"] == 32
    assert tgt.host.attrs["registers_per_block"] == 32768


def test_target_host_merge_0():
    tgt = tvm.target.Target(tvm.target.Target({"kind": "cuda", "host": "nvidia/jetson-nano"}), None)
    assert tgt.kind.name == "cuda"
    assert tgt.host.kind.name == "cuda"
    assert tgt.host.attrs["arch"] == "sm_53"
    assert tgt.host.attrs["max_shared_memory_per_block"] == 49152
    assert tgt.host.attrs["max_threads_per_block"] == 1024
    assert tgt.host.attrs["thread_warp_size"] == 32
    assert tgt.host.attrs["registers_per_block"] == 32768


def test_target_host_merge_1():
    tgt = tvm.target.Target({"kind": "cuda", "host": {"kind": "llvm"}})
    tgt = tvm.target.Target(tgt, tgt.host)
    assert tgt.kind.name == "cuda"
    assert tgt.host.kind.name == "llvm"


def test_target_host_merge_2():
    """Test picking the same host is ok."""
    tgt = tvm.target.Target(
        tvm.target.Target({"kind": "cuda", "host": {"kind": "llvm"}}),
        tvm.target.Target("llvm"),
    )
    assert tgt.kind.name == "cuda"
    assert tgt.host.kind.name == "llvm"


def test_target_tvm_object():
    """Test creating Target by using TVM Objects"""
    String = tvm_ffi.core.String
    tgt = tvm.target.Target(target={"kind": "cuda", "host": {"kind": "llvm"}})
    assert tgt.kind.name == "cuda"
    assert tgt.host.kind.name == "llvm"
    tgt = tvm.target.Target(target=String("cuda"), host=String("llvm"))
    assert tgt.kind.name == "cuda"
    assert tgt.host.kind.name == "llvm"


@pytest.mark.skip(reason="Causing infinite loop because of pytest and handle issue")
def test_target_host_merge_3():
    with pytest.raises(ValueError, match=r"target host has to be a string or dictionary."):
        tvm.target.Target(tvm.target.Target({"kind": "cuda", "host": {"kind": "llvm"}}), 12.34)


def test_target_with_host():
    tgt = tvm.target.Target("cuda")
    llvm = tvm.target.Target("llvm")
    tgt = tgt.with_host(llvm)
    assert tgt.kind.name == "cuda"
    assert tgt.host.kind.name == "llvm"
    cuda_host = tvm.target.Target("nvidia/jetson-nano")
    tgt = tgt.with_host(cuda_host)
    assert tgt.host.kind.name == "cuda"
    assert tgt.host.attrs["arch"] == "sm_53"
    assert tgt.host.attrs["max_shared_memory_per_block"] == 49152
    assert tgt.host.attrs["max_threads_per_block"] == 1024
    assert tgt.host.attrs["thread_warp_size"] == 32
    assert tgt.host.attrs["registers_per_block"] == 32768


def test_target_attr_bool_value():
    target0 = Target({"kind": "vulkan", "supports_float16": True})
    assert target0.attrs["supports_float16"] == 1
    target1 = Target({"kind": "vulkan", "supports_float16": True})
    assert target1.attrs["supports_float16"] == 1
    target2 = Target({"kind": "vulkan", "supports_float16": False})
    assert target2.attrs["supports_float16"] == 0
    target3 = Target({"kind": "vulkan", "supports_float16": False})
    assert target3.attrs["supports_float16"] == 0


def test_target_attr_l2_cache_size_bytes():
    target0 = Target("nvidia/nvidia-a100")
    assert int(target0.attrs.get("l2_cache_size_bytes", 0)) == 41943040
    target1 = Target("nvidia/geforce-rtx-4090")
    assert int(target1.attrs.get("l2_cache_size_bytes", 0)) == 75497472


def test_target_features():
    target_no_features = Target("cuda")
    assert target_no_features.features
    assert not target_no_features.features.is_test

    target_with_features = Target("test")
    assert target_with_features.features.is_test
    assert not target_with_features.features.is_missing


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("input_form", ["string", "device"])
def test_target_from_device_cuda(input_form):
    def run_and_check():
        dev = tvm.cuda()
        target = Target.from_device("cuda" if input_form == "string" else dev)
        assert target.kind.name == "cuda"
        assert target.attrs["max_threads_per_block"] == dev.max_threads_per_block
        assert int(target.attrs["max_shared_memory_per_block"]) == dev.max_shared_memory_per_block
        assert int(target.attrs["thread_warp_size"]) == dev.warp_size
        assert str(target.attrs.get("arch", "")) == "sm_" + dev.compute_version.replace(".", "")

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_rocm(), reason="need rocm")
@pytest.mark.parametrize("input_form", ["string", "device"])
def test_target_from_device_rocm(input_form):
    def run_and_check():
        dev = tvm.rocm()
        target = Target.from_device("rocm" if input_form == "string" else dev)
        assert target.kind.name == "rocm"
        assert target.attrs["mtriple"] == "amdgcn-and-amdhsa-hcc"
        assert target.attrs["max_threads_per_block"] == dev.max_threads_per_block
        assert int(target.attrs["max_shared_memory_per_block"]) == dev.max_shared_memory_per_block
        assert int(target.attrs["thread_warp_size"]) == dev.warp_size

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_vulkan(), reason="need vulkan")
@pytest.mark.parametrize("input_form", ["string", "device"])
def test_target_from_device_vulkan(input_form):
    def run_and_check():
        dev = tvm.vulkan()
        target = Target.from_device("vulkan" if input_form == "string" else dev)
        f_get_target_property = tvm.get_global_func("device_api.vulkan.get_target_property")
        assert target.kind.name == "vulkan"
        assert target.attrs["max_threads_per_block"] == dev.max_threads_per_block
        assert int(target.attrs["max_shared_memory_per_block"]) == dev.max_shared_memory_per_block
        assert int(target.attrs["thread_warp_size"]) == dev.warp_size
        assert target.attrs["supports_float16"] == f_get_target_property(dev, "supports_float16")
        assert target.attrs["supports_int16"] == f_get_target_property(dev, "supports_int16")
        assert target.attrs["supports_int8"] == f_get_target_property(dev, "supports_int8")
        assert target.attrs["supports_16bit_buffer"] == f_get_target_property(
            dev, "supports_16bit_buffer"
        )

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_opencl(), reason="need opencl")
@pytest.mark.parametrize("input_form", ["string", "device"])
def test_target_from_device_opencl(input_form):
    def run_and_check():
        dev = tvm.opencl()
        target = Target.from_device("opencl" if input_form == "string" else dev)
        assert target.kind.name == "opencl"
        assert target.attrs["max_threads_per_block"] == dev.max_threads_per_block
        assert int(target.attrs["max_shared_memory_per_block"]) == dev.max_shared_memory_per_block
        assert int(target.attrs["thread_warp_size"]) == dev.warp_size

    tvm.testing.run_with_gpu_lock(run_and_check)


def test_module_dict_from_deserialized_targets():
    target = Target("llvm")

    from tvm.script import tirx as T

    @T.prim_func(s_tir=True)
    def func():
        T.evaluate(0)

    func = func.with_attr("Target", target)
    target2 = tvm.ir.load_json(tvm.ir.save_json(target))
    mod = tvm.IRModule({"main": func})
    lib = tvm.compile(mod, target=target2)
    lib["func"]()


def test_json_roundtrip():
    """Test that Target(str(target)) roundtrips correctly."""
    target = Target({"kind": "llvm", "mcpu": "cortex-a53"})
    target2 = Target(str(target))
    assert target2.kind.name == "llvm"
    assert target2.attrs["mcpu"] == "cortex-a53"

    # Test with more complex target
    target = Target({"kind": "cuda", "arch": "sm_80", "max_threads_per_block": 1024})
    target2 = Target(str(target))
    assert target2.kind.name == "cuda"
    assert target2.attrs["arch"] == "sm_80"


def test_str_is_json():
    """Test that str() output is valid JSON."""
    target = Target({"kind": "llvm", "mcpu": "cortex-a53"})
    s = str(target)
    parsed = json.loads(s)
    assert parsed["kind"] == "llvm"
    assert parsed["mcpu"] == "cortex-a53"


def test_cli_string_rejected():
    """Test that CLI string form is rejected."""
    with pytest.raises(ValueError):
        Target("llvm -mcpu=cortex-a53")


def test_webgpu_target_subgroup_attrs():
    """Test WebGPU target defaults and supports_subgroups canonicalization."""
    # Default: thread_warp_size=1, supports_subgroups=False
    tgt_default = Target({"kind": "webgpu"})
    assert tgt_default.attrs["thread_warp_size"] == 1
    assert tgt_default.attrs["supports_subgroups"] == 0

    # With supports_subgroups=True: thread_warp_size is set to 32
    tgt_subgroups = Target({"kind": "webgpu", "supports_subgroups": True})
    assert tgt_subgroups.attrs["thread_warp_size"] == 32
    assert tgt_subgroups.attrs["supports_subgroups"] == 1

    for config in [
        {"kind": "webgpu", "thread_warp_size": 32},
        {"kind": "webgpu", "thread_warp_size": 32, "supports_subgroups": False},
    ]:
        with pytest.raises(ValueError, match="requires supports_subgroups=true"):
            Target(config)


if __name__ == "__main__":
    tvm.testing.main()
