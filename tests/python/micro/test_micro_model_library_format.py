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

import pathlib
import sys
import datetime
import json
import os
import tarfile

import numpy as np
import pytest
import platform

pytest.importorskip("tvm.micro")

import tvm
import tvm.relay
from tvm.relay.backend import Executor, Runtime
from tvm.relay.testing import byoc
import tvm.runtime.module
import tvm.testing
from tvm.contrib import utils
import tvm.micro as micro
from tvm.micro.testing.utils import get_conv2d_relay_module
import tvm.micro.model_library_format as model_library_format
from tvm.micro.model_library_format import _GENERATED_VERSION


@tvm.testing.requires_micro
def test_export_operator_model_library_format():
    target = tvm.target.target.micro("host")
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        A = tvm.te.placeholder((2,), dtype="int8")
        B = tvm.te.placeholder((1,), dtype="int8")
        C = tvm.te.compute(A.shape, lambda i: A[i] + B[0], name="C")
        sched = tvm.te.create_schedule(C.op)
        mod = tvm.build(
            sched,
            [A, B, C],
            tvm.target.Target(target, target),
            runtime=Runtime("crt", {"system-lib": True}),
            name="add",
        )

    temp_dir = utils.tempdir()
    mlf_tar_path = temp_dir.relpath("lib.tar")
    micro.export_model_library_format(mod, mlf_tar_path)

    tf = tarfile.open(mlf_tar_path)

    extract_dir = temp_dir.relpath("extract")
    os.mkdir(extract_dir)
    tf.extractall(extract_dir)

    with open(os.path.join(extract_dir, "metadata.json")) as json_f:
        metadata = json.load(json_f)
        assert metadata["version"] == _GENERATED_VERSION
        assert metadata["model_name"] == "add"
        export_datetime = datetime.datetime.strptime(
            metadata["export_datetime"], "%Y-%m-%d %H:%M:%SZ"
        )
        assert (datetime.datetime.now() - export_datetime) < datetime.timedelta(seconds=60 * 5)
        assert metadata["target"] == [str(target)]

        assert metadata["memory"]["add"][0]["dtype"] == "int8"
        assert metadata["memory"]["add"][0]["shape"] == [2]
        assert metadata["memory"]["add"][0]["size_bytes"] == 2

        assert metadata["memory"]["add"][1]["dtype"] == "int8"
        assert metadata["memory"]["add"][1]["shape"] == [1]
        assert metadata["memory"]["add"][1]["size_bytes"] == 1

        assert metadata["memory"]["add"][2]["dtype"] == "int8"
        assert metadata["memory"]["add"][2]["shape"] == [2]
        assert metadata["memory"]["add"][2]["size_bytes"] == 2

    assert os.path.exists(os.path.join(extract_dir, "codegen", "host", "src", "lib0.c"))
    assert os.path.exists(os.path.join(extract_dir, "codegen", "host", "src", "lib1.c"))

    assert (
        len(mod.ir_module_by_target) == 1
    ), f"expect 1 ir_model_by_target: {mod.ir_module_by_target!r}"
    for target, ir_mod in mod.ir_module_by_target.items():
        assert int(tvm.runtime.ndarray.device(str(target)).device_type) == 1
        with open(os.path.join(extract_dir, "src", "tir-1.txt")) as tir_f:
            assert tir_f.read() == str(ir_mod)


@tvm.testing.requires_micro
def test_export_multiple_operator_model_library_format():
    target = tvm.target.target.micro("host")
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        A = tvm.te.placeholder((2,), dtype="int8")
        B = tvm.te.placeholder((1,), dtype="int8")
        C = tvm.te.compute(A.shape, lambda i: A[i] + B[0], name="C")
        sched = tvm.te.create_schedule(C.op)
        mod = tvm.build(
            sched,
            [A, B, C],
            tvm.target.Target(target, target),
            runtime=Runtime("crt", {"system-lib": True}),
            name="add",
        )

    temp_dir = utils.tempdir()
    mlf_tar_path = temp_dir.relpath("lib.tar")

    with pytest.raises(RuntimeError) as exc:
        micro.export_model_library_format([mod, mod], mlf_tar_path)

        assert str(exc.exception) == ("Multiple operator is not supported.")


def validate_graph_json(extract_dir, factory):
    with open(
        os.path.join(extract_dir, "executor-config", "graph", f"{factory.libmod_name}.graph")
    ) as graph_f:
        graph_json = graph_f.read()
        assert graph_json == factory.graph_json

        # Just check it parses and looks roughly right.
        graph = json.loads(graph_json)
        assert "nodes" in graph
        assert len(graph["nodes"]) == 4
        assert "attrs" in graph


@tvm.testing.requires_micro
@pytest.mark.parametrize(
    "executor,runtime,should_generate_interface,json_constants_size_bytes",
    [
        (Executor("graph"), Runtime("crt", {"system-lib": True}), False, 8),
        (Executor("aot", {"link-params": True}), Runtime("crt"), False, 0),
        (
            Executor("aot", {"unpacked-api": True, "interface-api": "c"}),
            Runtime("crt"),
            True,
            0,
        ),
    ],
)
def test_export_model_library_format_c(
    executor, runtime, should_generate_interface, json_constants_size_bytes
):
    target = tvm.target.target.micro("host")
    with utils.TempDirectory.set_keep_for_debug(True):
        with tvm.transform.PassContext(
            opt_level=3, config={"tir.disable_vectorize": True, "tir.usmp.enable": False}
        ):
            relay_mod = tvm.relay.fromtext(
                """
            #[version = "0.0.5"]
            def @main(%a : Tensor[(1, 2), uint8], %b : Tensor[(1, 2), float32], %c : Tensor[(1, 2), float32]) {
            %0 = cast(%a, dtype="float32") + %b * %c;
            %0
            }"""
            )
            factory = tvm.relay.build(
                relay_mod,
                target,
                executor=executor,
                runtime=runtime,
                mod_name="add",
                params={"c": np.array([[2.0, 4.0]], dtype="float32")},
            )

        temp_dir = utils.tempdir()
        mlf_tar_path = temp_dir.relpath("lib.tar")

        micro.export_model_library_format(factory, mlf_tar_path)
        tf = tarfile.open(mlf_tar_path)

        extract_dir = temp_dir.relpath("extract")
        os.mkdir(extract_dir)
        tf.extractall(extract_dir)

        with open(os.path.join(extract_dir, "metadata.json")) as json_f:
            metadata = json.load(json_f)
            module_name = factory.libmod_name
            assert metadata["version"] == _GENERATED_VERSION
            assert metadata["modules"][module_name]["model_name"] == "add"
            export_datetime = datetime.datetime.strptime(
                metadata["modules"][module_name]["export_datetime"], "%Y-%m-%d %H:%M:%SZ"
            )
            assert (datetime.datetime.now() - export_datetime) < datetime.timedelta(seconds=60 * 5)
            assert metadata["modules"][module_name]["target"] == [str(target)]
            if executor.name == "graph":
                assert metadata["modules"][module_name]["memory"]["sids"] == [
                    {"storage_id": 0, "size_bytes": 2, "input_binding": "a"},
                    {"storage_id": 1, "size_bytes": 8, "input_binding": "b"},
                    {"storage_id": 2, "size_bytes": 8, "input_binding": "p0"},
                    {"storage_id": 3, "size_bytes": 8},
                ]
            assert metadata["modules"][module_name]["memory"]["functions"]["main"] == [
                {
                    "constants_size_bytes": json_constants_size_bytes,
                    "device": 1,
                    "inputs": {
                        "a": {"dtype": "uint8", "size": 2},
                        "b": {"dtype": "float32", "size": 8},
                    },
                    "io_size_bytes": 18,
                    "outputs": {"output": {"dtype": "float32", "size": 8}},
                    "workspace_size_bytes": 0,
                }
            ]
            assert metadata["modules"][module_name]["memory"]["functions"]["operator_functions"][0][
                "workspace"
            ] == [{"device": 1, "workspace_size_bytes": 0}]
            assert (
                "fused_cast_multiply_add"
                in metadata["modules"][module_name]["memory"]["functions"]["operator_functions"][0][
                    "function_name"
                ]
            )

        assert os.path.exists(os.path.join(extract_dir, "codegen", "host", "src", "add_lib0.c"))
        assert os.path.exists(os.path.join(extract_dir, "codegen", "host", "src", "add_lib1.c"))
        assert should_generate_interface == os.path.exists(
            os.path.join(extract_dir, "codegen", "host", "include", "tvmgen_add.h")
        )

        if executor.name == "graph":
            validate_graph_json(extract_dir, factory)

        with open(os.path.join(extract_dir, "src", f"{module_name}.relay")) as relay_f:
            assert relay_f.read() == str(relay_mod)

        with open(os.path.join(extract_dir, "parameters", "add.params"), "rb") as params_f:
            params = tvm.relay.load_param_dict(params_f.read())
            if json_constants_size_bytes != 0:
                assert "p0" in params
            else:
                assert len(params) == 0


@tvm.testing.requires_micro
def test_export_model_library_format_llvm():
    with utils.TempDirectory.set_keep_for_debug(True):
        target = tvm.target.target.micro("host")
        assert str(target)[:2] == "c "
        target = tvm.target.Target("llvm " + str(target)[2:])
        with tvm.transform.PassContext(opt_level=3):
            relay_mod = tvm.relay.fromtext(
                """
            #[version = "0.0.5"]
            def @main(%a : Tensor[(1, 2), uint8], %b : Tensor[(1, 2), float32], %c : Tensor[(1, 2), float32]) {
            %0 = cast(%a, dtype="float32") + %b * %c;
            %0
            }"""
            )
            factory = tvm.relay.build(
                relay_mod,
                target,
                runtime=Runtime("crt", {"system-lib": True}),
                mod_name="add",
                params={"c": np.array([[2.0, 4.0]], dtype="float32")},
            )

        temp_dir = utils.tempdir()
        mlf_tar_path = temp_dir.relpath("lib.tar")

        micro.export_model_library_format(factory, mlf_tar_path)
        tf = tarfile.open(mlf_tar_path)

        extract_dir = temp_dir.relpath("extract")
        os.mkdir(extract_dir)
        tf.extractall(extract_dir)

        with open(os.path.join(extract_dir, "metadata.json")) as json_f:
            metadata = json.load(json_f)
            module_name = factory.libmod_name
            assert metadata["version"] == _GENERATED_VERSION
            assert metadata["modules"][module_name]["model_name"] == "add"
            export_datetime = datetime.datetime.strptime(
                metadata["modules"][module_name]["export_datetime"], "%Y-%m-%d %H:%M:%SZ"
            )
            assert (datetime.datetime.now() - export_datetime) < datetime.timedelta(seconds=60 * 5)
            assert metadata["modules"][module_name]["target"] == [str(target)]
            assert metadata["modules"][module_name]["memory"]["sids"] == [
                {"storage_id": 0, "size_bytes": 2, "input_binding": "a"},
                {"storage_id": 1, "size_bytes": 8, "input_binding": "b"},
                {"storage_id": 2, "size_bytes": 8, "input_binding": "p0"},
                {"storage_id": 3, "size_bytes": 8},
            ]
            assert metadata["modules"][module_name]["memory"]["functions"]["main"] == [
                {
                    "constants_size_bytes": 8,
                    "device": 1,
                    "inputs": {
                        "a": {"dtype": "uint8", "size": 2},
                        "b": {"dtype": "float32", "size": 8},
                    },
                    "io_size_bytes": 18,
                    "outputs": {"output": {"dtype": "float32", "size": 8}},
                    "workspace_size_bytes": 0,
                }
            ]
            assert metadata["modules"][module_name]["memory"]["functions"]["operator_functions"][0][
                "workspace"
            ] == [{"device": 1, "workspace_size_bytes": 0}]
            assert (
                "fused_cast_multiply_add"
                in metadata["modules"][module_name]["memory"]["functions"]["operator_functions"][0][
                    "function_name"
                ]
            )

        assert os.path.exists(os.path.join(extract_dir, "codegen", "host", "lib", "add_lib0.o"))

        validate_graph_json(extract_dir, factory)

        with open(os.path.join(extract_dir, "src", f"{module_name}.relay")) as relay_f:
            assert relay_f.read() == str(relay_mod)

        with open(os.path.join(extract_dir, "parameters", "add.params"), "rb") as params_f:
            params = tvm.relay.load_param_dict(params_f.read())
            assert "p0" in params


@tvm.testing.requires_micro
@pytest.mark.parametrize(
    "executor,runtime",
    [(Executor("graph"), Runtime("crt", {"system-lib": True})), (Executor("aot"), Runtime("crt"))],
)
def test_export_model_library_format_workspace(executor, runtime):
    target = tvm.target.target.micro("host")
    with tvm.transform.PassContext(
        opt_level=3, config={"tir.disable_vectorize": True, "tir.usmp.enable": False}
    ):
        relay_mod = tvm.relay.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%p0: Tensor[(1, 56, 56, 128), int16], %p1: Tensor[(3, 3, 128, 1), int16], %p2: Tensor[(1, 1, 1, 128), int32]){
              %0 = nn.conv2d(%p0, %p1, padding=[1, 1, 1, 1], groups=128, channels=128, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWOI", out_dtype="int32") /* ty=Tensor[(1, 56, 56, 128), int32] */;
              %1 = add(%0, %p2) /* ty=Tensor[(1, 56, 56, 128), int32] */;
              %2 = fixed_point_multiply(%1, multiplier=2080045879, shift=-4) /* ty=Tensor[(1, 56, 56, 128), int32] */;
              %3 = clip(%2, a_min=0f, a_max=255f) /* ty=Tensor[(1, 56, 56, 128), int32] */;
              cast(%3, dtype="uint8") /* ty=Tensor[(1, 56, 56, 128), uint8] */
            }
            """
        )
        factory = tvm.relay.build(
            relay_mod,
            target,
            executor=executor,
            runtime=runtime,
            mod_name="qnn_conv2d",
        )

    temp_dir = utils.tempdir()
    mlf_tar_path = temp_dir.relpath("lib.tar")

    micro.export_model_library_format(factory, mlf_tar_path)
    tf = tarfile.open(mlf_tar_path)

    extract_dir = temp_dir.relpath("extract")
    os.mkdir(extract_dir)
    tf.extractall(extract_dir)

    with open(os.path.join(extract_dir, "metadata.json")) as json_f:
        metadata = json.load(json_f)
        module_name = factory.libmod_name
        assert metadata["version"] == _GENERATED_VERSION
        assert metadata["modules"][module_name]["model_name"] == "qnn_conv2d"
        export_datetime = datetime.datetime.strptime(
            metadata["modules"][module_name]["export_datetime"], "%Y-%m-%d %H:%M:%SZ"
        )
        assert (datetime.datetime.now() - export_datetime) < datetime.timedelta(seconds=60 * 5)
        assert metadata["modules"][module_name]["target"] == [str(target)]
        assert metadata["modules"][module_name]["memory"]["functions"]["main"] == [
            {
                "constants_size_bytes": 0,
                "device": 1,
                "inputs": {
                    "p0": {"dtype": "int16", "size": 802816},
                    "p1": {"dtype": "int16", "size": 2304},
                    "p2": {"dtype": "int32", "size": 512},
                },
                "io_size_bytes": 1207040,
                "outputs": {"output": {"dtype": "uint8", "size": 401408}},
                "workspace_size_bytes": 2466816,
            }
        ]
        assert metadata["modules"][module_name]["memory"]["functions"]["operator_functions"][0][
            "workspace"
        ] == [{"device": 1, "workspace_size_bytes": 2466816}]
        assert (
            "fused_nn_conv2d_add_fixed_point_multiply_clip_cast"
            in metadata["modules"][module_name]["memory"]["functions"]["operator_functions"][0][
                "function_name"
            ]
        )


@tvm.testing.requires_micro
def test_export_non_dso_exportable():
    module = tvm.support.FrontendTestModule()

    temp_dir = utils.tempdir()

    with pytest.raises(AssertionError) as exc:
        model_library_format._populate_codegen_dir([module], temp_dir.relpath("codegen"))

        assert str(exc.exception) == (
            "Don't know how to export non-c or non-llvm modules; found: ffi_testing"
        )


@tvm.testing.requires_micro
def test_export_byoc_c_module():
    """Test BYOC flow when it produces DSO-exportable modules.

    NOTE the general BYOC flow is not fully supported by Model Library Format right now.
    """
    x = tvm.relay.var("x", shape=(10, 10))
    w0 = tvm.relay.var("w0", shape=(10, 10))
    w1 = tvm.relay.var("w1", shape=(10, 10))
    w2 = tvm.relay.var("w2", shape=(10, 10))
    w3 = tvm.relay.var("w3", shape=(10, 10))
    w4 = tvm.relay.var("w4", shape=(10, 10))
    w5 = tvm.relay.var("w5", shape=(10, 10))
    w6 = tvm.relay.var("w6", shape=(10, 10))
    w7 = tvm.relay.var("w7", shape=(10, 10))

    # C compiler
    z0 = tvm.relay.add(x, w0)
    p0 = tvm.relay.subtract(z0, w1)
    q0 = tvm.relay.multiply(p0, w2)

    z1 = tvm.relay.add(x, w3)
    p1 = tvm.relay.subtract(z1, w4)
    q1 = tvm.relay.multiply(p1, w5)

    # Other parts on TVM
    z2 = tvm.relay.add(x, w6)
    q2 = tvm.relay.subtract(z2, w7)

    r = tvm.relay.concatenate((q0, q1, q2), axis=0)
    f = tvm.relay.Function([x, w0, w1, w2, w3, w4, w5, w6, w7], r)
    mod = tvm.IRModule()
    ann = byoc.CcompilerAnnotator()
    mod["main"] = ann.visit(f)
    mod = tvm.relay.transform.PartitionGraph("mod_name")(mod)
    mod = tvm.relay.transform.InferType()(mod)

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        factory = tvm.relay.build(mod, tvm.target.target.micro("host"), runtime=Runtime("crt"))

    temp_dir = utils.tempdir()
    mlf_tar_path = temp_dir.relpath("lib.tar")

    micro.export_model_library_format(factory, mlf_tar_path)

    with tarfile.open(mlf_tar_path, "r:*") as tf:
        tar_members = [ti.name for ti in tf.getmembers()]
        print("tar members", tar_members)
        assert "./metadata.json" in tar_members
        with tf.extractfile("./metadata.json") as f:
            metadata = json.load(f)
        main_md = metadata["modules"][factory.libmod_name]["memory"]["functions"]["main"]
        assert main_md == [
            {
                "constants_size_bytes": 0,
                "device": 1,
                "inputs": {
                    "w0": {"dtype": "float32", "size": 400},
                    "w1": {"dtype": "float32", "size": 400},
                    "w2": {"dtype": "float32", "size": 400},
                    "w3": {"dtype": "float32", "size": 400},
                    "w4": {"dtype": "float32", "size": 400},
                    "w5": {"dtype": "float32", "size": 400},
                    "w6": {"dtype": "float32", "size": 400},
                    "w7": {"dtype": "float32", "size": 400},
                    "x": {"dtype": "float32", "size": 400},
                },
                "io_size_bytes": 4800,
                "outputs": {"output": {"dtype": "float32", "size": 1200}},
                "workspace_size_bytes": 1200,
            }
        ]


@tvm.testing.requires_micro
def test_multiple_relay_modules_same_module_name():
    mod = get_conv2d_relay_module()

    executor = Executor("graph")
    runtime = Runtime("crt")
    target = tvm.target.target.micro("host")

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        factory1 = tvm.relay.build(mod, target, runtime=runtime, executor=executor, mod_name="mod")
        factory2 = tvm.relay.build(mod, target, runtime=runtime, executor=executor, mod_name="mod")

    temp_dir = utils.tempdir()
    mlf_tar_path = temp_dir.relpath("lib.tar")

    with pytest.raises(AssertionError, match="Multiple modules should have unique names"):
        micro.export_model_library_format([factory1, factory2], mlf_tar_path)


@tvm.testing.requires_micro
def test_multiple_relay_modules_graph():
    mod = get_conv2d_relay_module()

    executor = Executor("graph")
    runtime = Runtime("crt")
    target = tvm.target.target.micro("host")

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        factory1 = tvm.relay.build(mod, target, runtime=runtime, executor=executor, mod_name="mod1")
        factory2 = tvm.relay.build(mod, target, runtime=runtime, executor=executor, mod_name="mod2")

    temp_dir = utils.tempdir()
    mlf_tar_path = temp_dir.relpath("lib.tar")
    micro.export_model_library_format([factory1, factory2], mlf_tar_path)

    with tarfile.open(mlf_tar_path, "r:*") as tf:
        tar_members = [ti.name for ti in tf.getmembers()]
        print("tar members", tar_members)
        assert "./metadata.json" in tar_members
        assert "./codegen/host/src/mod1_lib0.c" in tar_members
        assert "./codegen/host/src/mod2_lib0.c" in tar_members

        with tf.extractfile("./metadata.json") as f:
            metadata = json.load(f)
        mod2_main_md = metadata["modules"]["mod2"]["memory"]["functions"]["main"]
        assert mod2_main_md == [
            {
                "constants_size_bytes": 0,
                "device": 1,
                "inputs": {
                    "data": {"dtype": "int8", "size": 12288},
                    "weight": {"dtype": "int8", "size": 600},
                },
                "io_size_bytes": 143960,
                "outputs": {"output": {"dtype": "int32", "size": 131072}},
                "workspace_size_bytes": 158088,
            }
        ]
        assert metadata["modules"]["mod1"]["model_name"] == "mod1"
        assert metadata["modules"]["mod2"]["model_name"] == "mod2"


@tvm.testing.requires_micro
def test_multiple_relay_modules_c():
    mod = get_conv2d_relay_module()

    executor = Executor("aot", {"unpacked-api": True, "interface-api": "c"})
    runtime = Runtime("crt")
    target = tvm.target.target.micro("host")

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        factory1 = tvm.relay.build(mod, target, runtime=runtime, executor=executor, mod_name="mod1")
        factory2 = tvm.relay.build(mod, target, runtime=runtime, executor=executor, mod_name="mod2")

    temp_dir = utils.tempdir()
    mlf_tar_path = temp_dir.relpath("lib.tar")

    micro.export_model_library_format([factory1, factory2], mlf_tar_path)

    tf = tarfile.open(mlf_tar_path)

    extract_dir = temp_dir.relpath("extract")
    os.mkdir(extract_dir)
    tf.extractall(extract_dir)

    assert os.path.exists(os.path.join(extract_dir, "codegen", "host", "src", "mod1_lib0.c"))
    assert os.path.exists(os.path.join(extract_dir, "codegen", "host", "src", "mod1_lib1.c"))
    assert os.path.exists(os.path.join(extract_dir, "codegen", "host", "src", "mod2_lib0.c"))
    assert os.path.exists(os.path.join(extract_dir, "codegen", "host", "src", "mod2_lib1.c"))

    assert os.path.exists(os.path.join(extract_dir, "codegen", "host", "include", "tvmgen_mod1.h"))
    assert os.path.exists(os.path.join(extract_dir, "codegen", "host", "include", "tvmgen_mod2.h"))

    # check CRT runtime directory
    assert os.path.exists(os.path.join(extract_dir, "runtime"))


@tvm.testing.requires_micro
def test_multiple_relay_modules_aot_graph():
    mod = get_conv2d_relay_module()

    executor1 = Executor("graph")
    executor2 = Executor("aot", {"unpacked-api": True, "interface-api": "c"})
    runtime = Runtime("crt")
    target = tvm.target.target.micro("host")

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        factory1 = tvm.relay.build(
            mod, target, runtime=runtime, executor=executor1, mod_name="mod1"
        )
        factory2 = tvm.relay.build(
            mod, target, runtime=runtime, executor=executor2, mod_name="mod2"
        )

    temp_dir = utils.tempdir()
    mlf_tar_path = temp_dir.relpath("lib.tar")

    micro.export_model_library_format([factory1, factory2], mlf_tar_path)

    tf = tarfile.open(mlf_tar_path)
    extract_dir = temp_dir.relpath("extract")
    os.mkdir(extract_dir)
    tf.extractall(extract_dir)

    assert os.path.exists(os.path.join(extract_dir, "codegen", "host", "src", "mod1_lib0.c"))
    assert os.path.exists(os.path.join(extract_dir, "codegen", "host", "src", "mod1_lib1.c"))
    assert os.path.exists(os.path.join(extract_dir, "codegen", "host", "src", "mod2_lib0.c"))
    assert os.path.exists(os.path.join(extract_dir, "codegen", "host", "src", "mod2_lib1.c"))

    assert os.path.exists(os.path.join(extract_dir, "codegen", "host", "include", "tvmgen_mod2.h"))

    with open(os.path.join(extract_dir, "metadata.json")) as f:
        metadata = json.load(f)

    assert metadata["modules"]["mod1"]["executors"] == ["graph"]
    assert metadata["modules"]["mod2"]["executors"] == ["aot"]
    assert metadata["version"] == _GENERATED_VERSION


@tvm.testing.requires_micro
def test_output_name_single():
    """Generate a conv2d Relay module for testing."""
    input_a = tvm.relay.var("input_a", shape=(3, 4, 5), dtype="int64")
    output_1 = input_a + tvm.relay.const(1, "int64")
    attrs = tvm.ir.make_node("DictAttrs", output_tensor_names=["test_output_a"])
    main_func = tvm.relay.Function([input_a], output_1, attrs=attrs)
    mod = tvm.IRModule.from_expr(main_func)
    mod = tvm.relay.transform.InferType()(mod)

    executor = Executor("aot", {"unpacked-api": True, "interface-api": "c"})
    runtime = Runtime("crt")
    target = tvm.target.target.micro("host")

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        factory = tvm.relay.build(mod, target, runtime=runtime, executor=executor, mod_name="mod1")
    temp_dir = utils.tempdir()
    mlf_tar_path = temp_dir.relpath("lib.tar")

    micro.export_model_library_format(factory, mlf_tar_path)

    tf = tarfile.open(mlf_tar_path)
    extract_dir = temp_dir.relpath("extract")
    os.mkdir(extract_dir)
    tf.extractall(extract_dir)

    with open(os.path.join(extract_dir, "metadata.json")) as f:
        metadata = json.load(f)

    assert metadata["modules"]["mod1"]["memory"]["functions"]["main"][0]["outputs"] == {
        "test_output_a": {"size": 480, "dtype": "int64"}
    }


@tvm.testing.requires_micro
def test_output_names_many():
    """Generate a conv2d Relay module for testing."""
    input_a = tvm.relay.var("input_a", shape=(3, 4, 5), dtype="int64")
    input_b = tvm.relay.var("input_b", shape=(3, 4), dtype="int32")
    input_c = tvm.relay.var("input_c", shape=(3,), dtype="float32")

    output_1 = input_a + tvm.relay.const(1, "int64")
    output_2 = input_b + tvm.relay.const(2)
    output_3 = input_b + tvm.relay.const(3)
    output_4 = input_c + tvm.relay.const(4.0)

    full_output = tvm.relay.Tuple(
        [output_1, tvm.relay.Tuple([tvm.relay.Tuple([output_2, output_3]), output_4])]
    )
    attrs = tvm.ir.make_node(
        "DictAttrs",
        output_tensor_names=["test_output_a", "test_output_b", "test_output_c", "test_output_d"],
    )
    main_func = tvm.relay.Function([input_a, input_b, input_c], full_output, attrs=attrs)
    mod = tvm.IRModule.from_expr(main_func)
    mod = tvm.relay.transform.InferType()(mod)

    executor = Executor("aot", {"unpacked-api": True, "interface-api": "c"})
    runtime = Runtime("crt")
    target = tvm.target.target.micro("host")

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        factory = tvm.relay.build(mod, target, runtime=runtime, executor=executor, mod_name="mod1")
    temp_dir = utils.tempdir()
    mlf_tar_path = temp_dir.relpath("lib.tar")

    micro.export_model_library_format(factory, mlf_tar_path)

    tf = tarfile.open(mlf_tar_path)
    extract_dir = temp_dir.relpath("extract")
    os.mkdir(extract_dir)
    tf.extractall(extract_dir)

    with open(os.path.join(extract_dir, "metadata.json")) as f:
        metadata = json.load(f)

    assert metadata["modules"]["mod1"]["memory"]["functions"]["main"][0]["outputs"] == {
        "test_output_a": {"size": 480, "dtype": "int64"},
        "test_output_b": {"size": 48, "dtype": "int32"},
        "test_output_c": {"size": 48, "dtype": "int32"},
        "test_output_d": {"size": 12, "dtype": "float32"},
    }


@tvm.testing.requires_micro
def test_template_files():
    """Check template files in generated model library format."""
    mod = get_conv2d_relay_module()

    executor = Executor("aot", {"unpacked-api": True, "interface-api": "c"})
    runtime = Runtime("crt")
    target = tvm.target.target.micro("host")

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        factory = tvm.relay.build(mod, target, runtime=runtime, executor=executor, mod_name="mod")

    temp_dir = utils.tempdir()
    mlf_tar_path = temp_dir / "lib.tar"
    micro.export_model_library_format(factory, mlf_tar_path)

    tf = tarfile.open(mlf_tar_path)
    extract_dir = temp_dir / "extract"
    os.mkdir(extract_dir)
    tf.extractall(extract_dir)

    assert (extract_dir / "templates" / "crt_config.h.template").is_file()
    assert (extract_dir / "templates" / "platform.c.template").is_file()


if __name__ == "__main__":
    tvm.testing.main()
