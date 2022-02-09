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
""" This file contains test that use USMP + AoT using C runtime APIs"""

from collections import OrderedDict
import sys

import numpy as np
import pytest

import tvm
from tvm import relay, TVMError
from tvm.ir.module import IRModule
from tvm.relay import testing, transform
from tvm.relay.testing import byoc
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.backend import Executor, Runtime
from tvm import WorkspaceMemoryPools, PoolInfo
from aot_test_utils import (
    AOTTestModel,
    AOTTestRunner,
    generate_ref_data,
    convert_to_relay,
    compile_and_run,
    compile_models,
    parametrize_aot_options,
    run_and_check,
)


def check_for_no_tvm_backendallocworkspace_calls(mod: tvm.runtime.module):
    """This checker checks whether any c-source produced has TVMBackendAllocWorkspace calls.
    If USMP is invoked, none of them should have TVMBAW calls"""
    dso_modules = mod._collect_dso_modules()
    for dso_mod in dso_modules:
        assert (
            dso_mod.type_key == "c"
        ), 'Current CRT AoT codegen flow should only produce type "c" runtime modules'
        source = dso_mod.get_source()
        source.count(
            "TVMBackendAllocWorkspace"
        ) == 0, "This is failing because USMP was unable to plan for every tir.allocate node"


@pytest.mark.parametrize(
    "workspace_byte_alignment,main_workspace_size",
    [
        (8, 17280),
        (16, 17280),
        (256, 17792),
    ],
)
def test_memory_planning(workspace_byte_alignment, main_workspace_size):
    mod, params = tvm.relay.testing.synthetic.get_workload()
    target = "c"
    runtime = Runtime("crt")
    executor = Executor(
        "aot",
        {
            "workspace-byte-alignment": workspace_byte_alignment,
        },
    )
    with tvm.transform.PassContext(
        opt_level=3,
        config={
            "tir.disable_vectorize": True,
            "tir.disable_storage_rewrite": True,
            "tir.usmp.enable": True,
            "tir.usmp.algorithm": "greedy_by_conflicts",
        },
    ):
        lib = tvm.relay.build(mod, target, executor=executor, runtime=runtime, params=params)
    assert (
        sum(lib.function_metadata["__tvm_main__"].workspace_sizes.values()) == main_workspace_size
    )


@parametrize_aot_options
@pytest.mark.parametrize("groups,weight_shape", [(1, 32), (32, 1)])
def test_conv2d(interface_api, use_unpacked_api, test_runner, groups, weight_shape):
    """Test a subgraph with a single conv2d operator."""
    dtype = "float32"
    ishape = (1, 32, 14, 14)
    wshape = (32, weight_shape, 3, 3)
    pass_config = {"tir.usmp.enable": True}
    test_runner = AOTTestRunner(
        makefile=test_runner.makefile,
        prologue=test_runner.prologue,
        epilogue=test_runner.epilogue,
        includes=test_runner.includes,
        parameters=test_runner.parameters,
        pass_config=pass_config,
    )

    data0 = relay.var("data", shape=ishape, dtype=dtype)
    weight0 = relay.var("weight", shape=wshape, dtype=dtype)
    out = relay.nn.conv2d(data0, weight0, kernel_size=(3, 3), padding=(1, 1), groups=groups)
    main_f = relay.Function([data0, weight0], out)
    mod = tvm.IRModule()
    mod["main"] = main_f
    mod = transform.InferType()(mod)

    i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    w1_data = np.random.uniform(0, 1, wshape).astype(dtype)

    inputs = OrderedDict([("data", i_data), ("weight", w1_data)])

    output_list = generate_ref_data(mod, inputs)
    compile_and_run(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list),
        test_runner,
        interface_api,
        use_unpacked_api,
    )
    compiled_test_mods = compile_models(
        models=AOTTestModel(module=mod, inputs=inputs, outputs=output_list),
        interface_api=interface_api,
        use_unpacked_api=use_unpacked_api,
        pass_config=test_runner.pass_config,
    )

    for compiled_model in compiled_test_mods:
        check_for_no_tvm_backendallocworkspace_calls(compiled_model.executor_factory.lib)

    run_and_check(
        models=compiled_test_mods,
        runner=test_runner,
        interface_api=interface_api,
    )


@pytest.mark.parametrize("merge_compiler_regions", [False, True])
def test_byoc_microtvm(merge_compiler_regions):
    """This is a simple test to check BYOC capabilities of AOT - with and without merging compiler regions to test for https://github.com/apache/tvm/issues/9036"""
    use_unpacked_api = False
    interface_api = "packed"
    test_runner = AOTTestRunner(pass_config={"tir.usmp.enable": True})

    x = relay.var("x", shape=(10, 10))
    w0 = relay.var("w0", shape=(10, 10))
    w1 = relay.var("w1", shape=(10, 10))

    # z0 = x + w0
    x_ = compiler_begin(x, "ccompiler")
    w0_ = compiler_begin(w0, "ccompiler")
    z0_ = relay.add(x_, w0_)
    z0 = compiler_end(z0_, "ccompiler")

    # z1 = z0 + w1
    z0__ = compiler_begin(z0, "ccompiler")
    w1_ = compiler_begin(w1, "ccompiler")
    z1_ = relay.add(z0__, w1_)
    z1 = compiler_end(z1_, "ccompiler")

    # z2 = z0 + z1
    z2 = relay.add(z0, z1)

    f = relay.Function([x, w0, w1], z2)
    mod = tvm.IRModule()
    mod["main"] = f

    if merge_compiler_regions:
        mod = transform.MergeCompilerRegions()(mod)

    mod = transform.PartitionGraph("mod_name")(mod)
    mod = transform.InferType()(mod)

    x_data = [("x", np.random.rand(10, 10).astype("float32"))]
    w_data = [("w{}".format(i), np.random.rand(10, 10).astype("float32")) for i in range(2)]

    map_inputs = OrderedDict(x_data + w_data)
    output_list = generate_ref_data(mod, map_inputs)

    compiled_test_mods = compile_models(
        AOTTestModel(name="my_mod", module=mod, inputs=map_inputs, outputs=output_list),
        interface_api=interface_api,
        use_unpacked_api=use_unpacked_api,
        pass_config=test_runner.pass_config,
    )

    for compiled_model in compiled_test_mods:
        check_for_no_tvm_backendallocworkspace_calls(compiled_model.executor_factory.lib)

    run_and_check(
        models=compiled_test_mods,
        runner=test_runner,
        interface_api=interface_api,
    )


def _get_relay_module_and_inputs_from_tflite_file(tflite_model_file):
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()
    mod, params = convert_to_relay(tflite_model_buf)

    inputs = dict()
    for param in mod["main"].params:
        name = str(param.name_hint)
        data_shape = [int(i) for i in param.type_annotation.shape]
        dtype = str(param.type_annotation.dtype)
        in_min, in_max = (np.iinfo(dtype).min, np.iinfo(dtype).max)
        data = np.random.randint(in_min, high=in_max, size=data_shape, dtype=dtype)
        inputs[name] = data

    return mod, inputs, params


MOBILENET_V1_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz",
    "mobilenet_v1_1.0_224_quant.tflite",
)
MOBILENET_V2_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz",
    "mobilenet_v2_1.0_224_quant.tflite",
)


@pytest.mark.parametrize(
    "model_url, usmp_algo, workspace_size,",
    [
        (MOBILENET_V1_URL, "greedy_by_size", 4845696),
        (MOBILENET_V1_URL, "greedy_by_conflicts", 4444288),
        (MOBILENET_V1_URL, "hill_climb", 3240064),
    ],
)
def test_tflite_model_u1_usecase(model_url, usmp_algo, workspace_size):
    """This checks for ML models and the memory used by them when using USMP with different algorithms"""
    pytest.importorskip("tflite")

    import tvm.relay.testing.tf as tf_testing

    use_unpacked_api = True
    interface_api = "c"
    test_runner = AOTTestRunner(
        pass_config={"tir.usmp.enable": True, "tir.usmp.algorithm": usmp_algo}
    )

    tflite_model_file = tf_testing.get_workload_official(
        model_url[0],
        model_url[1],
    )
    mod, inputs, params = _get_relay_module_and_inputs_from_tflite_file(tflite_model_file)
    output_list = generate_ref_data(mod, inputs, params)

    compiled_test_mods = compile_models(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list, params=params),
        interface_api=interface_api,
        use_unpacked_api=use_unpacked_api,
        pass_config=test_runner.pass_config,
    )

    for compiled_model in compiled_test_mods:
        check_for_no_tvm_backendallocworkspace_calls(compiled_model.executor_factory.lib)

    # Checking the workspace size
    assert (
        sum(
            compiled_model.executor_factory.function_metadata[
                "__tvm_main__"
            ].workspace_sizes.values()
        )
        == workspace_size
    )

    run_and_check(
        models=compiled_test_mods,
        runner=test_runner,
        interface_api=interface_api,
    )


def _get_workspace_size_define_macro(pool_name: str, model_name="default") -> str:
    """This function converts pool names to compiler generated
    workspace pool size macros"""

    prefix = "TVMGEN_" + model_name.upper() + "_"
    postfix = "_WORKSPACE_POOL_SIZE"
    return prefix + pool_name.upper() + postfix


@pytest.mark.parametrize(
    "model_url, usmp_algo",
    [
        (MOBILENET_V1_URL, "greedy_by_size"),
    ],
)
def test_tflite_model_u3_usecase_single_external_pool(model_url, usmp_algo):
    """This checks for inference with USMP using external pool placed in the application"""
    pytest.importorskip("tflite")

    import tvm.relay.testing.tf as tf_testing

    use_unpacked_api = True
    interface_api = "c"

    pool_name = "my_memory_pool"
    target = tvm.target.Target("c")
    workspace_memory_pools = WorkspaceMemoryPools(
        [PoolInfo(pool_name, {target: PoolInfo.READ_WRITE_ACCESS})]
    )
    test_runner = AOTTestRunner(
        pass_config={"tir.usmp.enable": True, "tir.usmp.algorithm": usmp_algo},
        prologue=f"""
        __attribute__((section(".data.tvm"), aligned(16)))
        static uint8_t {pool_name}[{_get_workspace_size_define_macro(pool_name)}];
        """,
    )

    tflite_model_file = tf_testing.get_workload_official(
        model_url[0],
        model_url[1],
    )
    mod, inputs, params = _get_relay_module_and_inputs_from_tflite_file(tflite_model_file)
    output_list = generate_ref_data(mod, inputs, params)

    compiled_test_mods = compile_models(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list, params=params),
        interface_api=interface_api,
        use_unpacked_api=use_unpacked_api,
        pass_config=test_runner.pass_config,
        workspace_memory_pools=workspace_memory_pools,
        target=target,
    )

    for compiled_model in compiled_test_mods:
        check_for_no_tvm_backendallocworkspace_calls(compiled_model.executor_factory.lib)

    run_and_check(
        models=compiled_test_mods,
        runner=test_runner,
        interface_api=interface_api,
    )


@pytest.mark.parametrize(
    "model_url, usmp_algo",
    [
        (MOBILENET_V1_URL, "greedy_by_size"),
    ],
)
def test_tflite_model_u3_usecase_two_external_pools(model_url, usmp_algo):
    """This checks for inference using two external pools placed in the application"""
    pytest.importorskip("tflite")

    import tvm.relay.testing.tf as tf_testing

    use_unpacked_api = True
    interface_api = "c"

    target = tvm.target.Target("c")
    workspace_memory_pools = WorkspaceMemoryPools(
        [
            PoolInfo(
                "my_memory_pool_1", {target: PoolInfo.READ_WRITE_ACCESS}, size_hint_bytes=2500000
            ),
            PoolInfo("my_memory_pool_2", {target: PoolInfo.READ_WRITE_ACCESS}),
        ]
    )
    test_runner = AOTTestRunner(
        pass_config={"tir.usmp.enable": True, "tir.usmp.algorithm": usmp_algo},
        prologue=f"""
        __attribute__((section(".data.tvm"), aligned(16)))
        static uint8_t my_memory_pool_1[{_get_workspace_size_define_macro("my_memory_pool_1")}];
        __attribute__((section(".data.tvm"), aligned(16)))
        static uint8_t my_memory_pool_2[{_get_workspace_size_define_macro("my_memory_pool_2")}];
        """,
    )

    tflite_model_file = tf_testing.get_workload_official(
        model_url[0],
        model_url[1],
    )
    mod, inputs, params = _get_relay_module_and_inputs_from_tflite_file(tflite_model_file)
    output_list = generate_ref_data(mod, inputs, params)

    compiled_test_mods = compile_models(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list, params=params),
        interface_api=interface_api,
        use_unpacked_api=use_unpacked_api,
        pass_config=test_runner.pass_config,
        workspace_memory_pools=workspace_memory_pools,
        target=target,
    )

    for compiled_model in compiled_test_mods:
        check_for_no_tvm_backendallocworkspace_calls(compiled_model.executor_factory.lib)

    run_and_check(
        models=compiled_test_mods,
        runner=test_runner,
        interface_api=interface_api,
    )


@pytest.mark.parametrize(
    "model_urls, usmp_algo",
    [
        ((MOBILENET_V1_URL, MOBILENET_V2_URL), "greedy_by_size"),
    ],
)
def test_tflite_model_u2_usecase_two_models_with_a_single_external_pool(model_urls, usmp_algo):
    """This checks for inference using a single large enough common pool"""
    pytest.importorskip("tflite")

    import tvm.relay.testing.tf as tf_testing

    use_unpacked_api = True
    interface_api = "c"

    target = tvm.target.Target("c")
    workspace_memory_pools = WorkspaceMemoryPools(
        [PoolInfo("my_memory_pool", {target: PoolInfo.READ_WRITE_ACCESS})]
    )
    test_runner = AOTTestRunner(
        pass_config={"tir.usmp.enable": True, "tir.usmp.algorithm": usmp_algo},
        prologue=f"""
        #define MAX(A, B) ((A > B) ? A : B)
        __attribute__((section(".data.tvm"), aligned(16)))
        static uint8_t my_memory_pool[MAX({_get_workspace_size_define_macro("my_memory_pool", "mod1")},{_get_workspace_size_define_macro("my_memory_pool", "mod2")})];
        """,
    )

    tflite_model_file1 = tf_testing.get_workload_official(
        model_urls[0][0],
        model_urls[0][1],
    )
    mod1, inputs1, params1 = _get_relay_module_and_inputs_from_tflite_file(tflite_model_file1)
    output_list1 = generate_ref_data(mod1, inputs1, params1)

    tflite_model_file2 = tf_testing.get_workload_official(
        model_urls[1][0],
        model_urls[1][1],
    )
    mod2, inputs2, params2 = _get_relay_module_and_inputs_from_tflite_file(tflite_model_file2)
    output_list2 = generate_ref_data(mod2, inputs2, params2)

    compiled_test_mods = compile_models(
        [
            AOTTestModel(
                name="mod1", module=mod1, inputs=inputs1, outputs=output_list1, params=params1
            ),
            AOTTestModel(
                name="mod2", module=mod2, inputs=inputs2, outputs=output_list2, params=params2
            ),
        ],
        interface_api=interface_api,
        use_unpacked_api=use_unpacked_api,
        pass_config=test_runner.pass_config,
        workspace_memory_pools=workspace_memory_pools,
        target=target,
    )

    for compiled_model in compiled_test_mods:
        check_for_no_tvm_backendallocworkspace_calls(compiled_model.executor_factory.lib)

    run_and_check(
        models=compiled_test_mods,
        runner=test_runner,
        interface_api=interface_api,
    )
