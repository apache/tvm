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
"""AOT with C++ Runtime Tests"""

import re
import textwrap

import numpy as np
import pytest

import tvm
from tvm import IRModule
from tvm import relay
from tvm.relay import backend, testing
from tvm.testing.aot import generate_ref_data


def test_error_c_interface():
    """Checks that an error occurs when using the packed API in combination with C interface"""

    two = relay.add(relay.const(1), relay.const(1))
    func = relay.Function([], two)

    with pytest.raises(
        tvm.TVMError,
        match=re.escape(
            'Need unpacked-api == false (got: 0) and interface-api == "packed" (got: c) when '
            "targeting c++ runtime"
        ),
    ):
        tvm.relay.build(
            IRModule.from_expr(func),
            target="llvm",
            executor=backend.Executor("aot", {"interface-api": "c"}),
        )


@pytest.mark.parametrize("enable_usmp", [True, False])
@pytest.mark.parametrize("target_kind", ["c", "llvm"])
def test_conv2d(enable_usmp, target_kind):
    """Tests compilation of convolutions"""
    relay_model = textwrap.dedent(
        """\
        #[version = "0.0.5"]
        def @main(%data : Tensor[(1, 3, 64, 64), uint8], %weight : Tensor[(3, 3, 5, 5), int8]) {
            %1 = nn.conv2d(
                 %data,
                 %weight,
                 padding=[2, 2],
                 channels=3,
                 kernel_size=[5, 5],
                 data_layout="NCHW",
                 kernel_layout="OIHW",
                 out_dtype="int32");
            %2 = cast(nn.max_pool2d(%1, pool_size=[3, 3]), dtype="int8");
            %3 = nn.conv2d(
                 %2,
                 %weight,
                 padding=[2, 2],
                 channels=3,
                 kernel_size=[5, 5],
                 data_layout="NCHW",
                 kernel_layout="OIHW",
                 out_dtype="int32");
            %4 = nn.max_pool2d(%3, pool_size=[3, 3]);
            %4
        }
    """
    )
    ir_mod = tvm.relay.fromtext(relay_model)

    main_func = ir_mod["main"]
    shape_dict = {p.name_hint: p.checked_type.concrete_shape for p in main_func.params}
    type_dict = {p.name_hint: p.checked_type.dtype for p in main_func.params}

    weight_data = np.random.randint(1, 255, shape_dict["weight"]).astype(type_dict["weight"])
    input_data = np.ones(shape_dict["data"]).astype(type_dict["data"])
    params = {"weight": weight_data}
    inputs = {"data": input_data}
    ref_outputs = generate_ref_data(ir_mod, inputs, params)

    with tvm.transform.PassContext(
        opt_level=3,
        config={
            "tir.disable_vectorize": True,
            "tir.usmp.enable": enable_usmp,
        },
    ):
        mod = tvm.relay.build(
            ir_mod,
            params=params,
            target=target_kind,
            executor=backend.Executor("aot", {"interface-api": "packed", "unpacked-api": False}),
        )
    temp_dir = tvm.contrib.utils.TempDirectory()
    test_so_path = temp_dir / "test.so"
    mod.export_library(test_so_path, cc="gcc", options=["-std=c11", "-g3", "-O0"])
    loaded_mod = tvm.runtime.load_module(test_so_path)
    runner = tvm.runtime.executor.AotModule(loaded_mod["default"](tvm.cpu(0)))
    runner.set_input(**inputs)

    assert runner.get_input_name(0) == "data"
    shape_dict, dtype_dict = runner.get_input_info()
    assert shape_dict == {"data": (1, 3, 64, 64)}
    assert dtype_dict == {"data": "uint8"}

    runner.run()
    assert (runner.get_output(0).numpy() == list(ref_outputs.values())[0]).all()


@pytest.mark.parametrize("enable_usmp", [True, False])
@pytest.mark.parametrize("target_kind", ["c", "llvm"])
def test_mobilenet(enable_usmp: bool, target_kind: str):
    """Full network test with Mobilenet"""
    ir_mod, params = testing.mobilenet.get_workload(batch_size=1)
    data_shape = [int(x) for x in ir_mod["main"].checked_type.arg_types[0].shape]
    data = np.random.uniform(size=data_shape).astype("float32")
    inputs = {"data": data}
    ref_outputs = generate_ref_data(ir_mod, inputs, params)

    with tvm.transform.PassContext(
        opt_level=3, config={"tir.disable_vectorize": True, "tir.usmp.enable": enable_usmp}
    ):
        mod = tvm.relay.build(
            ir_mod,
            params=params,
            target=target_kind,
            executor=backend.Executor("aot", {"interface-api": "packed"}),
        )

    temp_dir = tvm.contrib.utils.TempDirectory()
    test_so_path = temp_dir / "test.so"
    mod.export_library(test_so_path, cc="c++", options=["-std=gnu++17", "-g3", "-O0"])
    loaded_mod = tvm.runtime.load_module(test_so_path)
    runner = tvm.runtime.executor.AotModule(loaded_mod["default"](tvm.cpu(0)))
    runner.set_input(**inputs)
    runner.run()
    assert (runner.get_output(0).asnumpy() == list(ref_outputs.values())[0]).all()


def test_module_list():
    """Checks the correct list of module names is generated"""
    input_x = tvm.relay.var("x", tvm.relay.TensorType([1], dtype="float32"))
    expr = tvm.relay.add(input_x, tvm.relay.Constant(tvm.nd.array(np.array([1], dtype="float32"))))
    mod = tvm.relay.build(
        tvm.IRModule.from_expr(tvm.relay.Function([input_x], expr)),
        target="c",
        executor=tvm.relay.backend.Executor("aot", {"interface-api": "packed"}),
        mod_name="unusual_module_name_fred",
    )
    temp_dir = tvm.contrib.utils.TempDirectory()
    test_so_path = temp_dir / "test.so"
    mod.export_library(test_so_path, cc="gcc", options=["-std=c11"])
    loaded_mod = tvm.runtime.load_module(test_so_path)
    list_module_names = loaded_mod.get_function("list_module_names")
    names_expected = ["unusual_module_name_fred"]
    assert list(sorted(names_expected)) == list(sorted(list_module_names()))


def test_create_executor():
    x = tvm.relay.var("x", tvm.relay.TensorType([1], dtype="float32"))
    expr = tvm.relay.add(x, tvm.relay.Constant(tvm.nd.array(np.array([1], dtype="float32"))))
    actual = relay.create_executor(
        "aot", mod=tvm.IRModule.from_expr(tvm.relay.Function([x], expr)), target="c"
    ).evaluate()(np.array([2], dtype="float32"))

    np.isfinite(np.array([3], dtype="float32"))

    np.testing.assert_allclose(actual.numpy(), np.array([3], dtype="float32"))


def test_pass_wrong_device_arg():
    """Ensure an error is generated if the incorrect number of devices are passed"""
    x = tvm.relay.var("x", tvm.relay.TensorType([1], dtype="float32"))
    expr = tvm.relay.add(x, tvm.relay.Constant(tvm.nd.array(np.array([1], dtype="float32"))))
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.relay.build(
            tvm.IRModule.from_expr(tvm.relay.Function([x], expr)),
            target="c",
            executor=backend.Executor("aot", {"interface-api": "packed"}),
        )

    temp_dir = tvm.contrib.utils.TempDirectory()
    test_so_path = temp_dir / "test.so"
    mod.export_library(test_so_path, cc="gcc", options=["-std=c11", "-g3", "-O0"])
    loaded_mod = tvm.runtime.load_module(test_so_path)

    with pytest.raises(tvm.TVMError) as error:
        tvm.runtime.executor.AotModule(loaded_mod["default"](tvm.cpu(0), tvm.cpu(0)))

        assert (
            "Check failed: devices_.size() == 1 (2 vs. 1) : Expect exactly 1 device passed."
            in str(error.exception)
        )
    # TODO write asserts for # and type of device.


@pytest.mark.parametrize("target_kind", ["c", "llvm"])
@pytest.mark.parametrize("input_name", ["input:0", "input@0", "input_0"])
def test_aot_input_name_with_special_character(target_kind: str, input_name: str):
    """Test name transforms in AOT for input names with special characters."""
    dtype = "float32"
    input_1 = relay.var(input_name, shape=(10, 5), dtype=dtype)
    weight = relay.var("weight", shape=(1, 5), dtype=dtype)
    output = relay.add(input_1, weight)
    func = relay.Function([input_1, weight], output)

    input_data = np.random.rand(10, 5).astype(dtype)
    weight_data = np.random.rand(1, 5).astype(dtype)
    expected_output = input_data + weight_data
    params = {"weight": weight_data}

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.relay.build(
            tvm.IRModule.from_expr(func),
            target=target_kind,
            params=params,
            executor=tvm.relay.backend.Executor("aot", {"interface-api": "packed"}),
        )
    temp_dir = tvm.contrib.utils.TempDirectory()
    test_so_path = temp_dir / "test.so"
    mod.export_library(test_so_path, cc="c++", options=["-std=gnu++17", "-g3", "-O0"])
    # test both original name and transformed name
    for name in ["input_0", input_name]:
        loaded_mod = tvm.runtime.load_module(test_so_path)
        runner = tvm.runtime.executor.AotModule(loaded_mod["default"](tvm.cpu(0)))
        inputs = {name: input_data}
        runner.set_input(**inputs)

        input_ind = runner.get_input_index(name)
        assert (runner.get_input(input_ind).asnumpy() == input_data).all()

        runner.run()
        assert (runner.get_output(0).asnumpy() == expected_output).all()


@pytest.mark.parametrize("target_kind", ["c", "llvm"])
def test_aot_incorrect_input_name(target_kind: str):
    """Test passing incorrect input name."""
    dtype = "float32"
    correct_input_name = "input"
    incorrect_input_name = "input1"
    input1 = relay.var(correct_input_name, shape=(10, 5), dtype=dtype)
    weight = relay.var("weight", shape=(1, 5), dtype=dtype)
    output = relay.add(input1, weight)
    func = relay.Function([input1, weight], output)

    input_data = np.random.rand(10, 5).astype(dtype)
    weight_data = np.random.rand(1, 5).astype(dtype)
    params = {"weight": weight_data}

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.relay.build(
            tvm.IRModule.from_expr(func),
            target=target_kind,
            params=params,
            executor=tvm.relay.backend.Executor("aot", {"interface-api": "packed"}),
        )
    temp_dir = tvm.contrib.utils.TempDirectory()
    test_so_path = temp_dir / "test.so"
    mod.export_library(test_so_path, cc="c++", options=["-std=gnu++17", "-g3", "-O0"])

    loaded_mod = tvm.runtime.load_module(test_so_path)
    runner = tvm.runtime.executor.AotModule(loaded_mod["default"](tvm.cpu(0)))
    inputs = {incorrect_input_name: input_data}

    error_regex = r"Invalid input name."
    with pytest.raises(tvm.TVMError, match=error_regex):
        runner.set_input(**inputs)

    with pytest.raises(tvm.TVMError, match=error_regex):
        runner.get_input_index(incorrect_input_name)


if __name__ == "__main__":
    tvm.testing.main()
