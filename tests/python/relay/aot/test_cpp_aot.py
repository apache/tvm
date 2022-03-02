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


import re
import sys
import textwrap

import numpy as np
import pytest

import tvm
from tvm import relay, TVMError
from tvm.ir.module import IRModule
from tvm.relay import backend, testing, transform
from tvm.relay.testing import byoc
from tvm.relay.op.annotation import compiler_begin, compiler_end
from aot_test_utils import (
    AOTTestModel,
    AOT_DEFAULT_RUNNER,
    generate_ref_data,
    convert_to_relay,
    compile_and_run,
    compile_models,
    parametrize_aot_options,
)


def test_error_c_interface():
    interface_api = "c"
    use_unpacked_api = False
    test_runner = AOT_DEFAULT_RUNNER

    two = relay.add(relay.const(1), relay.const(1))
    func = relay.Function([], two)

    with pytest.raises(
        tvm.TVMError,
        match=re.escape(
            'Either need interface_api == "packed" (got: c) or '
            "unpacked-api == true (got: (bool)0) when targeting "
            "c runtime"
        ),
    ):
        compile_and_run(
            AOTTestModel(
                module=IRModule.from_expr(func), inputs={}, outputs=generate_ref_data(func, {})
            ),
            test_runner,
            interface_api,
            use_unpacked_api,
        )


enable_usmp = tvm.testing.parameter(True, False)


def test_conv2d(enable_usmp):
    RELAY_MODEL = textwrap.dedent(
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
    ir_mod = tvm.parser.fromtext(RELAY_MODEL)

    main_func = ir_mod["main"]
    shape_dict = {p.name_hint: p.checked_type.concrete_shape for p in main_func.params}
    type_dict = {p.name_hint: p.checked_type.dtype for p in main_func.params}

    weight_data = np.ones(shape_dict["weight"]).astype(type_dict["weight"])
    input_data = np.ones(shape_dict["data"]).astype(type_dict["data"])

    params = {"weight": weight_data}
    inputs = {"data": input_data}
    ref_outputs = generate_ref_data(ir_mod, inputs, params)

    with tvm.transform.PassContext(
        opt_level=3, config={"tir.disable_vectorize": True, "tir.usmp.enable": enable_usmp}
    ):
        mod = tvm.relay.build(
            ir_mod,
            params=params,
            target="c",
            executor=backend.Executor("aot", {"interface-api": "packed"}),
        )

    temp_dir = tvm.contrib.utils.TempDirectory()
    test_so_path = temp_dir / "test.so"
    mod.export_library(test_so_path, cc="gcc", options=["-std=c11"])
    loaded_mod = tvm.runtime.load_module(test_so_path)
    runner = tvm.runtime.executor.AotModule(loaded_mod["default"](tvm.cpu(0)))
    runner.set_input(**inputs)
    runner.run()
    assert (runner.get_output(0).asnumpy() == list(ref_outputs.values())[0]).all()


def test_mobilenet():
    ir_mod, params = testing.mobilenet.get_workload(batch_size=1)
    data_shape = [int(x) for x in ir_mod["main"].checked_type.arg_types[0].shape]
    data = np.random.uniform(size=data_shape).astype("float32")
    inputs = {"data": data}
    ref_outputs = generate_ref_data(ir_mod, inputs, params)

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.relay.build(
            ir_mod,
            params=params,
            target="c",
            executor=backend.Executor("aot", {"interface-api": "packed"}),
        )

    temp_dir = tvm.contrib.utils.TempDirectory()
    test_so_path = temp_dir / "test.so"
    mod.export_library(test_so_path, cc="gcc", options=["-std=c11"])
    loaded_mod = tvm.runtime.load_module(test_so_path)
    runner = tvm.runtime.executor.AotModule(loaded_mod["default"](tvm.cpu(0)))
    runner.set_input(**inputs)
    runner.run()
    assert (runner.get_output(0).asnumpy() == list(ref_outputs.values())[0]).all()


def test_create_executor():
    x = tvm.relay.var("x", tvm.relay.TensorType([1], dtype="float32"))
    expr = tvm.relay.add(x, tvm.relay.Constant(tvm.nd.array(np.array([1], dtype="float32"))))
    actual = relay.create_executor(
        "aot", mod=tvm.IRModule.from_expr(tvm.relay.Function([x], expr)), target="c -executor=aot"
    ).evaluate()(np.array([2], dtype="float32"))

    np.isfinite(np.array([3], dtype="float32"))

    np.testing.assert_allclose(actual.numpy(), np.array([3], dtype="float32"))


def test_pass_wrong_device_arg():
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
    mod.export_library(test_so_path, cc="gcc", options=["-std=c11"])
    loaded_mod = tvm.runtime.load_module(test_so_path)

    with pytest.raises(tvm.TVMError) as cm:
        tvm.runtime.executor.AotModule(loaded_mod["default"](tvm.cpu(0), tvm.cpu(0)))

        assert (
            "Check failed: devices_.size() == 1 (2 vs. 1) : Expect exactly 1 device passed."
            in str(cm.exception)
        )
    # TODO write asserts for # and type of device.


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
