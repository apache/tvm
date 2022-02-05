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


def print_mod_tree(m, indent=0):
    print(f"{' ' * indent} - {m!r}")
    for i in m.imported_modules:
        print_mod_tree(i, indent + 2)


unpacked_api = tvm.testing.parameter(True, False)


target_kind = tvm.testing.parameter("c", "llvm")


def test_conv2d(target_kind, unpacked_api):
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
    output_list = generate_ref_data(ir_mod, inputs, params)

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.relay.build(
            ir_mod,
            params=params,
            target=target_kind,
            executor=backend.Executor("aot", {"unpacked-api": unpacked_api, "interface-api": "packed"}),
        )

    print_mod_tree(mod.module)

    with tvm.contrib.utils.TempDirectory.set_keep_for_debug():
        mod.export_library("test.so", options=["-fpermissive"])
    mod.export_library("test.tar")
    runner = tvm.runtime.load_module("test.so")
    print_mod_tree(runner)
    runner = tvm.runtime.executor.AotModule(runner["default"](tvm.cpu(0)))
    runner.set_input(**inputs)
    runner.run()
    assert (runner.get_output(0).asnumpy() == output_list[0]).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
