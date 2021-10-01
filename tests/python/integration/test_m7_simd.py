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

from collections import OrderedDict
import sys

import numpy as np
import pytest

import tvm
from tvm import relay
from tvm.ir.module import IRModule
from tvm.relay import transform
from aot_test_utils import (
    AOTTestModel,
    AOT_DEFAULT_RUNNER,
    generate_ref_data,
    compile_and_run,
    parametrize_aot_options,
)



@parametrize_aot_options
@pytest.mark.parametrize("groups,weight_shape", [(1, 32), (32, 1)])
def test_conv2d(interface_api, use_unpacked_api, test_runner, groups, weight_shape):
    """Test a subgraph with a single conv2d operator."""
    dtype = "int8"
    ishape = (1, 32, 14, 14)
    wshape = (32, weight_shape, 3, 3)

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



@parametrize_aot_options
def test_multiple_models(interface_api, use_unpacked_api, test_runner):
    # Identity model without params
    x = relay.var("x", "float32")
    mod1 = relay.Function([x], x)
    one = np.array(1.0, "float32")
    inputs1 = {"x": one}
    output_list1 = generate_ref_data(mod1, inputs1)
    params1 = None

    # Convolution model
    RELAY_MODEL = """
#[version = "0.0.5"]
def @main(%data : Tensor[(1, 3, 64, 64), uint8], %weight : Tensor[(8, 3, 5, 5), int8]) {
    %1 = nn.conv2d(
         %data,
         %weight,
         padding=[2, 2],
         channels=8,
         kernel_size=[5, 5],
         data_layout="NCHW",
         kernel_layout="OIHW",
         out_dtype="int32");
  %1
}
"""
    mod2 = tvm.parser.fromtext(RELAY_MODEL)
    main_func = mod2["main"]
    shape_dict = {p.name_hint: p.checked_type.concrete_shape for p in main_func.params}
    type_dict = {p.name_hint: p.checked_type.dtype for p in main_func.params}

    weight_data = np.ones(shape_dict["weight"]).astype(type_dict["weight"])
    input_data = np.ones(shape_dict["data"]).astype(type_dict["data"])

    params2 = {"weight": weight_data}
    inputs2 = {"data": input_data}
    output_list2 = generate_ref_data(mod2, inputs2, params2)

    compile_and_run(
        [
            AOTTestModel(
                name="mod1", module=mod1, inputs=inputs1, outputs=output_list1, params=params1
            ),
            AOTTestModel(
                name="mod2", module=mod2, inputs=inputs2, outputs=output_list2, params=params2
            ),
        ],
        test_runner,
        interface_api,
        use_unpacked_api,
    )


#   target_str = f"c -keys=arm_cpu -mcpu={platform['mcpu']}  -march={platform['march']} -model={platform['model']} -runtime=c -link-params=1 --executor=aot --unpacked-api=1 --interface-api=c"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
