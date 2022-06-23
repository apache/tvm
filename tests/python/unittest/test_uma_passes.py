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
from tvm import topi
from tvm.relay.backend.contrib.uma._template.passes import my_ai_hw_conv2d_pass
import numpy as np
from tvm.contrib import utils, clang
import tvm.testing
from tvm import te


def test_external_conv2d():
    def c_to_llvm() -> str:
        cc_code = """
             extern "C" int my_hw_ai_conv2dnchw(float* data, float*  weight, float*  result) {
                    result[0] = 42.0;
                    result[1] = 3.14;

              return 0;
            }
        """
        temp = utils.tempdir()
        ll_path = temp.relpath("conv2d.ll")
        ll_code = clang.create_llvm(cc_code, output=ll_path)
        return ll_code

    target = tvm.target.Target(target="llvm", host="llvm")
    dev = tvm.device(target.kind.name, 0)

    ifmap = te.placeholder((1, 3, 224, 224), dtype="float32", name="ifmap")
    weights = te.placeholder((1, 3, 3, 3), dtype="float32", name="weights")
    ifmap_data = tvm.nd.array(np.random.uniform(size=(1, 3, 224, 224)).astype("float32"), dev)
    weight_data = tvm.nd.array(np.random.uniform(size=(1, 3, 3, 3)).astype("float32"), dev)
    result_data = tvm.nd.array(np.zeros((1, 1, 224, 224)).astype("float32"), dev)

    result = topi.nn.conv2d_nchw(ifmap, weights,  stride=1, padding=1, dilation=1)

    # Add pragma TE
    s = te.create_schedule(result.op)
    axis = result.op.axis
    s[result].pragma(axis[0], "import_llvm", c_to_llvm())
    with tvm.transform.PassContext(config={"tir.add_lower_pass": [(1, my_ai_hw_conv2d_pass)]}):
        mod = tvm.lower(s, [ifmap, weights, result], simple_mode=True)

    llvm_mod = tvm.build(mod, [ifmap, weights, result], target=target, name="test_external_conv2d")
    llvm_mod(ifmap_data, weight_data, result_data)

    print(result_data)
    tvm.testing.assert_allclose(result_data.numpy()[0, 0, 0, 0], 42.0, rtol=1e-5)
    tvm.testing.assert_allclose(result_data.numpy()[0, 0, 0, 1], 3.14, rtol=1e-5)
    tvm.testing.assert_allclose(result_data.numpy()[0, 0, 0, 2], 0.0, rtol=1e-5)


test_external_conv2d()