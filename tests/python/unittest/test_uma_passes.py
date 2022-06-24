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
from tvm import topi, IRModule
from tvm.relay.backend.contrib.uma._template.passes import my_ai_hw_conv2d_pass
import numpy as np
from tvm.contrib import utils, clang
import tvm.testing
from tvm import te
from tvm.relay.backend.contrib.uma.api.lower import UMALower
from tvm.relay.backend.contrib.uma.api.utils import PassPhase

conv2d_c_code = """
extern "C" int my_hw_ai_conv2dnchw(float* data, float*  weight, float*  result) {
  result[0] = 42.0;
  result[1] = 3.14;
  /*
  int ix = 224;
  int iy = 224;
  int ic = 3;
  int kx = 3;
  int ky = 3;

  int pad_size = ix * iy * ic;
  float*  pad_temp = new float[pad_size];
  if (pad_temp == nullptr) {
    return -1;
  }

  for (int i1 = 0; i1 < ic; ++i1) {
    for (int i2 = 0; i2 < ix; ++i2) {
      for (int i3 = 0; i3 < iy; ++i3) {
        ((float*)pad_temp)[(((i1 * 900) + (i2 * 30)) + i3)] = (((((1 <= i2) && (i2 < 29)) && (1 <= i3)) && (i3 < 29)) ? weight[((((i1 * 784) + (i2 * 28)) + i3) - 29)] : 0.000000e+00f);
      }
    }
  }
  
  for (int i11 = 0; i11 < 256; ++i11) {
    for (int i21 = 0; i21 < 14; ++i21) {
      for (int i31 = 0; i31 < 14; ++i31) {
        for (int i4 = 0; i4 < 256; ++i4) {
          for (int i5 = 0; i5 < kx; ++i5) {
            for (int i6 = 0; i6 < ky; ++i6) {
              int cse_var_1 = (((i11 * 196) + (i21 * 14)) + i31);
              if (((i4 == 0) && (i5 == 0)) && (i6 == 0)) {
                result[cse_var_1] = 0.000000e+00f;
              }
              result[cse_var_1] = (result[cse_var_1] + (((float*)pad_temp)[(((((i4 * 900) + (i21 * 60)) + (i5 * 30)) + (i31 * 2)) + i6)] * data[((((i11 * 2304) + (i4 * 9)) + (i5 * 3)) + i6)]));
            }
          }
        }
      }
    }
  }
  
  delete[] pad_temp;
  */
  return 0;
}
"""


def _c_to_llvm(c_code: str) -> str:
    temp = utils.tempdir()
    ll_path = temp.relpath("conv2d.ll")
    ll_code = clang.create_llvm([c_code], output=ll_path)
    return ll_code


def _conv2d_te_definition() -> list:
    ifmap = te.placeholder((1, 3, 224, 224), dtype="float32", name="ifmap")
    weights = te.placeholder((1, 3, 3, 3), dtype="float32", name="weights")
    result = topi.nn.conv2d_nchw(ifmap, weights, stride=1, padding=1, dilation=1)
    return [ifmap, weights, result]


def _pepare_conv2d_schedule():
    target = tvm.target.Target(target="llvm", host="llvm")
    dev = tvm.device(target.kind.name, 0)
    placeholders = _conv2d_te_definition()
    runtime_np_arrays = _generate_numpy_arrays(dev)
    sch_tir = _add_llvm_to_tir(placeholders, conv2d_c_code)
    return placeholders, runtime_np_arrays, sch_tir, target,


def _add_llvm_to_tir(placeholder: list, c_code_str: str):
    # How to do the same with TE
    # Add pragma TE
    # s = te.create_schedule(result.op)
    # axis = result.op.axis
    # s[result].pragma(axis[0], "import_llvm", c_to_llvm())
    # with tvm.transform.PassContext(config={"tir.add_lower_pass": [(1, my_ai_hw_conv2d_pass)]}):
    #     mod = tvm.lower(s, [ifmap, weights, result], simple_mode=True)
    #
    # llvm_mod = tvm.build(mod, [ifmap, weights, result], target=target, name="test_external_conv2d")
    # llvm_mod(ifmap_data, weight_data, result_data)

    func_tir = te.create_prim_func(placeholder)
    ir_module_from_te = IRModule({"main": func_tir})
    sch_tir = tvm.tir.Schedule(ir_module_from_te)
    conv2d_b = sch_tir.get_block("conv2d_nchw")
    conv2d_l = sch_tir.get_loops(conv2d_b)
    sch_tir.annotate(conv2d_l[0], "pragma_import_llvm", _c_to_llvm(c_code_str))
    return sch_tir


def _generate_numpy_arrays(dev):
    ifmap_data = tvm.nd.array(np.random.uniform(size=(1, 3, 224, 224)).astype("float32"), dev)
    weight_data = tvm.nd.array(np.random.uniform(size=(1, 3, 3, 3)).astype("float32"), dev)
    result_data = tvm.nd.array(np.zeros((1, 1, 224, 224)).astype("float32"), dev)
    return ifmap_data, weight_data, result_data


def test_lower_with_uma():
    placeholders, runtime_np_arrays, schedule, target = _pepare_conv2d_schedule()
    ifmap_data, weight_data, result_data = runtime_np_arrays

    uma_lower = UMALower("lower_test")
    uma_lower._tir_passes.append((PassPhase.TIR_PHASE_0, my_ai_hw_conv2d_pass))
    with tvm.transform.PassContext():
        tir_mod = uma_lower._lower_stir_to_nstir(schedule.mod["main"])

    llvm_mod = tvm.build(tir_mod, placeholders, target=target, name="test_external_conv2d")
    llvm_mod(ifmap_data, weight_data, result_data)

    tvm.testing.assert_allclose(result_data.numpy()[0, 0, 0, 0], 42.0, rtol=1e-5)
    tvm.testing.assert_allclose(result_data.numpy()[0, 0, 0, 1], 3.14, rtol=1e-5)
    tvm.testing.assert_allclose(result_data.numpy()[0, 0, 0, 2], 0.0, rtol=1e-5)
    print(result_data)


def test_lower_standalone():
    ifmap, ifmap_data, result, result_data, sch_tir, target, weight_data, weights = _pepare_conv2d_schedule()
    tir_mod = my_ai_hw_conv2d_pass(sch_tir.mod)
    llvm_mod = tvm.build(tir_mod, [ifmap, weights, result], target=target, name="test_external_conv2d")
    llvm_mod(ifmap_data, weight_data, result_data)

    tvm.testing.assert_allclose(result_data.numpy()[0, 0, 0, 0], 42.0, rtol=1e-5)
    tvm.testing.assert_allclose(result_data.numpy()[0, 0, 0, 1], 3.14, rtol=1e-5)
    tvm.testing.assert_allclose(result_data.numpy()[0, 0, 0, 2], 0.0, rtol=1e-5)

    print(result_data)


#test_lower_standalone()
test_lower_with_uma()
