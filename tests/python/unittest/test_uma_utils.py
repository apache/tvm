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
import numpy as np
from tvm.contrib import utils, clang
import tvm.testing
from tvm import te

conv2d_c_code = """
extern "C" int my_hw_ai_conv2dnchw(float* ifmap, float*  weights, float*  result,
                                   int oc, int iw, int ih, int ic, int kh, int kw) {

  int kw_low = kw / 2;
  int kh_low = kh / 2;
  int kw_high = iw + kw / 2;
  int kh_high = ih + kh / 2;

  int padded_iw = iw + 2 * kw_low; 
  int padded_ih = ih + 2 * kh_low;

  float* pad_temp = new float[(((ic * padded_iw * padded_ih) + (padded_ih * padded_iw)) + padded_iw)];

  if (pad_temp == nullptr) {
    return -1;
  }

  int shift = padded_iw * kh_low + kw_low;

  for (int i1 = 0; i1 < ic; ++i1) {
    for (int i2 = 0; i2 < padded_ih; ++i2) {
      for (int i3 = 0; i3 < padded_iw; ++i3) {
        ((float*)pad_temp)[(((i1 * padded_iw * padded_ih) + (i2 * padded_iw)) + i3)] = 
           (((((kh_low <= i2) && (i2 < kh_high)) && (kw_low <= i3)) && (i3 < kw_high)) ? ifmap[((((i1 * iw * ih) + (i2 * iw)) + i3) - shift)] : 0.000000e+00f);
      }
    }
  }
  for (int i11 = 0; i11 < oc; ++i11) { 
    for (int i21 = 0; i21 < ih; ++i21) { 
      for (int i31 = 0; i31 < iw; ++i31) { 
        for (int i4 = 0; i4 < ic; ++i4) { 
          for (int i5 = 0; i5 < kh; ++i5) { 
            for (int i6 = 0; i6 < kw; ++i6) { 
              int cse_var_1 = (((i11 * iw*ih) + (i21 * iw)) + i31);
              if (((i4 == 0) && (i5 == 0)) && (i6 == 0)) {
                result[cse_var_1] = 0.000000e+00f;
              }
              result[cse_var_1] = (result[cse_var_1] 
              + (((float*)pad_temp)[i4 * padded_iw * padded_ih + (i21+i5) * padded_iw + i31 + i6] 
              * weights[((((i11 * ic * kh * kw) + (i4 * kh * kw)) + (i5 * kw)) + i6)]));
            }
          }
        }
      }
    }
  }
  delete[] pad_temp;
  return 0;
}
"""


def _create_schedule(placeholder: list, c_code_str: str = "", use_external_conv2d_impl: bool = True):
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

    assert use_external_conv2d_impl and c_code_str != "" \
           or not use_external_conv2d_impl and c_code_str == ""

    def _c_to_llvm(c_code: str) -> str:
        temp = utils.tempdir()
        ll_path = temp.relpath("conv2d.ll")
        ll_code = clang.create_llvm([c_code], output=ll_path)
        return ll_code

    func_tir = te.create_prim_func(placeholder)
    ir_module_from_te = IRModule({"main": func_tir})
    sch_tir = tvm.tir.Schedule(ir_module_from_te)
    if use_external_conv2d_impl:
        conv2d_b = sch_tir.get_block("conv2d_nchw")
        conv2d_l = sch_tir.get_loops(conv2d_b)
        sch_tir.annotate(conv2d_l[0], "pragma_import_llvm", _c_to_llvm(c_code_str))
    return sch_tir


def _generate_io_arrays(shapes: dict, dev):
    n, w, h, ci, kw, kh, co = shapes["n"], shapes["w"], shapes["h"], shapes["ci"], shapes["kw"], shapes["kh"], shapes["co"],

    ifmap_data = tvm.nd.array(np.random.uniform(size=(n, ci, w, h)).astype("float32"), dev)
    weight_data = tvm.nd.array(np.random.uniform(size=(co, ci, kh, kw)).astype("float32"), dev)
    result_data = tvm.nd.array(np.zeros((n, co, w, h)).astype("float32"), dev)
    return ifmap_data, weight_data, result_data
