/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file intrin_rule_cuda.cc
 * \brief CUDA intrinsic rules.
 */
#include "../intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {
// Add float suffix to the intrinsics, CUDA fast math.
struct CUDAMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float()) {
      switch (t.bits()) {
        case 64: return name;
        case 32: return name + 'f';
        case 16: return 'h' + name;
        default: return "";
      }
    }
    return "";
  }
};

struct CUDAFastMath : public CUDAMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float() && t.bits() == 32) {
      return "__" + name + 'f';
    } else {
      return CUDAMath::operator()(t, name);
    }
    return "";
  }
};

struct CUDAFastMathTan : public CUDAMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float()) {
        switch (t.bits()) {
          case 64: return name;
          // `__tanf` seems to produce some values too deviant from numpy tan version.
          // So, let's use just `tanf` instead.
          case 32: return name + 'f';
          case 16: LOG(FATAL) << "cuda tan unsupported for float16";
          default: return "";
        }
    }
    return "";
  }
};

struct CUDAPopcount {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_uint()) {
      switch (t.bits()) {
        case 32: return "__popc";
        case 64: return "__popcll";
        default: return "";
      }
    }
    return "";
  }
};

static void DispatchCUDAShuffle(const TVMArgs& args, TVMRetValue* rv) {
  PrimExpr e = args[0];
  const CallNode* call = e.as<CallNode>();
  CHECK(call != nullptr);
  CHECK_EQ(call->args.size(), 4);  // value, warp_id, width, warp_size
  Array<PrimExpr> cuda_args{{call->args[0], call->args[1], call->args[2]}};
  *rv = CallNode::make(
      call->dtype, "__shfl", cuda_args, CallNode::PureExtern);
}

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.floor")
.set_body(DispatchExtern<CUDAMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.ceil")
.set_body(DispatchExtern<CUDAMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.trunc")
.set_body(DispatchExtern<CUDAMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.fabs")
.set_body(DispatchExtern<CUDAMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.round")
.set_body(DispatchExtern<CUDAMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.exp")
.set_body(DispatchExtern<CUDAFastMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.exp2")
.set_body(DispatchExtern<CUDAMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.exp10")
.set_body(DispatchExtern<CUDAFastMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.erf")
.set_body(DispatchExtern<CUDAMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.log")
.set_body(DispatchExtern<CUDAFastMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.log2")
.set_body(DispatchExtern<CUDAFastMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.log10")
.set_body(DispatchExtern<CUDAFastMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.tan")
.set_body(DispatchExtern<CUDAFastMathTan>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.cos")
.set_body(DispatchExtern<CUDAFastMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.cosh")
.set_body(DispatchExtern<CUDAMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.sin")
.set_body(DispatchExtern<CUDAFastMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.sinh")
.set_body(DispatchExtern<CUDAMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.atan")
.set_body(DispatchExtern<CUDAMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.tanh")
.set_body(DispatchExtern<CUDAMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.sqrt")
.set_body(DispatchExtern<CUDAMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.pow")
.set_body(DispatchExtern<CUDAMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.popcount")
.set_body(DispatchExtern<CUDAPopcount>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.tvm_warp_shuffle")
.set_body(DispatchCUDAShuffle);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.fmod")
.set_body(DispatchExtern<CUDAMath>);

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
