/*!
 *  Copyright (c) 2017 by Contributors
 * \file intrin_rule_cuda.cc
 * \brief CUDA intrinsic rules.
 */
#include "./intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {
// Add float suffix to the intrinsics, CUDA fast math.
struct CUDAMath {
  std::string operator()(Type t, std::string name) const {
    if (t.lanes() == 1) {
      if (t.is_float()) {
        switch (t.bits()) {
          case 64: return name;
          case 32: return name + 'f';
          case 16: return 'h' + name;
          default: return "";
        }
      }
    }
    return "";
  }
};

struct CUDAFastMath : public CUDAMath {
  std::string operator()(Type t, std::string name) const {
    if (t.lanes() == 1 && t.is_float() && t.bits() == 32) {
      return "__" + name + 'f';
    } else {
      return CUDAMath::operator()(t, name);
    }
    return "";
  }
};

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.exp")
.set_body(DispatchExtern<CUDAFastMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.log")
.set_body(DispatchExtern<CUDAFastMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.tanh")
.set_body(DispatchExtern<CUDAMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.sqrt")
.set_body(DispatchExtern<CUDAMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.pow")
.set_body(DispatchExtern<CUDAMath>);

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
