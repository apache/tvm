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
struct CUDAFastMath {
  std::string operator()(Type t, std::string name) const {
    if (t.lanes() == 1) {
      if (t.is_float()) {
        switch (t.bits()) {
          case 64: return name;
          case 32: return "__" + name + 'f';
          case 16: return 'h' + name;
          default: return "";
        }
      }
    }
    return "";
  }
};

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.exp")
.set_body(DispatchExtern<CUDAFastMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.log")
.set_body(DispatchExtern<CUDAFastMath>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cuda.tanh")
.set_body(DispatchExtern<CUDAFastMath>);

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
