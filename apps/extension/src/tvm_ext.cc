
/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Example package that uses TVM.
 * \file tvm_ext.cc
 */
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/packed_func_ext.h>

namespace tvm_ext {
using IntVector = std::vector<int>;
}  // namespace tvm_ext

namespace tvm {
namespace runtime {
template<>
struct extension_class_info<tvm_ext::IntVector> {
  static const int code = 17;
};
}  // namespace tvm
}  // namespace runtime

using namespace tvm;
using namespace tvm::runtime;

namespace tvm_ext {

TVM_REGISTER_EXT_TYPE(IntVector);

TVM_REGISTER_GLOBAL("tvm_ext.ivec_create")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    IntVector vec;
    for (int i = 0; i < args.size(); ++i) {
      vec.push_back(args[i].operator int());
    }
    *rv = vec;
  });

TVM_REGISTER_GLOBAL("tvm_ext.ivec_get")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    *rv = args[0].AsExtension<IntVector>()[args[1].operator int()];
  });


TVM_REGISTER_GLOBAL("tvm_ext.bind_add")
.set_body([](TVMArgs args_, TVMRetValue *rv_) {
    PackedFunc pf = args_[0];
    int b = args_[1];
    *rv_ = PackedFunc([pf, b](TVMArgs args, TVMRetValue *rv) {
        *rv = pf(b, args[0]);
      });
  });

TVM_REGISTER_GLOBAL("tvm_ext.sym_add")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    Var a = args[0];
    Var b = args[1];
    *rv = a + b;
  });

TVM_REGISTER_GLOBAL("device_api.ext_dev")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    *rv = (*tvm::runtime::Registry::Get("device_api.cpu"))();
  });
}  // namespace tvm_ext

// This callback approach allows extension allows tvm to extract
// This way can be helpful when we want to use a header only
// minimum version of TVM Runtime.
extern "C" int TVMExtDeclare(TVMFunctionHandle pregister) {
  const PackedFunc& fregister =
      *static_cast<PackedFunc*>(pregister);
  auto mul = [](TVMArgs args, TVMRetValue *rv) {
    int x = args[0];
    int y = args[1];
    *rv = x * y;
  };
  fregister("mul", PackedFunc(mul));
  return 0;
}
