
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
using namespace tvm;
using namespace tvm::runtime;

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
}  // namespace tvm_ext
