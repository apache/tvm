/*!
 *  Copyright (c) 2016 by Contributors
 *  Implementation of API functions related to Codegen
 * \file c_api_codegen.cc
 */
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/codegen.h>
#include <tvm/lowered_func.h>
#include <tvm/api_registry.h>

namespace tvm {
namespace codegen {

TVM_REGISTER_API(_codegen__Build)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args[0].IsNodeType<LoweredFunc>()) {
      *ret = Build({args[0]}, args[1]);
    } else {
      *ret = Build(args[0], args[1]);
    }
  });

TVM_REGISTER_API(_codegen__Enabled)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = TargetEnabled(args[0]);
  });
}  // namespace codegen
}  // namespace tvm
