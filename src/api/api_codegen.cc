/*!
 *  Copyright (c) 2016 by Contributors
 *  Implementation of API functions related to Codegen
 * \file c_api_codegen.cc
 */
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/codegen.h>
#include <tvm/api_registry.h>
#include "../codegen/codegen_c.h"

namespace tvm {
namespace codegen {

TVM_REGISTER_API(_codegen_CompileToC)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = CodeGenC().Compile(args[0], args[1]);
  });

TVM_REGISTER_API(_codegen_MakeAPI)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = MakeAPI(
        args[0], args[1], args[2], args[3]);
  });

TVM_REGISTER_API(_codegen_SplitHostDevice)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = SplitHostDevice(args[0]);
  });

// generate a dummy packed function for testing
void DummyHelloFunction(TVMArgs args, TVMRetValue* rv) {
  LOG(INFO) << args.size() << " arguments";
  for (int i = 0; i < args.size(); ++i) {
    switch (args.type_codes[i]) {
      case kNull: LOG(INFO) << i << ":nullptr"; break;
      case kFloat: LOG(INFO) << i << ": double=" << args.values[i].v_float64; break;
      case kInt: LOG(INFO) << i << ": long=" << args.values[i].v_int64; break;
      case kHandle: LOG(INFO) << i << ": handle=" << args.values[i].v_handle; break;
      case kArrayHandle: LOG(INFO) << i << ": array_handle=" << args.values[i].v_handle; break;
      default: LOG(FATAL) << "unhandled type " << runtime::TypeCode2Str(args.type_codes[i]);
    }
  }
}

TVM_REGISTER_API(_codegen_DummyHelloFunction)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = runtime::PackedFunc(DummyHelloFunction);
  });

TVM_REGISTER_API(_codegen_BuildStackVM)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = BuildStackVM(args[0]);
  });

}  // namespace codegen
}  // namespace tvm
