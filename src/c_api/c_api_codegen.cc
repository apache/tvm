/*!
 *  Copyright (c) 2016 by Contributors
 *  Implementation of API functions related to Codegen
 * \file c_api_codegen.cc
 */
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/codegen.h>

#include "./c_api_registry.h"
#include "../codegen/codegen_c.h"

namespace tvm {
namespace codegen {

using ArgStack = const std::vector<APIVariantValue>;
using RetValue = APIVariantValue;

TVM_REGISTER_API(_codegen_CompileToC)
.set_body([](const ArgStack& args, RetValue *ret) {
    *ret = CodeGenC().Compile(args.at(0), args.at(1));
  });

TVM_REGISTER_API(_codegen_MakeAPI)
.set_body([](const ArgStack& args, RetValue *ret) {
    *ret = MakeAPI(
        args.at(0), args.at(1), args.at(2), args.at(3));
  });

TVM_REGISTER_API(_codegen_SplitHostDevice)
.set_body([](const ArgStack& args, RetValue *ret) {
    *ret = SplitHostDevice(args.at(0));
  });


// generate a dummy packed function for testing
void DummyHelloFunction(const TVMValue* args, const int* type_code, int num_args) {
  LOG(INFO) << num_args << " arguments";
  for (int i = 0; i < num_args; ++i) {
    switch (type_code[i]) {
      case kNull: LOG(INFO) << i << ":nullptr"; break;
      case kFloat: LOG(INFO) << i << ": double=" << args[i].v_float64; break;
      case kInt: LOG(INFO) << i << ": long=" << args[i].v_int64; break;
      case kHandle: LOG(INFO) << i << ": handle=" << args[i].v_handle; break;
      default: LOG(FATAL) << "unhandled type " << TVMTypeCode2Str(type_code[i]);
    }
  }
}

TVM_REGISTER_API(_codegen_DummyHelloFunction)
.set_body([](const ArgStack& args, RetValue *ret) {
    *ret = runtime::PackedFunc(DummyHelloFunction);
  });

TVM_REGISTER_API(_codegen_BuildStackVM)
.set_body([](const ArgStack& args, RetValue *ret) {
    *ret = BuildStackVM(args.at(0));
  });

}  // namespace codegen
}  // namespace tvm
