/*!
 *  Copyright (c) 2016 by Contributors
 *  Implementation of API functions related to IR build
 * \file c_api_ir.cc
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

}  // namespace codegen
}  // namespace tvm
