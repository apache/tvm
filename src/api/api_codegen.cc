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
#include "../codegen/codegen_cuda.h"
#include "../codegen/codegen_opencl.h"

namespace tvm {
namespace codegen {

TVM_REGISTER_API(_codegen_CompileToC)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    std::string mode = "c";
    if (args.size() > 2) {
      mode = args[2].operator std::string();
    }
    if (mode == "c") {
      *ret = CodeGenC().Compile(args[0], args[1]);
    } else if (mode == "cuda") {
      *ret = CodeGenCUDA().Compile(args[0], args[1]);
    } else if (mode == "opencl") {
      *ret = CodeGenOpenCL().Compile(args[0], args[1]);
    } else {
      LOG(FATAL) << "cannot recognize mode";
    }
  });

TVM_REGISTER_API(_codegen_BuildStackVM)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = BuildStackVM(args[0],
                        std::unordered_map<LoweredFunc, PackedFunc>());
  });

TVM_REGISTER_API(_codegen_BuildNVRTC)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = BuildNVRTC(args[0], args[1]);
  });

TVM_REGISTER_API(_codegen_BuildOpenCL)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = BuildOpenCL(args[0], args[1]);
  });

}  // namespace codegen
}  // namespace tvm
