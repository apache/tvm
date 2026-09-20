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

/*! \file codegen_cuda_host.cc
 *  \brief C host wrappers with direct CUDA kernel launches.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/attrs.h>

#include <algorithm>
#include <array>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../../runtime/metadata.h"
#include "../../../target/source/codegen_c_host.h"

namespace tvm {
namespace codegen {

class CodeGenCUDAHost : public CodeGenCHost {
 public:
  void Init(Target target) {
    CodeGenCHost::Init(false, true, true, target->str(), {});
    check_error_ = name_supply_->FreshName("tvm_cuda_host_check");
    decl_stream << "#include <cuda_runtime.h>\n";
    decl_stream << "static int " << check_error_
                << "(cudaError_t error, bool allow_unloading = false) {\n"
                << "  if (error == cudaSuccess || "
                << "(allow_unloading && error == cudaErrorCudartUnloading)) return 0;\n"
                << "  const char* parts[] = {\"CUDA launch: \", cudaGetErrorString(error)};\n"
                << "  TVMFFIErrorSetRaisedFromCStrParts(\"CUDAError\", parts, 2);\n"
                << "  return -1;\n}\n";
  }

  void Dispatch_(const CallNode* op, std::ostream& os) override {
    if (!op->op.same_as(tirx::builtin::call_ffi_kernel())) {
      CodeGenCHost::Dispatch_(op, os);
      return;
    }
    const auto* attr = op->attrs.as<tirx::CallFFIKernelAttr>();
    TVM_FFI_CHECK(attr, ValueError) << "cuda_host kernel calls require CallFFIKernelAttr";
    TVM_FFI_CHECK(!op->args.empty() && op->args[0].as<StringImmNode>(), ValueError)
        << "cuda_host kernel calls require a string kernel symbol";
    const auto& symbol = op->args[0].as<StringImmNode>()->value;

    std::array<int, 6> axes;
    axes.fill(-1);
    int shared_memory = -1;
    size_t num_launch_values = 0;
    std::unordered_set<std::string> seen;
    for (size_t i = 0; i < attr->launch_params.size(); ++i) {
      std::string tag = attr->launch_params[i];
      TVM_FFI_CHECK(seen.insert(tag).second, ValueError)
          << "cuda_host duplicate launch parameter: " << tag;
      // These are flags, not values in the argument suffix.  Reject unsupported
      // launch semantics before decoding any operands, rather than dropping them.
      if (tag == runtime::launch_param::kUseProgramaticDependentLaunch ||
          tag == runtime::launch_param::kUseCooperativeLaunch ||
          tag == runtime::launch_param::kUseRequiredBlockDimension) {
        TVM_FFI_THROW(ValueError) << "cuda_host does not support launch flag: " << tag;
      } else if (tag == runtime::launch_param::kUseDynamicSharedMemoryTag) {
        TVM_FFI_CHECK_EQ(i + 1, attr->launch_params.size(), ValueError)
            << "cuda_host dynamic shared memory must be the last launch parameter";
        shared_memory = static_cast<int>(num_launch_values++);
      } else {
        int axis = -1;
        for (int j = 0; j < 3; ++j) {
          if (tag == std::string("blockIdx.") + "xyz"[j]) axis = j;
          if (tag == std::string("threadIdx.") + "xyz"[j]) axis = j + 3;
        }
        TVM_FFI_CHECK_GE(axis, 0, ValueError)
            << "cuda_host does not support launch parameter: " << tag;
        axes[axis] = static_cast<int>(num_launch_values++);
      }
    }
    TVM_FFI_CHECK_GE(op->args.size(), num_launch_values + 1, ValueError)
        << "cuda_host kernel call is missing launch arguments";
    size_t launch_begin = op->args.size() - num_launch_values;
    for (size_t i = launch_begin; i < op->args.size(); ++i) {
      auto type = op->args[i]->ty.as<PrimType>();
      TVM_FFI_CHECK(
          type.has_value() && type.value().IsScalar() && type.value().MatchesCode(kDLInt, kDLUInt),
          ValueError)
          << "cuda_host launch arguments must be scalar integers";
    }

    // Evaluate arguments once in their original order, including launch values.
    // Buffer expressions retain their pointer type for the direct kernel call.
    std::vector<std::string> arguments;
    for (size_t i = 1; i < op->args.size(); ++i) {
      std::string value = PrintExpr(op->args[i]);
      std::string name = name_supply_->FreshName("cuda_arg");
      PrintIndent();
      stream << "auto " << name << " = " << value << ";\n";
      arguments.push_back(name);
    }
    auto launch_arg = [&](int index) { return arguments[launch_begin - 1 + index]; };
    std::array<std::string, 6> dimensions;
    for (size_t i = 0; i < axes.size(); ++i) {
      if (axes[i] < 0) {
        dimensions[i] = "1";
      } else {
        dimensions[i] = name_supply_->FreshName("cuda_dim");
        PrintIndent();
        stream << "size_t " << dimensions[i] << " = static_cast<size_t>(" << launch_arg(axes[i])
               << ");\n";
        // Match LaunchParamConfig's handling of empty dynamic dimensions.
        PrintIndent();
        stream << "if (" << dimensions[i] << " == 0) " << dimensions[i] << " = 1;\n";
      }
    }
    std::string device = name_supply_->FreshName("cuda_device");
    std::string cuda_stream = name_supply_->FreshName("cuda_stream");
    PrintIndent();
    stream << "int " << device << ";\n";
    CheckError("cudaGetDevice(&" + device + ")");
    PrintIndent();
    stream << "cudaStream_t " << cuda_stream
           << " = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, " << device << "));\n";
    std::string bytes = shared_memory < 0 ? "0" : launch_arg(shared_memory);
    if (shared_memory >= 0) {
      PrintIndent();
      stream << "if (" << bytes << " >= 49152) {\n";
      int scope = BeginScope();
      CheckError("cudaFuncSetAttribute(::" + std::string(symbol) +
                 ", cudaFuncAttributeMaxDynamicSharedMemorySize, " + bytes + ")");
      EndScope(scope);
      PrintIndent();
      stream << "}\n";
    }
    PrintIndent();
    stream << "::" << symbol << "<<<dim3(" << dimensions[0] << ", " << dimensions[1] << ", "
           << dimensions[2] << "), dim3(" << dimensions[3] << ", " << dimensions[4] << ", "
           << dimensions[5] << "), " << bytes << ", " << cuda_stream << ">>>(";
    for (size_t i = 0; i + 1 < launch_begin; ++i) {
      if (i != 0) stream << ", ";
      stream << arguments[i];
    }
    stream << ");\n";
    CheckError("cudaGetLastError()", true);
    os << "0";
  }

 private:
  void CheckError(const std::string& call, bool allow_unloading = false) {
    PrintIndent();
    stream << "if (::" << check_error_ << "(" << call;
    if (allow_unloading) stream << ", true";
    stream << ") != 0) return -1;\n";
  }
  std::string check_error_;
};

ffi::Module BuildCUDAHost(IRModule mod, Target target) {
  CodeGenCUDAHost cg;
  cg.Init(target);
  std::vector<std::pair<GlobalVar, PrimFunc>> functions;
  for (auto [gvar, base_func] : mod->functions) {
    functions.emplace_back(gvar, base_func.as_or_throw<PrimFunc>());
  }
  std::sort(functions.begin(), functions.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.first->name_hint < rhs.first->name_hint;
  });
  for (const auto& [gvar, func] : functions) cg.DeclareFunction(gvar, func);
  for (const auto& [gvar, func] : functions) cg.AddFunction(gvar, func, true);
  return CSourceModuleCreate(cg.Finish(), "cu", cg.GetFunctionNames());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("target.build.cuda_host", BuildCUDAHost);
}

}  // namespace codegen
}  // namespace tvm
