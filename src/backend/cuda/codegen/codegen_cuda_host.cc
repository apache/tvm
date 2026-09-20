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
#include <tvm/tirx/type.h>

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
    // Retain C-host wrapper initialization, with an FFI/CUDA-only prelude.
    decl_stream.str("");
    decl_stream.clear();
    decl_stream << "#include <tvm/ffi/c_api.h>\n"
                << "#include <tvm/ffi/extra/c_env_api.h>\n"
                << "#include <tvm/ffi/extra/cuda/device_guard.h>\n"
                << "#include <tvm/ffi/function.h>\n"
                << "#include <cuda.h>\n#include <cuda_runtime.h>\n#include <math.h>\n"
                << "#define TVM_DLL TVM_FFI_DLL_EXPORT\n";
    InitGlobalContext();
    check_error_ = name_supply_->FreshName("tvm_cuda_host_check");
    decl_stream << "static int " << check_error_
                << "(cudaError_t error, bool allow_unloading = false) {\n"
                << "  if (error == cudaSuccess || "
                << "(allow_unloading && error == cudaErrorCudartUnloading)) return 0;\n"
                << "  const char* parts[] = {\"CUDA launch: \", cudaGetErrorString(error)};\n"
                << "  TVMFFIErrorSetRaisedFromCStrParts(\"CUDAError\", parts, 2);\n"
                << "  cudaGetLastError();\n"
                << "  return -1;\n}\n";
  }

  using CodeGenCHost::PrintType;

  void PrintType(const Type& type, std::ostream& os) override {
    if (type.as<tirx::TensorMapTypeNode>()) {
      os << "CUtensorMap";
    } else {
      CodeGenCHost::PrintType(type, os);
    }
  }

  void AddFunction(const GlobalVar& gvar, const PrimFunc& func) override {
    // A guard destructor may report a CUDA error on any return path. Keep it
    // inside the FFI exception boundary along with device selection and setup.
    InitFuncState(func);
    PrintFunctionSignature(GetFunctionName(gvar), func, stream);
    stream << " {\n  TVM_FFI_SAFE_CALL_BEGIN();\n  try {\n";
    int scope = BeginScope();
    PrintStmt(func->body);
    EndScope(scope);
    // CUDADeviceGuard reports errors by throwing. Clear the reported CUDA
    // last-error state so a subsequent successful launch is not blamed for it.
    stream << "  } catch (...) {\n    cudaGetLastError();\n    throw;\n  }\n"
           << "  TVM_FFI_SAFE_CALL_END();\n}\n\n";
    if (auto symbol = func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol)) {
      function_names_.push_back(symbol.value());
      if (func->HasNonzeroAttr(tirx::attr::kIsEntryFunc) && !has_tvm_ffi_main_func_) {
        function_names_.push_back(ffi::symbol::tvm_ffi_main);
        PrintFuncPrefix(stream);
        PrintType(func->ret_type, stream);
        stream << " " << ffi::symbol::tvm_ffi_main
               << "(void* self, void* args, int num_args, void* result) {\n"
               << "  return " << symbol.value() << "(self, args, num_args, result);\n}\n";
      }
    }
  }

  ffi::Array<ffi::String> GetFunctionNames() { return function_names_; }

  void Dispatch_(const CallNode* op, std::ostream& os) override {
    if (op->op.same_as(tirx::builtin::tvm_stack_alloca()) &&
        op->args[0].as_or_throw<StringImm>()->value == "tensormap") {
      auto count = op->args[1].as_or_throw<IntImm>()->value.as<int64_t>();
      TVM_FFI_CHECK(count.has_value() && *count > 0, ValueError)
          << "cuda_host requires a positive constant tensor-map allocation count";
      std::string name = name_supply_->FreshName("cuda_tensormap");
      PrintIndent();
      stream << "alignas(64) CUtensorMap " << name << "[" << *count << "];\n";
      os << name;
      return;
    }
    if (op->op.same_as(tirx::builtin::tvm_call_packed_lowered())) {
      const auto& name = op->args[0].as_or_throw<StringImm>()->value;
      if (name == "__tvm_set_device" || name == "runtime.cuTensorMapEncodeTiled" ||
          name == "runtime.cuTensorMapInit") {
        std::string args =
            "((TVMFFIAny*)" + PrintExpr(op->args[1]) + " + " + PrintExpr(op->args[2]) + ")";
        if (name == "__tvm_set_device") {
          std::string guard = name_supply_->FreshName("cuda_device_guard");
          PrintIndent();
          stream << "if (" << args << "[0].v_int64 != kDLCUDA) {\n"
                 << "  TVMFFIErrorSetRaisedFromCStr(\"ValueError\", "
                 << "\"cuda_host requires a CUDA device\");\n  return -1;\n}\n";
          PrintIndent();
          stream << "tvm::ffi::CUDADeviceGuard " << guard << "(" << args << "[1].v_int64);\n";
        } else {
          EmitTensorMapEncoder();
          PrintIndent();
          stream << "if (::" << encode_tensormap_ << "(" << args << ", " << PrintExpr(op->args[3])
                 << " - " << PrintExpr(op->args[2]) << ") != 0) return -1;\n";
        }
        os << "0";
        return;
      }
      std::string function_name = name;
      TVM_FFI_CHECK(function_name.rfind("runtime.", 0) != 0 &&
                        function_name.rfind("device_api.", 0) != 0 &&
                        function_name.rfind("__tvm_", 0) != 0,
                    ValueError)
          << "cuda_host does not provide TVM runtime service: " << name;
    }
    if (auto call_op = op->op.as<Op>()) {
      if (op_attr_global_symbol_.count(call_op.value())) {
        const auto& symbol = op_attr_global_symbol_[call_op.value()];
        TVM_FFI_CHECK(std::string(symbol).rfind("TVMBackend", 0) != 0, ValueError)
            << "cuda_host does not provide TVM runtime operation: " << symbol;
      }
    }
    if (op->op.same_as(tirx::builtin::call_extern()) ||
        op->op.same_as(tirx::builtin::call_pure_extern())) {
      const auto& symbol = op->args[0].as_or_throw<StringImm>()->value;
      TVM_FFI_CHECK(std::string(symbol).rfind("TVMBackend", 0) != 0, ValueError)
          << "cuda_host does not provide TVM runtime operation: " << symbol;
    }
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
      if (i < launch_begin) {
        if (auto* ptr = op->args[i]->ty.as<PointerTypeNode>()) {
          if (ptr->element_type.as<tirx::TensorMapTypeNode>()) {
            // The device ABI takes a grid-constant descriptor by value, while
            // the host IR carries a pointer to its aligned stack storage.
            value = "*(" + value + ")";
          }
        }
      }
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
  void EmitTensorMapEncoder() {
    if (!encode_tensormap_.empty()) return;
    encode_tensormap_ = name_supply_->FreshName("cuda_encode_tensormap");
    decl_stream << "static int " << encode_tensormap_ << "(const TVMFFIAny* values, int count) {\n";
    decl_stream << R"(
  TVM_FFI_SAFE_CALL_BEGIN();
  auto arg = [&](int i) { return tvm::ffi::AnyView::CopyFromTVMFFIAny(values[i]); };
  TVM_FFI_CHECK(count >= 4, ValueError) << "Invalid tensor-map argument count";
  int rank = arg(2).cast<int>();
  TVM_FFI_CHECK(rank >= 1 && rank <= 5, ValueError) << "Tensor-map rank must be 1..5";
  int base_count = 4 * rank + 7;
  TVM_FFI_CHECK(count == base_count || count == base_count + 1, ValueError)
      << "Invalid tensor-map argument count";
  DLDataType dtype = arg(1).cast<DLDataType>();
  TVM_FFI_CHECK(dtype.lanes == 1, ValueError) << "Tensor-map dtype must be scalar";
  CUtensorMapDataType cuda_dtype;
  if ((dtype.code == kDLUInt || dtype.code == kDLInt) && dtype.bits == 8) {
    cuda_dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else if (dtype.code == kDLUInt && dtype.bits == 16) {
    cuda_dtype = CU_TENSOR_MAP_DATA_TYPE_UINT16;
  } else if (dtype.code == kDLUInt && dtype.bits == 32) {
    cuda_dtype = CU_TENSOR_MAP_DATA_TYPE_UINT32;
  } else if (dtype.code == kDLUInt && dtype.bits == 64) {
    cuda_dtype = CU_TENSOR_MAP_DATA_TYPE_UINT64;
  } else if (dtype.code == kDLInt && dtype.bits == 32) {
    cuda_dtype = CU_TENSOR_MAP_DATA_TYPE_INT32;
  } else if (dtype.code == kDLInt && dtype.bits == 64) {
    cuda_dtype = CU_TENSOR_MAP_DATA_TYPE_INT64;
  } else if (dtype.code == kDLFloat && dtype.bits == 16) {
    cuda_dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  } else if (dtype.code == kDLFloat && dtype.bits == 32) {
    cuda_dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  } else if (dtype.code == kDLFloat && dtype.bits == 64) {
    cuda_dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
  } else if (dtype.code == kDLBfloat && dtype.bits == 16) {
    cuda_dtype = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  } else if (dtype.code == kDLFloat8_e4m3fn || dtype.code == kDLFloat8_e5m2) {
    cuda_dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else {
    TVM_FFI_THROW(ValueError) << "Unsupported cuda_host tensor-map dtype";
  }
  if (count > base_count && arg(base_count).cast<int>() != -1) {
    int forced = arg(base_count).cast<int>();
    TVM_FFI_CHECK(forced == CU_TENSOR_MAP_DATA_TYPE_TFLOAT32 &&
                      dtype.code == kDLFloat && dtype.bits == 32,
                  ValueError)
        << "cuda_host only supports a TFLOAT32 tensor-map dtype override";
    cuda_dtype = CU_TENSOR_MAP_DATA_TYPE_TFLOAT32;
  }
  cuuint64_t shape[5], strides[5];
  cuuint32_t box[5], element_strides[5];
  int index = 4;
  for (int i = 0; i < rank; ++i) shape[i] = arg(index++).cast<uint64_t>();
  for (int i = 0; i < rank - 1; ++i) strides[i] = arg(index++).cast<uint64_t>();
  for (int i = 0; i < rank; ++i) box[i] = arg(index++).cast<uint32_t>();
  for (int i = 0; i < rank; ++i) element_strides[i] = arg(index++).cast<uint32_t>();
  auto interleave = static_cast<CUtensorMapInterleave>(arg(index++).cast<int>());
  auto swizzle = static_cast<CUtensorMapSwizzle>(arg(index++).cast<int>());
  auto promotion = static_cast<CUtensorMapL2promotion>(arg(index++).cast<int>());
  auto fill = static_cast<CUtensorMapFloatOOBfill>(arg(index++).cast<int>());
  CUresult error = cuTensorMapEncodeTiled(static_cast<CUtensorMap*>(arg(0).cast<void*>()),
      cuda_dtype, rank, arg(3).cast<void*>(), shape, strides, box, element_strides,
      interleave, swizzle, promotion, fill);
  if (error != CUDA_SUCCESS) {
    const char* message = "cuTensorMapEncodeTiled failed";
    cuGetErrorString(error, &message);
    TVMFFIErrorSetRaisedFromCStr("CUDAError", message);
    return -1;
  }
  TVM_FFI_SAFE_CALL_END();
}
)";
  }

  std::string check_error_;
  std::string encode_tensormap_;
  ffi::Array<ffi::String> function_names_;
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
  for (const auto& [gvar, func] : functions) cg.AddFunction(gvar, func);
  return CSourceModuleCreate(cg.Finish(), "cu", cg.GetFunctionNames());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("target.build.cuda_host", BuildCUDAHost);
}

}  // namespace codegen
}  // namespace tvm
