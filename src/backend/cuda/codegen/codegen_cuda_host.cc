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
#include <tvm/tirx/op.h>
#include <tvm/tirx/type.h>

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <unordered_map>
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
    // Host C types spell some device types differently (bfloat16 is uint16_t on
    // the host), so convert each argument to the kernel's own parameter type.
    launch_ = name_supply_->FreshName("tvm_cuda_host_launch");
    decl_stream << "template <typename... Params, typename... Args>\n"
                << "static cudaError_t " << launch_
                << "(void (*kernel)(Params...), const cudaLaunchConfig_t* config, "
                << "Args... args) {\n"
                << "  static_assert(sizeof...(Params) == sizeof...(Args), "
                << "\"kernel argument count mismatch\");\n"
                << "  return cudaLaunchKernelEx(config, kernel, ((Params)args)...);\n}\n";
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
    if (!func->body.has_value()) {
      DeclareFunction(gvar, func);
      return;
    }
    // A guard destructor may report a CUDA error on any return path. Keep it
    // inside the FFI exception boundary along with device selection and setup.
    InitFuncState(func);
    PrintFunctionSignature(GetFunctionName(gvar), func, stream);
    stream << " {\n  TVM_FFI_SAFE_CALL_BEGIN();\n  try {\n";
    int scope = BeginScope();
    PrintStmt(func->body.value());
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

  void Dispatch_(const BindNode* op) override {
    // C accepts the implicit void* conversion emitted for descriptor handles;
    // the CUDA translation unit is C++, which requires an explicit cast.
    auto* ptr = op->var->ty.as<PointerTypeNode>();
    if (print_ssa_form_ || !ptr || !ptr->element_type.as<tirx::TensorMapTypeNode>()) {
      CodeGenC::Dispatch_(op);
      return;
    }
    std::string value = PrintExpr(op->value);
    PrintIndent();
    stream << "CUtensorMap* " << AllocVarID(op->var.get()) << " = (CUtensorMap*)(" << value
           << ");\n";
  }

  void Dispatch_(const CallNode* op, std::ostream& os) override {
    if (op->op.same_as(tirx::tvm_stack_alloca_op()) &&
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
    if (op->op.same_as(tirx::tvm_call_packed_lowered_op())) {
      const auto& name = op->args[0].as_or_throw<StringImm>()->value;
      if (name == "runtime.cuTensorMapEncodeTiled" || name == "runtime.cuTensorMapInit") {
        TVM_FFI_THROW(ValueError)
            << "cuda_host requires tensormap_encode_tiled instead of a packed tensor-map encoder";
      }
      if (name == "__tvm_set_device") {
        std::string args =
            "((TVMFFIAny*)" + PrintExpr(op->args[1]) + " + " + PrintExpr(op->args[2]) + ")";
        std::string guard = name_supply_->FreshName("cuda_device_guard");
        PrintIndent();
        stream << "if (" << args << "[0].v_int64 != kDLCUDA) {\n"
               << "  TVMFFIErrorSetRaisedFromCStr(\"ValueError\", "
               << "\"cuda_host requires a CUDA device\");\n  return -1;\n}\n";
        PrintIndent();
        stream << "tvm::ffi::CUDADeviceGuard " << guard << "(" << args << "[1].v_int64);\n";
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
    if (op->op.same_as(tirx::call_extern_op()) || op->op.same_as(tirx::call_pure_extern_op())) {
      const auto& symbol = op->args[0].as_or_throw<StringImm>()->value;
      TVM_FFI_CHECK(std::string(symbol).rfind("TVMBackend", 0) != 0, ValueError)
          << "cuda_host does not provide TVM runtime operation: " << symbol;
    }
    if (op->op.same_as(tirx::tensormap_encode_tiled_op())) {
      PrintTensorMapEncode(op);
      os << "0";
      return;
    }
    if (!op->op.same_as(tirx::call_ffi_kernel_op())) {
      CodeGenCHost::Dispatch_(op, os);
      return;
    }
    const auto* attr = op->attrs.as<tirx::CallFFIKernelAttr>();
    TVM_FFI_CHECK(attr, ValueError) << "cuda_host kernel calls require CallFFIKernelAttr";
    TVM_FFI_CHECK(!op->args.empty() && op->args[0].as<StringImmNode>(), ValueError)
        << "cuda_host kernel calls require a string kernel symbol";
    const auto& symbol = op->args[0].as<StringImmNode>()->value;

    // Match LaunchParamConfig's layout: grid, block, cluster and preferred cluster.
    std::array<int, 12> axes;
    axes.fill(-1);
    int shared_memory = -1;
    size_t num_launch_values = 0;
    bool use_programmatic_dependent_launch = false;
    bool use_cooperative_launch = false;
    std::unordered_set<std::string> seen;
    for (size_t i = 0; i < attr->launch_params.size(); ++i) {
      std::string tag = attr->launch_params[i];
      TVM_FFI_CHECK(seen.insert(tag).second, ValueError)
          << "cuda_host duplicate launch parameter: " << tag;
      // These are flags, not values in the argument suffix.  Reject unsupported
      // launch semantics before decoding any operands, rather than dropping them.
      if (tag == runtime::launch_param::kUseProgramaticDependentLaunch) {
        use_programmatic_dependent_launch = true;
      } else if (tag == runtime::launch_param::kUseCooperativeLaunch) {
        use_cooperative_launch = true;
      } else if (tag == runtime::launch_param::kUseRequiredBlockDimension) {
        TVM_FFI_THROW(ValueError) << "cuda_host does not support launch flag: " << tag;
      } else if (tag == runtime::launch_param::kUseDynamicSharedMemoryTag) {
        TVM_FFI_CHECK_EQ(i + 1, attr->launch_params.size(), ValueError)
            << "cuda_host dynamic shared memory must be the last launch parameter";
        shared_memory = static_cast<int>(num_launch_values++);
      } else {
        static const char* kScopes[] = {"blockIdx.", "threadIdx.", "clusterCtaIdx.",
                                        "preferredClusterCtaIdx."};
        int axis = -1;
        for (int rank = 0; rank < 4; ++rank) {
          for (int j = 0; j < 3; ++j) {
            if (tag == std::string(kScopes[rank]) + "xyz"[j]) axis = rank * 3 + j;
          }
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
    std::array<std::string, 12> dimensions;
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
    auto has_axis = [&](int rank) {
      return axes[rank * 3] >= 0 || axes[rank * 3 + 1] >= 0 || axes[rank * 3 + 2] >= 0;
    };
    bool use_cluster = has_axis(2);
    bool use_preferred_cluster = has_axis(3);
    if (use_cluster) {
      CheckError("cudaFuncSetAttribute(::" + std::string(symbol) +
                 ", cudaFuncAttributeNonPortableClusterSizeAllowed, 1)");
    }
    std::string config = name_supply_->FreshName("cuda_config");
    PrintIndent();
    stream << "cudaLaunchConfig_t " << config << " = {};\n";
    PrintIndent();
    stream << config << ".gridDim = dim3(" << dimensions[0] << ", " << dimensions[1] << ", "
           << dimensions[2] << ");\n";
    PrintIndent();
    stream << config << ".blockDim = dim3(" << dimensions[3] << ", " << dimensions[4] << ", "
           << dimensions[5] << ");\n";
    PrintIndent();
    stream << config << ".dynamicSmemBytes = " << bytes << ";\n";
    PrintIndent();
    stream << config << ".stream = " << cuda_stream << ";\n";
    int max_attrs = use_cluster + use_preferred_cluster + use_programmatic_dependent_launch +
                    use_cooperative_launch;
    if (max_attrs > 0) {
      std::string attrs = name_supply_->FreshName("cuda_attrs");
      PrintIndent();
      stream << "cudaLaunchAttribute " << attrs << "[" << max_attrs << "] = {};\n";
      PrintIndent();
      stream << config << ".attrs = " << attrs << ";\n";
      auto set_dimension = [&](const std::string& id, const std::string& field, int rank) {
        PrintIndent();
        stream << attrs << "[" << config << ".numAttrs].id = " << id << ";\n";
        for (int j = 0; j < 3; ++j) {
          PrintIndent();
          stream << attrs << "[" << config << ".numAttrs].val." << field << "." << "xyz"[j] << " = "
                 << dimensions[rank * 3 + j] << ";\n";
        }
        PrintIndent();
        stream << "++" << config << ".numAttrs;\n";
      };
      if (use_cluster) {
        set_dimension("cudaLaunchAttributeClusterDimension", "clusterDim", 2);
      }
      if (use_preferred_cluster) {
        // Match the CUDA runtime module: a unit preferred cluster adds no attribute.
        PrintIndent();
        stream << "if (" << dimensions[9] << " != 1 || " << dimensions[10] << " != 1 || "
               << dimensions[11] << " != 1) {\n";
        int scope = BeginScope();
        set_dimension("cudaLaunchAttributePreferredClusterDimension", "preferredClusterDim", 3);
        EndScope(scope);
        PrintIndent();
        stream << "}\n";
      }
      auto set_flag = [&](const std::string& id, const std::string& field) {
        PrintIndent();
        stream << attrs << "[" << config << ".numAttrs].id = " << id << ";\n";
        PrintIndent();
        stream << attrs << "[" << config << ".numAttrs++].val." << field << " = 1;\n";
      };
      if (use_programmatic_dependent_launch) {
        set_flag("cudaLaunchAttributeProgrammaticStreamSerialization",
                 "programmaticStreamSerializationAllowed");
      }
      if (use_cooperative_launch) {
        set_flag("cudaLaunchAttributeCooperative", "cooperative");
      }
    }
    std::string launch = launch_ + "(::" + std::string(symbol) + ", &" + config;
    for (size_t i = 0; i + 1 < launch_begin; ++i) launch += ", " + arguments[i];
    CheckError(launch + ")", true);
    os << "0";
  }

 private:
  void CheckError(const std::string& call, bool allow_unloading = false) {
    PrintIndent();
    stream << "if (::" << check_error_ << "(" << call;
    if (allow_unloading) stream << ", true";
    stream << ") != 0) return -1;\n";
  }
  void PrintTensorMapEncode(const CallNode* op) {
    const auto* attr = op->attrs.as<tirx::TensorMapEncodeTiledAttr>();
    TVM_FFI_CHECK(attr && attr->rank >= 1 && attr->rank <= 5 &&
                      op->args.size() == static_cast<size_t>(4 * attr->rank + 1),
                  ValueError)
        << "Invalid tensormap_encode_tiled attributes or operands";
    static const std::unordered_map<std::string, std::string> dtype_names{
        {"int8", "UINT8"},        {"uint8", "UINT8"},
        {"uint16", "UINT16"},     {"uint32", "UINT32"},
        {"uint64", "UINT64"},     {"int32", "INT32"},
        {"int64", "INT64"},       {"float16", "FLOAT16"},
        {"float32", "FLOAT32"},   {"float64", "FLOAT64"},
        {"bfloat16", "BFLOAT16"}, {"float8_e4m3fn", "UINT8"},
        {"float8_e5m2", "UINT8"}, {"float4_e2m1fn", "16U4_ALIGN16B"}};
    auto dtype = dtype_names.find(ffi::DLDataTypeToString(attr->descriptor_dtype));
    TVM_FFI_CHECK(dtype != dtype_names.end(), ValueError)
        << "Unsupported cuda_host tensor-map descriptor dtype: "
        << ffi::DLDataTypeToString(attr->descriptor_dtype);
    std::string cuda_dtype = "CU_TENSOR_MAP_DATA_TYPE_" + dtype->second;
    if (attr->force_cu_dtype != -1) {
      TVM_FFI_CHECK(attr->force_cu_dtype == 11 && attr->descriptor_dtype.code == kDLFloat &&
                        attr->descriptor_dtype.bits == 32 && attr->descriptor_dtype.lanes == 1,
                    ValueError)
          << "cuda_host only supports a TFLOAT32 tensor-map dtype override";
      cuda_dtype = "CU_TENSOR_MAP_DATA_TYPE_TFLOAT32";
    }
    for (int64_t value : {attr->interleave, attr->swizzle, attr->l2_promotion, attr->oob_fill}) {
      TVM_FFI_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(), ValueError)
          << "cuda_host tensor-map options must fit a nonnegative CUDA enum";
    }
    // Preserve operand evaluation order and reject narrowing that could turn an
    // invalid dynamic dimension or stride into a valid but different CUDA input.
    std::vector<std::string> values;
    for (size_t i = 0; i < op->args.size(); ++i) {
      std::string value = PrintExpr(op->args[i]);
      std::string name = name_supply_->FreshName("tensormap_arg");
      PrintIndent();
      stream << "auto " << name << " = " << value << ";\n";
      values.push_back(name);
    }
    for (size_t i = 2; i < op->args.size(); ++i) {
      const std::string& name = values[i];
      auto type = op->args[i]->ty.as<PrimType>();
      TVM_FFI_CHECK(type && type.value().IsScalar() && type.value().bits() <= 64 &&
                        type.value().MatchesCode(kDLInt, kDLUInt),
                    ValueError)
          << "cuda_host tensor-map dimensions and strides must be scalar integers";
      if (type.value().MatchesCode(kDLInt)) {
        PrintIndent();
        stream << "TVM_FFI_CHECK(" << name << " >= 0, ValueError) "
               << "<< \"Negative tensor-map dimension or stride\";\n";
      }
      if (i >= static_cast<size_t>(2 * attr->rank + 1)) {
        PrintIndent();
        stream << "TVM_FFI_CHECK(static_cast<uint64_t>(" << name
               << ") <= 4294967295ULL, ValueError) "
               << "<< \"Tensor-map dimension or stride exceeds uint32\";\n";
      }
    }
    size_t index = 2;
    auto array = [&](const char* type, int64_t count) {
      std::string name = name_supply_->FreshName("tensormap_values");
      PrintIndent();
      stream << type << " " << name << "[" << std::max<int64_t>(count, 1) << "] = {";
      for (int64_t i = 0; i < count; ++i) {
        if (i) stream << ", ";
        stream << "static_cast<" << type << ">(" << values[index++] << ")";
      }
      stream << "};\n";
      return name;
    };
    std::string shape = array("cuuint64_t", attr->rank);
    std::string strides = array("cuuint64_t", attr->rank - 1);
    std::string box = array("cuuint32_t", attr->rank);
    std::string element_strides = array("cuuint32_t", attr->rank);
    std::string error = name_supply_->FreshName("tensormap_error");
    PrintIndent();
    stream << "CUresult " << error << " = cuTensorMapEncodeTiled(static_cast<CUtensorMap*>("
           << values[0] << "), " << cuda_dtype << ", " << attr->rank << ", " << values[1] << ", "
           << shape << ", " << strides << ", " << box << ", " << element_strides
           << ", static_cast<CUtensorMapInterleave>(" << attr->interleave
           << "), static_cast<CUtensorMapSwizzle>(" << attr->swizzle
           << "), static_cast<CUtensorMapL2promotion>(" << attr->l2_promotion
           << "), static_cast<CUtensorMapFloatOOBfill>(" << attr->oob_fill << "));\n";
    PrintIndent();
    stream << "if (" << error << " != CUDA_SUCCESS) {\n"
           << "  const char* message = \"cuTensorMapEncodeTiled failed\";\n"
           << "  cuGetErrorString(" << error << ", &message);\n"
           << "  TVMFFIErrorSetRaisedFromCStr(\"CUDAError\", message);\n"
           << "  return -1;\n}\n";
  }

  std::string check_error_;
  std::string launch_;
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
