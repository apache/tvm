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

#if defined(TVM_LLVM_VERSION) && TVM_LLVM_VERSION >= 70

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#if TVM_LLVM_VERSION >= 100
#include <llvm/IR/IntrinsicsHexagon.h>
#endif
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#if TVM_LLVM_VERSION >= 100
#include <llvm/Support/Alignment.h>
#endif
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <tvm/runtime/module.h>
#include <tvm/target/codegen.h>
#include <tvm/tir/analysis.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../runtime/hexagon/hexagon_module.h"
#include "../build_common.h"
#include "codegen_cpu.h"
#include "llvm_instance.h"

namespace tvm {
namespace codegen {

// Hexagon code generation
class CodeGenHexagon final : public CodeGenCPU {
 public:
  void Init(const std::string& module_name, LLVMTarget* llvm_target,
            Optional<String> system_lib_prefix, bool dynamic_lookup,
            bool target_c_runtime) override;
  void InitTarget() final;

  using CodeGenCPU::VisitStmt_;
  llvm::Value* VisitExpr_(const BufferLoadNode* op) override;
  llvm::Value* CreateIntrinsic(const CallNode* op) override;

  llvm::Value* CreateCallExtern(Type ret_type, String global_symbol, const Array<PrimExpr>& args,
                                bool skip_first_arg) override;
  llvm::Value* CreateCallExternQHL(Type ret_type, String global_symbol, const Array<PrimExpr>& args,
                                   bool skip_first_arg);

  llvm::Module* GetModulePtr() const { return module_.get(); }

  uint64_t GetTypeSizeInBits(llvm::Type* type) const {
#if TVM_LLVM_VERSION >= 160
    return data_layout_->getTypeSizeInBits(type).getFixedValue();
#elif TVM_LLVM_VERSION >= 100
    return data_layout_->getTypeSizeInBits(type).getFixedSize();
#else
    return data_layout_->getTypeSizeInBits(type);
#endif
  }

 protected:
  void CreatePrintf(const std::string& format, llvm::ArrayRef<llvm::Value*> format_args) final;

 private:
  TypedPointer CreateBufferPtr(llvm::Value* buffer_ptr, DataType buffer_element_dtype,
                               llvm::ArrayRef<llvm::Value*> indices, DataType value_dtype) final;
  TypedPointer CreateStructRefPtr(DataType t, llvm::Value* buf, llvm::Value* index, int kind);

  bool IsQHLFunction(const std::string& func);

  llvm::Value* VectorLookupLoad(Buffer buffer, DataType buffer_type, Array<PrimExpr> indices);
  llvm::Value* Intrinsic(llvm::Intrinsic::ID, llvm::ArrayRef<llvm::Value*> args);
  std::vector<std::string> fqhl_list_ = {
      "tvm_vect_qhmath_hvx_cos_ahf",     "tvm_vect_qhmath_hvx_tanh_ahf",
      "tvm_vect_qhmath_hvx_sigmoid_ahf", "tvm_vect_qhmath_hvx_sin_ahf",
      "tvm_vect_qhmath_hvx_sqrt_ahf",    "tvm_vect_qhmath_hvx_exp_ahf",
      "tvm_vect_qhmath_hvx_tan_ahf",     "tvm_vect_qhmath_hvx_floor_ahf",
      "tvm_vect_qhmath_hvx_ceil_ahf",    "tvm_vect_qhmath_hvx_pow_ahf"};
};

void CodeGenHexagon::Init(const std::string& module_name, LLVMTarget* llvm_target,
                          Optional<String> system_lib_prefix, bool dynamic_lookup,
                          bool target_c_runtime) {
  CodeGenCPU::Init(module_name, llvm_target, system_lib_prefix, dynamic_lookup, target_c_runtime);
}

void CodeGenHexagon::InitTarget() {
  native_vector_bits_ = 64;                       // Assume "scalar" vectors at first.
  const auto hvx_length_feature = "+hvx-length";  // +hvx-length{64|128}b
  for (const std::string& f : llvm_target_->GetTargetFeatures()) {
    llvm::StringRef fs(f);
    if (!fs.startswith(hvx_length_feature)) continue;

    ICHECK(fs.endswith("b")) << "malformed target feature: " << f;
    int hvx_bytes = 0;
    size_t len_begin = std::strlen(hvx_length_feature);
    ICHECK(!fs.substr(len_begin, fs.size() - len_begin - 1).getAsInteger(10, hvx_bytes))
        << "invalid HVX length in feature string: " << f;
    ICHECK(hvx_bytes == 64 || hvx_bytes == 128)
        << "invalid HVX vector length: " << hvx_bytes << ", should be 64 or 128";
    native_vector_bits_ = hvx_bytes * 8;
    // There should only be one hvx-length...
    break;
  }
  CodeGenCPU::InitTarget();
}

llvm::Value* CodeGenHexagon::CreateCallExternQHL(Type ret_type, String global_symbol,
                                                 const Array<PrimExpr>& args, bool skip_first_arg) {
  int num_lanes = args[1].dtype().lanes();
  int vector_length = native_vector_bits_ / args[1].dtype().bits();
  num_lanes = ((num_lanes + vector_length - 1) / vector_length) * vector_length;
  std::vector<llvm::Value*> vect_split;
  for (int i = 0; i < num_lanes / vector_length; ++i) {
    std::vector<llvm::Value*> sub_vect_val;
    std::vector<llvm::Type*> arg_types;
    for (size_t k = skip_first_arg; k < args.size(); ++k)
      sub_vect_val.push_back(
          CodeGenCPU::CreateVecSlice(MakeValue(args[k]), i * vector_length, vector_length));
    for (llvm::Value* v : sub_vect_val) {
      arg_types.push_back(v->getType());
    }
    llvm::FunctionType* ftype = llvm::FunctionType::get(arg_types[0], arg_types, false);
    llvm::Function* f = module_->getFunction(MakeStringRef(global_symbol));
    if (f == nullptr) {
      f = llvm::Function::Create(ftype, llvm::Function::ExternalLinkage,
                                 MakeStringRef(global_symbol), module_.get());
    }
#if TVM_LLVM_VERSION >= 90
    auto ext_callee = llvm::FunctionCallee(f);
#else
    auto ext_callee = f;
#endif
    vect_split.push_back(builder_->CreateCall(ext_callee, sub_vect_val));
  }
  return CodeGenCPU::CreateVecConcat(vect_split);
}

bool CodeGenHexagon::IsQHLFunction(const std::string& func) {
  return std::find(fqhl_list_.begin(), fqhl_list_.end(), func) != fqhl_list_.end();
}

llvm::Value* CodeGenHexagon::CreateCallExtern(Type ret_type, String global_symbol,
                                              const Array<PrimExpr>& args, bool skip_first_arg) {
  int num_lanes = args[1].dtype().lanes();
  int vector_length = native_vector_bits_ / args[1].dtype().bits();
  if (IsQHLFunction(global_symbol) && (num_lanes > vector_length))
    return CreateCallExternQHL(ret_type, global_symbol, args, skip_first_arg);
  return CodeGenCPU::CreateCallExtern(ret_type, global_symbol, args, skip_first_arg);
}

llvm::Value* CodeGenHexagon::VisitExpr_(const BufferLoadNode* op) {
  if (!op->buffer.same_as(op->buffer->data)) {
    // Check if we can generate a vector lookup.
    if (!op->indices[0].as<RampNode>()) {
      if (auto* vlut = VectorLookupLoad(op->buffer, op->dtype, op->indices)) {
        return vlut;
      }
    }
  }
  return CodeGenCPU::VisitExpr_(op);
}

llvm::Value* CodeGenHexagon::CreateIntrinsic(const CallNode* op) {
#if TVM_LLVM_VERSION >= 150
  if (op->op.same_as(builtin::start_profile_intrinsic()) ||
      op->op.same_as(builtin::end_profile_intrinsic())) {
    llvm::Value* id = MakeValue(op->args[0]);
    auto instrprof_id = llvm::Intrinsic::hexagon_instrprof_custom;
    llvm::Function* func = llvm::Intrinsic::getDeclaration(module_.get(), instrprof_id);
    llvm::GlobalVariable* name_var = module_->getGlobalVariable("handler_name");
    if (!name_var) {
      llvm::StringRef init_str = "lwp_handler";
      llvm::Constant* init = llvm::ConstantDataArray::getString(module_->getContext(), init_str);

      name_var = new llvm::GlobalVariable(*module_, init->getType(), true,
                                          llvm::GlobalValue::InternalLinkage, init, "handler_name");
    }
    llvm::Type* t_int8_p_ = t_int8_->getPointerTo();
    return builder_->CreateCall(func, {llvm::ConstantExpr::getBitCast(name_var, t_int8_p_), id});
  }
#endif
  return CodeGenCPU::CreateIntrinsic(op);
}

void CodeGenHexagon::CreatePrintf(const std::string& format,
                                  llvm::ArrayRef<llvm::Value*> format_args) {
  // This function generates LLVM instructions to call HAP_debug_v2,
  // as if the FARF macro in `HAP_farf.h` were called as
  // FARF(ALWAYS, format, format_args[0], format_args[1], ...)
  std::string func_name = "HAP_debug_v2";

  llvm::Function* func = module_->getFunction(func_name);
  if (func == nullptr) {
    llvm::FunctionType* ftype = llvm::FunctionType::get(
        t_void_, {t_int32_, t_char_->getPointerTo(), t_int32_, t_char_->getPointerTo()}, true);
    func = llvm::Function::Create(ftype, llvm::Function::ExternalLinkage, func_name, module_.get());
  }

  llvm::Value* format_str = builder_->CreateGlobalStringPtr(format, "printf_format_str");

  // The value of FARF_ALWAYS_LEVEL, defined as HAP_LEVEL_HIGH
  llvm::Value* level = ConstInt32(2);

  // There is no such filename/line number for this print statement
  llvm::Value* filename = builder_->CreateGlobalStringPtr("generated-LLVM-code", "dummy_filename");
  llvm::Value* line_number = ConstInt32(1);

  std::vector<llvm::Value*> func_args = {level, filename, line_number, format_str};
  func_args.insert(func_args.end(), format_args.begin(), format_args.end());

  builder_->CreateCall(func, func_args);
}

CodeGenLLVM::TypedPointer CodeGenHexagon::CreateBufferPtr(llvm::Value* buffer_ptr,
                                                          DataType buffer_element_dtype,
                                                          llvm::ArrayRef<llvm::Value*> indices,
                                                          DataType value_dtype) {
  // Flat indices get delegated to the LLVM codegen.
  if (indices.size() == 1) {
    return CodeGenCPU::CreateBufferPtr(buffer_ptr, buffer_element_dtype, indices, value_dtype);
  }

  ICHECK_EQ(indices.size(), 2) << "CodegenHexagon supports 1-d and 2-d physical buffers, received "
                               << indices.size() << "-d buffer indices";

  // Use the first index to identify the pointer.
  DataType dtype_void_ptr = DataType::Handle();
  CodeGenLLVM::TypedPointer buffer_chunk_ptr_ptr =
      CodeGenCPU::CreateBufferPtr(buffer_ptr, dtype_void_ptr, {indices[0]}, dtype_void_ptr);
  llvm::Value* buffer_chunk_ptr =
      builder_->CreateLoad(buffer_chunk_ptr_ptr.type, buffer_chunk_ptr_ptr.addr);

  // Then delegate the CodeGenLLVM to find the value from the second
  // index.
  return CodeGenCPU::CreateBufferPtr(buffer_chunk_ptr, buffer_element_dtype, {indices[1]},
                                     value_dtype);
}

CodeGenLLVM::TypedPointer CodeGenHexagon::CreateStructRefPtr(DataType t, llvm::Value* buf,
                                                             llvm::Value* index, int kind) {
  static const std::map<int, int> field_index = {
      {builtin::kArrData, 0},      {builtin::kArrDeviceType, 1}, {builtin::kArrDeviceId, 1},
      {builtin::kArrNDim, 2},      {builtin::kArrTypeCode, 3},   {builtin::kArrTypeBits, 3},
      {builtin::kArrTypeLanes, 3}, {builtin::kArrShape, 4},      {builtin::kArrStrides, 5},
      {builtin::kArrByteOffset, 6}};
  static const std::map<int, int> subfield_index = {
      {builtin::kArrDeviceType, 0}, {builtin::kArrDeviceId, 1},  {builtin::kArrTypeCode, 0},
      {builtin::kArrTypeBits, 1},   {builtin::kArrTypeLanes, 2},
  };

  if (kind < builtin::kArrKindBound_) {
    if (buf->getType() == t_void_p_) {
      buf = builder_->CreatePointerCast(buf, t_tvm_array_->getPointerTo());
    } else {
      ICHECK_EQ(buf->getType(), t_tvm_array_->getPointerTo());
    }
    /* The following "kinds" are accessing the members of DLTensor:
       typedef struct {
         void* data;            kArrData
         DLDevice device;       kArrDeviceType (device.device_type)
                                kArrDeviceId (device.device_id)
         int ndim;              kArrNDim
         DLDataType dtype;      kArrTypeCode (dtype.code)
                                kArrTypeBits (dtype.bits)
                                kArrTypeLanes (dtype.lanes)
         int64_t* shape;        kArrShape
         int64_t* strides;      kArrStrides
         uint64_t byte_offset;  kArrByteOffset
       } DLTensor;
    */
    llvm::Value* base_gep = builder_->CreateInBoundsGEP(t_tvm_array_, buf, index, "base_gep");
    if (kind == builtin::kArrAddr) {
      return TypedPointer(t_void_p_, base_gep);
    }
    llvm::Value* field_gep = builder_->CreateInBoundsGEP(
        t_tvm_array_, base_gep, {ConstInt32(0), ConstInt32(field_index.at(kind))}, "field_gep");
    llvm::Type* field_type = t_tvm_array_->getStructElementType(field_index.at(kind));
    switch (kind) {
      // These fields have no sub-fields.
      case builtin::kArrData:
      case builtin::kArrNDim:
      case builtin::kArrShape:
      case builtin::kArrStrides:
      case builtin::kArrByteOffset:
        return TypedPointer(field_type, field_gep);
    }
    llvm::Value* subfield_gep = builder_->CreateInBoundsGEP(
        field_type, field_gep, {ConstInt32(0), ConstInt32(subfield_index.at(kind))},
        "subfield_gep");
    llvm::Type* subfield_type = field_type->getStructElementType(subfield_index.at(kind));
    return TypedPointer(subfield_type, subfield_gep);
  }

  if (kind == builtin::kTVMValueContent) {
    /* TVMValue is a union:
       typedef union {
         int64_t v_int64;
         double v_float64;
         void* v_handle;
         const char* v_str;
         TVMType v_type;
         DLDevice v_device;
       } TVMValue;
    */
    ICHECK_EQ(t.lanes(), 1);
    ICHECK(t.is_handle() || t.bits() == 64);
    if (t.is_int()) {
      buf = builder_->CreatePointerCast(buf, t_int64_->getPointerTo());
      return TypedPointer(t_int64_, builder_->CreateInBoundsGEP(t_int64_, buf, index));
    } else if (t.is_float()) {
      buf = builder_->CreatePointerCast(buf, t_float64_->getPointerTo());
      return TypedPointer(t_float64_, builder_->CreateInBoundsGEP(t_float64_, buf, index));
    } else {
      ICHECK(t.is_handle());
      buf = builder_->CreatePointerCast(buf, t_tvm_value_->getPointerTo());
      buf = builder_->CreateInBoundsGEP(t_tvm_value_, buf, index);
      return TypedPointer(t_void_p_, builder_->CreatePointerCast(buf, t_void_p_->getPointerTo()));
    }
  }

  assert(!"Unknown kind");
  return TypedPointer();
}

llvm::Value* CodeGenHexagon::Intrinsic(llvm::Intrinsic::ID IntID,
                                       llvm::ArrayRef<llvm::Value*> args) {
  llvm::Function* intf = llvm::Intrinsic::getDeclaration(module_.get(), IntID);
#if TVM_LLVM_VERSION >= 90
  auto intf_callee = llvm::FunctionCallee(intf);
#else
  auto intf_callee = intf;
#endif
  std::vector<llvm::Value*> conv_args;
  llvm::FunctionType* intf_type = intf->getFunctionType();
  ICHECK(args.size() == intf_type->getNumParams());

  for (int i = 0, e = args.size(); i != e; ++i) {
    llvm::Value* arg = args[i];
    auto* need_type = llvm::dyn_cast<llvm::VectorType>(intf_type->getParamType(i));
    auto* have_type = llvm::dyn_cast<llvm::VectorType>(arg->getType());
    if (need_type != nullptr && have_type != nullptr && need_type != have_type) {
      int need_width = GetTypeSizeInBits(need_type);
      int have_width = GetTypeSizeInBits(have_type);
      if (need_width == have_width) {
        if (need_width == native_vector_bits_ || need_width == 2 * native_vector_bits_) {
          arg = builder_->CreateBitCast(arg, need_type);
        }
      }  // TODO(joshherr-quic): add handling of v128i1 <-> v1024i1
    }
    conv_args.push_back(arg);
  }
  return builder_->CreateCall(intf_callee, conv_args);
}

llvm::Value* CodeGenHexagon::VectorLookupLoad(Buffer buffer, DataType buffer_type,
                                              Array<PrimExpr> indices) {
  PrimExpr index = indices[0];
  if (!index.dtype().is_vector()) {
    return nullptr;
  }

  if (buffer_type.bits() != 8) return nullptr;

  int table_elem_count = arith::Analyzer().Simplify(buffer->shape[0]).as<IntImmNode>()->value;
  if (table_elem_count <= 0 || table_elem_count > 256) return nullptr;

  auto int32 = DataType::Int(32);
  auto native_vector_bytes = native_vector_bits_ / 8;

  // Indexes
  llvm::Value* trunc = MakeValue(Cast(index.dtype().with_bits(8), index));
  llvm::Value* index_pad = CreateVecPad(trunc, native_vector_bytes);

  // Values
  std::vector<llvm::Value*> vloads;
  DataType table_type = buffer_type.with_lanes(table_elem_count);

  auto table_all =
      MakeValue(BufferLoad(buffer, {
                                       Ramp(IntImm(int32, 0), IntImm(int32, 1), table_elem_count),
                                   }));

  // The number of value vectors should be a power of 2.
  int table_vec_count = llvm::PowerOf2Ceil(GetVectorBytes(table_type) / native_vector_bytes);
  int table_vec_length = native_vector_bytes / buffer_type.bytes();
  for (int i = 0; i != table_vec_count; ++i) {
    // CreateVecSlice will generate undefs for elements outside the source vector.
    vloads.push_back(CreateVecSlice(table_all, i * table_vec_length, table_vec_length));
  }

#define VLO(x) Intrinsic(llvm::Intrinsic::hexagon_V6_lo_128B, {x})
#define VHI(x) Intrinsic(llvm::Intrinsic::hexagon_V6_hi_128B, {x})
#define VXOR(x, y) Intrinsic(llvm::Intrinsic::hexagon_V6_vxor_128B, {x, y})
#define VSHUFF(x) Intrinsic(llvm::Intrinsic::hexagon_V6_vshuffb_128B, {x})
#define VSPLATB(x) Intrinsic(llvm::Intrinsic::hexagon_V6_lvsplatb_128B, {x})
#define VLUT32(x, y, z) Intrinsic(llvm::Intrinsic::hexagon_V6_vlutvvbi_128B, {x, y, z})
#define VLUT32_OR(v, x, y, z) \
  Intrinsic(llvm::Intrinsic::hexagon_V6_vlutvvb_oracci_128B, {v, x, y, z})

  // Shuffle table bytes:
  // 127, 63,  126, 62,........68, 4,  67, 3,  66, 2,  65, 1,  64, 0
  std::vector<llvm::Value*> table;
  for (int i = 0; i != table_vec_count; ++i) table.push_back(VSHUFF(vloads[i]));

  // Get each 32 byte sub-table's output
  std::vector<llvm::Value*> results;
  int table_iters = table_elem_count / 32;
  for (int i = 0; i < table_iters; ++i)
    results.push_back(VLUT32(index_pad, table[i / 4], ConstInt32(i % 8)));

  // Combine outputs
  llvm::Value* result = results[0];
  for (int i = 1; i < table_iters; ++i) result = VXOR(result, results[i]);

  llvm::Type* res_type = result->getType();
  llvm::Type* ret_type = DTypeToLLVMType(buffer_type);
  if (res_type == ret_type) {
    return result;
  }

  int res_bits = GetTypeSizeInBits(res_type);
  int ret_bits = GetTypeSizeInBits(ret_type);
  ICHECK_GE(res_bits, ret_bits);
  if (ret_bits < res_bits) {
#if TVM_LLVM_VERSION >= 110
    llvm::Type* res_byte_type = llvm::VectorType::get(t_int8_, res_bits / 8, /*Scalable*/ false);
#else
    llvm::Type* res_byte_type = llvm::VectorType::get(t_int8_, res_bits / 8);
#endif
    result = CreateVecSlice(builder_->CreateBitCast(result, res_byte_type), 0, ret_bits / 8);
  }
  if (result->getType() != ret_type) {
    return builder_->CreateBitCast(result, ret_type);
  }
  return result;

#undef VLUT32_OR
#undef VLUT32
#undef VSPLATB
#undef VSHUFF
#undef VXOR
#undef VHI
#undef VLO
}

namespace {
DMLC_ATTRIBUTE_UNUSED std::ostream& operator<<(std::ostream& os, const llvm::Module& m) {
  std::string ms;
  llvm::raw_string_ostream sos(ms);
  sos << m;
  os << sos.str();
  return os;
}

void ProcessLLVMOptions(const std::vector<std::string>& llvm_vec) {
  if (llvm_vec.empty()) return;

  // LLVM options.
  std::vector<const char*> starts;
  std::transform(llvm_vec.begin(), llvm_vec.end(), std::back_inserter(starts),
                 std::mem_fn(&std::string::c_str));
  const char** args = &starts.front();

  llvm::cl::ParseCommandLineOptions(llvm_vec.size(), args);
}
}  // namespace

runtime::Module BuildHexagon(IRModule mod, Target target) {
  LLVMInstance llvm_instance;
  With<LLVMTarget> llvm_target(llvm_instance, target);

  auto split = [](const std::string& str, char delim = ' ') {
    std::vector<std::string> vec;
    std::string tmp;
    for (std::istringstream iss(str); std::getline(iss, tmp, delim);) {
      vec.push_back(tmp);
    }
    return vec;
  };
  std::string llvm_options_str = "llvm";
  if (const auto& llvm_options = target->GetAttr<Array<String>>("llvm-options")) {
    for (const String& s : llvm_options.value()) llvm_options_str += "," + s;
  }
  // Postprocess the LLVM options string: replace '@' with '=', and ',' with ' '.
  for (int i = 0, e = llvm_options_str.size(); i != e; ++i) {
    switch (llvm_options_str[i]) {
      case '@':
        llvm_options_str[i] = '=';
        break;
      case ',':
        llvm_options_str[i] = ' ';
        break;
    }
  }

  // The vector of LLVM options is treated at "argv" from "main(argc, argv)". The entry at
  // position 0 is the name of the executable, and is ignored by the LLVM cl::option parser.
  // Make sure it's set to "llvm" (tvm.target.hexagon does that).
  std::vector<std::string> llvm_options_vec = split(llvm_options_str);
  assert(llvm_options_vec.size() >= 1 && llvm_options_vec[0] == "llvm");
  llvm_options_vec.insert(std::next(llvm_options_vec.begin()),
                          {"-hexagon-small-data-threshold=0",
                           "-force-target-max-vector-interleave=1", "-hexagon-autohvx=1"});

  // Process extra command line options for LLVM. Make sure it's only
  // done once.
  static bool CallOnce = (ProcessLLVMOptions(llvm_options_vec), true);
  (void)CallOnce;

  auto cg = std::make_unique<CodeGenHexagon>();

  std::string entry_func;

  for (auto kv : mod->functions) {
    if (!kv.second->IsInstance<PrimFuncNode>()) {
      // (@jroesch): we relax constraints here, Relay functions will just be ignored.
      DLOG(INFO) << "Can only lower IR Module with PrimFuncs, but got " << kv.second->GetTypeKey();
      continue;
    }
    auto f = Downcast<PrimFunc>(kv.second);
    if (f->HasNonzeroAttr(tir::attr::kIsEntryFunc)) {
      auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
      ICHECK(global_symbol.defined());
      entry_func = global_symbol.value();
    }
  }

  cg->Init("TVMHexagonModule", llvm_target.get(), NullOpt, false, false);
  cg->AddFunctionsOrdered(mod->functions.begin(), mod->functions.end());
  if (entry_func.length() != 0) {
    cg->AddMainFunction(entry_func);
  }

  // Uncomment to get the LLVM module right out of codegen, before optimizations.
  // std::cerr << "HexagonModule.0 {\n" << *cg->GetModulePtr() << "}\n";
  std::unique_ptr<llvm::Module> module = cg->Finish();

  enum CodeGenFileType { Asm, Obj, IR, BC };

  auto EmitToString = [&llvm_target](const llvm::Module& m, CodeGenFileType cgft) {
    std::string out;

    if (cgft == IR || cgft == BC) {
      llvm::raw_string_ostream os(out);
      if (cgft == IR)
        m.print(os, nullptr);
      else
        llvm::WriteBitcodeToFile(m, os);
    } else if (cgft == Asm || cgft == Obj) {
#if TVM_LLVM_VERSION <= 90
      auto ft = cgft == Asm ? llvm::TargetMachine::CodeGenFileType::CGFT_AssemblyFile
                            : llvm::TargetMachine::CodeGenFileType::CGFT_ObjectFile;
#elif TVM_LLVM_VERSION <= 170
      auto ft = cgft == Asm ? llvm::CGFT_AssemblyFile : llvm::CGFT_ObjectFile;
#else
      auto ft =
          cgft == Asm ? llvm::CodeGenFileType::AssemblyFile : llvm::CodeGenFileType::ObjectFile;
#endif

      llvm::SmallString<16384> ss;  // Will grow on demand.
      llvm::raw_svector_ostream os(ss);
      std::unique_ptr<llvm::Module> cm = llvm::CloneModule(m);
      llvm::legacy::PassManager pass;
      llvm::TargetMachine* tm = llvm_target->GetOrCreateTargetMachine();
      ICHECK(tm->addPassesToEmitFile(pass, os, nullptr, ft) == 0) << "Cannot emit target code";
      pass.run(*cm.get());
      out.assign(ss.c_str(), ss.size());
    }

    return out;
  };

  auto SaveToFile = [](const std::string& data, const std::string& suffix) {
    llvm::SmallString<64> file_name;
    int fd;
    std::error_code ec = llvm::sys::fs::createTemporaryFile("tvm", suffix, fd, file_name);
    ICHECK_EQ(static_cast<bool>(ec), false) << ec.message();
    llvm::raw_fd_ostream file(fd, true);
    file << data;
    ICHECK(!file.has_error()) << file.error().message();
    // If there is an error, execution will never get here, but return
    // {ec, name} anyway to allow caller to handle error conditions.
    // This way the "ICHECK" above can be removed with minimal effort.
    return std::make_pair(file.error(), std::string(file_name.c_str()));
  };

  std::string asm_str = EmitToString(*module.get(), Asm);
  std::string obj_str = EmitToString(*module.get(), Obj);
  std::string ir_str = EmitToString(*module.get(), IR);
  std::string bc_str = EmitToString(*module.get(), BC);

  std::string o_name = SaveToFile(obj_str, "o").second;
  std::string so_name(o_name, 0, o_name.size() - 1);
  so_name += "so";

  const auto* f = tvm::runtime::Registry::Get("tvm.contrib.hexagon.link_shared");
  ICHECK(f != nullptr) << "tvm.contrib.hexagon.link_shared does not to exist, "
                          "do import tvm.contrib.hexagon";

  Array<PrimExpr> o_names = {StringImm(o_name)};
  Map<String, String> extra_args;
  if (target->attrs.count("mcpu")) {
    std::string mcpu = Downcast<String>(target->attrs.at("mcpu"));
    ICHECK(llvm::StringRef(mcpu).startswith("hexagon"))
        << "unexpected -mcpu value in target:" << mcpu;
    extra_args.Set("hex_arch", llvm::StringRef(mcpu).drop_front(strlen("hexagon")).str());
  }
  int rc = (*f)(so_name, o_names, extra_args);
  ICHECK(rc == 0) << "Failed to link " << so_name;

  return HexagonModuleCreate(so_name, "so", ExtractFuncInfo(mod), asm_str, obj_str, ir_str, bc_str);
}

TVM_REGISTER_GLOBAL("target.build.hexagon").set_body_typed(BuildHexagon);

TVM_REGISTER_GLOBAL("tvm.codegen.llvm.target_hexagon")
    .set_body([](const TVMArgs& targs, TVMRetValue* rv) {
      *rv = static_cast<void*>(new CodeGenHexagon());
    });

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
