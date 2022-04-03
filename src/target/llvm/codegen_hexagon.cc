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

#include <llvm/Bitcode/BitcodeWriter.h>
#if TVM_LLVM_VERSION <= 90
#include <llvm/IR/Intrinsics.h>
#else
#include <llvm/IR/IntrinsicsHexagon.h>
#endif
#include <llvm/Support/CommandLine.h>
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

namespace tvm {
namespace codegen {

static std::string get_name(const PrimFunc& f) {
  auto global_symbol = f->GetAttr<runtime::String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenLLVM: Expect PrimFunc to have the global_symbol attribute";
  return std::string(global_symbol.value());
}

// Hexagon code generation
class CodeGenHexagon final : public CodeGenCPU {
 public:
 void Init(const std::string& module_name, llvm::TargetMachine* tm, llvm::LLVMContext* ctx,
            bool system_lib, bool dynamic_lookup, bool target_c_runtime) override;
  void InitTarget(llvm::TargetMachine* tm) final;

  llvm::Value* CreateIntrinsic(const CallNode* op) override;
  llvm::Value* CreateCallExtern(Type ret_type, String global_symbol, const Array<PrimExpr>& args,
                                bool skip_first_arg) override;
  llvm::Module* GetModulePtr() const { return module_.get(); }

 protected:
  void CreatePrintf(const std::string& format, llvm::ArrayRef<llvm::Value*> format_args) final;

 private:
  TypedPointer CreateBufferPtr(llvm::Value* buffer_ptr, DataType buffer_element_dtype,
                               llvm::ArrayRef<llvm::Value*> indices, DataType value_dtype) final;
  TypedPointer CreateStructRefPtr(DataType t, llvm::Value* buf, llvm::Value* index, int kind);

  llvm::GlobalVariable* InitContextPtr(llvm::Type* type, std::string name);
  llvm::Value* GetContextPtr(llvm::GlobalVariable* gv);
  std::vector<std::pair<std::string, llvm::Value*>> export_system_symbols_;

  // global to packed function handle
  std::unordered_map<std::string, llvm::GlobalVariable*> func_handle_map_;

  // create call into tvm packed function.
  llvm::Value* CreateCallPacked(const CallNode* op);

  std::map<std::string, llvm::Type*> types_for_alloca_;
};

void CodeGenHexagon::Init(const std::string& module_name, llvm::TargetMachine* tm, llvm::LLVMContext* ctx,
            bool system_lib, bool dynamic_lookup, bool target_c_runtime) {
  CodeGenCPU::Init(module_name, tm, ctx, system_lib, dynamic_lookup, target_c_runtime);

  types_for_alloca_ = {
    {"shape", t_tvm_shape_index_},
    {"arg_value", t_tvm_value_},
    {"arg_tcode", t_int_},
    {"array", t_tvm_array_},
  };
}
void CodeGenHexagon::InitTarget(llvm::TargetMachine* tm) {
  native_vector_bits_ = 64;  // Assume "scalar" vectors at first.
  llvm::StringRef fs = tm->getTargetFeatureString();
  size_t npos = llvm::StringRef::npos;
  const auto hvx_length_feature = "+hvx-length";  // +hvx-length{64|128}b
  size_t len_begin = fs.find(hvx_length_feature);
  size_t len_end = len_begin != npos ? fs.find('b', len_begin) : npos;
  if (len_end != npos) {
    int hvx_bytes = 0;
    len_begin += std::strlen(hvx_length_feature);
    ICHECK(!fs.substr(len_begin, len_end - len_begin).getAsInteger(10, hvx_bytes))
        << "invalid HVX length in feature string: " << fs.str();
    ICHECK(hvx_bytes == 64 || hvx_bytes == 128)
        << "invalid HVX vector length: " << hvx_bytes << ", should be 64 or 128";
    native_vector_bits_ = hvx_bytes * 8;
  }
  CodeGenLLVM::InitTarget(tm);
}

llvm::Value* CodeGenHexagon::CreateCallExtern(Type ret_type, String global_symbol,
                                              const Array<PrimExpr>& args, bool skip_first_arg) {
  std::vector<llvm::Value*> arg_values;
  for (size_t i = skip_first_arg; i < args.size(); ++i) {
    arg_values.push_back(MakeValue(args[i]));
  }
  std::vector<llvm::Type*> arg_types;
  for (llvm::Value* v : arg_values) {
    arg_types.push_back(v->getType());
  }
  llvm::FunctionType* ftype = llvm::FunctionType::get(GetLLVMType(ret_type), arg_types, false);
  // Check if it is available in global function table as injected function.
  auto it = gv_func_map_.find(global_symbol);
  if (it != gv_func_map_.end()) {
    if (it->second == nullptr) {
      gv_func_map_[global_symbol] = InitContextPtr(ftype->getPointerTo(), "__" + global_symbol);
      it = gv_func_map_.find(global_symbol);
    }
#if TVM_LLVM_VERSION >= 90
    auto ext_callee = llvm::FunctionCallee(ftype, GetContextPtr(it->second));
#else
    auto ext_callee = GetContextPtr(it->second);
#endif
    return builder_->CreateCall(ext_callee, arg_values);
  } else {
    llvm::Function* f = module_->getFunction(global_symbol);
    if (f == nullptr) {
      f = llvm::Function::Create(ftype, llvm::Function::ExternalLinkage,
                                 global_symbol.operator llvm::StringRef(), module_.get());
    }
#if TVM_LLVM_VERSION >= 90
    auto ext_callee = llvm::FunctionCallee(f);
#else
    auto ext_callee = f;
#endif
    return builder_->CreateCall(ext_callee, arg_values);
  }
}

llvm::GlobalVariable* CodeGenHexagon::InitContextPtr(llvm::Type* p_type, std::string name) {
  llvm::GlobalVariable* gv = new llvm::GlobalVariable(
      *module_, p_type, false, llvm::GlobalValue::LinkOnceAnyLinkage, nullptr, name);
#if TVM_LLVM_VERSION >= 100
  gv->setAlignment(llvm::Align(data_layout_->getTypeAllocSize(p_type)));
#else
  gv->setAlignment(data_layout_->getTypeAllocSize(p_type));
#endif
  gv->setInitializer(llvm::Constant::getNullValue(p_type));
  gv->setDLLStorageClass(llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);
  return gv;
}

llvm::Value* CodeGenHexagon::GetContextPtr(llvm::GlobalVariable* gv) {
  ICHECK(gv != nullptr);
#if TVM_LLVM_VERSION >= 110
  llvm::LoadInst* faddr =
      builder_->CreateAlignedLoad(gv->getValueType(), gv, llvm::Align(gv->getAlignment()));
#elif TVM_LLVM_VERSION >= 80
  llvm::LoadInst* faddr = builder_->CreateAlignedLoad(gv->getValueType(), gv, gv->getAlignment());
#else
  llvm::LoadInst* faddr = builder_->CreateAlignedLoad(gv, gv->getAlignment());
#endif
  faddr->setMetadata("tbaa",
                     md_builder_->createTBAAStructTagNode(md_tbaa_ctx_ptr_, md_tbaa_ctx_ptr_, 0));
  return faddr;
}

llvm::Value* CodeGenHexagon::CreateCallPacked(const CallNode* op) {
  // There is always a call to __tvm_set_device in a standalone op,
  // and we can't have calls to packed functions, because they need
  // a Module object to work (or at least TVMBackendGetFuncFromEnv
  // function).
  const std::string& name = op->args[0].as<StringImmNode>()->value;
  if (name == "__tvm_set_device") {
    return ConstInt32(0);
  }

  return CodeGenCPU::CreateCallPacked(op);
}

llvm::Value* CodeGenHexagon::CreateIntrinsic(const CallNode* op) {
  if (op->op.same_as(builtin::tvm_call_packed_lowered())) {
    return CreateCallPacked(op);
  } else if (op->op.same_as(builtin::tvm_call_trace_packed_lowered())) {
    return CreateCallTracePacked(op);
  } else if (op->op.same_as(builtin::tvm_struct_get())) {
    ICHECK_EQ(op->args.size(), 3);
    int kind = op->args[2].as<IntImmNode>()->value;
    TypedPointer ref =
        CreateStructRefPtr(op->dtype, MakeValue(op->args[0]), MakeValue(op->args[1]), kind);
    if (kind == builtin::kArrAddr) {
      return builder_->CreatePointerCast(ref.addr, t_void_p_);
    }
    return builder_->CreateLoad(ref.type, ref.addr);
  } else if (op->op.same_as(builtin::tvm_struct_set())) {
    ICHECK_EQ(op->args.size(), 4);
    int kind = op->args[2].as<IntImmNode>()->value;
    ICHECK(kind != builtin::kArrAddr);
    TypedPointer ref = CreateStructRefPtr(op->args[3].dtype(), MakeValue(op->args[0]),
                                          MakeValue(op->args[1]), kind);
    llvm::Value* value = MakeValue(op->args[3]);
    if (value->getType()->isPointerTy()) {
      value = builder_->CreatePointerCast(value, ref.type);
    }
    builder_->CreateStore(value, ref.addr);
    return ConstInt32(0);
  } else if (op->op.same_as(builtin::tvm_stack_alloca())) {
    ICHECK_EQ(op->args.size(), 2);
    const std::string& name = op->args[0].as<StringImmNode>()->value;
    llvm::Value* size = ConstInt32(op->args[1].as<IntImmNode>()->value);
    return builder_->CreateAlloca(types_for_alloca_.at(name), size);
  } else if (op->op.same_as(builtin::tvm_throw_last_error())) {
    llvm::Value* neg_1 = ConstInt32(-1);
    builder_->CreateRet(neg_1);
    auto next_block = std::next(builder_->GetInsertBlock()->getIterator());
    llvm::BasicBlock* new_bb = llvm::BasicBlock::Create(*ctx_, "cont", function_, &*next_block);
    builder_->SetInsertPoint(new_bb);
    return neg_1;
  }

  return CodeGenLLVM::CreateIntrinsic(op);
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
    return CodeGenLLVM::CreateBufferPtr(buffer_ptr, buffer_element_dtype, indices, value_dtype);
  }

  ICHECK_EQ(indices.size(), 2) << "CodegenHexagon supports 1-d and 2-d physical buffers, received "
                               << indices.size() << "-d buffer indices";

  // Use the first index to identify the pointer.
  DataType dtype_void_ptr = DataType::Handle();
  CodeGenLLVM::TypedPointer buffer_chunk_ptr_ptr =
      CodeGenLLVM::CreateBufferPtr(buffer_ptr, dtype_void_ptr, {indices[0]}, dtype_void_ptr);
  llvm::Value* buffer_chunk_ptr =
      builder_->CreateLoad(buffer_chunk_ptr_ptr.type, buffer_chunk_ptr_ptr.addr);

  // Then delegate the CodeGenLLVM to find the value from the second
  // index.
  return CodeGenLLVM::CreateBufferPtr(buffer_chunk_ptr, buffer_element_dtype, {indices[1]},
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

namespace {
// Check if the function matches the TVMBackendPackedCFunc prototype.
bool UsesExportABI(const PrimFunc& f) {
  if (f->attrs.defined()) {
    auto it = f->attrs->dict.find("calling_conv");
    return it != f->attrs->dict.end() &&
           Downcast<Integer>((*it).second) == CallingConv::kCPackedFunc;
  }
  return false;
}

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
  // Make sure all targets are registered. InitializeLLVM can be called
  // multiple times, after the first call all subsequent calls are no-ops.
  InitializeLLVM();

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

  std::unique_ptr<llvm::TargetMachine> tm = GetLLVMTargetMachine(target);
  std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext());
  std::unique_ptr<CodeGenHexagon> cg(new CodeGenHexagon());

  std::vector<PrimFunc> funcs;
  std::string entry_func;
  Map<String, LinkedParam> linked_params;
  bool could_have_linked_params = mod->ShouldLinkParameters();

  for (auto kv : mod->functions) {
    if (could_have_linked_params &&
        kv.first->name_hint == ::tvm::runtime::symbol::tvm_lookup_linked_param) {
      // If `f` is the linked-params function, extract the parameters from the
      // attribute dictionary, and skip the codegen.
      auto attrs_dict = Downcast<Map<String, ObjectRef>>(kv.second->attrs->dict);
      CHECK(attrs_dict.find(::tvm::tir::attr::kLinkedParams) != attrs_dict.end())
          << "no " << ::tvm::tir::attr::kLinkedParams << " attribute found!";

      CHECK(linked_params.empty()) << "Multiple linked-param functions";
      linked_params =
          Downcast<Map<String, LinkedParam>>(attrs_dict[::tvm::tir::attr::kLinkedParams]);
      continue;
    }
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
    funcs.emplace_back(f);
  }

  cg->Init("TVMHexagonModule", tm.get(), ctx.get(), false, false, false);
  cg->AddFunctionsOrdered(funcs.begin(), funcs.end());
  if (entry_func.length() != 0) {
    cg->AddMainFunction(entry_func);
  }

  if (!linked_params.empty()) {
    cg->LinkParameters(linked_params);
  }

  // Uncomment to get the LLVM module right out of codegen, before optimizations.
  // std::cerr << "HexagonModule.0 {\n" << *cg->GetModulePtr() << "}\n";
  std::unique_ptr<llvm::Module> module = cg->Finish();

  enum CodeGenFileType { Asm, Obj, IR, BC };

  auto EmitToString = [&tm](const llvm::Module& m, CodeGenFileType cgft) {
    std::string out;

    if (cgft == IR || cgft == BC) {
      llvm::raw_string_ostream os(out);
      if (cgft == IR)
        m.print(os, nullptr);
      else
        llvm::WriteBitcodeToFile(m, os);
    } else if (cgft == Asm || cgft == Obj) {
      using namespace llvm;
#if TVM_LLVM_VERSION <= 90
      auto ft = cgft == Asm ? TargetMachine::CodeGenFileType::CGFT_AssemblyFile
                            : TargetMachine::CodeGenFileType::CGFT_ObjectFile;
#else
      auto ft = cgft == Asm ? llvm::CGFT_AssemblyFile : llvm::CGFT_ObjectFile;
#endif

      SmallString<16384> ss;  // Will grow on demand.
      llvm::raw_svector_ostream os(ss);
      std::unique_ptr<llvm::Module> cm = CloneModule(m);
      legacy::PassManager pass;
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
    llvm::StringRef mcpu = Downcast<String>(target->attrs.at("mcpu"));
    ICHECK(mcpu.startswith("hexagon")) << "unexpected -mcpu value in target:" << mcpu.str();
    extra_args.Set("hex_arch", mcpu.drop_front(strlen("hexagon")).str());
  }
  int rc = (*f)(so_name, o_names, extra_args);
  ICHECK(rc == 0) << "Failed to link " << so_name;

  // Move it to ExtractFuncInfo?
  std::set<std::string> export_abi;
  for (auto kv : mod->functions) {
    auto f = Downcast<PrimFunc>(kv.second);
    if (UsesExportABI(f)) export_abi.insert(get_name(f));
  }
  return HexagonModuleCreate(so_name, "so", ExtractFuncInfo(mod), asm_str, obj_str, ir_str, bc_str,
                             export_abi);
}

TVM_REGISTER_GLOBAL("target.build.hexagon").set_body_typed(BuildHexagon);

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
