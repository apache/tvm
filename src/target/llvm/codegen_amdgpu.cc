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

/*!
 * \file codegen_amdgpu.cc
 * \brief AMDGPU code generator.
 */
#ifdef TVM_LLVM_VERSION

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/registry.h>
#include "codegen_llvm.h"
#include "../build_common.h"
#include "../../runtime/rocm/rocm_module.h"

namespace tvm {
namespace codegen {

namespace {

// calls the device api to get the max threads per block
static inline int DetectROCMmaxThreadsPerBlock() {
  TVMContext tvm_ctx;
  tvm_ctx.device_type = kDLROCM;
  tvm_ctx.device_id = 0;
  tvm::runtime::DeviceAPI* api = tvm::runtime::DeviceAPI::Get(tvm_ctx, true);
  if (api != nullptr) {
    TVMRetValue val;
    api->GetAttr(tvm_ctx, tvm::runtime::kExist, &val);
    if (val.operator int() == 1) {
      tvm::runtime::DeviceAPI::Get(tvm_ctx)->
        GetAttr(tvm_ctx, tvm::runtime::kMaxThreadsPerBlock, &val);
      return val.operator int();
    }
  }
  LOG(WARNING) << "Cannot get maximum number of threads for AMD codegen";
  return 256;  // see the discussion at PR #4342 for the choice of default
}

}  // namespace

// AMDGPU code generator.
class CodeGenAMDGPU : public CodeGenLLVM {
 public:
  void AddFunction(const PrimFunc& f) final {
    // add function as void return value
    CodeGenLLVM::AddFunctionInternal(f, true);
    function_->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
    std::ostringstream attr;
    attr << "1," << DetectROCMmaxThreadsPerBlock();
    function_->addFnAttr("amdgpu-flat-work-group-size", attr.str());
  }

  void VisitStmt_(const AllocateNode* op) final {
    CHECK(!is_zero(op->condition));
    llvm::Value* buf = nullptr;

    int32_t constant_size = op->constant_allocation_size();
    CHECK_GT(constant_size, 0)
         << "Can only handle constant size stack allocation in GPU";

    StorageInfo& info = alloc_storage_info_[op->buffer_var.get()];
    if (constant_size % 4 == 0 && info.alignment == 0) {
      info.alignment = GetTempAllocaAlignment(op->dtype, constant_size);
    }
    // maximum necessary alignment in the AMD devices
    if (info.alignment > 16) {
      info.alignment = 16;
    }
    if (info.scope.rank == runtime::StorageRank::kLocal) {
      // const int local_address_space = 5;
      // TODO(tqchen): for higher version of LLVM, local address space can be set.
      llvm::AllocaInst* alloca = WithFunctionEntry([&]() {
          return builder_->CreateAlloca(
              DTypeToLLVMType(op->dtype), ConstInt32(constant_size));
        });
      if (alloca->getAlignment() < static_cast<uint32_t>(info.alignment)) {
#if TVM_LLVM_VERSION >= 100
        alloca->setAlignment(llvm::Align(info.alignment));
#else
        alloca->setAlignment(info.alignment);
#endif
      }
      buf = alloca;
    } else {
      CHECK(info.scope.rank == runtime::StorageRank::kShared)
          << "Can only allocate shared or local memory inside kernel";
      // Shared memory: address space  == 3
      const unsigned shared_address_space = 3;
      llvm::Type* type = llvm::ArrayType::get(
          DTypeToLLVMType(op->dtype), constant_size);
      // Allocate shared memory in global, address_space = 3
      llvm::GlobalVariable *global = new llvm::GlobalVariable(
          *module_, type, false, llvm::GlobalValue::PrivateLinkage, 0, ".shared",
          nullptr, llvm::GlobalValue::NotThreadLocal, shared_address_space);
#if TVM_LLVM_VERSION >= 100
      global->setAlignment(llvm::Align(info.alignment));
#else
      global->setAlignment(info.alignment);
#endif
      buf = global;
    }

    buf = builder_->CreatePointerCast(
        buf, DTypeToLLVMType(op->dtype)->getPointerTo(
            buf->getType()->getPointerAddressSpace()));
    CHECK(!var_map_.count(op->buffer_var.get()));
    var_map_[op->buffer_var.get()] = buf;
    this->VisitStmt(op->body);
  }

  // Return the thread index via intrinsics.
  llvm::Value* GetThreadIndex(const IterVar& iv) final {
    runtime::ThreadScope ts = runtime::ThreadScope::make(iv->thread_tag);
    llvm::Intrinsic::ID intrin_id = ::llvm::Intrinsic::amdgcn_workitem_id_x;
    if (ts.rank == 1) {
      switch (ts.dim_index) {
        case 0: intrin_id = ::llvm::Intrinsic::amdgcn_workitem_id_x; break;
        case 1: intrin_id = ::llvm::Intrinsic::amdgcn_workitem_id_y; break;
        case 2: intrin_id = ::llvm::Intrinsic::amdgcn_workitem_id_z; break;
        default: LOG(FATAL) << "unknown workitem idx";
      }
    } else {
      CHECK_EQ(ts.rank, 0);
      switch (ts.dim_index) {
        case 0: intrin_id = ::llvm::Intrinsic::amdgcn_workgroup_id_x; break;
        case 1: intrin_id = ::llvm::Intrinsic::amdgcn_workgroup_id_y; break;
        case 2: intrin_id = ::llvm::Intrinsic::amdgcn_workgroup_id_z; break;
        default: LOG(FATAL) << "unknown workgroup idx";
      }
    }
    llvm::Function* f = llvm::Intrinsic::getDeclaration(module_.get(), intrin_id);
    return builder_->CreateCall(f, {});
  }

  llvm::Value* CreateStorageSync(const CallNode* op) final {
    const std::string& sync = op->args[0].as<StringImmNode>()->value;
    if (sync == "warp") {
      return nullptr;
    } else if (sync == "shared") {
      llvm::Function* f = llvm::Intrinsic::getDeclaration(
          module_.get(),
          ::llvm::Intrinsic::amdgcn_s_barrier);
      return builder_->CreateCall(f, {});
    } else {
      LOG(FATAL) << "Do not support sync " << sync;
      return nullptr;
    }
  }

  void InitPassManagerBuilder(llvm::PassManagerBuilder* builder) final {
    // Additional optimization hook to tweak the builder.
  }

  unsigned GetGlobalAddressSpace() const final {
    return 1;
  }

 protected:
  void InitTarget(llvm::TargetMachine* tm) final {
    // Maximum vector lane = float4
    native_vector_bits_ = 4 * 32;
    CodeGenLLVM::InitTarget(tm);
  }
};

inline int DetectROCMComputeVersion(const std::string& target) {
  size_t pos = target.find("=gfx");
  if (pos != std::string::npos) {
    int value;
    std::stringstream is(target.substr(pos + 4));
    if (is >> value) return value;
  }
  TVMContext tvm_ctx;
  tvm_ctx.device_type = kDLROCM;
  tvm_ctx.device_id = 0;
  tvm::runtime::DeviceAPI* api = tvm::runtime::DeviceAPI::Get(tvm_ctx, true);
  if (api != nullptr) {
    TVMRetValue val;
    api->GetAttr(tvm_ctx, tvm::runtime::kExist, &val);
    if (val.operator int() == 1) {
      tvm::runtime::DeviceAPI::Get(tvm_ctx)->GetAttr(tvm_ctx, tvm::runtime::kGcnArch, &val);
      return val.operator int();
    }
  }
  LOG(WARNING) << "Cannot find -mcpu to specify rocm compute version assume gfx900";
  return 900;
}

runtime::Module BuildAMDGPU(IRModule mod, std::string target) {
#if TVM_LLVM_VERSION < 90
  LOG(FATAL) << "AMDGPU backend requires at least LLVM 9";
  // Lower versions will crash when loading the bitcode, see
  // issue #4087 for a discussion
#endif
  InitializeLLVM();
  CHECK(target.length() >= 4 &&
        target.substr(0, 4) == "rocm");
  std::ostringstream config;
  config << "-mtriple=amdgcn-amd-amdhsa-hcc -mcpu=gfx"
         << DetectROCMComputeVersion(target)
         << " -mattr=-code-object-v3 "
         << target.substr(4, target.length() - 4);
  std::unique_ptr<llvm::TargetMachine> tm = GetLLVMTargetMachine(config.str());
  std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext());
  // careful: cg will hold a naked pointer reference to ctx, so it should
  // have a shorter lifetime than the ctx.
  std::unique_ptr<CodeGenAMDGPU> cg(new CodeGenAMDGPU());

  cg->Init("TVMAMDGPUModule", tm.get(), ctx.get(), false, false);

  for (auto kv :  mod->functions) {
    CHECK(kv.second->IsInstance<PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto f = Downcast<PrimFunc>(kv.second);
    cg->AddFunction(f);
  }

  const auto *find_rocm_bitcodes =
      tvm::runtime::Registry::Get("tvm_callback_rocm_bitcode_path");
  Array<runtime::String> bitcode_files = (*find_rocm_bitcodes)();

  for (auto &bitcode_path : bitcode_files) {
    std::string path = bitcode_path;
    llvm::SMDiagnostic err;
    std::unique_ptr<llvm::Module> mlib = llvm::parseIRFile(path, err, *ctx);
    if (mlib.get() == nullptr) {
      std::string msg(err.getMessage());
      LOG(FATAL) << "Fail to load bitcode file " << path << "\n"
                 << "line " << err.getLineNo() << ":" << msg;
    }
    mlib->setTargetTriple(tm->getTargetTriple().str());
    mlib->setDataLayout(tm->createDataLayout());
    for (llvm::Function &f : mlib->functions()) {
      f.addFnAttr(llvm::Attribute::AlwaysInline);
    }
    cg->AddLinkModule(std::move(mlib));
  }

  std::unique_ptr<llvm::Module> module = cg->Finish();
  llvm::SmallString<8> dataObj, data_ll, dataAsm;
  llvm::raw_svector_ostream destObj(dataObj), dest_ll(data_ll), destAsm(dataAsm);
  destObj.SetUnbuffered();
  dest_ll.SetUnbuffered();
  destAsm.SetUnbuffered();
  module->print(dest_ll, nullptr);
#if TVM_LLVM_VERSION <= 60
  std::unique_ptr<llvm::Module> mAsm = llvm::CloneModule(module.get());
  std::unique_ptr<llvm::Module> mObj = llvm::CloneModule(module.get());
#else
  std::unique_ptr<llvm::Module> mAsm = llvm::CloneModule(*module.get());
  std::unique_ptr<llvm::Module> mObj = llvm::CloneModule(*module.get());
#endif
  llvm::legacy::PassManager pass;

#if TVM_LLVM_VERSION <= 60
  CHECK(tm->addPassesToEmitFile(
            pass, destObj, llvm::TargetMachine::CGFT_ObjectFile) == 0)
            << "Cannot emit target CGFT_ObjectFile";
#elif TVM_LLVM_VERSION <= 90
  CHECK(tm->addPassesToEmitFile(
            pass, destObj, nullptr, llvm::TargetMachine::CGFT_ObjectFile) == 0)
            << "Cannot emit target CGFT_ObjectFile";
#else
  CHECK(tm->addPassesToEmitFile(
            pass, destObj, nullptr, llvm::CGFT_ObjectFile) == 0)
            << "Cannot emit target CGFT_ObjectFile";
#endif
  pass.run(*mObj);
  std::string obj(dataObj.begin(), dataObj.end());

  llvm::legacy::PassManager passAsm;
#if TVM_LLVM_VERSION <= 60
  CHECK(tm->addPassesToEmitFile(passAsm, destAsm,
                                llvm::TargetMachine::CGFT_AssemblyFile) == 0)
      << "Cannot emit target CGFT_AssemblyFile";
#elif TVM_LLVM_VERSION <= 90
  CHECK(tm->addPassesToEmitFile(passAsm, destAsm, nullptr,
                                llvm::TargetMachine::CGFT_AssemblyFile) == 0)
      << "Cannot emit target CGFT_AssemblyFile";
#else
  CHECK(tm->addPassesToEmitFile(passAsm, destAsm, nullptr,
                                llvm::CGFT_AssemblyFile) == 0)
      << "Cannot emit target CGFT_AssemblyFile";
#endif
  passAsm.run(*mAsm);
  std::string assembly(dataAsm.begin(), dataAsm.end());

  const auto* f = tvm::runtime::Registry::Get("tvm_callback_rocm_link");
  CHECK(f != nullptr) << "Require tvm_callback_rocm_link to exist, do import tvm.contrib.rocm";

  TVMByteArray arr;
  arr.data = &obj[0];
  arr.size = obj.length();

  std::string hsaco = (*f)(arr);
  std::string ll(data_ll.begin(), data_ll.end());
  return ROCMModuleCreate(hsaco, "hsaco", ExtractFuncInfo(mod), ll, assembly);
}

TVM_REGISTER_GLOBAL("target.build.rocm")
.set_body_typed(BuildAMDGPU);

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
