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

#include <llvm/ADT/SmallString.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#if TVM_LLVM_VERSION >= 100
#include <llvm/IR/IntrinsicsAMDGPU.h>
#endif
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IRReader/IRReader.h>
#if TVM_LLVM_VERSION >= 100
#include <llvm/Support/Alignment.h>
#endif
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include "../../runtime/rocm/rocm_module.h"
#include "../build_common.h"
#include "codegen_llvm.h"
#include "llvm_instance.h"

namespace tvm {
namespace codegen {

namespace {

// calls the device api to get the max threads per block
static inline int DetectROCMmaxThreadsPerBlock() {
  Device tvm_dev;
  tvm_dev.device_type = kDLROCM;
  tvm_dev.device_id = 0;
  tvm::runtime::DeviceAPI* api = tvm::runtime::DeviceAPI::Get(tvm_dev, true);
  if (api != nullptr) {
    TVMRetValue val;
    api->GetAttr(tvm_dev, tvm::runtime::kExist, &val);
    if (val.operator int() == 1) {
      tvm::runtime::DeviceAPI::Get(tvm_dev)->GetAttr(tvm_dev, tvm::runtime::kMaxThreadsPerBlock,
                                                     &val);
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
  CodeGenAMDGPU() = default;
  virtual ~CodeGenAMDGPU() = default;

  void AddFunction(const PrimFunc& f) final {
    // add function as void return value
    CodeGenLLVM::AddFunctionInternal(f, true);
    function_->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
    std::ostringstream attr;
    attr << "1," << DetectROCMmaxThreadsPerBlock();
    function_->addFnAttr("amdgpu-flat-work-group-size", attr.str());
  }

  void VisitStmt_(const AllocateNode* op) final {
    ICHECK(!is_zero(op->condition));
    llvm::Value* buf = nullptr;
    StorageInfo& info = alloc_storage_info_[op->buffer_var.get()];
    auto storage_scope = runtime::StorageScope::Create(GetPtrStorageScope(op->buffer_var));

    if (storage_scope.rank == runtime::StorageRank::kShared && storage_scope.tag == ".dyn") {
      LOG(WARNING) << "Dynamic shared memory support for rocm is experimental.";
      buf = AllocateSharedMemory(op->dtype, 0, 3, std::min(info.alignment, 16),
                                 llvm::GlobalValue::ExternalLinkage);
    } else {
      size_t constant_size = op->ConstantAllocationSize();
      ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation in GPU";

      if (constant_size % 4 == 0 && info.alignment == 0) {
        info.alignment = GetTempAllocaAlignment(op->dtype, constant_size);
      }
      // maximum necessary alignment in the AMD devices
      if (info.alignment > 16) {
        info.alignment = 16;
      }
      if (storage_scope.rank == runtime::StorageRank::kLocal) {
        // const int local_address_space = 5;
        // TODO(tqchen): for higher version of LLVM, local address space can be set.
        llvm::AllocaInst* alloca = WithFunctionEntry([&]() {
          return builder_->CreateAlloca(DTypeToLLVMType(op->dtype), ConstInt32(constant_size));
        });
#if TVM_LLVM_VERSION >= 110
        auto alignment = static_cast<unsigned>(alloca->getAlign().value());
#else
        unsigned alignment = alloca->getAlignment();
#endif
        if (alignment < static_cast<unsigned>(info.alignment)) {
#if TVM_LLVM_VERSION >= 100
          alloca->setAlignment(llvm::Align(info.alignment));
#else
          alloca->setAlignment(info.alignment);
#endif
        }
        buf = alloca;
      } else {
        ICHECK(storage_scope.rank == runtime::StorageRank::kShared)
            << "Can only allocate shared or local memory inside kernel";
        // Shared memory: address space  == 3
        buf = AllocateSharedMemory(op->dtype, constant_size, 3, info.alignment,
                                   llvm::GlobalValue::PrivateLinkage);
      }
    }

    buf = builder_->CreatePointerCast(
        buf, DTypeToLLVMType(op->dtype)->getPointerTo(buf->getType()->getPointerAddressSpace()));
    ICHECK(!var_map_.count(op->buffer_var.get()));
    var_map_[op->buffer_var.get()] = buf;
    this->VisitStmt(op->body);
  }

  // Return the thread index via intrinsics.
  llvm::Value* GetThreadIndex(const IterVar& iv) final {
    runtime::ThreadScope ts = runtime::ThreadScope::Create(iv->thread_tag);
    llvm::Intrinsic::ID intrin_id = llvm::Intrinsic::amdgcn_workitem_id_x;
    if (ts.rank == 1) {
      switch (ts.dim_index) {
        case 0:
          intrin_id = llvm::Intrinsic::amdgcn_workitem_id_x;
          break;
        case 1:
          intrin_id = llvm::Intrinsic::amdgcn_workitem_id_y;
          break;
        case 2:
          intrin_id = llvm::Intrinsic::amdgcn_workitem_id_z;
          break;
        default:
          LOG(FATAL) << "unknown workitem idx";
      }
    } else {
      ICHECK_EQ(ts.rank, 0);
      switch (ts.dim_index) {
        case 0:
          intrin_id = llvm::Intrinsic::amdgcn_workgroup_id_x;
          break;
        case 1:
          intrin_id = llvm::Intrinsic::amdgcn_workgroup_id_y;
          break;
        case 2:
          intrin_id = llvm::Intrinsic::amdgcn_workgroup_id_z;
          break;
        default:
          LOG(FATAL) << "unknown workgroup idx";
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
      llvm::Function* f =
          llvm::Intrinsic::getDeclaration(module_.get(), llvm::Intrinsic::amdgcn_s_barrier);
      return builder_->CreateCall(f, {});
    } else {
      LOG(FATAL) << "Do not support sync " << sync;
      return nullptr;
    }
  }

#if TVM_LLVM_VERSION < 160
  // This function only works with the legacy pass manager.
  void InitPassManagerBuilder(llvm::PassManagerBuilder* builder) final {
    // Additional optimization hook to tweak the builder.
  }
#endif

  unsigned GetGlobalAddressSpace() const final { return 1; }

  llvm::Value* CreateIntrinsic(const CallNode* op) final {
    if (op->op.same_as(builtin::atomic_add())) {
      ICHECK(op->args[1]->dtype.bits() == 32) << "Only supports 32 bit atomic for now";
      llvm::Value* v0 = MakeValue(op->args[0]);
      llvm::Value* v1 = MakeValue(op->args[1]);
      if (op->args[1]->dtype.is_float()) {
#if TVM_LLVM_VERSION >= 90
#if TVM_LLVM_VERSION >= 130
        return builder_->CreateAtomicRMW(llvm::AtomicRMWInst::FAdd, v0, v1, llvm::MaybeAlign(),
                                         llvm::AtomicOrdering::Monotonic);
#else
        return builder_->CreateAtomicRMW(llvm::AtomicRMWInst::FAdd, v0, v1,
                                         llvm::AtomicOrdering::Monotonic);
#endif
#else
        LOG(FATAL) << "Floating point atomic requires LLVM 9 or newer";
#endif
      }
#if TVM_LLVM_VERSION >= 130
      return builder_->CreateAtomicRMW(llvm::AtomicRMWInst::Add, v0, v1, llvm::MaybeAlign(),
                                       llvm::AtomicOrdering::Monotonic);
#else
      return builder_->CreateAtomicRMW(llvm::AtomicRMWInst::Add, v0, v1,
                                       llvm::AtomicOrdering::Monotonic);
#endif
    }
    return CodeGenLLVM::CreateIntrinsic(op);
  }

 protected:
  void InitTarget() final {
    // Maximum vector lane = float4
    native_vector_bits_ = 4 * 32;
    CodeGenLLVM::InitTarget();
  }
};

runtime::Module BuildAMDGPU(IRModule mod, Target target) {
  LLVMInstance llvm_instance;

  With<LLVMTarget> llvm_target(llvm_instance, target);
#if TVM_LLVM_VERSION < 90
  LOG(FATAL) << "AMDGPU backend requires at least LLVM 9";
  // Lower versions will crash when loading the bitcode, see
  // issue #4087 for a discussion
#endif
  auto cg = std::make_unique<CodeGenAMDGPU>();

  cg->Init("TVMAMDGPUModule", llvm_target.get(), false, false, false);

  cg->AddFunctionsOrdered(mod->functions.begin(), mod->functions.end(), [](auto& kv) {
    ICHECK(kv.second->template IsInstance<PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    return Downcast<PrimFunc>(kv.second);
  });

  llvm::TargetMachine* tm = llvm_target->GetOrCreateTargetMachine();
  const auto* find_rocm_bitcodes = tvm::runtime::Registry::Get("tvm_callback_rocm_bitcode_path");
  Array<runtime::String> bitcode_files = (*find_rocm_bitcodes)();

  for (auto& bitcode_path : bitcode_files) {
    std::unique_ptr<llvm::Module> mlib = llvm_instance.LoadIR(bitcode_path);
    mlib->setTargetTriple(llvm_target->GetTargetTriple());
    mlib->setDataLayout(tm->createDataLayout());

    for (llvm::Function& f : mlib->functions()) {
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
  ICHECK(tm->addPassesToEmitFile(pass, destObj, llvm::TargetMachine::CGFT_ObjectFile) == 0)
      << "Cannot emit target CGFT_ObjectFile";
#elif TVM_LLVM_VERSION <= 90
  ICHECK(tm->addPassesToEmitFile(pass, destObj, nullptr, llvm::TargetMachine::CGFT_ObjectFile) == 0)
      << "Cannot emit target CGFT_ObjectFile";
#else
  ICHECK(tm->addPassesToEmitFile(pass, destObj, nullptr, llvm::CGFT_ObjectFile) == 0)
      << "Cannot emit target CGFT_ObjectFile";
#endif
  pass.run(*mObj);
  std::string obj(dataObj.begin(), dataObj.end());

  llvm::legacy::PassManager passAsm;
#if TVM_LLVM_VERSION <= 60
  ICHECK(tm->addPassesToEmitFile(passAsm, destAsm, llvm::TargetMachine::CGFT_AssemblyFile) == 0)
      << "Cannot emit target CGFT_AssemblyFile";
#elif TVM_LLVM_VERSION <= 90
  ICHECK(tm->addPassesToEmitFile(passAsm, destAsm, nullptr,
                                 llvm::TargetMachine::CGFT_AssemblyFile) == 0)
      << "Cannot emit target CGFT_AssemblyFile";
#else
  ICHECK(tm->addPassesToEmitFile(passAsm, destAsm, nullptr, llvm::CGFT_AssemblyFile) == 0)
      << "Cannot emit target CGFT_AssemblyFile";
#endif
  passAsm.run(*mAsm);
  std::string assembly(dataAsm.begin(), dataAsm.end());

  const auto* f = tvm::runtime::Registry::Get("tvm_callback_rocm_link");
  ICHECK(f != nullptr) << "Require tvm_callback_rocm_link to exist, do import tvm.contrib.rocm";

  TVMByteArray arr;
  arr.data = &obj[0];
  arr.size = obj.length();

  std::string hsaco = (*f)(arr);
  std::string ll(data_ll.begin(), data_ll.end());
  return ROCMModuleCreate(hsaco, "hsaco", ExtractFuncInfo(mod), ll, assembly);
}

TVM_REGISTER_GLOBAL("target.build.rocm").set_body_typed(BuildAMDGPU);

TVM_REGISTER_GLOBAL("tvm.codegen.llvm.target_rocm")
    .set_body([](const TVMArgs& targs, TVMRetValue* rv) {
      *rv = static_cast<void*>(new CodeGenAMDGPU());
    });

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
