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
 * \file codegen_nvptx.cc
 * \brief NVPTX code generator.
 */
#ifdef TVM_LLVM_VERSION

#include <llvm/ADT/SmallString.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#if TVM_LLVM_VERSION >= 100
#include <llvm/IR/IntrinsicsNVPTX.h>
#endif
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IRReader/IRReader.h>
#if TVM_LLVM_VERSION >= 100
#include <llvm/Support/Alignment.h>
#endif
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <tvm/runtime/device_api.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../../runtime/cuda/cuda_module.h"
#include "../build_common.h"
#include "codegen_llvm.h"
#include "llvm_instance.h"

namespace tvm {
namespace codegen {

// NVPTX code generator.
class CodeGenNVPTX : public CodeGenLLVM {
 public:
  void AddFunction(const PrimFunc& f) final {
    // add function as void return value
    CodeGenLLVM::AddFunctionInternal(f, true);
    // annotate as kernel function
    llvm::LLVMContext* ctx = llvm_target_->GetContext();
    module_->getOrInsertNamedMetadata("nvvm.annotations")
        ->addOperand(llvm::MDNode::get(
            *ctx, {llvm::ValueAsMetadata::get(function_), llvm::MDString::get(*ctx, "kernel"),
                   llvm::ValueAsMetadata::get(ConstInt32(1))}));
  }

  void VisitStmt_(const AllocateNode* op) final {
    ICHECK(!is_zero(op->condition));
    llvm::Value* buf = nullptr;
    StorageInfo& info = alloc_storage_info_[op->buffer_var.get()];
    // maximum necessary alignment in the NV devices
    if (info.alignment > 16) {
      info.alignment = 16;
    }

    auto storage_scope = runtime::StorageScope::Create(GetPtrStorageScope(op->buffer_var));
    if (storage_scope.rank == runtime::StorageRank::kShared && storage_scope.tag == ".dyn") {
      // Shared memory: address space  == 3
      buf =
          AllocateSharedMemory(op->dtype, 0, 3, info.alignment, llvm::GlobalValue::ExternalLinkage);
    } else {
      size_t constant_size = op->ConstantAllocationSize();
      ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation in GPU";

      if (constant_size % 4 == 0 && info.alignment == 0) {
        info.alignment = GetTempAllocaAlignment(op->dtype, constant_size);
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
    llvm::Intrinsic::ID intrin_id = llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x;
    if (ts.rank == 1) {
      switch (ts.dim_index) {
        case 0:
          intrin_id = llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x;
          break;
        case 1:
          intrin_id = llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y;
          break;
        case 2:
          intrin_id = llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z;
          break;
        default:
          LOG(FATAL) << "unknown thread idx";
      }
    } else {
      ICHECK_EQ(ts.rank, 0);
      switch (ts.dim_index) {
        case 0:
          intrin_id = llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x;
          break;
        case 1:
          intrin_id = llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y;
          break;
        case 2:
          intrin_id = llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z;
          break;
        default:
          LOG(FATAL) << "unknown thread idx";
      }
    }
    llvm::Function* f = llvm::Intrinsic::getDeclaration(module_.get(), intrin_id);
    return builder_->CreateCall(f, {});
  }

  llvm::Value* CreateStorageSync(const CallNode* op) final {
    const std::string& sync = op->args[0].as<StringImmNode>()->value;
    if (sync == "warp") {
      // TODO(tqchen) warp sync in CUDA9
      return nullptr;
    } else if (sync == "shared" || sync == "shared.dyn") {
      llvm::Function* f =
          llvm::Intrinsic::getDeclaration(module_.get(), llvm::Intrinsic::nvvm_barrier0);
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

  void Optimize() final {
    for (auto& f : *module_) {
      auto fname = static_cast<std::string>(f.getName());
      if (fname.substr(0, 4) != "__nv") continue;
      // This is to strip off unused __nv_* functions from the final module
      // The one that is actually used will be inlined at call site
      // Adapted from Halide's runtime linker
      if (!f.isDeclaration() && !f.hasFnAttribute(llvm::Attribute::NoInline)) {
        f.setLinkage(llvm::GlobalValue::AvailableExternallyLinkage);
      }
    }
    CodeGenLLVM::Optimize();
  }

  llvm::Value* CreateIntrinsic(const CallNode* op) override;

 protected:
  void InitTarget() final {
    // Maximum vector lane = float4
    native_vector_bits_ = 4 * 32;
    CodeGenLLVM::InitTarget();
  }
};

// Check if this is a warp shuffle intrinsic call and match its
// corresponding nvvm intrinsic. Return true if the match is successful.
static bool GetWarpShuffleIntrinsic(const CallNode* op, llvm::Intrinsic::ID* id) {
  // Only 32 bit data type is supported.
  if (op->dtype.is_vector() || op->dtype.bits() != 32) {
    return false;
  }

  // Intrinsic lookup table.
  // It is difficult to emit _sync verion that works on Pascal.
  // We ignore the mask and only emit the non-sync version for nvptx.
  llvm::Intrinsic::ID ids[] = {
      llvm::Intrinsic::nvvm_shfl_idx_i32,  llvm::Intrinsic::nvvm_shfl_idx_f32,
      llvm::Intrinsic::nvvm_shfl_up_i32,   llvm::Intrinsic::nvvm_shfl_up_f32,
      llvm::Intrinsic::nvvm_shfl_down_i32, llvm::Intrinsic::nvvm_shfl_down_f32};

  int offset = 0;
  if (op->op.same_as(builtin::tvm_warp_shuffle())) {
    offset = 0;
  } else if (op->op.same_as(builtin::tvm_warp_shuffle_up())) {
    offset = 2;
  } else if (op->op.same_as(builtin::tvm_warp_shuffle_down())) {
    offset = 4;
  } else {
    return false;
  }

  *id = ids[offset + op->dtype.is_float()];
  return true;
}

llvm::Value* CodeGenNVPTX::CreateIntrinsic(const CallNode* op) {
  llvm::Intrinsic::ID id = llvm::Intrinsic::not_intrinsic;
  if (GetWarpShuffleIntrinsic(op, &id)) {
    std::vector<llvm::Value*> arg_value;
    std::vector<llvm::Type*> arg_type;
    // Ignore the first mask operand and remove the last
    // redundant warp_size..
    size_t n_args = op->args.size() - 1;
    for (size_t i = 1; i < n_args; ++i) {
      arg_value.push_back(MakeValue(op->args[i]));
      arg_type.push_back(arg_value.back()->getType());
    }
    llvm::Type* return_type = arg_type[0];
    llvm::Function* func = GetIntrinsicDecl(id, return_type, arg_type);
    return builder_->CreateCall(func, arg_value);
  } else if (op->op.same_as(builtin::tvm_warp_activemask())) {
    // Only nvptx target may keep this intrinsic at this point.
    // PTX assembly: asm "activemask.b32 r1;"
    auto fty = llvm::FunctionType::get(t_int32_, false);
    auto val = llvm::InlineAsm::get(fty, "activemask.b32 %0", "=r", true);
    return builder_->CreateCall(val);
  } else if (op->op.same_as(builtin::atomic_add())) {
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

int GetCUDAComputeVersion(const Target& target) {
  Optional<String> mcpu = target->GetAttr<String>("mcpu");
  ICHECK(mcpu.defined()) << "InternalError: \"-mcpu\" is undefined in the NVPTX target";
  std::string sm_version = mcpu.value();
  return std::stoi(sm_version.substr(3));
}

runtime::Module BuildNVPTX(IRModule mod, Target target) {
  LLVMInstance llvm_instance;
  With<LLVMTarget> llvm_target(llvm_instance, target);

  int compute_ver = GetCUDAComputeVersion(target);
  auto cg = std::make_unique<CodeGenNVPTX>();

  cg->Init("TVMPTXModule", llvm_target.get(), false, false, false);

  cg->AddFunctionsOrdered(mod->functions.begin(), mod->functions.end(), [](auto& kv) {
    ICHECK(kv.second->template IsInstance<PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    return Downcast<PrimFunc>(kv.second);
  });

  llvm::TargetMachine* tm = llvm_target->GetOrCreateTargetMachine();
  const auto* flibdevice_path = tvm::runtime::Registry::Get("tvm_callback_libdevice_path");
  if (flibdevice_path != nullptr) {
    std::string path = (*flibdevice_path)(compute_ver);
    if (path.length() != 0) {
      std::unique_ptr<llvm::Module> mlib = llvm_instance.LoadIR(path);
      mlib->setTargetTriple(llvm_target->GetTargetTriple());
      mlib->setDataLayout(tm->createDataLayout());
      cg->AddLinkModule(std::move(mlib));
    }
  }
  std::unique_ptr<llvm::Module> module = cg->Finish();
  llvm::SmallString<8> data_ptx, data_ll;
  llvm::raw_svector_ostream dest_ptx(data_ptx), dest_ll(data_ll);
  dest_ptx.SetUnbuffered();
  dest_ll.SetUnbuffered();
  // print ll
  module->print(dest_ll, nullptr);
  std::string ll(data_ll.begin(), data_ll.end());
  // emit ptx
  llvm::legacy::PassManager pass;
#if TVM_LLVM_VERSION <= 60
  ICHECK(tm->addPassesToEmitFile(pass, dest_ptx, llvm::TargetMachine::CGFT_AssemblyFile) == 0)
      << "Cannot emit target CGFT_ObjectFile";
#elif TVM_LLVM_VERSION <= 90
  ICHECK(tm->addPassesToEmitFile(pass, dest_ptx, nullptr, llvm::TargetMachine::CGFT_AssemblyFile) ==
         0)
      << "Cannot emit target CGFT_ObjectFile";
#else
  ICHECK(tm->addPassesToEmitFile(pass, dest_ptx, nullptr, llvm::CGFT_AssemblyFile) == 0)
      << "Cannot emit target CGFT_ObjectFile";
#endif
  pass.run(*module);
  std::string ptx(data_ptx.begin(), data_ptx.end());
  return CUDAModuleCreate(ptx, "ptx", ExtractFuncInfo(mod), ll);
}

TVM_REGISTER_GLOBAL("target.build.nvptx").set_body_typed(BuildNVPTX);

TVM_REGISTER_GLOBAL("tvm.codegen.llvm.target_nvptx")
    .set_body([](const TVMArgs& targs, TVMRetValue* rv) {
      *rv = static_cast<void*>(new CodeGenNVPTX());
    });

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
