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

#include <tvm/runtime/device_api.h>
#include "codegen_llvm.h"
#include "../build_common.h"
#include "../../runtime/cuda/cuda_module.h"

namespace tvm {
namespace codegen {

// NVPTX code generator.
class CodeGenNVPTX : public CodeGenLLVM {
 public:
  void AddFunction(const PrimFunc& f) final {
    // add function as void return value
    CodeGenLLVM::AddFunctionInternal(f, true);
    // annotate as kernel function
    module_->getOrInsertNamedMetadata("nvvm.annotations")
        ->addOperand(llvm::MDNode::get(*ctx_, {
              llvm::ValueAsMetadata::get(function_),
              llvm::MDString::get(*ctx_, "kernel"),
              llvm::ValueAsMetadata::get(ConstInt32(1)) }));
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
    // maximum necessary alignment in the NV devices
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
    llvm::Intrinsic::ID intrin_id = ::llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x;
    if (ts.rank == 1) {
      switch (ts.dim_index) {
        case 0: intrin_id = ::llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x; break;
        case 1: intrin_id = ::llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y; break;
        case 2: intrin_id = ::llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z; break;
        default: LOG(FATAL) << "unknown thread idx";
      }
    } else {
      CHECK_EQ(ts.rank, 0);
      switch (ts.dim_index) {
        case 0: intrin_id = ::llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x; break;
        case 1: intrin_id = ::llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y; break;
        case 2: intrin_id = ::llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z; break;
        default: LOG(FATAL) << "unknown thread idx";
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
    } else if (sync == "shared") {
      llvm::Function* f = llvm::Intrinsic::getDeclaration(
          module_.get(),
          ::llvm::Intrinsic::nvvm_barrier0);
      return builder_->CreateCall(f, {});
    } else {
      LOG(FATAL) << "Do not support sync " << sync;
      return nullptr;
    }
  }

  void InitPassManagerBuilder(llvm::PassManagerBuilder* builder) final {
    // Additional optimization hook to tweak the builder.
  }

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

 protected:
  void InitTarget(llvm::TargetMachine* tm) final {
    // Maximum vector lane = float4
    native_vector_bits_ = 4 * 32;
    CodeGenLLVM::InitTarget(tm);
  }
};

inline int DetectCUDAComputeVersion() {
  TVMContext tvm_ctx;
  tvm_ctx.device_type = kDLGPU;
  tvm_ctx.device_id = 0;
  TVMRetValue val;
  tvm::runtime::DeviceAPI::Get(tvm_ctx)->GetAttr(
      tvm_ctx, tvm::runtime::kExist, &val);
  if (val.operator int() == 1) {
    tvm::runtime::DeviceAPI::Get(tvm_ctx)->GetAttr(
        tvm_ctx, tvm::runtime::kComputeVersion, &val);
    std::string version = val;
    std::istringstream is(version);
    double ver;
    is >> ver;
    return static_cast<int>(ver * 10);
  } else {
    return 20;
  }
}

runtime::Module BuildNVPTX(IRModule mod, std::string target) {
  InitializeLLVM();
  CHECK(target.length() >= 5 &&
        target.substr(0, 5) == "nvptx");
  int compute_ver = DetectCUDAComputeVersion();
  std::ostringstream config;
  config << "-mtriple=nvptx64-nvidia-cuda -mcpu=sm_"
         << compute_ver
         << target.substr(5, target.length() - 5);
  std::unique_ptr<llvm::TargetMachine> tm = GetLLVMTargetMachine(config.str());
  std::unique_ptr<CodeGenNVPTX> cg(new CodeGenNVPTX());
  std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext());

  cg->Init("TVMPTXModule", tm.get(), ctx.get(), false, false);

  for (auto kv :  mod->functions) {
    CHECK(kv.second->IsInstance<PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto f = Downcast<PrimFunc>(kv.second);
    cg->AddFunction(f);
  }

  const auto* flibdevice_path =
      tvm::runtime::Registry::Get("tvm_callback_libdevice_path");
  if (flibdevice_path != nullptr) {
    std::string path = (*flibdevice_path)(compute_ver);
    if (path.length() != 0) {
      llvm::SMDiagnostic err;
      std::unique_ptr<llvm::Module> mlib = llvm::parseIRFile(path, err, *ctx);
      if (mlib.get() == nullptr) {
        std::string msg(err.getMessage());
        LOG(FATAL) << "Fail to load bitcode file " << path << "\n"
                   << "line " << err.getLineNo() << ":" << msg;
      }
      mlib->setTargetTriple(tm->getTargetTriple().str());
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
  CHECK(tm->addPassesToEmitFile(
      pass, dest_ptx, llvm::TargetMachine::CGFT_AssemblyFile) == 0)
      << "Cannot emit target CGFT_ObjectFile";
#elif TVM_LLVM_VERSION <= 90
  CHECK(tm->addPassesToEmitFile(
      pass, dest_ptx, nullptr, llvm::TargetMachine::CGFT_AssemblyFile) == 0)
      << "Cannot emit target CGFT_ObjectFile";
#else
  CHECK(tm->addPassesToEmitFile(
      pass, dest_ptx, nullptr, llvm::CGFT_AssemblyFile) == 0)
      << "Cannot emit target CGFT_ObjectFile";
#endif
  pass.run(*module);
  std::string ptx(data_ptx.begin(), data_ptx.end());
  return CUDAModuleCreate(ptx, "ptx", ExtractFuncInfo(mod), ll);
}

TVM_REGISTER_GLOBAL("target.build.nvptx")
.set_body_typed(BuildNVPTX);

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
