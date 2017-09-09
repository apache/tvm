/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_nvptx.cc
 * \brief NVPTX code generator.
 */
#ifdef TVM_LLVM_VERSION
#if TVM_CUDA_RUNTIME

#include <tvm/runtime/device_api.h>
#include "./codegen_llvm.h"
#include "../build_common.h"
#include "../../pass/ir_util.h"
#include "../../runtime/cuda/cuda_module.h"

namespace tvm {
namespace codegen {

// NVPTX code generator.
class CodeGenNVPTX : public CodeGenLLVM {
 public:
  void AddFunction(const LoweredFunc& f) final {
    // add function as void return value
    CodeGenLLVM::AddFunctionInternal(f, true);
    // annotate as kernel function
    module_->getOrInsertNamedMetadata("nvvm.annotations")
        ->addOperand(llvm::MDNode::get(*ctx_, {
              llvm::ValueAsMetadata::get(function_),
              llvm::MDString::get(*ctx_, "kernel"),
              llvm::ValueAsMetadata::get(ConstInt32(1)) }));
  }

  void VisitStmt_(const Allocate* op) final {
    CHECK(!is_zero(op->condition));
    llvm::Value* buf = nullptr;
    if (op->new_expr.defined()) {
      CHECK_EQ(op->free_function, "nop");
      buf = MakeValue(op->new_expr);
    } else {
      int32_t constant_size = op->constant_allocation_size();
      CHECK_GT(constant_size, 0)
          << "Can only handle constant size stack allocation in GPU";
      StorageInfo& info = alloc_storage_info_[op->buffer_var.get()];
      if (constant_size % 4 == 0 && info.alignment == 0) {
        info.alignment = GetTempAllocaAlignment(op->type, constant_size);
      }
      // maximum necessary alignment in the NV devices
      if (info.alignment > 16) {
        info.alignment = 16;
      }
      if (info.scope.rank == 2) {
        // const int local_address_space = 5;
        // TODO(tqchen): for higher version of LLVM, local address space can be set.
        llvm::AllocaInst* alloca = builder_->CreateAlloca(
            LLVMType(op->type), ConstInt32(constant_size));
        if (alloca->getAlignment() < static_cast<uint32_t>(info.alignment)) {
          alloca->setAlignment(info.alignment);
        }
        buf = alloca;
      } else {
        CHECK_EQ(info.scope.rank, 1)
            << "Can only allocate shared or local memory inside kernel";
        // Shared memory: address space  == 3
        const unsigned shared_address_space = 3;
        llvm::Type* type = llvm::ArrayType::get(LLVMType(op->type), constant_size);
        // Allocate shared memory in global, address_space = 3
        llvm::GlobalVariable *global = new llvm::GlobalVariable(
            *module_, type, false, llvm::GlobalValue::PrivateLinkage, 0, ".shared",
            nullptr, llvm::GlobalValue::NotThreadLocal, shared_address_space);
        global->setAlignment(info.alignment);
        buf = global;
      }
    }
    buf = builder_->CreatePointerCast(
        buf, LLVMType(op->type)->getPointerTo(
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

  llvm::Value* CreateStorageSync(const Call* op) final {
    const std::string& sync = op->args[0].as<StringImm>()->value;
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

 protected:
  void InitTarget(llvm::TargetMachine* tm) final {
    // Maximum vector lane = float4
    native_vector_bits_ = 4 * 32;
    CodeGenLLVM::InitTarget(tm);
  }
};

runtime::Module BuildNVPTX(Array<LoweredFunc> funcs, std::string target) {
  CHECK(target.length() >= 5 &&
        target.substr(0, 5) == "nvptx");
  llvm::TargetMachine* tm = GetLLVMTargetMachine(
      "-mtriple=nvptx64-nvidia-cuda -mcpu=sm_20" +
      target.substr(5, target.length() - 5));
  std::unique_ptr<CodeGenNVPTX> cg(new CodeGenNVPTX());
  std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext());
  cg->Init(funcs[0]->name, tm, ctx.get(), false, false);
  for (LoweredFunc f :  funcs) {
    cg->AddFunction(f);
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
  CHECK(tm->addPassesToEmitFile(
      pass, dest_ptx, llvm::TargetMachine::CGFT_AssemblyFile) == 0)
      << "Cannot emit target CGFT_ObjectFile";
  pass.run(*module);
  std::string ptx(data_ptx.begin(), data_ptx.end());
  return CUDAModuleCreate(ptx, "ptx", ExtractFuncInfo(funcs), ll);
}

TVM_REGISTER_API("codegen.build_nvptx")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildNVPTX(args[0], args[1]);
  });

}  // namespace codegen
}  // namespace tvm
#endif   // TVM_CUDA_RUNTIME
#endif  // TVM_LLVM_VERSION
