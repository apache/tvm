/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_amdgpu.cc
 * \brief AMDGPU code generator.
 */
#ifdef TVM_LLVM_VERSION
#if TVM_ROCM_RUNTIME

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/registry.h>
#include "./codegen_llvm.h"
#include "../build_common.h"
#include "../../pass/ir_util.h"
#include "../../runtime/rocm/rocm_module.h"

namespace tvm {
namespace codegen {

// AMDGPU code generator.
class CodeGenAMDGPU : public CodeGenLLVM {
 public:
  void AddFunction(const LoweredFunc& f) final {
    // add function as void return value
    CodeGenLLVM::AddFunctionInternal(f, true);
    function_->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
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
      // maximum necessary alignment in the AMD devices
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

  llvm::Value* CreateStorageSync(const Call* op) final {
    const std::string& sync = op->args[0].as<StringImm>()->value;
    if (sync == "warp") {
      // TODO(tqchen) warp sync in CUDA9
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

  unsigned GetGlobalAddressSpace() {
    return 1;
  }

 protected:
  void InitTarget(llvm::TargetMachine* tm) final {
    // Maximum vector lane = float4
    native_vector_bits_ = 4 * 32;
    CodeGenLLVM::InitTarget(tm);
  }
};

runtime::Module BuildAMDGPU(Array<LoweredFunc> funcs, std::string target) {
  CHECK(target.length(
) >= 4 &&
        target.substr(0, 4) == "rocm");

  TVMContext tvmCtx;
  tvmCtx.device_type = kROCM;
  tvmCtx.device_id = 0;
  TVMRetValue val;
  tvm::runtime::DeviceAPI::Get(tvmCtx)->GetAttr(tvmCtx, tvm::runtime::kExist, &val);
  if (val.operator int() == 1) {
    tvm::runtime::DeviceAPI::Get(tvmCtx)->GetAttr(tvmCtx, tvm::runtime::kComputeVersion, &val);
  } else {
    val = 803;
  }

  llvm::TargetMachine* tm = \
    GetLLVMTargetMachine("-mtriple=amdgcn-amd-amdhsa-hcc -mcpu=gfx" + \
    std::to_string(val.operator int())+ target.substr(4, target.length() - 4));

  std::unique_ptr<CodeGenAMDGPU> cg(new CodeGenAMDGPU());
  std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext());
  cg->Init(funcs[0]->name, tm, ctx.get(), false, false);
  for (LoweredFunc f :  funcs) {
    cg->AddFunction(f);
  }

  std::unique_ptr<llvm::Module> module = cg->Finish();

  llvm::SmallString<8> dataObj, data_ll, dataAsm;
  llvm::raw_svector_ostream destObj(dataObj), dest_ll(data_ll), destAsm(dataAsm);
  destObj.SetUnbuffered();
  dest_ll.SetUnbuffered();
  destAsm.SetUnbuffered();
  module->print(dest_ll, nullptr);
  std::unique_ptr<llvm::Module> mAsm = llvm::CloneModule(module.get());
  std::unique_ptr<llvm::Module> mObj = llvm::CloneModule(module.get());
  llvm::legacy::PassManager pass;

  CHECK(tm->addPassesToEmitFile(
            pass, destObj, llvm::TargetMachine::CGFT_ObjectFile) == 0)
            << "Cannot emit target CGFT_ObjectFile";
  pass.run(*mObj);
  std::string obj(dataObj.begin(), dataObj.end());

  const auto* f = tvm::runtime::Registry::Get("tvm_callback_rocm_link");
  CHECK(f != nullptr) << "Require tvm_callback_rocm_link to exist, do import tvm.contrib.rocm";

  TVMByteArray arr;
  arr.data = &obj[0];
  arr.size = obj.length();

  std::string hsaco = (*f)(arr);
  std::string ll(data_ll.begin(), data_ll.end());

  return ROCMModuleCreate(hsaco, "hsaco", ExtractFuncInfo(funcs), ll);
}

TVM_REGISTER_API("codegen.build_rocm")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildAMDGPU(args[0], args[1]);
  });

}  // namespace codegen
}  // namespace tvm
#endif   // TVM_ROCM_RUNTIME
#endif  // TVM_LLVM_VERSION
