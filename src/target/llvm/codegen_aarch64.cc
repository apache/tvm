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
 * \file src/target/llvm/codegen_aarch64.cc
 * \brief AArch64 specific LLVM code generator.
 */
#ifdef TVM_LLVM_VERSION

#include <llvm/IR/Intrinsics.h>
#include <llvm/Target/TargetMachine.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/runtime/logging.h>

#include "codegen_cpu.h"
#include "llvm_instance.h"

namespace tvm {

namespace codegen {

class CodeGenAArch64 final : public CodeGenCPU {
 public:
  CodeGenAArch64() = default;
  virtual ~CodeGenAArch64() = default;

  llvm::Function* DeclareFunction(const GlobalVar& gvar, const Function& f) override;
  void AddFunction(const GlobalVar& gvar, const Function& f) override;
  void SetTargetAttributes(llvm::Function* func) override;

 private:
  void SetComputeScopeAttributes(llvm::Function* func) override;
  void SetPStateAttributes(llvm::Function* func, const Function& f);

  ffi::Optional<Function> current_function_;
  llvm::Function* packed_function_{nullptr};
};

llvm::Function* CodeGenAArch64::DeclareFunction(const GlobalVar& gvar, const Function& f) {
  llvm::Function* func = CodeGenCPU::DeclareFunction(gvar, f);
  if (f->GetAttr<CallingConv>(tvm::attr::kCallingConv) != CallingConv::kCPackedFunc) {
    SetPStateAttributes(func, f);
  }
  return func;
}

void CodeGenAArch64::AddFunction(const GlobalVar& gvar, const Function& f) {
  llvm::Function* func = DeclareFunction(gvar, f);
  current_function_ = f;
  packed_function_ = f->GetAttr<CallingConv>(tvm::attr::kCallingConv) == CallingConv::kCPackedFunc
                         ? func
                         : nullptr;
  CodeGenCPU::AddFunction(gvar, f);
  current_function_ = std::nullopt;
  packed_function_ = nullptr;
}

void CodeGenAArch64::SetComputeScopeAttributes(llvm::Function* func) {
  CodeGenCPU::SetComputeScopeAttributes(func);
  // MakePackedAPI keeps Function attrs on the packed wrapper, but the SME contract belongs
  // to its outlined compute function. Runtime callbacks and nested helpers have separate ABIs.
  if (function_ == packed_function_) {
    SetPStateAttributes(func, current_function_.value());
  }
}

void CodeGenAArch64::SetPStateAttributes(llvm::Function* func, const Function& f) {
  // These string Function attrs are exposed directly through T.func_attr and with_attr.
  if (auto sm = f->GetAttr<ffi::String>(tvm::codegen::aarch64::attr::kPStateSM)) {
    // A locally streaming body does not change a bodyless declaration's interface.
    if (sm.value() != "body" || f->body.has_value()) {
      func->addFnAttr(MakeStringRef("aarch64_pstate_sm_" + sm.value()));
    }
  }
  if (auto za = f->GetAttr<ffi::String>(tvm::codegen::aarch64::attr::kPStateZA)) {
#if TVM_LLVM_VERSION >= 190
    // LLVM 19 renamed the new/shared policies. Keep the legacy spelling for preserved:
    // aarch64_preserves_za describes a shared interface, unlike the old private interface.
    if (za.value() == "new") {
      func->addFnAttr("aarch64_new_za");
    } else if (za.value() == "shared") {
      func->addFnAttr("aarch64_inout_za");
    } else
#endif
    {
      func->addFnAttr(MakeStringRef("aarch64_pstate_za_" + za.value()));
    }
  }
}

void CodeGenAArch64::SetTargetAttributes(llvm::Function* func) {
  // Add vscale_range() function attribute when appropriate.
  if (llvm_target_->TargetHasCPUFeature("sve") || llvm_target_->TargetHasCPUFeature("sme")) {
    // Compute max_val = largest power-of-two <= vector_width/8.
    // Guard against calling llvm_get_vector_width_fn when no target is active —
    // Target::Current() returns an undefined Target outside a compilation context.
    static auto llvm_get_vector_width_fn =
        tvm::ffi::Function::GetGlobalRequired("target.llvm_get_vector_width");
    unsigned int max_val = 0;
    if (auto target = Target::Current(); target.defined()) {
      unsigned int vector_width =
          static_cast<unsigned int>(llvm_get_vector_width_fn(target).cast<int>());
      for (unsigned int i = 0;; ++i) {
        unsigned int power = 1u << i;
        if (power > (vector_width / 8)) break;
        max_val = power;
      }
    }
    if (max_val > 0) {
      func->addFnAttr(
          llvm::Attribute::getWithVScaleRangeArgs(*llvm_target_->GetContext(), 1, max_val));
    }
  }
  CodeGenCPU::SetTargetAttributes(func);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("tvm.codegen.llvm.target_aarch64",
                               [](const ffi::PackedArgs& targs, ffi::Any* rv) {
                                 *rv = static_cast<void*>(new CodeGenAArch64());
                               });
}

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
