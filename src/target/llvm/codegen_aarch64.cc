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
#include <tvm/runtime/registry.h>

#include "../../arith/scalable_expression.h"
#include "codegen_cpu.h"
#include "llvm_instance.h"

namespace tvm {
namespace codegen {

class CodeGenAArch64 final : public CodeGenCPU {
 public:
  CodeGenAArch64() = default;
  virtual ~CodeGenAArch64() = default;

  void VisitStmt_(const AttrStmtNode* op);
  void AddFunction(const GlobalVar& gvar, const PrimFunc& f);
  void SetTargetAttributes(llvm::Function* func);

  bool func_has_pstate_sm = false;
  bool func_has_pstate_za = false;
};

void CodeGenAArch64::AddFunction(const GlobalVar& gvar, const PrimFunc& f) {
  func_has_pstate_sm = false;
  func_has_pstate_za = false;
  CodeGenCPU::AddFunction(gvar, f);
}

void CodeGenAArch64::SetTargetAttributes(llvm::Function* func) {
#if TVM_LLVM_VERSION >= 130
  // Add vscale_range() function attribute when appropriate.
  if (llvm_target_->TargetHasCPUFeature("sve") || llvm_target_->TargetHasCPUFeature("sme")) {
    unsigned int max_val =
        *std::max_element(arith::kAArch64VScaleValues.begin(), arith::kAArch64VScaleValues.end());
    func->addFnAttr(
        llvm::Attribute::getWithVScaleRangeArgs(*llvm_target_->GetContext(), 1, max_val));
  }
#endif
  CodeGenCPU::SetTargetAttributes(func);
}

/*!
 * \brief Visit and handle AArch64 specific pragmas. To be AArch64 specific,
 * the expectation is that they are prepended with "pragma_aarch64".
 */
void CodeGenAArch64::VisitStmt_(const AttrStmtNode* op) {
  std::string attr_key = op->attr_key;

  if (!tir::attr::IsPragmaKey(attr_key)) {
    CodeGenCPU::VisitStmt_(op);
    return;
  }
  bool is_aarch64_specific_pragma = attr_key.substr(7, 7) == "aarch64";
  if (!is_aarch64_specific_pragma) {
    CodeGenCPU::VisitStmt_(op);
    return;
  }

  const auto* attr_value = op->value.as<StringImmNode>();
  ICHECK(attr_value) << "Expect " << attr_key << " to have a String value but was "
                     << op->value->GetTypeKey();

  std::string aarch64_attr_key = attr_key.substr(7);
  if (aarch64_attr_key == "aarch64_pstate_sm") {
    ICHECK(!func_has_pstate_sm) << "Multiple definitions of " << op->attr_key
                                << " attribute found in the function "
                                << function_->getName().data();
    function_->addFnAttr({aarch64_attr_key + "_" + attr_value->value});
    func_has_pstate_sm = true;
  } else if (aarch64_attr_key == "aarch64_pstate_za") {
    ICHECK(!func_has_pstate_za) << "Multiple definitions of " << op->attr_key
                                << " attribute found in the function "
                                << function_->getName().data();
    function_->addFnAttr({aarch64_attr_key + "_" + attr_value->value});
    func_has_pstate_za = true;
  } else {
    LOG(WARNING) << "Unknown pragma " << op->attr_key;
  }
  this->VisitStmt(op->body);
}

TVM_REGISTER_GLOBAL("tvm.codegen.llvm.target_aarch64")
    .set_body([](const TVMArgs& targs, TVMRetValue* rv) {
      *rv = static_cast<void*>(new CodeGenAArch64());
    });

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
