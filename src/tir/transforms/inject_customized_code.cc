/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file inject_customized_code.cc
 * \brief Transform customized code into prepend stmt.
 */

#include <tvm/target/target.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

#include "../../support/utils.h"
#include "../schedule/utils.h"
#include "./ir_utils.h"

namespace tvm {
namespace tir {

namespace inject_customized_code {

/*! Structure that represents the provided annotation per block or loop. */
struct CustomziedCodeInfo {
  int width;
};

class CustomziedCodeInjector : private StmtExprMutator {
 public:
  static Stmt Inject(const PrimFunc& func) {
    CustomziedCodeInjector injector;
    return injector(func->body);
  }

 private:
  explicit CustomziedCodeInjector() {}
  Stmt VisitStmt_(const ForNode* op) final {
    // Step 1: Recursively rewrite the children first.
    For for_node = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    if (HasCustomizedCodePrependAnnotation(op)) {
      String code = Downcast<String>(op->annotations.at(attr::inject_customized_code_prepend));
      // Create a CustomizedCode
      auto _code = CustomizedCode(code);
      Array<Stmt> seq;
      seq.push_back(_code);
      seq.push_back(for_node);
      auto seq_node = SeqStmt(seq);
      // Step 2: Rewrite the current node.
      return std::move(seq_node);
    } else if (HasCustomizedCodePostpendAnnotation(op)) {
      String code = Downcast<String>(op->annotations.at(attr::inject_customized_code_postpend));
      // Create a CustomizedCode
      auto _code = CustomizedCode(code);
      Array<Stmt> seq;
      seq.push_back(for_node);
      seq.push_back(_code);
      auto seq_node = SeqStmt(seq);
      // Step 2: Rewrite the current node.
      return std::move(seq_node);
    }
    return std::move(for_node);
  }

  bool HasCustomizedCodePrependAnnotation(const ForNode* op) const {
    auto it = op->annotations.find(attr::inject_customized_code_prepend);
    bool has_annotation = it != op->annotations.end();
    if (has_annotation) {
      return true;
    }
    return false;
  }

  bool HasCustomizedCodePostpendAnnotation(const ForNode* op) const {
    auto it = op->annotations.find(attr::inject_customized_code_postpend);
    bool has_annotation = it != op->annotations.end();
    if (has_annotation) {
      return true;
    }
    return false;
  }
};

}  // namespace inject_customized_code_prepend

namespace transform {

/*!
 * \brief Transform annotated block into threa block rasteration form.
 * \return The IR transform pass.
 */
Pass InjectCustomizedCode() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* fptr = f.CopyOnWrite();
    fptr->body = inject_customized_code::CustomziedCodeInjector::Inject(f);
    fptr->body = ConvertSSA(std::move(fptr->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InjectCustomizedCode", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InjectCustomizedCode").set_body_typed(InjectCustomizedCode);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
