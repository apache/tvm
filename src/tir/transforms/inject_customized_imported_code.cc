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

namespace customized_imported_code {

/*! Structure that represents the provided annotation per block or loop. */
struct CustomziedImportedCodeInfo {
  int width;
};

class CustomziedImportedCodeInjector : private StmtExprMutator {
 public:
  static Stmt Inject(const PrimFunc& func) {
    CustomziedImportedCodeInjector injector;
    return injector(func->body);
  }

 private:
  explicit CustomziedImportedCodeInjector(){}
  Stmt VisitStmt_(const ForNode* op) final {
    // Step 1: Recursively rewrite the children first.
    For for_node = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    if (!HasCustomizedImportedCodeAnnotation(op)) {
      return std::move(for_node);
    }

    String code = Downcast<String>(op->annotations.at(attr::customized_imported_code));

    // Create a RasterNode
    auto imported_code = ImportedCode(code);
    Array<Stmt> seq;
    seq.push_back(imported_code);
    seq.push_back(for_node);
    auto seq_node = SeqStmt(seq);
    // Step 2: Rewrite the current node.
    // combine raster with for_node into a stmt

    return std::move(seq_node);
  }

  bool HasCustomizedImportedCodeAnnotation(const ForNode* op) const {
    auto it = op->annotations.find(attr::customized_imported_code);
    bool has_annotation = it != op->annotations.end();
    if (has_annotation) {
      return true;
    }
    return false;
  }
};

}  // namespace customized_code

namespace transform {

/*!
 * \brief Transform annotated block into threa block rasteration form.
 * \return The IR transform pass.
 */
Pass InjectCustomizedImportedCode() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* fptr = f.CopyOnWrite();
    fptr->body =
        customized_imported_code::CustomziedImportedCodeInjector::Inject(f);
    fptr->body = ConvertSSA(std::move(fptr->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InjectCustomizedImportedCode", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InjectCustomizedImportedCode").set_body_typed(InjectCustomizedImportedCode);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
