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
 *  Out of bounds array access static analyzer.
 */

#include <tvm/tir/transform.h>

#include "../../arith/ir_visitor_with_analyzer.h"
#include "../../printer/text_printer.h"
#include "../schedule/error.h"

namespace tvm {
namespace tir {
namespace transform {
struct OOBLocation {
  Buffer buf;
  size_t dimension;
  ObjectRef index;
  arith::IntSet index_bounds;
  arith::IntSet shape_bounds;
};

class OOBError : public ScheduleError {
 public:
  OOBError(IRModule mod, std::vector<OOBLocation> locations) : mod_(mod), locations_(locations) {}
  String FastErrorString() const final { return "Out of bound memory access"; }

  String DetailRenderTemplate() const final {
    std::stringstream s;
    for (const auto& oob : locations_) {
      s << "Out of bounds memory access on buffer " << oob.buf->name << " dimension "
        << oob.dimension << ".";
      s << " index " << oob.index << " with bounds [" << oob.index_bounds.min() << ", "
        << oob.index_bounds.max() << "] is outside the range [0, " << oob.shape_bounds.min()
        << "].";
      s << "\n";
    }
    return s.str();
  }
  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final {
    std::vector<ObjectRef> locs;
    for (auto loc : locations_) {
      locs.push_back(loc.index);
    }
    return locs;
  }

 private:
  IRModule mod_;
  std::vector<OOBLocation> locations_;
};
class OOBCheckerVisitor final : public arith::IRVisitorWithAnalyzer {
  using IRVisitorWithAnalyzer::VisitExpr_;
  using IRVisitorWithAnalyzer::VisitStmt_;

 public:
  void VisitStmt_(const BufferStoreNode* node) final {
    for (size_t i = 0; i < node->buffer->shape.size(); i++) {
      CheckBounds(node, i);
    }
    IRVisitorWithAnalyzer::VisitStmt_(node);
  }
  void VisitExpr_(const BufferLoadNode* node) final {
    for (size_t i = 0; i < node->buffer->shape.size(); i++) {
      CheckBounds(node, i);
    }
    IRVisitorWithAnalyzer::VisitExpr_(node);
  }

  template <class T>
  void CheckBounds(const T* node, size_t i) {
    auto ind_bounds = analyzer_.int_set(node->indices[i]);
    auto shape_bounds = analyzer_.int_set(node->buffer->shape[i]);
    // Only show an error if we can prove that the access occurs out of bounds.
    // In some cases we may not be able to prove that the access is in or out
    // of bounds and we would like to ignore these cases.
    if (analyzer_.CanProve(ind_bounds.max() >= shape_bounds.min()) ||
        analyzer_.CanProve(ind_bounds.min() < 0)) {
      errors.push_back({node->buffer, i, node->indices[i], ind_bounds, shape_bounds});
    }
  }

  std::vector<OOBLocation> errors;
};

transform::Pass OOBChecker() {
  auto pass_func = [=](tir::PrimFunc func, IRModule mod, transform::PassContext ctx) {
    OOBCheckerVisitor checker;
    checker(func->body);
    if (checker.errors.size() > 0) {
      // mod doesn't contain our function, so we construct a new mod with out function
      IRModule func_mod({{GlobalVar("main"), func}});
      LOG(FATAL) << OOBError(func_mod, checker.errors).RenderReport("Out of bounds checker");
    }
    return func;
  };
  return transform::CreatePrimFuncPass(pass_func, 0, "tir.analysis.OOBChecker", {});
}

TVM_REGISTER_GLOBAL("tir.analysis.OOBChecker").set_body_typed(OOBChecker);
}  // namespace transform
}  // namespace tir
}  // namespace tvm
