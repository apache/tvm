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
 * \file common.h
 * \brief Common utilities for TL transforms
 */

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include <queue>

#include "../../../arith/ir_mutator_with_analyzer.h"
#include "../../op/parallel.h"
#include "../loop_partition.h"
#include "../loop_vectorize.h"

namespace tvm {
namespace tl {

using namespace tir;
using arith::IRMutatorWithAnalyzer;

class FragmentAccessDetector : public StmtExprVisitor {
 public:
  FragmentAccessDetector() = default;

  void Collect(Stmt stmt) { VisitStmt(stmt); }

  bool HasFragmentAccess() { return has_fragment_access_; }

 private:
  void VisitExpr_(const BufferLoadNode* op) final {
    // Check if the buffer is in global scope
    if (IsFragementBuffer(op->buffer)) {
      has_fragment_access_ = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    // Check if the buffer is in global scope
    if (IsFragementBuffer(op->buffer)) {
      has_fragment_access_ = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  // Helper function to determine if a buffer is local.fragement
  bool IsFragementBuffer(const Buffer& buffer) {
    // The storage scope is often encoded in the buffer->data var name or associated attributes.
    String scope = buffer.scope();
    return scope == "local.fragment";
  }

  bool has_fragment_access_{false};
};

/*!
 * \brief ParallelLoopFuser
 * This class is used to fuse a chain of parallel loops into one loop.
 * The loops must:
 *  - All be parallel (ForKind::kParallel)
 *  - Have bounds from 0 to their extent
 * Once fused, a single loop variable will replace the chain, and the
 * original loop variables will be derived by division and modulo operations.
 *
 * This can be helpful for inferring layout for the fragment in a subsequent pass.
 */
class ParallelLoopFuser : public IRMutatorWithAnalyzer {
 public:
  static Stmt Fuse(Stmt stmt) {
    arith::Analyzer analyzer;
    ParallelLoopFuser substituter(&analyzer);
    return substituter.VisitStmt(stmt);
  }

 private:
  ParallelLoopFuser(arith::Analyzer* analyzer) : IRMutatorWithAnalyzer(analyzer) {};

  Stmt VisitStmt_(const ForNode* op) final {
    // Gather consecutive parallel loops
    std::vector<const ForNode*> loop_chain;
    const ForNode* current = op;
    // check if has fragment access
    FragmentAccessDetector detector;
    detector.Collect(op->body);
    // Do not fuse if there is a fragment access
    if (detector.HasFragmentAccess()) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    while (true) {
      if (current->kind != ForKind::kParallel) break;
      if (!is_zero(current->min)) break;
      loop_chain.push_back(current);

      const ForNode* inner_for = current->body.as<ForNode>();
      if (!inner_for) {
        break;
      }
      current = inner_for;
    }

    // If only one loop found or loop chain size is 1, no fusion needed.
    if (loop_chain.size() <= 1) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    // At this point we have multiple nested parallel loops starting at zero
    // We will fuse them all.
    PrimExpr fused_extent = make_const(DataType::Int(32), 1);
    for (auto it = loop_chain.rbegin(); it != loop_chain.rend(); ++it) {
      fused_extent = fused_extent * (*it)->extent;
    }

    std::string fused_name;
    for (auto it = loop_chain.begin(); it != loop_chain.end(); ++it) {
      fused_name += (*it)->loop_var->name_hint + "_";
    }

    fused_name += "fused";

    // Create a new fused loop var
    Var fused_var(fused_name, DataType::Int(32));

    // The body of the last loop in the chain:
    const ForNode* innermost_loop = loop_chain.back();
    Stmt body = innermost_loop->body;

    // We need to substitute all loop variables in the chain.
    // The scheme:
    // Suppose we have loops (i in [0,M], j in [0,N], k in [0,O])
    // fused loop var f in [0, M*N*O]
    // i = f / (N*O)
    // j = (f % (N*O)) / O
    // k = f % O
    //
    // Generalizing for a chain of lengths L:
    // extents: E_0, E_1, ... E_{L-1}
    // index_i = (f / (E_{i+1}*...*E_{L-1})) % E_i
    // For the last one, it's just f % E_{L-1} if i == L-1.

    // Compute the "stride" products for each loop variable
    // stride[i] = product of extents of loops after i
    // for L loops: stride[L-1] = 1
    // stride[L-2] = E_{L-1}
    // stride[L-3] = E_{L-1} * E_{L-2}
    // ...
    std::vector<PrimExpr> extents;
    extents.reserve(loop_chain.size());
    for (auto l : loop_chain) {
      extents.push_back(l->extent);
    }

    std::vector<PrimExpr> strides(loop_chain.size(), make_const(DataType::Int(32), 1));
    for (int i = static_cast<int>(loop_chain.size()) - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * extents[i + 1];
    }

    // We'll create a substitution map for all loop variables
    // index_i = (f / strides[i]) % extents[i]
    // We'll define a helper lambda:
    auto create_index_expr = [&](int i) {
      return FloorMod(FloorDiv(fused_var, strides[i]), extents[i]);
    };

    Map<Var, PrimExpr> var_map;
    for (size_t i = 0; i < loop_chain.size(); i++) {
      const ForNode* loop = loop_chain[i];
      var_map.Set(loop->loop_var, analyzer_->Simplify(create_index_expr(static_cast<int>(i))));
    }

    // Perform the substitution
    body = Substitute(body, var_map);

    // Create the fused loop
    For fused_for = For(fused_var, 0, fused_extent, ForKind::kParallel, body);

    return fused_for;
  }
};

}  // namespace tl
}  // namespace tvm
