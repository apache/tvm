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
 * \file tl/op/parallel.h
 * \brief Infer layout from ops and parallel for
 */

#ifndef TVM_TL_OP_PARALLEL_H_
#define TVM_TL_OP_PARALLEL_H_

#include <tvm/target/target.h>
#include <tvm/tir/stmt_functor.h>

#include "../layout/layout.h"
#include "op.h"

namespace tvm {
namespace tl {

using namespace tir;

class ParallelOp;

class ParallelLoopNestVisitor : public StmtExprVisitor {
 private:
  ParallelLoopNestVisitor(ParallelOp* op) : p(op){};
  void VisitStmt_(const ForNode* op) final;
  void VisitStmt_(const BufferStoreNode* op) final;
  void VisitExpr_(const BufferLoadNode* op) final;

  ParallelOp* p;

  friend class ParallelOp;
};

class ParallelOp : public Operator {
 public:
  ParallelOp(For root);
  LayoutMap InferLayout(const LayoutInferArgs& T, InferLevel level) final;

  Fragment GetLoopLayout() const { return loop_layout_; }
  For GetRoot() const { return root_; }
  Map<Buffer, Array<PrimExpr>> GetIndiceMap() const { return indice_map_; }
  Optional<PrimExpr> GetPredicate(Var thread_var) const;

 private:
  Fragment CompleteBufferFragment(const Buffer& buffer);
  bool IsCommonAccessIndice(const Buffer& buffer) const;
  void AddPredicate(PrimExpr expr) {
    predicate_ = predicate_.defined() ? And(expr, predicate_.value()) : expr;
  }

  For root_;

  ParallelLoopNestVisitor V;

  Map<Buffer, Array<PrimExpr>> indice_map_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_is_write_;
  Array<IterVar> loop_vars_;

  Fragment loop_layout_;
  arith::Analyzer analyzer_;
  Optional<PrimExpr> predicate_;

  friend class ParallelLoopNestVisitor;
};

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_OP_PARALLEL_H_
