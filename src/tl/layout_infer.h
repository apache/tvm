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
 * \file tl/layout_infer.h
 * \brief Infer layout from ops and parallel for
 */

#ifndef TVM_TL_LAYOUT_INFER_H_
#define TVM_TL_LAYOUT_INFER_H_

#include <tvm/target/target.h>
#include <tvm/tir/stmt_functor.h>

#include "layout.h"
#include "op.h"

namespace tvm {
namespace tl {

using namespace tir;

using LayoutMap = Map<Buffer, Layout>;

enum class InferLevel {
  kFree = 0,
  kCommon = 1,
  kStrict = 2,
};

class LayoutInferBase {
 public:
  virtual LayoutMap Inference(const LayoutMap& layout_map, InferLevel level) { return {}; };
};

class ForNodeLayoutInfer : public LayoutInferBase, StmtExprVisitor {
 public:
  ForNodeLayoutInfer(const ForNode* root, IterVar thread_var);
  Map<Buffer, Layout> Inference(const Map<Buffer, Layout>& layout_map, InferLevel level) final;

  Fragment GetLoopLayout() const { return loop_layout_; }
  const ForNode* GetRoot() const { return root_; }
  Map<Buffer, Array<PrimExpr>> GetIndiceMap() const { return indice_map_; }
  PrimExpr GetPredicate() const { return predicate_; }

 private:
  Fragment CompleteBufferFragment(const Buffer& buffer);
  bool IsCommonAccessIndice(const Buffer& buffer) const;
  void VisitStmt_(const ForNode* op) final;
  void VisitStmt_(const BufferStoreNode* op) final;
  void VisitExpr_(const BufferLoadNode* op) final;
  void AddPredicate(PrimExpr expr) {
    predicate_ = predicate_.defined() ? And(expr, predicate_) : expr;
  }
  const ForNode* root_;
  IterVar thread_var_;

  Map<Buffer, Array<PrimExpr>> indice_map_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_is_write_;
  Array<IterVar> loop_vars_;
  Fragment loop_layout_;
  arith::Analyzer analyzer_;
  PrimExpr predicate_;
};

class GemmOpLayoutInfer : public LayoutInferBase {
 public:
  GemmOpLayoutInfer(const GemmArgs& gemm_args, size_t block_size, const TargetNode* target);
  LayoutMap Inference(const LayoutMap& layout_map, InferLevel level) final;

 private:
  const GemmArgs args;
  const size_t block_size_;
  bool completed_ = false;
  const TargetNode* target_;
};

class ReduceOpLayoutInfer : public LayoutInferBase {
 public:
  ReduceOpLayoutInfer(const ReduceArgs& reduce_args, size_t block_size);
  LayoutMap Inference(const LayoutMap& layout_map, InferLevel level) final;

 private:
  const ReduceArgs args;
  const size_t block_size_;
};

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_LAYOUT_INFER_H_
