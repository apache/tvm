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
#include "memhammer_rewrite_rule.h"

namespace tvm {
namespace tir {

Stmt CopyLoopChain(const std::vector<const ForNode*> loops, const Stmt& inner_body, int ith = -1,
                   Stmt* ith_loop = nullptr) {
  Stmt ret = inner_body;
  for (int i = static_cast<int>(loops.size() - 1); i >= 0; i--) {
    ObjectPtr<ForNode> new_loop = make_object<ForNode>(*loops[i]);
    new_loop->body = ret;
    ret = For(new_loop);
    if (ith == i) {
      *ith_loop = ret;
    }
  }
  return ret;
}

/*!
 * \brief lift all the thread binding loops
 * \param stmt the top loop
 * \return a pair. The first is the transformed stmt.
 *         The second is the lowest thread binding loop.
 */
std::pair<Stmt, For> LiftThreadBindingLoops(Stmt stmt) {
  std::vector<const ForNode*> normal_loops;
  std::vector<const ForNode*> thread_binding_loops;
  Stmt body = stmt;
  while (const ForNode* loop = body.as<ForNode>()) {
    if (loop->kind == ForKind::kThreadBinding) {
      thread_binding_loops.push_back(loop);
    } else {
      normal_loops.push_back(loop);
    }
    body = loop->body;
  }
  body = CopyLoopChain(normal_loops, body);
  For compute_location;
  body = CopyLoopChain(thread_binding_loops, body,
                       static_cast<int>(thread_binding_loops.size()) - 1, &compute_location);

  return std::make_pair(body, compute_location);
}

/*!
 * \brief Analyze the access pattern for buffer rank promotion.
 * Rank promotion is a transformation that reshapes the buffer
 * but doesn't change its underlying data layout.
 * After the reshape, we expect that all dimensions of the access indices
 * will be in the form of floormod(floordiv(x, a), b).
 * Rank promotion removes strided access, thus enabling further buffer compacting
 */
class IndexPatternFinder : public ExprVisitor {
 public:
  IndexPatternFinder(const Map<Var, Range>& var_range, Array<PrimExpr>* resulting_index)
      : var_range_(var_range), resulting_index_(resulting_index) {}
  struct Operator {
    enum class OpKind { Mul, FloorDiv, FloorMod };
    OpKind kind;
    int64_t operand;
  };

  /*!
   * \brief Calculate the new buffer shape after rank promotion.
   * For each dimension of original shape, it will be compacted.
   * \param indices The access indices of the buffer
   * \param var_range The iter range of the vars in the indices
   * \param rewrite_indices The access indices after rank promotion
   * \return The new buffer shape after rank promotion.
   */
  static Array<PrimExpr> getRankPromotedShape(Array<PrimExpr> indices,
                                              const Map<Var, Range>& var_range,
                                              Array<PrimExpr>* rewrite_indices) {
    Map<Var, arith::IntSet> var_dom = arith::AsIntSet(var_range);
    Array<PrimExpr> new_shape;
    for (const PrimExpr& expr : indices) {
      Array<PrimExpr> indices_dim;
      IndexPatternFinder extractor(var_range, &indices_dim);
      extractor(expr);
      if (!extractor.success_) {
        return {};
      }
      Array<PrimExpr> access_shape = extractor.access_shape_;
      PrimExpr product_shape = 1;
      for (PrimExpr e : access_shape) {
        product_shape *= e;
      }
      new_shape.push_back(product_shape);
      PrimExpr flatten_index = 0;
      for (int i = 0; i < static_cast<int>(access_shape.size()); i++) {
        flatten_index = flatten_index * access_shape[i] + indices_dim[i];
      }
      rewrite_indices->push_back(flatten_index);
    }
    return new_shape;
  }

 private:
  void VisitExpr_(const VarNode* op) final {
    if (!success_) {
      return;
    }
    if (Optional<Range> range = var_range_.Get(GetRef<Var>(op))) {
      PrimExpr index = GetRef<Var>(op);
      int64_t max = range.value()->extent.as<IntImmNode>()->value;
      int64_t extent = max;
      for (int i = static_cast<int>(operator_stack.size()) - 1; i >= 0; i--) {
        Operator o = operator_stack[i];
        switch (o.kind) {
          case Operator::OpKind::Mul:
            max *= o.operand;
            index = index * Integer(o.operand);
            break;
          case Operator::OpKind::FloorDiv:
            if (max % o.operand != 0 && o.operand % max != 0) {
              success_ = false;
              return;
            }
            max = max / o.operand;
            if (extent > max) {
              extent = std::max(static_cast<int64_t>(1), max);
            }
            if (max % extent != 0) {
              success_ = false;
              return;
            }
            index = floordiv(index, Integer(o.operand));
            break;
          case Operator::OpKind::FloorMod:
            int64_t step = max / extent;
            if (step % o.operand != 0 && o.operand % step != 0) {
              success_ = false;
              return;
            }
            if (step % o.operand == 0) {
              extent = 1;
              max = 0;
            } else {
              extent = std::max(static_cast<int64_t>(1), std::min(extent, o.operand / step));
              max = extent * step;
            }
            index = floormod(index, Integer(o.operand));
        }
      }
      if (extent > 1) {
        ICHECK(max % extent == 0);
        access_shape_.push_back(Integer(extent));
        resulting_index_->push_back(floordiv(index, max / extent));
      }
    }
  }

  void VisitExpr_(const FloorDivNode* op) final {
    int64_t b = op->b.as<IntImmNode>()->value;
    operator_stack.push_back(Operator{Operator::OpKind::FloorDiv, b});
    ExprVisitor::VisitExpr_(op);
    operator_stack.pop_back();
  }

  void VisitExpr_(const FloorModNode* op) final {
    int64_t b = op->b.as<IntImmNode>()->value;
    operator_stack.push_back(Operator{Operator::OpKind::FloorMod, b});
    ExprVisitor::VisitExpr_(op);
    operator_stack.pop_back();
  }

  void VisitExpr_(const MulNode* op) final {
    int64_t b = op->b.as<IntImmNode>()->value;
    operator_stack.push_back(Operator{Operator::OpKind::Mul, b});
    ExprVisitor::VisitExpr_(op);
    operator_stack.pop_back();
  }

  Map<Var, Range> var_range_;
  Array<PrimExpr> access_shape_;
  Array<PrimExpr>* resulting_index_;
  std::vector<Operator> operator_stack;
  bool success_ = true;
};

class BufferLoadReplacer : public StmtExprMutator {
 public:
  BufferLoadReplacer(const Buffer& tgt_buffer, const BufferLoad& new_buffer_load)
      : tgt_buffer_(tgt_buffer), new_buffer_load_(new_buffer_load) {}

  PrimExpr VisitExpr_(const BufferLoadNode* op) {
    if (op->buffer.same_as(tgt_buffer_)) {
      return new_buffer_load_;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

 private:
  Buffer tgt_buffer_;
  BufferLoad new_buffer_load_;
};

/*!
 * \brief Insert a cache stage to the compute location
 * \param stmt the stmt
 * \param is_write_cache whether to write a read cache or write cache
 * \param storage_scope the storage scope of the new cache
 * \param compute_location the compute location.
 * \param outer_loops the outer loops of this stmt
 * \param alloc_buffer the new cache block
 * \return a pair. The first is the stmt after transformation.
 *         The second is the SeqStmt that contains 2 stages (one original and another inserted).
 */
std::pair<Stmt, SeqStmt> InsertCacheStage(Stmt stmt, bool is_write_cache, String storage_scope,
                                          Optional<For> compute_location,
                                          const Array<For>& outer_loops, Buffer* alloc_buffer) {
  Stmt body = stmt;
  std::vector<const ForNode*> loops;
  std::vector<const ForNode*> loops_under_compute_location;
  std::vector<const ForNode*> relaxed_thread_loops;
  bool need_relax = !compute_location.defined();
  Map<Var, Range> var_range;
  PrimExpr vector_bytes = -1;
  // Step 1. Perform rank promotion on the buffer access, turning a strided-changing dimension into
  // several contiguous-changing dimensions
  // Step 1.1 collect loop var range for rank promotion
  while (const ForNode* loop = body.as<ForNode>()) {
    if (need_relax) {
      var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
      loops_under_compute_location.push_back(loop);
    } else {
      loops.push_back(loop);
    }
    if (loop == compute_location.value_or(For()).get()) {
      need_relax = true;
    }
    if (loop->kind == ForKind::kVectorized) {
      vector_bytes = loop->extent;
    }
    body = loop->body;
  }
  Optional<PrimExpr> predicate;
  if (const auto* op = body.as<IfThenElseNode>()) {
    // the predicate is generated by coalescing
    predicate = op->condition;
    body = op->then_case;
  }
  for (const For& loop : outer_loops) {
    if (loop->kind == ForKind::kThreadBinding) {
      const String& thread_tag = loop->thread_binding.value()->thread_tag;
      if (CanRelaxStorageUnderThread(runtime::StorageScope::Create(storage_scope),
                                     runtime::ThreadScope::Create(thread_tag))) {
        var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
        relaxed_thread_loops.push_back(loop.get());
      }
    }
  }

  arith::Analyzer analyzer;
  const BufferLoadNode* target_buffer_load = nullptr;
  if (is_write_cache) {
    tir::PreOrderVisit(stmt, [&](const ObjectRef& obj) {
      if (const auto* buffer_load = obj.as<BufferLoadNode>()) {
        if (buffer_load->buffer.scope() == "wmma.accumulator" ||
            buffer_load->buffer.scope() == "m16n8k8.matrixC") {
          if (target_buffer_load == nullptr) {
            target_buffer_load = buffer_load;
          } else {
            CHECK(target_buffer_load->buffer.same_as(buffer_load->buffer))
                << "More than one target buffer found";
            ICHECK(target_buffer_load->indices.size() == buffer_load->indices.size());
            for (size_t i = 0; i < target_buffer_load->indices.size(); i++) {
              CHECK(
                  analyzer.CanProveEqual(target_buffer_load->indices[i], buffer_load->indices[i]));
            }
          }
        }
      }
      return true;
    });
    CHECK(target_buffer_load);
  }

  const BufferStoreNode* buf_store = TVM_TYPE_AS(body, BufferStoreNode);
  Array<PrimExpr> cache_indices;
  Array<PrimExpr> new_shape;
  bool use_rank_promotion = false;
  if (!is_write_cache && buf_store->value.as<BufferLoadNode>()) {
    Array<PrimExpr> indices =
        is_write_cache ? buf_store->indices : buf_store->value.as<BufferLoadNode>()->indices;
    new_shape = IndexPatternFinder::getRankPromotedShape(indices, var_range, &cache_indices);
    // write cache disabled for now
    // rank promotion for write cache cannot guarantee the shape fits wmma.accumulator
    if (!new_shape.empty()) {
      use_rank_promotion = true;
    }
  }
  Array<Var> new_loop_vars;
  Map<Var, PrimExpr> subst_map;
  if (!use_rank_promotion) {
    cache_indices.clear();
    for (const ForNode* loop : relaxed_thread_loops) {
      new_shape.push_back(loop->extent);
    }
    for (const ForNode* loop : loops_under_compute_location) {
      new_shape.push_back(loop->extent);
    }
  }

  for (int i = 0; i < static_cast<int>(relaxed_thread_loops.size()); i++) {
    const ForNode* loop = relaxed_thread_loops[i];
    Var new_loop_var = loop->loop_var.copy_with_suffix("_cache");
    new_loop_vars.push_back(new_loop_var);
    subst_map.Set(loop->loop_var, new_loop_var);
    if (!use_rank_promotion) {
      cache_indices.push_back(loop->loop_var);
    }
  }
  for (int i = 0; i < static_cast<int>(loops_under_compute_location.size()); i++) {
    const ForNode* loop = loops_under_compute_location[i];
    Var new_loop_var = loop->loop_var.copy_with_suffix("_cache");
    new_loop_vars.push_back(new_loop_var);
    subst_map.Set(loop->loop_var, new_loop_var);
    if (!use_rank_promotion) {
      cache_indices.push_back(loop->loop_var);
    }
  }
  Array<PrimExpr> subst_indices;
  Array<PrimExpr> subst_cache_indices;
  if (is_write_cache) {
    for (PrimExpr e : buf_store->indices) {
      subst_indices.push_back(Substitute(e, subst_map));
    }
  }
  for (PrimExpr e : cache_indices) {
    subst_cache_indices.push_back(Substitute(e, subst_map));
  }

  Buffer new_buffer;
  if (is_write_cache) {
    // this is needed for global <- cast(load(wmma))
    // shared stage should have the same dtype as wmma
    new_buffer = WithScope(target_buffer_load->buffer, storage_scope);
  } else {
    new_buffer = WithScope(buf_store->buffer, storage_scope);
  }
  BufferNode* buffer_ptr = new_buffer.CopyOnWrite();
  buffer_ptr->shape = new_shape;
  *alloc_buffer = new_buffer;

  Stmt generate_body;
  if (is_write_cache) {
    // copy from wmma to new cache buffer
    BufferLoad new_buffer_load{new_buffer, cache_indices};
    generate_body =
        BufferLoadReplacer(target_buffer_load->buffer, new_buffer_load)(GetRef<Stmt>(buf_store));
    generate_body = Substitute(generate_body, subst_map);
  } else {
    generate_body =
        BufferStore(new_buffer, Substitute(buf_store->value, subst_map), subst_cache_indices);
  }

  if (predicate.defined()) {
    // generated by coalescing
    CHECK_EQ(loops_under_compute_location.size(), 2);
    PrimExpr subst_value = 0;
    PrimExpr subst_predicate = Substitute(predicate.value(), subst_map);
    generate_body = IfThenElse(subst_predicate, generate_body);
  }

  for (int i = static_cast<int>(loops_under_compute_location.size()) - 1; i >= 0; i--) {
    const ForNode* orig_loop = loops_under_compute_location[i];
    ObjectPtr<ForNode> new_loop = make_object<ForNode>(*orig_loop);
    new_loop->loop_var = new_loop_vars[i + relaxed_thread_loops.size()];
    new_loop->body = generate_body;
    generate_body = For(new_loop);
  }
  for (int i = static_cast<int>(relaxed_thread_loops.size()) - 1; i >= 0; i--) {
    const ForNode* orig_loop = relaxed_thread_loops[i];
    ObjectPtr<ForNode> new_loop = make_object<ForNode>(*orig_loop);
    new_loop->loop_var = new_loop_vars[i];
    new_loop->body = generate_body;
    new_loop->kind = ForKind::kSerial;
    new_loop->thread_binding = NullOpt;
    new_loop->annotations = {};
    generate_body = For(new_loop);
  }
  Stmt rewrite_body;
  if (is_write_cache) {
    BufferLoad new_buffer_load{new_buffer, cache_indices};
    rewrite_body = BufferStore(new_buffer, GetRef<BufferLoad>(target_buffer_load), cache_indices);
  } else {
    rewrite_body =
        BufferStore(buf_store->buffer, BufferLoad(new_buffer, cache_indices), buf_store->indices);
  }
  if (predicate.defined()) {
    rewrite_body = IfThenElse(predicate.value(), rewrite_body);
  }
  for (int i = static_cast<int>(loops_under_compute_location.size()) - 1; i >= 0; i--) {
    const ForNode* orig_loop = loops_under_compute_location[i];
    ObjectPtr<ForNode> new_loop = make_object<ForNode>(*orig_loop);
    new_loop->body = rewrite_body;
    rewrite_body = For(new_loop);
  }
  SeqStmt insert_location;
  if (is_write_cache) {
    generate_body = insert_location = SeqStmt({rewrite_body, generate_body});
  } else {
    generate_body = insert_location = SeqStmt({generate_body, rewrite_body});
  }
  generate_body = CopyLoopChain(loops, generate_body);
  return std::make_pair(generate_body, insert_location);
}

Stmt CreateLocalStage::Rewrite(const Stmt& stmt, const ConstraintSet& constraints,
                               OutputSet* output) const {
  Stmt body;
  For compute_location;
  std::tie(body, compute_location) = LiftThreadBindingLoops(std::move(stmt));
  Buffer cache_buffer;
  Stmt after_caching = InsertCacheStage(body, false, "local", compute_location,
                                        constraints.outer_loops, &cache_buffer)
                           .first;
  if (cache_buffer.defined()) {
    output->alloc_buffer.push_back(cache_buffer);
  }
  return after_caching;
}

}  // namespace tir
}  // namespace tvm
