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
#include "./ir_comparator.h"

namespace tvm {

namespace tir {

/******** Tensorize Comparator ********/

class TensorIntrinMismatchError : public ScheduleError {
 public:
  explicit TensorIntrinMismatchError(IRModule lhs_mod, Stmt lhs_stmt, Stmt rhs_stmt) :
    lhs_mod_(std::move(lhs_mod)), lhs_stmt_(std::move(lhs_stmt)), rhs_stmt_(std::move(rhs_stmt)) {
    ICHECK(lhs_stmt_->IsInstance<ForNode>() || lhs_stmt_->IsInstance<BlockNode>());
  }

  String FastErrorString() const final {
    return "ScheduleError: The stmt doesn't match the tensor intrin.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The stmt {0} doesn't match the tensor intrin\n " << rhs_stmt_;
    return os.str();
  }

  IRModule mod() const final { return lhs_mod_; }

  Array<ObjectRef> LocationsOfInterest() const final { return {lhs_stmt_}; }

 private:
  IRModule lhs_mod_;
  Stmt lhs_stmt_;
  Stmt rhs_stmt_;
};

/* Override the dispatcher to make sure RHS is always valid */
bool TensorizeComparator::VisitStmt(const Stmt& n, const Stmt& other) {
  bool equal = (n->type_index() == other->type_index()) && StmtComparator::VisitStmt(n, other);
  if (!equal && assert_mode_ && (n->IsInstance<ForNode>() || n->IsInstance<BlockNode>())) {
    throw TensorIntrinMismatchError(lhs_mod_, n, other);
  }
  return equal;
}

bool TensorizeComparator::VisitExpr(const PrimExpr& n, const PrimExpr& other) {
  return (n->type_index() == other->type_index()) && ExprComparator::VisitExpr(n, other);
}

bool TensorizeComparator::VisitStmt_(const ForNode* op, const Stmt& other) {
  const auto* rhs = other.as<ForNode>();
  if (!DefEqual(op->loop_var, rhs->loop_var)) return false;
  if (!VisitExpr(op->min, rhs->min)) return false;
  if (!VisitExpr(op->extent, rhs->extent)) return false;
  if (op->thread_binding.defined() != rhs->thread_binding.defined()) return false;
  if (op->thread_binding.defined() &&
      !VisitExpr(op->thread_binding.value(), rhs->thread_binding.value())) {
    return false;
  }
  if (op->kind != rhs->kind) return false;
  if (!CompareAnnotationMap(op->annotations, rhs->annotations)) return false;
  return VisitStmt(op->body, rhs->body);
}

bool TensorizeComparator::VisitStmt_(const SeqStmtNode* op, const Stmt& other) {
  const auto* rhs = other.as<SeqStmtNode>();
  return CompareArray(op->seq, rhs->seq, &TensorizeComparator::VisitStmt);
}

bool TensorizeComparator::VisitStmt_(const BufferStoreNode* op, const Stmt& other) {
  const auto* rhs = other.as<BufferStoreNode>();
  return CompareBufferAccess(op, rhs) && VisitExpr(op->value, rhs->value);
}

bool TensorizeComparator::VisitStmt_(const BlockRealizeNode* op, const Stmt& other) {
  const auto* rhs = other.as<BlockRealizeNode>();
  // Skip Compare binding values if the block is scope block (the outermost one).
  if (!is_scope_block) {
    size_t offset = op->iter_values.size() - rhs->iter_values.size();
    if (rhs->iter_values.size() > op->iter_values.size()) return false;
    if (is_inner_block) {
      // weak pattern matching for the inner block (the son of the scope block)
      // where the pattern is v + iter <=> expr + iter
      for (size_t i = 0; i < rhs->iter_values.size(); ++i) {
        PrimExpr lhs_expr, rhs_expr;
        Optional<Var> lhs_iter, rhs_iter;
        auto detect = [](const PrimExpr& binding) -> std::pair<PrimExpr, Optional<Var>> {
          arith::PVar<PrimExpr> expr;
          arith::PVar<Var> iter;
          if (iter.Match(binding)) {
            return std::make_pair(0, iter.Eval());
          } else if ((expr + iter).Match(binding)) {
            return std::make_pair(expr.Eval(), iter.Eval());
          } else if ((iter + expr).Match(binding)) {
            return std::make_pair(expr.Eval(), iter.Eval());
          } else {
            return std::make_pair(expr.Eval(), NullOpt);
          }
        };
        std::tie(lhs_expr, lhs_iter) = detect(op->iter_values[i + offset]);
        std::tie(rhs_expr, rhs_iter) = detect(rhs->iter_values[i]);
        CHECK((lhs_iter && rhs_iter) || (!lhs_iter && !rhs_iter)) << "Incompatible binding";
        if (lhs_iter) VisitExpr(lhs_iter.value(), rhs_iter.value());
        if (is_zero(rhs_expr)) {
          CHECK(is_zero(lhs_expr)) << "Incompatible binding";
        } else {
          const auto* bv = rhs_expr.as<VarNode>();
          if (!bv) {
            VisitExpr(lhs_expr, rhs_expr);
          } else {
            auto it = equal_map_.find(GetRef<Var>(bv));
            if (it == equal_map_.end()) {
              equal_map_[GetRef<Var>(bv)] = lhs_expr;
            } else {
              CHECK(it->second->IsInstance<PrimExprNode>());
              VisitExpr(lhs_expr, Downcast<PrimExpr>(it->second));
            }
          }
        }
      }
    } else {
      for (size_t i = 0; i < rhs->iter_values.size(); ++i) {
        if (!VisitExpr(op->iter_values[i + offset], rhs->iter_values[i])) return false;
      }
      const Block& block = op->block;
      for (size_t i = 0; i < offset; ++i) {
        Var block_var = Downcast<Var>(op->iter_values[i]);
        auto it = equal_map_.find(block_var);
        equal_map_[block->iter_vars[i]->var] = (it == equal_map_.end() ? block_var : it->second);
      }
    }
  }

  return VisitExpr(op->predicate, rhs->predicate) && VisitStmt(op->block, rhs->block);
}

bool TensorizeComparator::VisitStmt_(const BlockNode* op, const Stmt& other) {
  const auto* rhs = other.as<BlockNode>();
  // Check block equality.
  // All iter vars and buffer regions including the order shoudl match.
  // When checking iter vars, DefEqual is used to remap variables.
  // Only the inner most several axis are compared. Other iter vars are added to extra_block_vars.
  if (op->iter_vars.size() < rhs->iter_vars.size()) return false;

  if (is_scope_block) {
    //lhs_scope_block = op;
  }
  // size_t offset = op->iter_vars.size() - rhs->iter_vars.size();
  // for (size_t i = 0; i < rhs->iter_vars.size(); ++i) {
  //   auto lhs_var = op->iter_vars[i + offset], rhs_var = rhs->iter_vars[i];
  //   // Skip iter dom
  //   if (!DefEqual(lhs_var->var, rhs_var->var)) {
  //     return false;
  //   }
  //   if (lhs_var->iter_type != rhs_var->iter_type) {
  //     return false;
  //   }
  // }

  // if (is_scope_block) {
  //   for (size_t i = 0; i < offset; ++i) {
  //     extra_block_vars_.push_back(op->iter_vars[i]);
  //   }
  // }

  // if (!is_scope_block) {
  //   if (!CompareArray(op->writes, rhs->writes, &TensorizeComparator::CompareBufferRegion)) {
  //     return false;
  //   }
  //   if (!CompareArray(op->reads, rhs->reads, &TensorizeComparator::CompareBufferRegion)) {
  //     return false;
  //   }
  //   if (!CompareAnnotationMap(op->annotations, rhs->annotations)) {
  //     return false;
  //   }
  //   if (!CompareArray(op->alloc_buffers, rhs->alloc_buffers, &TensorizeComparator::CompareBuffer)) {
  //     return false;
  //   }
  // }
  is_scope_block = false;
  return VisitStmt(op->body, rhs->body);
}

// Exprs
#define TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(OpName)                            \
  bool TensorizeComparator::VisitExpr_(const OpName* op, const PrimExpr& other) { \
    const auto* rhs = other.as<OpName>();                                         \
    return VisitExpr(op->a, rhs->a) && VisitExpr(op->b, rhs->b);                  \
  }

TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(AddNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(SubNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(MulNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(DivNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(ModNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(EQNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(NENode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(LTNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(LENode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(GTNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(GENode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(AndNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(OrNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(MinNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(MaxNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(FloorDivNode);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(FloorModNode);

bool TensorizeComparator::VisitExpr_(const IntImmNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<IntImmNode>();
  return CompareType(op->dtype, rhs->dtype) && op->value == rhs->value;
}

bool TensorizeComparator::VisitExpr_(const FloatImmNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<FloatImmNode>();
  return CompareType(op->dtype, rhs->dtype) && op->value == rhs->value;
}

bool TensorizeComparator::VisitExpr_(const CastNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<CastNode>();
  return CompareType(op->dtype, rhs->dtype) && VisitExpr(op->value, rhs->value);
}

bool TensorizeComparator::VisitExpr_(const VarNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<VarNode>();
  auto lhs = GetRef<Var>(op);
  if (lhs.same_as(other)) return true;
  if (!CompareType(op->dtype, rhs->dtype)) return false;
  auto it = equal_map_.find(lhs);
  return it != equal_map_.end() && it->second.same_as(other);
}

bool TensorizeComparator::VisitExpr_(const BufferLoadNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<BufferLoadNode>();
  return CompareBufferAccess(op, rhs);
}

bool TensorizeComparator::DefEqual(const ObjectRef& lhs, const ObjectRef& rhs) {
  if (lhs.same_as(rhs)) return true;
  if (lhs->type_index() != rhs->type_index()) return false;
  auto it = equal_map_.find(lhs);
  // If there is already a mapping
  if (it != equal_map_.end()) return it->second.same_as(rhs);
  equal_map_[lhs] = rhs;
  return true;
}

bool TensorizeComparator::CompareAnnotation(const std::pair<String, ObjectRef>& lhs,
                                            const std::pair<String, ObjectRef>& rhs) {
  if (lhs.first != rhs.first) return false;
  if (!lhs.second.same_as(rhs.second)) return false;
  return VisitExpr(Downcast<PrimExpr>(lhs.second), Downcast<PrimExpr>(rhs.second));
}

bool TensorizeComparator::CompareAnnotationMap(const Map<String, ObjectRef>& lhs,
                                               const Map<String, ObjectRef>& rhs) {
  if (lhs.same_as(rhs)) return true;
  if (lhs.size() != rhs.size()) return false;

  auto sort_map =
      [](const Map<String, ObjectRef>& map) -> std::vector<std::pair<String, ObjectRef>> {
    std::vector<std::pair<String, ObjectRef>> ret;
    ret.reserve(map.size());
    for (const auto& pair : map) {
      ret.emplace_back(pair);
    }
    sort(ret.begin(), ret.end());
    return ret;
  };

  auto lhs_array = sort_map(lhs), rhs_array = sort_map(rhs);

  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!CompareAnnotation(lhs_array[i], rhs_array[i])) return false;
  }
  return true;
}

bool TensorizeComparator::CompareBuffer(const Buffer& lhs, const Buffer& rhs) {
  if (lhs.same_as(rhs)) return true;
  // Remap both buffer itself and buffer data
  // Skip buffer shape
  bool equal = DefEqual(lhs, rhs) && DefEqual(lhs->data, rhs->data) &&
               CompareType(lhs->dtype, rhs->dtype) && lhs.scope() == rhs.scope();
  if (equal) {
    rhs_buffer_map_[rhs] = lhs;
  } else if (assert_mode_) {
    LOG(FATAL) << "Buffers are not matching between:" << lhs << " and " << rhs << lhs->dtype << rhs->dtype<<lhs.scope() << rhs.scope();
  }
  return equal;
}

bool TensorizeComparator::CompareBufferRegion(const BufferRegion& lhs, const BufferRegion& rhs) {
  // Only for block region declaration
  if (!CompareBuffer(lhs->buffer, rhs->buffer)) return false;
  // Number of indices in desc_block must be smaller than it in AST
  if (rhs->region.size() > lhs->region.size()) return false;

  std::vector<Range> lhs_region;
  for (const auto& range : lhs->region) {
    lhs_region.push_back(Range::FromMinExtent(range->min, range->extent));
  }
  size_t offset = lhs_region.size() - rhs->region.size();
  // initialize buffer indices
  bool need_update = false;
  if (!buffer_indices_.count(lhs->buffer)) {
    need_update = true;
    buffer_indices_[lhs->buffer] = std::vector<PrimExpr>();
  } else {
    if (offset != buffer_indices_[lhs->buffer].size()) return false;
  }
  std::vector<PrimExpr>& indices = buffer_indices_[lhs->buffer];
  for (size_t i = 0; i < offset; ++i) {
    const Range& range = lhs_region[i];
    // High-dim region must be element-wise
    if (!is_one(range->extent)) return false;
    if (need_update) {
      indices.push_back(range->min);
    } else {
      // The order matters since we only map inner block_var to outside block_var
      if (!VisitExpr(range->min, indices[i])) return false;
    }
  }
  for (size_t i = 0; i < rhs->region.size(); ++i) {
    if (!CompareRange(lhs_region[i + offset], rhs->region[i])) return false;
  }
  return true;
}

// Comparator for BufferStoreNode and BufferLoadNode
template <typename T>
bool TensorizeComparator::CompareBufferAccess(const T* lhs, const T* rhs) {
  if (!CompareBuffer(lhs->buffer, rhs->buffer)) return false;
  if (rhs->indices.size() > lhs->indices.size()) return false;
  // otherwise compare the leading indices
  size_t offset = lhs->indices.size() - rhs->indices.size();
  for (size_t i = 0; i < rhs->indices.size(); ++i) {
    if (!VisitExpr(lhs->indices[i + offset], rhs->indices[i])) return false;
  }
  return true;
}

template <typename T, typename F>
bool TensorizeComparator::CompareArray(const Array<T>& lhs, const Array<T>& rhs, F cmp) {
  if (lhs.same_as(rhs)) return true;
  if (lhs.size() != rhs.size()) return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!(this->*cmp)(lhs[i], rhs[i])) return false;
  }
  return true;
}

bool TensorizeComparator::CompareRange(const Range& lhs, const Range& rhs) {
  return VisitExpr(lhs->min, rhs->min) && VisitExpr(lhs->extent, rhs->extent);
}

bool TensorizeComparator::CompareType(const DataType& lhs, const DataType& rhs) {
  if (lhs == rhs) return true;
  return lhs.code() == rhs.code() && lhs.bits() == rhs.bits() && lhs.lanes() == rhs.lanes();
}

}  // namespace tir
}  // namespace tvm
