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
  explicit TensorIntrinMismatchError(IRModule lhs_mod, Stmt lhs_stmt, Stmt rhs_stmt,
                                     std::vector<std::string> error_messages)
      : lhs_mod_(std::move(lhs_mod)),
        lhs_stmt_(std::move(lhs_stmt)),
        rhs_stmt_(std::move(rhs_stmt)),
        error_messages_(std::move(error_messages)) {
    ICHECK(lhs_stmt_->IsInstance<ForNode>() || lhs_stmt_->IsInstance<BlockNode>());
  }

  String FastErrorString() const final {
    return "ScheduleError: The stmt doesn't match the tensor intrin.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The stmt {0} doesn't match the tensor intrin\n " << rhs_stmt_;
    for (const auto& msg : error_messages_) {
      os << msg << std::endl;
    }
    return os.str();
  }

  IRModule mod() const final { return lhs_mod_; }

  Array<ObjectRef> LocationsOfInterest() const final { return {lhs_stmt_}; }

 private:
  IRModule lhs_mod_;
  Stmt lhs_stmt_;
  Stmt rhs_stmt_;
  std::vector<std::string> error_messages_;
};

/* Override the dispatcher to make sure RHS is always valid */
bool TensorizeComparator::VisitStmt(const Stmt& n, const Stmt& other) {
  bool equal = n.same_as(other) ||
               ((n->type_index() == other->type_index()) && StmtComparator::VisitStmt(n, other));
  if (!equal && assert_mode_ && (n->IsInstance<ForNode>() || n->IsInstance<BlockNode>())) {
    throw TensorIntrinMismatchError(lhs_mod_, n, other, std::move(error_messages_));
  }
  return equal;
}

bool TensorizeComparator::VisitExpr(const PrimExpr& n, const PrimExpr& other) {
  bool equal =
      n.same_as(other) || ((n->type_index() == other->type_index()) && n->dtype == other->dtype &&
                           ExprComparator::VisitExpr(n, other));
  if (!equal && assert_mode_) {
    std::ostringstream os;
    os << "Expression mismatch: " << n << " vs " << other;
    EmitError(os.str());
  }
  return equal;
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
  if (!is_scope_block) {
    if (!CompareArray(op->iter_values, rhs->iter_values, &TensorizeComparator::VisitExpr)) {
      return false;
    }
  }
  return VisitExpr(op->predicate, rhs->predicate) && VisitStmt(op->block, rhs->block);
}

bool TensorizeComparator::VisitStmt_(const BlockNode* op, const Stmt& other) {
  const auto* rhs = other.as<BlockNode>();
  // Check block equality.
  // All iter vars and buffer regions including the order should match.
  // When checking iter vars, DefEqual is used to remap variables.
  if (!is_scope_block) {
    if (!CompareArray(op->iter_vars, rhs->iter_vars, &TensorizeComparator::CompareIterVar)) {
      return false;
    }
    if (!CompareAnnotationMap(op->annotations, rhs->annotations)) {
      return false;
    }
    if (!CompareArray(op->alloc_buffers, rhs->alloc_buffers, &TensorizeComparator::CompareBuffer)) {
      return false;
    }
  }
  if (!CompareArray(op->writes, rhs->writes, &TensorizeComparator::CompareBufferRegion)) {
    return false;
  }
  if (!CompareArray(op->reads, rhs->reads, &TensorizeComparator::CompareBufferRegion)) {
    return false;
  }
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
  return op->value == rhs->value;
}

bool TensorizeComparator::VisitExpr_(const FloatImmNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<FloatImmNode>();
  return op->value == rhs->value;
}

bool TensorizeComparator::VisitExpr_(const CastNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<CastNode>();
  return VisitExpr(op->value, rhs->value);
}

bool TensorizeComparator::VisitExpr_(const VarNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<VarNode>();
  auto lhs = GetRef<Var>(op);
  if (lhs.same_as(other)) return true;
  if (op->dtype != rhs->dtype) return false;
  auto it = equal_map_.find(lhs);
  return it != equal_map_.end() && it->second.same_as(other);
}

bool TensorizeComparator::VisitExpr_(const BufferLoadNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<BufferLoadNode>();
  return CompareBufferAccess(op, rhs);
}

bool TensorizeComparator::VisitExpr_(const SelectNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<SelectNode>();
  return VisitExpr(op->condition, rhs->condition) && VisitExpr(op->true_value, rhs->true_value) &&
         VisitExpr(op->false_value, rhs->false_value);
}

bool TensorizeComparator::DefEqual(const Var& lhs, const Var& rhs) {
  if (lhs.same_as(rhs)) return true;
  auto it = equal_map_.find(lhs);
  // If there is already a mapping
  if (it != equal_map_.end()) return it->second.same_as(rhs);
  // Otherwise remap lhs to rhs
  equal_map_[lhs] = rhs;
  analyzer_.Bind(lhs, rhs);
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
    std::vector<std::pair<String, ObjectRef>> ret(map.begin(), map.end());
    sort(ret.begin(), ret.end());
    return ret;
  };

  std::vector<std::pair<String, ObjectRef>> lhs_array = sort_map(lhs);
  std::vector<std::pair<String, ObjectRef>> rhs_array = sort_map(rhs);

  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!CompareAnnotation(lhs_array[i], rhs_array[i])) return false;
  }
  return true;
}

bool TensorizeComparator::CompareBuffer(const Buffer& lhs, const Buffer& rhs) {
  if (lhs.same_as(rhs)) return true;
  auto it = rhs_buffer_map_.find(rhs);
  bool equal;
  if (it != rhs_buffer_map_.end()) {
    equal = (*it).second.same_as(lhs);
  } else {
    // Remap both buffer itself and buffer data, skip buffer shape
    equal =
        DefEqual(lhs->data, rhs->data) && lhs->dtype == rhs->dtype && lhs.scope() == rhs.scope();
    if (equal) {
      rhs_buffer_map_[rhs] = lhs;
    }
  }
  return equal;
}

bool TensorizeComparator::CompareBufferRegion(const BufferRegion& lhs, const BufferRegion& rhs) {
  if (!CompareBuffer(lhs->buffer, rhs->buffer)) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "Buffer mismatch: " << lhs->buffer << " vs " << rhs->buffer;
      EmitError(os.str());
    }
    return false;
  }
  int offset = static_cast<int>(lhs->region.size()) - static_cast<int>(rhs->region.size());
  // Number of indices in RHS (desc of the tensor intrinsic) must be smaller than it in LHS
  if (offset < 0) return false;

  auto it = buffer_indices_.find(lhs->buffer);
  if (it == buffer_indices_.end()) {
    // Update base indices for the buffer, this can only happen if it is visiting the scope block.
    ICHECK(is_scope_block);
    std::vector<PrimExpr> indices_base;
    indices_base.reserve(lhs->region.size());
    for (int i = 0; i < offset; i++) {
      // High-dim region must be element-wise
      if (!is_one(lhs->region[i]->extent)) return false;
      indices_base.emplace_back(lhs->region[i]->min);
    }
    for (size_t i = 0; i < rhs->region.size(); i++) {
      // save base index
      indices_base.emplace_back(lhs->region[i + offset]->min);
      // check extent match
      if (!analyzer_.CanProveEqual(lhs->region[i + offset]->extent, rhs->region[i]->extent)) {
        return false;
      }
    }
    buffer_indices_.emplace(lhs->buffer, std::move(indices_base));
  } else {
    // Check the base indices are consistent.
    const std::vector<PrimExpr>& indices_base = it->second;
    for (int i = 0; i < offset; i++) {
      // High-dim region must be element-wise
      if (!is_one(lhs->region[i]->extent)) return false;
      if (!analyzer_.CanProveEqual(indices_base[i], lhs->region[i]->min)) return false;
    }
    for (size_t i = 0; i < rhs->region.size(); i++) {
      // check extent match
      if (!analyzer_.CanProveEqual(lhs->region[i + offset]->extent, rhs->region[i]->extent)) {
        return false;
      }
      PrimExpr normalized_lhs_min = (lhs->region[i + offset]->min - indices_base[i + offset]);
      if (!analyzer_.CanProveEqual(normalized_lhs_min, rhs->region[i]->min)) {
        return false;
      }
    }
  }
  return true;
}

// Comparator for BufferStoreNode and BufferLoadNode
template <typename T>
bool TensorizeComparator::CompareBufferAccess(const T* lhs, const T* rhs) {
  if (!CompareBuffer(lhs->buffer, rhs->buffer)) return false;
  int offset = static_cast<int>(lhs->indices.size()) - static_cast<int>(rhs->indices.size());
  if (offset < 0) return false;
  auto it = buffer_indices_.find(lhs->buffer);
  ICHECK(it != buffer_indices_.end());
  const std::vector<PrimExpr>& indices_base = (*it).second;
  ICHECK_EQ(indices_base.size(), rhs->indices.size() + offset);
  for (size_t i = 0; i < rhs->indices.size(); i++) {
    PrimExpr normalized_lhs_index = lhs->indices[i + offset] - indices_base[i + offset];
    if (!analyzer_.CanProveEqual(normalized_lhs_index, rhs->indices[i])) {
      if (assert_mode_) {
        std::ostringstream os;
        os << "Buffer indices mismatch: " << lhs->indices[i + offset] << " vs " << rhs->indices[i];
        EmitError(os.str());
      }
      return false;
    }
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

bool TensorizeComparator::CompareIterVar(const IterVar& lhs, const IterVar& rhs) {
  return DefEqual(lhs->var, rhs->var) && lhs->iter_type == rhs->iter_type;
}

void TensorizeComparator::EmitError(const std::string& error_message) {
  error_messages_.push_back(error_message);
}

}  // namespace tir
}  // namespace tvm
