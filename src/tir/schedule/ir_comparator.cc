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

#include "../../arith/scalable_expression.h"

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
    os << "The stmt {0} doesn't match the tensor intrin\nThe pattern attempting to be matched:\n"
       << lhs_stmt_ << "\nDoes not match the tensorize description:\n"
       << rhs_stmt_ << '\n';
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
  bool equal = n.same_as(other) ||
               ((n->type_index() == other->type_index()) &&
                n.dtype().code() == other.dtype().code() && ExprComparator::VisitExpr(n, other)) ||
               (tvm::arith::ContainsVscaleCall(n) && analyzer_.CanProveEqual(n, other));

  if (!equal && assert_mode_) {
    std::ostringstream os;
    os << "Expression mismatch: " << n << " vs " << other;
    EmitError(os.str());
  }
  return equal;
}

bool TensorizeComparator::VisitExpr_(const CallNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<CallNode>();
  if (!rhs->op.same_as(op->op)) return false;
  if (op->dtype.code() != rhs->dtype.code()) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "CallNode data type codes do not match: op->dtype.code()=" << op->dtype.code()
         << " vs rhs->dtype.code()=" << rhs->dtype.code();
      EmitError(os.str());
    }
    return false;
  }
  if (!CompareArray(op->args, rhs->args, &TensorizeComparator::VisitExpr)) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "CallNode iter_values do not match: op->iter_values=" << op->args
         << " vs rhs->iter_values=" << rhs->args;
      EmitError(os.str());
    }
    return false;
  }
  return true;
}

bool TensorizeComparator::VisitStmt_(const ForNode* op, const Stmt& other) {
  const auto* rhs = other.as<ForNode>();
  if (!DefEqual(op->loop_var, rhs->loop_var)) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "ForNode loop vars do not match: op->loop_var=" << op->loop_var
         << " vs rhs->loop_var=" << rhs->loop_var;
      EmitError(os.str());
    }
    return false;
  }
  if (!VisitExpr(op->min, rhs->min)) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "ForNode min values do not match: op->min=" << op->min << " vs rhs->min=" << rhs->min;
      EmitError(os.str());
    }
    return false;
  }
  if (!VisitExpr(op->extent, rhs->extent)) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "ForNode extent values do not match: op->extent=" << op->extent
         << " vs rhs->extent=" << rhs->extent;
      EmitError(os.str());
    }
    return false;
  }
  if (op->thread_binding.defined() != rhs->thread_binding.defined()) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "ForNode thread_bindings do not match: op->thread_binding.defined()="
         << op->thread_binding.defined()
         << " vs rhs->thread_binding.defined()=" << rhs->thread_binding.defined();
      EmitError(os.str());
    }
    return false;
  }
  if (op->thread_binding.defined() &&
      !VisitExpr(op->thread_binding.value(), rhs->thread_binding.value())) {
    return false;
  }
  if (op->kind != rhs->kind) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "ForNode kinds do not match: op->kind=" << op->kind << " vs rhs->kind=" << rhs->kind;
      EmitError(os.str());
    }
    return false;
  }
  if (!CompareAnnotationMap(op->annotations, rhs->annotations)) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "ForNode annotation maps do not match: op->annotations=" << op->annotations
         << " vs rhs->annotations=" << rhs->annotations;
      EmitError(os.str());
    }
    return false;
  }
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
      if (assert_mode_) {
        std::ostringstream os;
        os << "BlockRealizeNode iter_values do not match: op->iter_values=" << op->iter_values
           << " vs rhs->iter_values=" << rhs->iter_values;
        EmitError(os.str());
      }
      return false;
    }
  }
  return VisitExpr(op->predicate, rhs->predicate) && VisitStmt(op->block, rhs->block);
}

bool TensorizeComparator::VisitStmt_(const BlockNode* op, const Stmt& other) {
  const auto* rhs = other.as<BlockNode>();
  for (const IterVar& iter : op->iter_vars) {
    lhs_analyzer_.Bind(iter->var, iter->dom);
  }
  // Check block equality.
  // All iter vars and buffer regions including the order should match.
  // When checking iter vars, DefEqual is used to remap variables.
  if (!is_scope_block) {
    if (!CompareArray(op->iter_vars, rhs->iter_vars, &TensorizeComparator::CompareIterVar)) {
      if (assert_mode_) {
        std::ostringstream os;
        os << "BlockNode iter_vars do not match: op->alloc_buffers=" << op->iter_vars
           << " vs rhs->alloc_buffers=" << rhs->iter_vars;
        EmitError(os.str());
      }
      return false;
    }
    if (!CompareArray(op->alloc_buffers, rhs->alloc_buffers, &TensorizeComparator::CompareBuffer)) {
      if (assert_mode_) {
        std::ostringstream os;
        os << "BlockNode alloc_buffers do not match: op->alloc_buffers=" << op->alloc_buffers
           << " vs rhs->alloc_buffers=" << rhs->alloc_buffers;
        EmitError(os.str());
      }
      return false;
    }
  }
  if (!CompareArray(op->writes, rhs->writes, &TensorizeComparator::CompareBufferRegion)) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "BlockNode write buffers do not match: op->writes=" << op->writes
         << " vs rhs->writes=" << rhs->writes;
      EmitError(os.str());
    }
    return false;
  }
  if (!CompareArray(op->reads, rhs->reads, &TensorizeComparator::CompareBufferRegion)) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "BlockNode read buffers regions do not match: op->reads=" << op->reads
         << " vs rhs->reads=" << rhs->reads;
      EmitError(os.str());
    }
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
  if (op->value != rhs->value) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "IntImmNode values do not match: op->value=" << op->value
         << " vs rhs->value=" << rhs->value;
      EmitError(os.str());
    }
    return false;
  }
  return true;
}

bool TensorizeComparator::VisitExpr_(const FloatImmNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<FloatImmNode>();
  if (op->value != rhs->value) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "FloatImmNode values do not match: op->value=" << op->value
         << " vs rhs->value=" << rhs->value;
      EmitError(os.str());
    }
    return false;
  }
  return true;
}

bool TensorizeComparator::VisitExpr_(const CastNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<CastNode>();
  return VisitExpr(op->value, rhs->value);
}

bool TensorizeComparator::VisitExpr_(const VarNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<VarNode>();
  auto lhs = GetRef<Var>(op);
  if (lhs.same_as(other)) return true;
  if (op->dtype.code() != rhs->dtype.code()) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "VarNode data type codes do not match: op->dtype.code()=" << op->dtype.code()
         << " vs rhs->dtype.code()=" << rhs->dtype.code();
      EmitError(os.str());
    }
    return false;
  }
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
  // Cast if necessary. This allows the workload and the tensor intrin to have different dtypes in
  // the indices.
  analyzer_.Bind(lhs, cast(lhs.dtype(), rhs));
  return true;
}

bool TensorizeComparator::CompareAnnotation(const std::pair<String, ObjectRef>& lhs,
                                            const std::pair<String, ObjectRef>& rhs) {
  if (lhs.first != rhs.first) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "CompareAnnotation key mismatch: lhs.first=" << lhs.first
         << " vs rhs.first=" << rhs.first;
      EmitError(os.str());
    }
    return false;
  }
  return VisitExpr(Downcast<PrimExpr>(lhs.second), Downcast<PrimExpr>(rhs.second));
}

bool TensorizeComparator::CompareAnnotationMap(const Map<String, ObjectRef>& lhs,
                                               const Map<String, ObjectRef>& rhs) {
  if (lhs.same_as(rhs)) return true;
  if (lhs.size() != rhs.size()) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "CompareAnnotationMap size mismatch: lhs.size()=" << lhs.size()
         << " vs rhs.size()=" << rhs.size();
      EmitError(os.str());
    }
    return false;
  }

  auto sort_map =
      [](const Map<String, ObjectRef>& map) -> std::vector<std::pair<String, ObjectRef>> {
    std::vector<std::pair<String, ObjectRef>> ret(map.begin(), map.end());
    sort(ret.begin(), ret.end());
    return ret;
  };

  std::vector<std::pair<String, ObjectRef>> lhs_array = sort_map(lhs);
  std::vector<std::pair<String, ObjectRef>> rhs_array = sort_map(rhs);

  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!CompareAnnotation(lhs_array[i], rhs_array[i])) {
      if (assert_mode_) {
        std::ostringstream os;
        os << "CompareAnnotationMap annotations mismatch within AnnotationMap.";
        EmitError(os.str());
      }
      return false;
    }
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
    } else {
      if (assert_mode_) {
        std::ostringstream os;
        os << "CompareBuffer buffer mismatch. data: " << lhs->data << " vs " << rhs->data
           << ", dtypes: " << lhs->dtype << " vs " << rhs->dtype << ", scope(): " << lhs.scope()
           << " vs " << rhs.scope();
        EmitError(os.str());
      }
    }
  }
  return equal;
}

bool TensorizeComparator::CompareBufferRegion(const BufferRegion& lhs, const BufferRegion& rhs) {
  if (!CompareBuffer(lhs->buffer, rhs->buffer)) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "CompareBufferRegion returning false due to buffer mismatch: lhs->buffer="
         << lhs->buffer << " vs rhs->buffer=" << rhs->buffer;
      EmitError(os.str());
    }
    return false;
  }
  int offset = static_cast<int>(lhs->region.size()) - static_cast<int>(rhs->region.size());
  // Number of indices in RHS (desc of the tensor intrinsic) must be smaller than it in LHS
  if (offset < 0) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "CompareBufferRegion returning false because buffer region sizes do not match: "
            "lhs->region.size()="
         << lhs->region.size() << " vs rhs->region.size()=" << rhs->region.size();
      EmitError(os.str());
    }
    return false;
  }

  auto it = buffer_indices_.find(lhs->buffer);
  if (it == buffer_indices_.end()) {
    // Update base indices for the buffer, this can only happen if it is visiting the scope block.
    ICHECK(is_scope_block);
    std::vector<PrimExpr> indices_base;
    indices_base.reserve(lhs->region.size());
    for (int i = 0; i < offset; i++) {
      // High-dim region must be element-wise
      if (!is_one(lhs->region[i]->extent)) {
        if (assert_mode_) {
          std::ostringstream os;
          os << "CompareBufferRegion returning false because buffer extent high-dim region must be "
                "element-wise. lhs->region[i]->extent="
             << lhs->region[i]->extent;
          EmitError(os.str());
        }
        return false;
      }
      indices_base.emplace_back(lhs->region[i]->min);
    }
    for (size_t i = 0; i < rhs->region.size(); i++) {
      // save base index
      indices_base.emplace_back(lhs->region[i + offset]->min);
      // check extent match
      if (!analyzer_.CanProveEqual(lhs->region[i + offset]->extent, rhs->region[i]->extent)) {
        if (assert_mode_) {
          std::ostringstream os;
          os << "CompareBufferRegion buffer extent mismatch: lhs->region[i + offset]="
             << lhs->region[i + offset] << " vs rhs->region[i]=" << rhs->region[i];
          EmitError(os.str());
        }
        return false;
      }
    }
    buffer_indices_.emplace(lhs->buffer, std::move(indices_base));
  } else {
    // Check the base indices are consistent.
    const std::vector<PrimExpr>& indices_base = it->second;
    for (int i = 0; i < offset; i++) {
      // High-dim region must be element-wise
      if (!is_one(lhs->region[i]->extent)) {
        if (assert_mode_) {
          std::ostringstream os;
          os << "CompareBufferRegion returning false because buffer extent high-dim region must be "
                "element-wise. lhs->region[i]->extent="
             << lhs->region[i]->extent;
          EmitError(os.str());
        }
        return false;
      }
      if (!lhs_analyzer_.CanProveEqual(indices_base[i], lhs->region[i]->min)) {
        if (assert_mode_) {
          std::ostringstream os;
          os << "Buffer base index consistency check failed due to unequal index base: "
                "indices_base[i]="
             << indices_base[i] << " vs lhs->region[i]->min=" << lhs->region[i]->min;
          EmitError(os.str());
        }
        return false;
      }
    }
    for (size_t i = 0; i < rhs->region.size(); i++) {
      // check extent match
      if (!analyzer_.CanProveEqual(lhs->region[i + offset]->extent, rhs->region[i]->extent)) {
        if (assert_mode_) {
          std::ostringstream os;
          os << "CompareBufferRegion buffer region extent mismatch. lhs->region[i + offset]="
             << lhs->region[i + offset] << " vs rhs->region[i]=" << rhs->region[i];
          EmitError(os.str());
        }
        return false;
      }
      PrimExpr normalized_lhs_min =
          lhs_analyzer_.Simplify((lhs->region[i + offset]->min - indices_base[i + offset]));
      if (!analyzer_.CanProveEqual(normalized_lhs_min, rhs->region[i]->min)) {
        if (assert_mode_) {
          std::ostringstream os;
          os << "CompareBufferRegion buffer region min mismatch. lhs->region[i + offset]="
             << lhs->region[i + offset] << " vs rhs->region[i]=" << rhs->region[i];
          EmitError(os.str());
        }
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
  if (offset < 0) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "CompareBufferAccess returning false because buffer indices sizes do not match: "
            "lhs->indices.size()="
         << lhs->indices.size() << " vs rhs->indices.size()=" << rhs->indices.size();
      EmitError(os.str());
    }
    return false;
  }
  auto it = buffer_indices_.find(lhs->buffer);
  ICHECK(it != buffer_indices_.end());
  const std::vector<PrimExpr>& indices_base = (*it).second;
  ICHECK_EQ(indices_base.size(), rhs->indices.size() + offset);
  for (size_t i = 0; i < rhs->indices.size(); i++) {
    PrimExpr normalized_lhs_index = lhs->indices[i + offset] - indices_base[i + offset];
    if (!analyzer_.CanProveEqual(normalized_lhs_index, rhs->indices[i])) {
      if (assert_mode_) {
        std::ostringstream os;
        os << "CompareBufferAccess buffer indices mismatch. lhs->indices[i + offset]="
           << lhs->indices[i + offset] << " vs rhs->indices[i]=" << rhs->indices[i];
        EmitError(os.str());
      }
      return false;
    }
  }
  return true;
}

template <typename T, typename Self, typename F>
bool TensorizeComparator::CompareArray(const Array<T>& lhs, const Array<T>& rhs, F Self::*cmp) {
  if (lhs.same_as(rhs)) return true;
  if (lhs.size() != rhs.size()) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "CompareArray array size mismatch. lhs.size()=" << lhs.size()
         << " vs rhs.size()=" << rhs.size();
      EmitError(os.str());
    }
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!(static_cast<Self*>(this)->*cmp)(lhs[i], rhs[i])) return false;
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

/******** AutoTensorize Extractor ********/

bool AutoTensorizeComparator::VisitExprDefault_(const Object* op, const PrimExpr& other) {
  return false;
}

bool AutoTensorizeComparator::VisitStmtDefault_(const Object* op, const Stmt& other) {
  return false;
}

bool AutoTensorizeComparator::VisitStmt_(const BlockNode* op, const Stmt& other) {
  const auto* rhs = other.as<BlockNode>();
  // Check block equality.
  // All iter vars and buffer regions including the order should match.
  // When checking iter vars, DefEqual is used to remap variables.
  if (!is_scope_block) {
    if (!CompareArray(op->iter_vars, rhs->iter_vars, &AutoTensorizeComparator::CompareIterVar)) {
      return false;
    }
    if (!CompareAnnotationMap(op->annotations, rhs->annotations)) {
      return false;
    }
    if (!CompareArray(op->alloc_buffers, rhs->alloc_buffers,
                      &AutoTensorizeComparator::CompareBuffer)) {
      return false;
    }
    for (const IterVar& block_iter : op->iter_vars) {
      inner_iter_dom_map_.Set(block_iter->var, arith::IntSet::FromRange(block_iter->dom));
    }
  } else {
    auto collect_iter = [&](const BlockNode* op, std::vector<IterVar>& iters) -> bool {
      for (const auto& iter : op->iter_vars) {
        analyzer_.Bind(iter->var, iter->dom);
        if (iter->iter_type == IterVarType::kDataPar ||
            iter->iter_type == IterVarType::kCommReduce) {
          iters.push_back(iter);
        } else {
          return false;
        }
      }
      return true;
    };
    if (!collect_iter(op, lhs_iters_)) {
      return false;
    }
    if (!collect_iter(rhs, rhs_iters_)) {
      return false;
    }
  }
  is_scope_block = false;
  return VisitStmt(op->body, rhs->body);
}

bool AutoTensorizeComparator::CompareBuffer(const Buffer& lhs, const Buffer& rhs) {
  if (lhs.same_as(rhs)) return true;
  auto it = rhs_buffer_map_.find(rhs);
  bool equal;
  if (it != rhs_buffer_map_.end()) {
    equal = (*it).second.same_as(lhs);
  } else {
    // Remap both buffer itself and buffer data, skip buffer shape and scope
    equal = DefEqual(lhs->data, rhs->data) && lhs->dtype == rhs->dtype;
    if (equal) {
      rhs_buffer_map_[rhs] = lhs;
      lhs_buffer_map_[lhs] = rhs;
    }
  }
  return equal;
}

bool AutoTensorizeComparator::VisitStmt_(const BufferStoreNode* op, const Stmt& other) {
  const auto* rhs = other.as<BufferStoreNode>();
  return CompareBufferAccess(op, rhs) && VisitExpr(op->value, rhs->value);
}

bool AutoTensorizeComparator::VisitExpr_(const BufferLoadNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<BufferLoadNode>();
  return CompareBufferAccess(op, rhs);
}

template <typename T>
bool AutoTensorizeComparator::CompareBufferAccess(const T* lhs, const T* rhs) {
  if (!CompareBuffer(lhs->buffer, rhs->buffer)) return false;
  auto it_lhs = lhs_buffer_indices_map_.find(lhs->buffer);
  if (it_lhs == lhs_buffer_indices_map_.end()) {
    if (rhs_buffer_indices_map_.find(rhs->buffer) != rhs_buffer_indices_map_.end()) {
      return false;
    }
    std::vector<PrimExpr> lhs_indices;
    for (const PrimExpr& index : lhs->indices) {
      lhs_indices.push_back(SimplifyNonTrivialExpr(index, &analyzer_));
    }

    auto is_scalar_access = [](const Array<PrimExpr>& indices, PrimExpr index) {
      // Check if the indexing is of the form C[0]
      if (indices.size() > 1) return false;
      auto int_imm = index.template as<IntImmNode>();
      if (int_imm && int_imm->value == 0) return true;
      return false;
    };

    for (const auto& index : rhs->indices) {
      if (!index.template as<VarNode>() && !is_scalar_access(rhs->indices, index)) return false;
    }
    lhs_buffer_indices_map_[lhs->buffer] = lhs_indices;
    rhs_buffer_indices_map_[rhs->buffer] = rhs->indices;
  } else {
    auto it_rhs = rhs_buffer_indices_map_.find(rhs->buffer);
    if (it_rhs == rhs_buffer_indices_map_.end()) {
      return false;
    }
    auto indices_check = [&](const Array<PrimExpr>& indices,
                             const Array<PrimExpr>& old_indices) -> bool {
      if (indices.size() != old_indices.size()) {
        return false;
      }
      for (size_t i = 0; i < indices.size(); ++i) {
        if (!analyzer_.CanProveEqual(indices[i], old_indices[i])) {
          return false;
        }
      }
      return true;
    };
    if (!indices_check(lhs->indices, it_lhs->second)) return false;
    if (!indices_check(rhs->indices, it_rhs->second)) return false;
  }
  return true;
}

}  // namespace tir
}  // namespace tvm
