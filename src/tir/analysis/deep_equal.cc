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
 * \file tir/analysis/deep_equal.cc
 * \brief Deep equality checking.
 */
#include <tvm/node/object_path.h>
#include <tvm/node/reflection.h>
#include <tvm/node/structural_equal.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>  // For the class StmtExprMutator
namespace tvm {
namespace tir {

class SortExprByHashVisitor : public ExprVisitor {
 public:
  void VisitExpr_(const VarNode* op) final;
  void VisitExpr_(const SizeVarNode* op) final;
  void VisitExpr_(const LoadNode* op) final;
  void VisitExpr_(const BufferLoadNode* op) final;
  void VisitExpr_(const ProducerLoadNode* op) final;
  void VisitExpr_(const LetNode* op) final;
  void VisitExpr_(const CallNode* op) final;
  void VisitExpr_(const AddNode* op) final;
  void VisitExpr_(const SubNode* op) final;
  void VisitExpr_(const MulNode* op) final;
  void VisitExpr_(const DivNode* op) final;
  void VisitExpr_(const ModNode* op) final;
  void VisitExpr_(const FloorDivNode* op) final;
  void VisitExpr_(const FloorModNode* op) final;
  void VisitExpr_(const MinNode* op) final;
  void VisitExpr_(const MaxNode* op) final;
  void VisitExpr_(const EQNode* op) final;
  void VisitExpr_(const NENode* op) final;
  void VisitExpr_(const LTNode* op) final;
  void VisitExpr_(const LENode* op) final;
  void VisitExpr_(const GTNode* op) final;
  void VisitExpr_(const GENode* op) final;
  void VisitExpr_(const AndNode* op) final;
  void VisitExpr_(const OrNode* op) final;
  void VisitExpr_(const ReduceNode* op) final;
  void VisitExpr_(const CastNode* op) final;
  void VisitExpr_(const NotNode* op) final;
  void VisitExpr_(const SelectNode* op) final;
  void VisitExpr_(const RampNode* op) final;
  void VisitExpr_(const BroadcastNode* op) final;
  void VisitExpr_(const ShuffleNode* op) final;
  void VisitExpr_(const IntImmNode* op) final;
  void VisitExpr_(const FloatImmNode* op) final;
  void VisitExpr_(const StringImmNode* op) final;
  void VisitExpr_(const AnyNode* op) final;

  std::vector<std::pair<int, std::vector<PrimExpr>>> op_stack;
  int cur_max_tree_idx = 0;
  int pre_max_tree_idx = 0;

 private:
  std::string pre_bin_op = "null";
  int stack_idx = 0;
  int cur_tree_idx = 0;
};

#define TVM_DEFINE_BIN_OP_SORT_BY_HASH_VISITOR(OpName)                 \
  void SortExprByHashVisitor::VisitExpr_(const OpName* op) {           \
    std::string cur_bin_op = op->_type_key;                            \
    std::string cur_pre_bin_op = pre_bin_op;                           \
    int cur_stack_idx = stack_idx;                                     \
    if (cur_bin_op != cur_pre_bin_op || cur_pre_bin_op == "null") {    \
      std::vector<PrimExpr> expr_stack;                                \
      if (cur_tree_idx + 1 > pre_max_tree_idx) {                       \
        return;                                                        \
      }                                                                \
      op_stack.emplace_back(std::make_pair(cur_tree_idx, expr_stack)); \
      cur_tree_idx += 1;                                               \
      cur_max_tree_idx = std::max(cur_max_tree_idx, cur_tree_idx);     \
      cur_stack_idx = op_stack.size();                                 \
      stack_idx = cur_stack_idx;                                       \
      cur_pre_bin_op = cur_bin_op;                                     \
      pre_bin_op = cur_pre_bin_op;                                     \
    }                                                                  \
    int cur_tree_idx_temp = cur_tree_idx;                              \
    if ((op->a).as<OpName>() == nullptr) {                             \
      op_stack[stack_idx - 1].second.emplace_back(op->a);              \
    }                                                                  \
    if ((op->b).as<OpName>() == nullptr) {                             \
      op_stack[stack_idx - 1].second.emplace_back(op->b);              \
    }                                                                  \
    this->VisitExpr(op->a);                                            \
    pre_bin_op = cur_pre_bin_op;                                       \
    stack_idx = cur_stack_idx;                                         \
    cur_tree_idx = cur_tree_idx_temp;                                  \
    this->VisitExpr(op->b);                                            \
    pre_bin_op = cur_pre_bin_op;                                       \
    stack_idx = cur_stack_idx;                                         \
    cur_tree_idx = cur_tree_idx_temp;                                  \
  }

#define TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_VISITOR(OpName)    \
  void SortExprByHashVisitor::VisitExpr_(const OpName* op) { \
    std::string cur_pre_bin_op = "null";                     \
    pre_bin_op = cur_pre_bin_op;                             \
    this->VisitExpr(op->a);                                  \
    pre_bin_op = cur_pre_bin_op;                             \
    this->VisitExpr(op->b);                                  \
  }

#define TVM_DEFINE_PASS_OP_SORT_BY_HASH_VISITOR(OpName) \
  void SortExprByHashVisitor::VisitExpr_(const OpName* op) { return; }

TVM_DEFINE_BIN_OP_SORT_BY_HASH_VISITOR(AddNode)
TVM_DEFINE_BIN_OP_SORT_BY_HASH_VISITOR(MulNode)
TVM_DEFINE_BIN_OP_SORT_BY_HASH_VISITOR(AndNode)
TVM_DEFINE_BIN_OP_SORT_BY_HASH_VISITOR(OrNode)

TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_VISITOR(SubNode)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_VISITOR(DivNode)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_VISITOR(ModNode)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_VISITOR(FloorDivNode)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_VISITOR(FloorModNode)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_VISITOR(MinNode)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_VISITOR(MaxNode)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_VISITOR(EQNode)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_VISITOR(NENode)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_VISITOR(LTNode)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_VISITOR(LENode)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_VISITOR(GTNode)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_VISITOR(GENode)

TVM_DEFINE_PASS_OP_SORT_BY_HASH_VISITOR(VarNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_VISITOR(SizeVarNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_VISITOR(LoadNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_VISITOR(BufferLoadNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_VISITOR(ProducerLoadNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_VISITOR(LetNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_VISITOR(CallNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_VISITOR(ReduceNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_VISITOR(CastNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_VISITOR(NotNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_VISITOR(SelectNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_VISITOR(RampNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_VISITOR(BroadcastNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_VISITOR(ShuffleNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_VISITOR(IntImmNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_VISITOR(FloatImmNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_VISITOR(StringImmNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_VISITOR(AnyNode)

class SortExprByHashMutator : public StmtExprMutator {
 public:
  void Init() {
    pre_bin_op = "null";
    stack_idx = 0;
    cur_tree_idx = 0;
    full_stack_size = 0;
  }

  PrimExpr Rewrite(const PrimExpr& op) {
    Init();
    SortExprByHashVisitor sort_visitor;
    sort_visitor.pre_max_tree_idx = pre_max_tree_idx;
    sort_visitor(op);
    for (auto& stack_pair : sort_visitor.op_stack) {
      if (stack_pair.first == sort_visitor.cur_max_tree_idx - 1) {
        std::sort(stack_pair.second.begin(), stack_pair.second.end(),
                  [](PrimExpr expr_a, PrimExpr expr_b) {
                    int64_t hash_a = tvm::StructuralHash()(expr_a);
                    int64_t hash_b = tvm::StructuralHash()(expr_b);
                    return hash_a < hash_b;
                  });
      }
    }
    op_stack.swap(sort_visitor.op_stack);
    pre_max_tree_idx = sort_visitor.cur_max_tree_idx;
    PrimExpr result = StmtExprMutator::VisitExpr(op);
    pre_max_tree_idx = sort_visitor.cur_max_tree_idx - 1;
    return result;
  }

  PrimExpr VisitExpr_(const VarNode* op) final;
  PrimExpr VisitExpr_(const SizeVarNode* op) final;
  PrimExpr VisitExpr_(const LoadNode* op) final;
  PrimExpr VisitExpr_(const BufferLoadNode* op) final;
  PrimExpr VisitExpr_(const ProducerLoadNode* op) final;
  PrimExpr VisitExpr_(const LetNode* op) final;
  PrimExpr VisitExpr_(const CallNode* op) final;
  PrimExpr VisitExpr_(const AddNode* op) final;
  PrimExpr VisitExpr_(const SubNode* op) final;
  PrimExpr VisitExpr_(const MulNode* op) final;
  PrimExpr VisitExpr_(const DivNode* op) final;
  PrimExpr VisitExpr_(const ModNode* op) final;
  PrimExpr VisitExpr_(const FloorDivNode* op) final;
  PrimExpr VisitExpr_(const FloorModNode* op) final;
  PrimExpr VisitExpr_(const MinNode* op) final;
  PrimExpr VisitExpr_(const MaxNode* op) final;
  PrimExpr VisitExpr_(const EQNode* op) final;
  PrimExpr VisitExpr_(const NENode* op) final;
  PrimExpr VisitExpr_(const LTNode* op) final;
  PrimExpr VisitExpr_(const LENode* op) final;
  PrimExpr VisitExpr_(const GTNode* op) final;
  PrimExpr VisitExpr_(const GENode* op) final;
  PrimExpr VisitExpr_(const AndNode* op) final;
  PrimExpr VisitExpr_(const OrNode* op) final;
  PrimExpr VisitExpr_(const ReduceNode* op) final;
  PrimExpr VisitExpr_(const CastNode* op) final;
  PrimExpr VisitExpr_(const NotNode* op) final;
  PrimExpr VisitExpr_(const SelectNode* op) final;
  PrimExpr VisitExpr_(const RampNode* op) final;
  PrimExpr VisitExpr_(const BroadcastNode* op) final;
  PrimExpr VisitExpr_(const ShuffleNode* op) final;
  PrimExpr VisitExpr_(const IntImmNode* op) final;
  PrimExpr VisitExpr_(const FloatImmNode* op) final;
  PrimExpr VisitExpr_(const StringImmNode* op) final;
  PrimExpr VisitExpr_(const AnyNode* op) final;

  int pre_max_tree_idx = 0;

 private:
  std::vector<std::pair<int, std::vector<PrimExpr>>> op_stack;
  std::string pre_bin_op = "null";
  int stack_idx = 0;
  int full_stack_size = 0;
  int cur_tree_idx = 0;
};

#define TVM_DEFINE_BIN_OP_SORT_BY_HASH_MUTATOR(Op)                                            \
  PrimExpr SortExprByHashMutator::VisitExpr_(const Op##Node* op) {                            \
    std::string cur_bin_op = op->_type_key;                                                   \
    std::string cur_pre_bin_op = pre_bin_op;                                                  \
    int cur_stack_idx = stack_idx;                                                            \
    if (cur_bin_op != cur_pre_bin_op) {                                                       \
      if (cur_tree_idx + 1 > pre_max_tree_idx) {                                              \
        return GetRef<PrimExpr>(op);                                                          \
      }                                                                                       \
      if (cur_tree_idx + 1 == pre_max_tree_idx) {                                             \
        PrimExpr expr_sorted =                                                                \
            Op(op_stack[full_stack_size].second[0], op_stack[full_stack_size].second[1]);     \
        for (std::size_t idx = 0; idx < op_stack[full_stack_size].second.size() - 2; idx++) { \
          expr_sorted = Op(expr_sorted, op_stack[full_stack_size].second[idx + 2]);           \
        }                                                                                     \
        full_stack_size += 1;                                                                 \
        cur_stack_idx = full_stack_size;                                                      \
        cur_tree_idx += 1;                                                                    \
        return expr_sorted;                                                                   \
      }                                                                                       \
      full_stack_size += 1;                                                                   \
      cur_stack_idx = full_stack_size;                                                        \
      cur_tree_idx += 1;                                                                      \
      stack_idx = cur_stack_idx;                                                              \
      cur_pre_bin_op = cur_bin_op;                                                            \
      pre_bin_op = cur_pre_bin_op;                                                            \
    }                                                                                         \
    PrimExpr a;                                                                               \
    PrimExpr b;                                                                               \
    int cur_tree_idx_temp = cur_tree_idx;                                                     \
    GetRef<PrimExpr>(op);                                                                     \
    if (!((op->a).as<Op##Node>() == nullptr && cur_tree_idx == pre_max_tree_idx)) {           \
      a = this->VisitExpr(op->a);                                                             \
    } else {                                                                                  \
      a = op->a;                                                                              \
    }                                                                                         \
    pre_bin_op = cur_pre_bin_op;                                                              \
    stack_idx = cur_stack_idx;                                                                \
    cur_tree_idx = cur_tree_idx_temp;                                                         \
    if (!((op->b).as<Op##Node>() == nullptr && cur_tree_idx == pre_max_tree_idx)) {           \
      b = this->VisitExpr(op->b);                                                             \
    } else {                                                                                  \
      b = op->b;                                                                              \
    }                                                                                         \
    pre_bin_op = cur_pre_bin_op;                                                              \
    stack_idx = cur_stack_idx;                                                                \
    cur_tree_idx = cur_tree_idx_temp;                                                         \
    return Op(a, b);                                                                          \
  }

#define TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_MUTATOR(Op)              \
  PrimExpr SortExprByHashMutator::VisitExpr_(const Op##Node* op) { \
    std::string cur_pre_bin_op = "null";                           \
    pre_bin_op = cur_pre_bin_op;                                   \
    PrimExpr a = this->VisitExpr(op->a);                           \
    pre_bin_op = cur_pre_bin_op;                                   \
    PrimExpr b = this->VisitExpr(op->b);                           \
    return Op(a, b);                                               \
  }

#define TVM_DEFINE_PASS_OP_SORT_BY_HASH_MUTATOR(OpName) \
  PrimExpr SortExprByHashMutator::VisitExpr_(const OpName* op) { return GetRef<PrimExpr>(op); }

TVM_DEFINE_BIN_OP_SORT_BY_HASH_MUTATOR(Add)
TVM_DEFINE_BIN_OP_SORT_BY_HASH_MUTATOR(Mul)
TVM_DEFINE_BIN_OP_SORT_BY_HASH_MUTATOR(And)
TVM_DEFINE_BIN_OP_SORT_BY_HASH_MUTATOR(Or)

TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_MUTATOR(Sub)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_MUTATOR(Div)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_MUTATOR(Mod)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_MUTATOR(FloorDiv)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_MUTATOR(FloorMod)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_MUTATOR(Min)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_MUTATOR(Max)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_MUTATOR(EQ)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_MUTATOR(NE)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_MUTATOR(LT)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_MUTATOR(LE)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_MUTATOR(GT)
TVM_DEFINE_BIN_OP_NO_SORT_BY_HASH_MUTATOR(GE)

TVM_DEFINE_PASS_OP_SORT_BY_HASH_MUTATOR(VarNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_MUTATOR(SizeVarNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_MUTATOR(LoadNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_MUTATOR(BufferLoadNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_MUTATOR(ProducerLoadNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_MUTATOR(LetNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_MUTATOR(CallNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_MUTATOR(ReduceNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_MUTATOR(CastNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_MUTATOR(NotNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_MUTATOR(SelectNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_MUTATOR(RampNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_MUTATOR(BroadcastNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_MUTATOR(ShuffleNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_MUTATOR(IntImmNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_MUTATOR(FloatImmNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_MUTATOR(StringImmNode)
TVM_DEFINE_PASS_OP_SORT_BY_HASH_MUTATOR(AnyNode)

class DeepCmpSEqualHandler : public SEqualReducer::Handler {
 public:
  // use direct recursion.
  bool SEqualReduce(const ObjectRef& lhs, const ObjectRef& rhs, bool map_free_vars,
                    const Optional<ObjectPathPair>&) final {
    if (lhs.same_as(rhs)) return true;
    if (!lhs.defined() && rhs.defined()) return false;
    if (!rhs.defined() && lhs.defined()) return false;
    if (lhs->type_index() != rhs->type_index()) return false;
    return vtable_->SEqualReduce(lhs.get(), rhs.get(), SEqualReducer(this, nullptr, false)) &&
           !fail_;
  }

  void DeferFail(const ObjectPathPair&) final { fail_ = true; }

  ObjectRef MapLhsToRhs(const ObjectRef& lhs) final { return ObjectRef(nullptr); }
  void MarkGraphNode() final {}

 private:
  // reflection vtable
  ReflectionVTable* vtable_ = ReflectionVTable::Global();
  bool fail_ = false;
};

bool ExprDeepEqual::operator()(const PrimExpr& lhs, const PrimExpr& rhs) const {
  // quick path
  if (lhs.same_as(rhs)) return true;
  if (!lhs.defined() && rhs.defined()) return false;
  if (!rhs.defined() && lhs.defined()) return false;
  if (lhs->type_index() != rhs->type_index()) return false;
  if (auto* plhs = lhs.as<IntImmNode>()) {
    auto* prhs = rhs.as<IntImmNode>();
    return plhs->dtype == prhs->dtype && plhs->value == prhs->value;
  }
  if (lhs.as<AnyNode>()) {
    return false;
  }
  return DeepCmpSEqualHandler().SEqualReduce(lhs, rhs, false, NullOpt);
}

class CommutativeDeepEqual : public ExprDeepEqual {
 public:
  bool operator()(const PrimExpr& lhs, const PrimExpr& rhs) const {
    // quick path
    if (lhs.same_as(rhs)) return true;
    if (!lhs.defined() && rhs.defined()) return false;
    if (!rhs.defined() && lhs.defined()) return false;
    if (lhs->type_index() != rhs->type_index()) return false;
    if (auto* plhs = lhs.as<IntImmNode>()) {
      auto* prhs = rhs.as<IntImmNode>();
      return plhs->dtype == prhs->dtype && plhs->value == prhs->value;
    }
    if (lhs.as<AnyNode>()) {
      return false;
    }
    SortExprByHashMutator sort;
    sort.pre_max_tree_idx = INT32_MAX;
    auto sort_lhs = sort.Rewrite(lhs);
    while (sort.pre_max_tree_idx != -1) {
      sort_lhs = sort.Rewrite(sort_lhs);
    }
    sort.pre_max_tree_idx = INT32_MAX;
    auto sort_rhs = sort.Rewrite(rhs);
    while (sort.pre_max_tree_idx != -1) {
      sort_rhs = sort.Rewrite(sort_rhs);
    }
    return DeepCmpSEqualHandler().SEqualReduce(sort_lhs, sort_rhs, false, NullOpt);
  }
};

TVM_REGISTER_GLOBAL("tir.analysis.expr_deep_equal")
    .set_body_typed([](const PrimExpr& lhs, const PrimExpr& rhs) {
      return ExprDeepEqual()(lhs, rhs);
    });

TVM_REGISTER_GLOBAL("tir.analysis.commutative_deep_equal")
    .set_body_typed([](const PrimExpr& lhs, const PrimExpr& rhs) {
      return CommutativeDeepEqual()(lhs, rhs);
    });

}  // namespace tir
}  // namespace tvm
