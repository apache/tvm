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
 * \file loop_vectorize.cc
 * \brief A tool to automatically vectorize a for loop
 */

#include "loop_vectorize.h"

#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>

#include <numeric>

#include "../../arith/int_operator.h"
#include "../../arith/ir_visitor_with_analyzer.h"
#include "../layout/layout.h"
#include "../layout/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

struct VectorizePlanResult {
  int vector_size;
  bool dynamic;
  PrimExpr condition;
};

class VectorizePlanner : public arith::IRVisitorWithAnalyzer {
 public:
  VectorizePlanner() = default;

  int Plan(const For& node) {
    this->operator()(node);
    // Always Enable vectorization
    // if (!has_nonlocal_memory_access_) return 1;
    return vector_size_;
  }

  bool GetDynamic() {
    return dynamic_;
  }

  PrimExpr GetCondition() {
    return condition_;
  }

 private:
  void VisitStmt_(const ForNode* node) final {
    inner_for_ = node;
    iter_map_.Set(node->loop_var, Range(node->min, node->extent));
    arith::IRVisitorWithAnalyzer::VisitStmt_(node);
  }

  void VisitExpr_(const BufferLoadNode* node) final {
    if (node->buffer.scope() == "shared" || node->buffer.scope() == "global" ||
        node->buffer.scope() == "shared.dyn")
      has_nonlocal_memory_access_ = true;
    if (node->buffer->shape.size() == 1 && node->buffer->shape[0].as<IntImmNode>()->value == 1) {
      // TODO(lei): This should be improved as 
      // constant buffer that tl hack to use as local register.
      return arith::IRVisitorWithAnalyzer::VisitExpr_(node);
    }
    UpdateVectorSize(node->indices, node->buffer);
    return arith::IRVisitorWithAnalyzer::VisitExpr_(node);
  }

  void VisitStmt_(const BufferStoreNode* node) final {
    if (node->buffer.scope() == "shared" || node->buffer.scope() == "global" ||
        node->buffer.scope() == "shared.dyn")
      has_nonlocal_memory_access_ = true;
    UpdateVectorSize(node->indices, node->buffer);
    return arith::IRVisitorWithAnalyzer::VisitStmt_(node);
  }

  void VisitStmt_(const IfThenElseNode* node) final {
    CheckConditionVectorized(node->condition);
    return arith::IRVisitorWithAnalyzer::VisitStmt_(node);
  }

  void VisitExpr_(const CallNode* node) final {
    if (node->op == builtin::if_then_else()) {
      CheckConditionVectorized(node->args[0]);
    } else if (node->op == builtin::call_extern()) {
      // do not vectorize extern calls
      vector_size_ = 1;
    }
    return arith::IRVisitorWithAnalyzer::VisitExpr_(node);
  }

  void CheckConditionVectorized(const PrimExpr& cond) {
    // TODO: perform some checks here
  }

  void UpdateVectorSize(const Array<PrimExpr> indices, const Buffer& buffer) {
    if (!inner_for_) return;
    auto extent_ptr = inner_for_->extent.as<IntImmNode>();
    if (!extent_ptr) return;

    const DataType& access_type = buffer->dtype;
    int max_vector_size = arith::ZeroAwareGCD(128 / access_type.bits(), extent_ptr->value);

    auto mod_set = analyzer_.modular_set(buffer->shape.back());
    // when dynamic shape like [m, k]: coeff=1, base=0, GCD will block conditionally tail vectorize
    if (buffer->shape.back().as<IntImmNode>()) {
      max_vector_size = arith::ZeroAwareGCD(max_vector_size, mod_set->coeff);
      max_vector_size = arith::ZeroAwareGCD(max_vector_size, mod_set->base);
      vector_size_ = arith::ZeroAwareGCD(max_vector_size, vector_size_);
      while (!IndiceCanVectorize(buffer.OffsetOf(indices).back(), inner_for_->loop_var,
                                inner_for_->extent, vector_size_, &analyzer_)) {
        vector_size_ /= 2;
      }
    } else if (vector_size_ <= vector_load_bits_max_ / buffer->dtype.bits()) {
      // dynamic shape load: get the vectorization condition
      dynamic_ = true;
      PrimExpr offset = buffer.OffsetOf(indices).back();
      condition_ = (FloorMod(offset, vector_size_) == 0);
    }
  }

  static const int vector_load_bits_max_ = 128;

  const ForNode* inner_for_;
  Map<Var, Range> iter_map_;
  bool has_nonlocal_memory_access_ = false;
  int vector_size_ = 128;
  // conditionally vectorize
  bool dynamic_ = false;
  PrimExpr condition_;
};

class VectorizeDynamicCallRemover : public StmtExprMutator {
  public:
    VectorizeDynamicCallRemover(Var inner_var, int vector_size):
      inner_var_(inner_var), vector_size_(vector_size) {}
  private:
    PrimExpr VisitExpr_(const CallNode* op) final {
      if (op->op.same_as(builtin::if_then_else())) {
        PrimExpr cond = this->VisitExpr(op->args[0]);
        Map<Var, PrimExpr> vmap;
        // Currently remove upper bound check
        vmap.Set(inner_var_, 0);
        cond = Substitute(cond, vmap);
        Array<PrimExpr> new_args{cond, op->args[1], op->args[2]};
        return Call(op->dtype, op->op, new_args, op->span);
      } else {
        // TODO: For other calls
        return GetRef<PrimExpr>(op);
      }
    }

    Var inner_var_;
    int vector_size_;
};

class VectorizeRewriter : public StmtExprMutator {
 public:
  VectorizeRewriter(VectorizePlanResult plan):
    vector_size_(plan.vector_size), condition_(plan.condition), dynamic_(plan.dynamic) {}

 private:
  Stmt VisitStmt_(const ForNode* node) final {
    inner_for_ = node;
    auto ret = StmtExprMutator::VisitStmt_(node);
    if (inner_for_ == node) {  // rewrite the innermost loop
      For fnode = ret.as<For>().value();
      auto old_var = fnode->loop_var;
      auto extent_ptr = as_const_int(fnode->extent);
      ICHECK(extent_ptr) << fnode->extent;
      int extent = *extent_ptr;
      ICHECK(extent % vector_size_ == 0);
      ICHECK(is_zero(fnode->min));
      if (!dynamic_) { // check dynamic shape
        if (extent == vector_size_) {
          fnode.CopyOnWrite()->kind = ForKind::kVectorized;
          return fnode;
        } else {
          Var inner_var = Var("vec");
          Var outer_var = Var(old_var->name_hint);
          Map<Var, PrimExpr> vmap;
          vmap.Set(fnode->loop_var, outer_var * vector_size_ + inner_var);
          Stmt body = Substitute(fnode->body, vmap);
          body = For(inner_var, 0, vector_size_, ForKind::kVectorized, body);
          body = For(outer_var, 0, extent / vector_size_, fnode->kind, body, fnode->thread_binding,
                    fnode->annotations, fnode->span);
          return body;
        }
      } else {
          Var inner_var = Var("vec");
          Var outer_var = Var(old_var->name_hint);
          Map<Var, PrimExpr> vmap;
          vmap.Set(fnode->loop_var, outer_var * vector_size_ + inner_var);
          Stmt body = Substitute(fnode->body, vmap);
          // add condition ifthenelse here
          Map<Var, PrimExpr> vmap_condition;
          vmap_condition.Set(fnode->loop_var, outer_var * vector_size_);
          PrimExpr condition = Substitute(condition_, vmap_condition);

          VectorizeDynamicCallRemover remover(inner_var, vector_size_);
          body = remover(body);

          For vectorize_for = For(inner_var, 0, vector_size_, ForKind::kVectorized, body);
          For serial_for = For(inner_var, 0, vector_size_, ForKind::kSerial, body);
          body = IfThenElse(condition, vectorize_for, serial_for);
          body = For(outer_var, 0, extent / vector_size_, fnode->kind, body, fnode->thread_binding,
                    fnode->annotations, fnode->span);
          return body;
      }
    } else {
      return ret;
    }
  }

  const ForNode* inner_for_;
  const int vector_size_;
  const PrimExpr condition_;
  const bool dynamic_;
};

int GetVectorizeSize(const For& loop) { return VectorizePlanner().Plan(loop); }

VectorizePlanResult GetVectorizePlanResult(const For& loop) {
  VectorizePlanner planner;
  int vector_size = planner.Plan(loop);
  bool dynamic = planner.GetDynamic();
  PrimExpr condition = planner.GetCondition();
  return {vector_size, dynamic, condition};
}

// Use the same code as tir.transform.vectorize_loop
class VectorizeChecker : public ExprMutator {
 public:
  VectorizeChecker(Var var, int extent, arith::Analyzer* analyzer) : var_(var), extent_(extent) {}

 private:
  PrimExpr VisitExpr_(const AddNode* op) final {
    return AddSubVec(op, [](PrimExpr a, PrimExpr b) { return a + b; });
  }

  PrimExpr VisitExpr_(const SubNode* op) final {
    return AddSubVec(op, [](PrimExpr a, PrimExpr b) { return a - b; });
  }

  PrimExpr VisitExpr_(const MulNode* op) final {
    PrimExpr a = this->VisitExpr(op->a);
    PrimExpr b = this->VisitExpr(op->b);
    if (a.same_as(op->a) && b.same_as(op->b)) {
      return GetRef<PrimExpr>(op);
    } else {
      int lanes = std::max(a.dtype().lanes(), b.dtype().lanes());
      if (lanes != 1) {
        const RampNode* b_ramp = b.as<RampNode>();
        const RampNode* a_ramp = a.as<RampNode>();
        if (a_ramp && b.dtype().lanes() == 1 && analyzer_->CanProve(b > 0)) {
          return Ramp(a_ramp->base * b, a_ramp->stride * b, a_ramp->lanes);
        }
        if (b_ramp && a.dtype().lanes() == 1 && analyzer_->CanProve(a > 0)) {
          return Ramp(b_ramp->base * a, b_ramp->stride * a, b_ramp->lanes);
        }
      }
      return Mul(BroadcastTo(a, lanes), BroadcastTo(b, lanes));
    }
    return BinaryVec<Mul>(op);
  }
  PrimExpr VisitExpr_(const DivNode* op) final { return BinaryVec<Div>(op); }
  PrimExpr VisitExpr_(const ModNode* op) final { return BinaryVec<Mod>(op); }
  PrimExpr VisitExpr_(const FloorDivNode* op) final { return BinaryVec<FloorDiv>(op); }
  PrimExpr VisitExpr_(const FloorModNode* op) final { return BinaryVec<FloorMod>(op); }
  PrimExpr VisitExpr_(const MinNode* op) final { return BinaryVec<Min>(op); }
  PrimExpr VisitExpr_(const MaxNode* op) final { return BinaryVec<Max>(op); }
  PrimExpr VisitExpr_(const EQNode* op) final { return BinaryVec<EQ>(op); }
  PrimExpr VisitExpr_(const NENode* op) final { return BinaryVec<NE>(op); }
  PrimExpr VisitExpr_(const LTNode* op) final { return BinaryVec<LT>(op); }
  PrimExpr VisitExpr_(const LENode* op) final { return BinaryVec<LE>(op); }
  PrimExpr VisitExpr_(const GTNode* op) final { return BinaryVec<GT>(op); }
  PrimExpr VisitExpr_(const GENode* op) final { return BinaryVec<GE>(op); }
  PrimExpr VisitExpr_(const AndNode* op) final { return BinaryVec<And>(op); }
  PrimExpr VisitExpr_(const OrNode* op) final { return BinaryVec<Or>(op); }

  PrimExpr VisitExpr_(const NotNode* op) final {
    PrimExpr a = this->VisitExpr(op->a);
    if (a.same_as(op->a)) {
      return GetRef<PrimExpr>(op);
    } else {
      return !(a);
    }
  }

  PrimExpr VisitExpr_(const RampNode* op) final {
    PrimExpr base = this->VisitExpr(op->base);
    PrimExpr stride = this->VisitExpr(op->stride);
    int op_lanes = static_cast<int>(Downcast<IntImm>(op->lanes)->value);
    if (base.dtype().lanes() > 1 && stride.dtype().lanes() == 1) {
      const RampNode* base_ramp = base.as<RampNode>();
      if (analyzer_->CanProve(base_ramp->stride ==
                              stride * make_const(stride.dtype(), op_lanes))) {
        return Ramp(base_ramp->base, stride, op_lanes * base_ramp->lanes);
      }
    }
    int lanes = std::max(base.dtype().lanes(), stride.dtype().lanes());
    base = BroadcastTo(base, lanes);
    stride = BroadcastTo(stride, lanes);
    Array<PrimExpr> elems;
    for (int i = 0; i < lanes; ++i) {
      elems.push_back(
          Ramp(Shuffle::ExtractElement(base, i), Shuffle::ExtractElement(stride, i), lanes));
    }
    return Shuffle::Concat(elems);
  }

  PrimExpr VisitExpr_(const SelectNode* op) final {
    PrimExpr cond = this->VisitExpr(op->condition);
    PrimExpr t = this->VisitExpr(op->true_value);
    PrimExpr f = this->VisitExpr(op->false_value);
    if (cond.same_as(op->condition) && t.same_as(op->true_value) && f.same_as(op->false_value)) {
      return GetRef<PrimExpr>(op);
    } else {
      int lanes = std::max(std::max(cond.dtype().lanes(), t.dtype().lanes()), f.dtype().lanes());
      return Select(cond, BroadcastTo(t, lanes), BroadcastTo(f, lanes));
    }
  }

  PrimExpr VisitExpr_(const CastNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    if (value.same_as(op->value)) {
      return GetRef<PrimExpr>(op);
    } else {
      return Cast(op->dtype.with_lanes(value.dtype().lanes()), value);
    }
  }

  // Variable
  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    if (var.same_as(var_)) {
      return Ramp(var, 1, extent_);
    }
    return std::move(var);
  }

  // IfThenElse expr
  PrimExpr MutateIfThenElseExpr_(const CallNode* op) {
    PrimExpr cond = this->VisitExpr(op->args[0]);
    if (cond.dtype().is_vector()) {
      return GetRef<PrimExpr>(op);
    }
    PrimExpr t = this->VisitExpr(op->args[1]);
    PrimExpr f = this->VisitExpr(op->args[2]);
    if (cond.same_as(op->args[0]) && t.same_as(op->args[1]) && f.same_as(op->args[2])) {
      return GetRef<PrimExpr>(op);
    } else {
      int lanes = std::max(t.dtype().lanes(), f.dtype().lanes());
      t = BroadcastTo(t, lanes);
      f = BroadcastTo(f, lanes);
      return Call(op->dtype.with_lanes(lanes), op->op, {cond, t, f});
    }
  }
  // Call
  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::if_then_else())) {
      return MutateIfThenElseExpr_(op);
    } else if (op->op.same_as(builtin::texture2d_load())) {
      int lane = 0;
      Array<PrimExpr> fcd = MutateArray({op->args.back()}, &lane);
      auto new_args = op->args;
      new_args.pop_back();
      new_args.push_back(fcd[0]);
      return Call(op->dtype.with_lanes(4), op->op, new_args);
    } else if (op->op.same_as(builtin::texture2d_store())) {
      int lane = 0;
      // Vectorize the value to store
      Array<PrimExpr> value{op->args.back()};
      Array<PrimExpr> mutated_value = MutateArray(value, &lane);
      Array<PrimExpr> new_args{op->args[0], op->args[1], op->args[2], mutated_value[0]};
      return Call(op->dtype.with_lanes(lane), op->op, new_args);
    }
    auto optional_op = op->op.as<Op>();
    bool vectorizable = optional_op && op_vectorizable_.get(optional_op.value(), false);

    if (!vectorizable) {
      // Cannot vectorize this op
      Array<PrimExpr> new_args;
      for (auto arg : op->args) {
        auto new_arg = this->VisitExpr(arg);
        if (new_arg.dtype().is_vector()) {
          return GetRef<PrimExpr>(op);
        }
        new_args.push_back(new_arg);
      }
      if (op->args.same_as(new_args)) {
        return GetRef<PrimExpr>(op);
      } else {
        return Call(op->dtype, op->op, new_args);
      }
    } else {
      int lane = 0;
      Array<PrimExpr> new_args = MutateArray(op->args, &lane);
      // normal code path.
      if (op->args.same_as(new_args)) {
        return GetRef<PrimExpr>(op);
      } else {
        return Call(op->dtype.with_lanes(lane), op->op, new_args);
      }
    }
  }

  static inline PrimExpr BroadcastTo(PrimExpr e, int lanes) {
    if (e.dtype().lanes() == lanes) return e;
    if (const BroadcastNode* op = e.as<BroadcastNode>()) {
      int op_lanes = static_cast<int>(Downcast<IntImm>(op->lanes)->value);
      if (lanes % op_lanes == 0) {
        return Broadcast(op->value, lanes);
      }
    }
    ICHECK_EQ(e.dtype().lanes(), 1)
        << "Cannot broadcast lane=" << e.dtype().lanes() << " to " << lanes;
    return Broadcast(e, lanes);
  }
  // mutate array, with given lane requirement
  // when finished, p_lane updates the lane requirement.
  Array<PrimExpr> MutateArray(Array<PrimExpr> arr, int* p_lanes) {
    if (arr.size() == 0) return arr;
    int& lanes = *p_lanes;
    bool changed = false;
    std::vector<PrimExpr> new_arr(arr.size());
    for (size_t i = 0; i < arr.size(); i++) {
      PrimExpr old_elem = arr[i];
      PrimExpr new_elem = this->VisitExpr(old_elem);
      if (!new_elem.same_as(old_elem)) changed = true;
      new_arr[i] = new_elem;
      lanes = std::max(lanes, new_elem.dtype().lanes());
    }

    for (size_t i = 0; i < arr.size(); ++i) {
      if (new_arr[i].dtype().lanes() != lanes) {
        new_arr[i] = BroadcastTo(new_arr[i], lanes);
        changed = true;
      }
    }
    if (!changed) return arr;
    return Array<PrimExpr>(new_arr);
  }
  template <typename TOp, typename T>
  PrimExpr BinaryVec(const T* op) {
    static_assert(std::is_same<typename TOp::ContainerType, T>::value, "constraint");
    PrimExpr a = this->VisitExpr(op->a);
    PrimExpr b = this->VisitExpr(op->b);
    if (a.same_as(op->a) && b.same_as(op->b)) {
      return GetRef<PrimExpr>(op);
    } else {
      int lanes = std::max(a.dtype().lanes(), b.dtype().lanes());
      return TOp(BroadcastTo(a, lanes), BroadcastTo(b, lanes));
    }
  }
  template <typename T, typename FCompute>
  PrimExpr AddSubVec(const T* op, FCompute fcompute) {
    PrimExpr a = this->VisitExpr(op->a);
    PrimExpr b = this->VisitExpr(op->b);
    if (a.same_as(op->a) && b.same_as(op->b)) {
      return GetRef<PrimExpr>(op);
    } else {
      int lanes = std::max(a.dtype().lanes(), b.dtype().lanes());
      if (lanes != 1) {
        const RampNode* b_ramp = b.as<RampNode>();
        const RampNode* a_ramp = a.as<RampNode>();
        if (a.dtype().lanes() == 1 && b_ramp) {
          return Ramp(fcompute(a, b_ramp->base),
                      fcompute(make_zero(b_ramp->stride.dtype()), b_ramp->stride), b_ramp->lanes);
        }
        if (b.dtype().lanes() == 1 && a_ramp) {
          return Ramp(fcompute(a_ramp->base, b), a_ramp->stride, a_ramp->lanes);
        }
      }
      return fcompute(BroadcastTo(a, lanes), BroadcastTo(b, lanes));
    }
  }

  Var var_;
  int extent_;
  arith::Analyzer* analyzer_;
  OpAttrMap<bool> op_vectorizable_ = Op::GetAttrMap<bool>("TVectorizable");
};

bool IndiceCanVectorize(PrimExpr expr, Var var, PrimExpr iter_var_size, int target_vectorized_size,
                        arith::Analyzer* analyzer) {
  ICHECK(target_vectorized_size >= 1);
  if (target_vectorized_size == 1) return true;
  if (!analyzer->CanProveEqual(FloorMod(iter_var_size, target_vectorized_size), 0)) return false;
  Var v0("v0"), v1("v1");
  analyzer->Bind(v0, Range(0, target_vectorized_size));
  analyzer->Bind(v1, Range(0, FloorDiv(iter_var_size, target_vectorized_size)));
  PrimExpr expr_transformed =
      analyzer->Simplify(Substitute(expr, {{var, v0 + v1 * target_vectorized_size}}));
  VectorizeChecker T(v0, target_vectorized_size, analyzer);
  PrimExpr expr_vectorized = T(expr_transformed);
  auto ramp_node = expr_vectorized.as<RampNode>();
  if (!ramp_node) {
    // Broadcast value
    if (expr_vectorized.dtype().lanes() == 1)
      return true;
    else
      return false;
  } else {
    return is_one(ramp_node->stride);
  }
}

For VectorizeLoop(const For& loop, int vectorize_hint) {
  VectorizePlanResult res{128, false, 0};
  if (vectorize_hint <= 0) {
    res = GetVectorizePlanResult(loop);
    vectorize_hint = res.vector_size;
    // vectorize_hint = GetVectorizeSize(loop);
  }
  if (vectorize_hint == 1) return loop;
  auto rewriter = VectorizeRewriter(res);
  return Downcast<For>(rewriter(loop));
}

}  // namespace tl
}  // namespace tvm
