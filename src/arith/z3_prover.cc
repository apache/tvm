#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <z3++.h>

#include <sstream>
#include <string>
#include <unordered_map>

#include "tvm/ffi/cast.h"
#include "tvm/ffi/object.h"
#include "tvm/ffi/string.h"
#include "tvm/ir/expr.h"
#include "tvm/node/structural_equal.h"
#include "tvm/node/structural_hash.h"
#include "tvm/runtime/data_type.h"
#include "tvm/tir/expr_functor.h"
#include "tvm/arith/analyzer.h"

namespace tvm::arith {

using namespace tir;
using namespace ffi;

class Z3Prover::Impl : ExprFunctor<z3::expr(const PrimExpr &)>, public Object {
  struct Scope {
    std::vector<std::pair<PrimExpr, std::optional<z3::expr>>> leaf_node_updates;
    std::vector<PrimExpr> constraint;
  };
public:
  z3::context ctx;
  z3::solver solver{ctx};
  Impl() {
    scope_stack.push_back({});
    ctx.set("model", false);
    SetTimeoutMs(5);
  }
  void CopyFrom(const Z3Prover::Impl & other_) {
    for(auto & item: other_.scope_stack) {
      for(auto & constr: item.constraint) {
        AddConstraint(constr);
      }
    }
  }
  using Base = ExprFunctor<z3::expr(const PrimExpr &)>;
  using ExprMap = std::unordered_map<const PrimExpr, z3::expr, StructuralHash, StructuralEqual>;
  bool force_memorize {false};
  std::function<void()> EnterConstraint(const PrimExpr& constraint, bool is_assume=false) {
    EnterWithScope();
    return [this](){return ExitWithScope();};
  }
  void EnterWithScope() {
    solver.push();
    scope_stack.push_back({});
  }
  void ExitWithScope() {
    for (const auto &[e, v] : scope_stack.back().leaf_node_updates) {
      if (v.has_value()) {
        leaf_node_map.emplace(e, v.value());
      } else {
        leaf_node_map.erase(e);
      }
    }
    scope_stack.pop_back();
    solver.pop();
  }
  static bool IsValidDType(const DataType & dtype) {
    return (dtype.is_int() || dtype.is_uint()) && dtype.lanes() == 1;
  }
  void Bind(const Var &var, const PrimExpr &value, bool allow_override = false) {
    if (!IsValidDType(var->dtype)) return;
    auto var_expr = GetLeafNode(var.as<VarNode>(), true, allow_override);
    auto value_expr = VisitInt(value);
    add(var_expr == value_expr);
  }
  void Bind(const Var &var, const Range &range, bool allow_override = false) {
    if (!IsValidDType(var->dtype)) return;
    auto var_expr = GetLeafNode(var.as<VarNode>(), true, allow_override);
    auto min_expr = VisitInt(range->min);
    auto extent_expr = VisitInt(range->extent);
    add(var_expr >= min_expr);
    add(var_expr < (min_expr + extent_expr));
  }
  void AddConstraint(const PrimExpr &constraint, bool is_assume=false) {
    force_memorize = is_assume;
    add(VisitBool(constraint));
    force_memorize = false;
  }
  bool CanProve(const PrimExpr &expr) {
    if (!IsValidDType(expr->dtype)) return false;
    z3::check_result result = z3::unknown;
    try {
      z3::expr_vector vec(ctx);
      vec.push_back(!VisitBool(expr));
      result = solver.check(vec);
    } catch(std::exception & e) {
      std::string reason = e.what();
      if(reason == "max. steps exceeded") {
        return false;
      }
      LOG(FATAL) << "Z3 encountered an error: " << e.what();
    }
    return result == z3::unsat;
  }
  ffi::String GetProblem(const PrimExpr & expr) {
    EnterWithScope();
    add(!VisitBool(expr));
    auto result = solver.to_smt2();
    ExitWithScope();
    return result;
  }
  ffi::String Statistics() {
    std::stringstream ss;
    ss << solver.statistics();
    return ss.str();
  }
  void SetMaxStep(unsigned max_step) {
    solver.set("max_steps", max_step);
  }
  void SetTimeoutMs(unsigned timeout_ms) {
    solver.set("timeout", timeout_ms);
  }
  ffi::String GetSMTLIB2() {
    return solver.to_smt2();
  }
  ffi::String GetSMTLIB2(const PrimExpr & e) {
    EnterWithScope();
    AddConstraint(!e);
    auto res = solver.to_smt2();
    ExitWithScope();
    return res;
  }
  // static void RegisterReflection() {
  //   namespace refl = tvm::ffi::reflection;
  //   auto set_param_impl = [](Z3ProverNode * node, const String & param, const Any & value) {
  //     if(value.type_index() == TypeIndex::kTVMFFIBool) {
  //       return node->solver.set(param.c_str(), value.cast<bool>());
  //     }
  //     if(value.type_index() == TypeIndex::kTVMFFIInt) {
  //       return node->solver.set(param.c_str(), value.cast<unsigned>());
  //     }
  //     if(value.type_index() == TypeIndex::kTVMFFIFloat) {
  //       return node->solver.set(param.c_str(), value.cast<double>());
  //     }
  //     if(auto v = value.as<String>()) {
  //       return node->solver.set(param.c_str(), v->c_str());
  //     }
  //     LOG(FATAL) << "Z3Prover::SetParam only supports unsigned, double, bool, and string.";
  //   };
  //   auto bind_impl = [](Z3ProverNode * self, const Var & var, const ObjectRef & obj, bool allow_override) {
  //     if(obj->IsInstance<PrimExprNode>()) {
  //       return self->Bind(var, Downcast<PrimExpr>(obj), allow_override);
  //     }
  //     if(obj->IsInstance<RangeNode>()) {
  //       return self->Bind(var, Downcast<Range>(obj), allow_override);
  //     }
  //     LOG(FATAL) << "Z3Prover::Bind only supports PrimExpr and Range.";
  //   };
  //   using Self = Z3ProverNode;
  //   refl::ObjectDef<Z3ProverNode>()
  //       .def("_SetParam", set_param_impl)
  //       .def("_Bind", bind_impl)
  //       .def("_AddConstraint", &Self::AddConstraint)
  //       .def("set_max_step", &Self::SetMaxStep)
  //       .def("set_timeout_ms", &Self::SetTimeoutMs)
  //       .def("can_prove", &Self::CanProve)
  //       .def("get_smtlib2", &Self::GetSMTLIB2)
  //       .def("get_problem", &Self::GetProblem)
  //       .def("enter_with_scope", &Self::EnterWithScope)
  //       .def("exit_with_scope", &Self::ExitWithScope)
  //       .def("get_statistics", &Self::Statistics);
  // }
private:
  std::vector<Scope> scope_stack;
  std::unordered_set<std::string> used_names;
  ExprMap leaf_node_map;
  void add(z3::expr e) {
    solver.add(e);
    scope_stack.back().constraint.emplace_back(e);
  }
  std::string GetNewName(const std::string & name) {
    if(used_names.count(name) == 0) {
      used_names.insert(name);
      return name;
    }
    int idx = 1;
    std::string check_name = name + "$" + std::to_string(idx);
    while(used_names.count(check_name)) {
      idx ++;
      check_name = name + "$" + std::to_string(idx);
    }
    used_names.insert(check_name);
    return check_name;
  }
  z3::expr GetLeafNode(const PrimExprNode *op, bool memorize = false, bool override = false) {
    auto ref = ffi::GetRef<PrimExpr>(op);
    if (!override && leaf_node_map.count(ref)) {
      return leaf_node_map.at(ref);
    }
    auto dtype = op->dtype;
    std::stringstream ss;
    ss << ref;
    std::string name = GetNewName(ss.str());
    z3::expr e = ctx.int_const(name.c_str());
    auto max_val = Downcast<IntImm>(max_value(dtype))->value;
    auto min_val = Downcast<IntImm>(min_value(dtype))->value;
    add(e <= ctx.int_val(max_val));
    add(e >= ctx.int_val(min_val));
    if (memorize || force_memorize) {
      if (leaf_node_map.count(ref)) {
        scope_stack.back().leaf_node_updates.emplace_back(ref, leaf_node_map.at(ref));
      } else {
        scope_stack.back().leaf_node_updates.emplace_back(ref, std::nullopt);
      }
      leaf_node_map.emplace(ref, e);
    }
    return e;
  }
  z3::expr VisitInt(const PrimExpr &expr) {
    auto e = VisitExpr(expr);
    if (e.is_bool()) {
      return z3::ite(e, ctx.int_val(1), ctx.int_val(0));
    } else {
      return e;
    }
  }
  z3::expr VisitBool(const PrimExpr &e) {
    auto expr = VisitExpr(e);
    if (expr.is_bool()) {
      return expr;
    } else {
      return expr != ctx.int_val(0);
    }
  }
  z3::expr VisitExpr_(const CastNode * op) override {
    if(!IsValidDType(op->value->dtype)) return GetLeafNode(op);
    return VisitInt(op->value);
  }
  using Z3BinOp = z3::expr(*)(const z3::expr &, const z3::expr &);
  z3::expr VisitArith(Z3BinOp signed_op, const PrimExprNode *op, const PrimExpr &a, const PrimExpr &b) {
    if (IsValidDType(a->dtype) && IsValidDType(b->dtype)) {
      return signed_op(VisitInt(a), VisitInt(b));
    } else {
      return GetLeafNode(op);
    }
  }
  z3::expr VisitExpr_(const MinNode *op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    return z3::ite(a < b, a, b);
  }
  z3::expr VisitExpr_(const MaxNode *op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    return z3::ite(a > b, a, b);
  }
  z3::expr VisitExpr_(const LetNode *op) override { 
    if (IsValidDType(op->var->dtype)) {
      add(VisitExpr(op->var == op->value));
    }
    return VisitExpr(op->body);
  }
  z3::expr VisitExpr_(const CallNode *op) override { return GetLeafNode(op, true); }
  z3::expr VisitExpr_(const VarNode *op) override { return GetLeafNode(op, true); }
  z3::expr VisitExpr_(const BufferLoadNode *op) override { return GetLeafNode(op); }
  z3::expr VisitExpr_(const ProducerLoadNode *op) override { return GetLeafNode(op); }
  z3::expr VisitExpr_(const ReduceNode *op) override { return GetLeafNode(op); }
  z3::expr VisitExpr_(const AddNode *op) override { return VisitArith(z3::operator +, op, op->a, op->b); }
  z3::expr VisitExpr_(const SubNode *op) override { return VisitArith(z3::operator -, op, op->a, op->b); }
  z3::expr VisitExpr_(const MulNode *op) override { return VisitArith(z3::operator *, op, op->a, op->b); }
  z3::expr VisitExpr_(const DivNode *op) override { return VisitArith(z3::operator /, op, op->a, op->b); }
  z3::expr VisitExpr_(const ModNode *op) override { return VisitArith(z3::operator %, op, op->a, op->b); }
  z3::expr VisitExpr_(const FloorDivNode *op) override { return VisitArith(z3::operator/, op, op->a, op->b); }
  z3::expr VisitExpr_(const FloorModNode *op) override { return VisitArith(z3::operator %, op, op->a, op->b); }
  z3::expr VisitExpr_(const EQNode *op) override { return VisitArith(z3::operator==, op, op->a, op->b); }
  z3::expr VisitExpr_(const NENode *op) override { return VisitArith(z3::operator!=, op, op->a, op->b); }
  z3::expr VisitExpr_(const LTNode *op) override { return VisitArith(z3::operator<, op, op->a, op->b); }
  z3::expr VisitExpr_(const LENode *op) override { return VisitArith(z3::operator<=, op, op->a, op->b); }
  z3::expr VisitExpr_(const GTNode *op) override { return VisitArith(z3::operator>, op, op->a, op->b); }
  z3::expr VisitExpr_(const GENode *op) override { return VisitArith(z3::operator>=, op, op->a, op->b); }
  z3::expr VisitExpr_(const AndNode *op) override { return VisitBool(op->a) && VisitBool(op->b); }
  z3::expr VisitExpr_(const OrNode *op) override { return VisitBool(op->a) || VisitBool(op->b); }
  z3::expr VisitExpr_(const NotNode *op) override { return !VisitBool(op->a); }
  z3::expr VisitExpr_(const SelectNode *op) override { return z3::ite(VisitBool(op->condition), VisitInt(op->true_value), VisitInt(op->false_value)); }
  z3::expr VisitExpr_(const RampNode *op) override { LOG(FATAL) << "Z3Prover does not support RampNode."; }
  z3::expr VisitExpr_(const BroadcastNode *op) override { LOG(FATAL) << "Z3Prover does not support BroadcastNode."; }
  z3::expr VisitExpr_(const ShuffleNode *op) override { LOG(FATAL) << "Z3Prover does not support ShuffleNode."; }
  z3::expr VisitExpr_(const IntImmNode *op) override { return ctx.int_val(op->value); }
  z3::expr VisitExpr_(const FloatImmNode *op) override { LOG(FATAL) << "Z3Prover only supports scalar integer expressions."; }
  z3::expr VisitExpr_(const StringImmNode *op) override { LOG(FATAL) << "Z3Prover only supports scalar integer expressions."; }
};

TVM_DLL bool Z3Prover::CanProve(const PrimExpr & expr) {
  return impl_->CanProve(expr);
}
TVM_DLL void Z3Prover::Bind(const Var& var, const Range& new_range, bool allow_override) {
  return impl_->Bind(var, new_range, allow_override);
}
TVM_DLL void Z3Prover::Bind(const Var& var, const PrimExpr& expr, bool allow_override) {
  return impl_->Bind(var, expr, allow_override);
}
std::function<void()> Z3Prover::EnterConstraint(const PrimExpr& constraint, bool is_assume) {
  return impl_->EnterConstraint(constraint, is_assume);
}
ffi::String Z3Prover::GetSMTLIB2(const ffi::Optional<PrimExpr> expr) {
  if(expr.has_value()) {
    return impl_->GetSMTLIB2(expr.value());
  } else {
    return impl_->GetSMTLIB2();
  }
}
void Z3Prover::SetTimeoutMs(unsigned timeout_ms) {
  impl_->SetTimeoutMs(timeout_ms);
}
void Z3Prover::SetMaxStep(unsigned max_step) {
  impl_->SetMaxStep(max_step);
}
void Z3Prover::CopyFrom(const Z3Prover & other) {
  impl_->CopyFrom(*other.impl_);
}
Z3Prover::Z3Prover(Analyzer* parent): impl_(new Impl) {}
TVM_DLL Z3Prover::~Z3Prover() {
  delete impl_;
}

} // namespace tvm::arith