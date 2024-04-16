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
 * \file tvm/relax/transform/remove_symbolic_expression_in_subroutine.cc
 *
 * \brief Replace symbolic expressions with single variables, when possible.
 *
 * For example, if a subroutine accepts symbolic shape parameters `N`
 * and `M`, and the variables `N` and `M` are only ever used to
 * compute `N*M`, then the subroutine could instead accept a symbolic
 * shape parameter `new_var = N*M`.  This can allow shape parameters
 * to be inferred from tensor shapes, rather than requiring additional
 * arguments.
 */

#include <tvm/node/object_path.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr_functor.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace relax {

namespace {

// Utility templates for unordered map/set that use structural hash/equal.

template <typename Key, typename Value>
using StructMap = std::unordered_map<Key, Value, StructuralHash, StructuralEqual>;

template <typename Key>
using StructSet = std::unordered_set<Key, StructuralHash, StructuralEqual>;

/* \brief Collect symbolic expressions that may be inferred from a function signature
 *
 * \param func The function whose signature should be inspected
 *
 * \return A map from PrimExpr to the location where it occurs in the signature
 */
StructMap<PrimExpr, std::string> CollectInferableExpressions(const Function& func) {
  StructMap<PrimExpr, std::string> output;

  auto mark = [&](const PrimExpr& expr, const ObjectPath& path) {
    if (!output.count(expr)) {
      std::stringstream ss;
      ss << path;
      output[expr] = ss.str();
    }
  };

  std::function<void(const StructInfo&, const ObjectPath&)> visit = [&](const StructInfo& sinfo,
                                                                        const ObjectPath& path) {
    if (auto tensor = sinfo.as<TensorStructInfoNode>()) {
      if (auto opt_shape = tensor->GetShape()) {
        auto shape_path = path->Attr("shape");
        auto shape = opt_shape.value();
        for (size_t i = 0; i < shape.size(); i++) {
          mark(shape[i], shape_path->ArrayIndex(i));
        }
      }
    } else if (auto tuple = sinfo.as<TupleStructInfoNode>()) {
      for (size_t i = 0; i < tuple->fields.size(); i++) {
        visit(tuple->fields[i], path->ArrayIndex(i));
      }
    }
  };

  for (const auto& param : func->params) {
    visit(GetStructInfo(param), ObjectPath::Root(param->name_hint()));
  }

  return output;
}

/* \brief Collect expressions that are required in a function body
 *
 * This recurses into StructInfo and sub-expressions, but does not
 * recurse beyond any expression in `inferable_expressions`.  This
 * allows the transform to determine whether a `tir::Var` ever occurs
 * outside of an expression that can be inferred.
 */
class RequiredExpressionCollector : private StructInfoVisitor,
                                    private ExprVisitor,
                                    private tir::ExprVisitor {
 public:
  static StructSet<PrimExpr> Collect(
      const Function& func, const StructMap<PrimExpr, std::string>& inferable_expressions) {
    RequiredExpressionCollector visitor(inferable_expressions);
    visitor.VisitExpr(func->body);
    return visitor.required_expressions_;
  }

 private:
  explicit RequiredExpressionCollector(
      const StructMap<PrimExpr, std::string>& inferable_expressions)
      : inferable_expressions_(inferable_expressions) {}

  using relax::ExprVisitor::VisitExpr;
  using tir::ExprVisitor::VisitExpr;

  // Required in order to recurse from `TensorStructInfo` into its
  // `ShapeExpr`.  This hands control from `StructInfoVisitor` into
  // `ExprVisitor`.
  void VisitStructInfoExprField(const Expr& expr) override { VisitExpr(expr); }

  // Required in order to recurse into `ShapeStructInfo`.  This hands
  // control from `ExprVisitor` back to `StructInfoVisitor`.
  void VisitExprDepStructInfoField(const StructInfo& struct_info) override {
    VisitStructInfo(struct_info);
  }

  void VisitPrimExpr(const PrimExpr& expr) override {
    required_expressions_.insert(expr);
    if (!inferable_expressions_.count(expr)) {
      tir::ExprVisitor::VisitExpr(expr);
    }
  }

  void VisitStructInfoExprField(const PrimExpr& expr) override { VisitPrimExpr(expr); }

  const StructMap<PrimExpr, std::string>& inferable_expressions_;
  StructSet<PrimExpr> required_expressions_;
};

/* \brief Replace occurrences of a PrimExpr in the symbolic variables
 *
 * In most cases, the `tvm::relax::Bind` utility should be used
 * instead.  Here, though, we are replacing a `PrimExpr` with a
 * `tir::Var`, whereas `tvm::relax::Bind` supports the more standard
 * case of replacing a `tir::Var` with a `PrimExpr`.
 */
class SymbolicSubexprReplacer : public relax::ExprMutator,
                                public StructInfoMutator,
                                public tir::ExprMutator {
 public:
  using relax::ExprMutator::operator();
  using relax::ExprMutator::VisitExpr;
  using tir::ExprMutator::operator();
  using tir::ExprMutator::VisitExpr;

  explicit SymbolicSubexprReplacer(StructMap<PrimExpr, tir::Var> replacements)
      : replacements_(replacements) {}

  StructInfo VisitExprDepStructInfoField(const StructInfo& struct_info) override {
    return VisitStructInfo(struct_info);
  }
  Expr VisitStructInfoExprField(const Expr& expr) override { return VisitExpr(expr); }
  PrimExpr VisitStructInfoExprField(const PrimExpr& expr) override { return VisitExpr(expr); }
  PrimExpr VisitPrimExpr(const PrimExpr& expr) override { return VisitExpr(expr); }

  PrimExpr VisitExpr(const PrimExpr& expr) override {
    if (auto it = replacements_.find(expr); it != replacements_.end()) {
      return it->second;
    } else {
      return tir::ExprMutator::VisitExpr(expr);
    }
  }

  StructMap<PrimExpr, tir::Var> replacements_;
};

}  // namespace

Function RemoveSymbolicExpressionInSubroutine(Function func) {
  bool is_exposed_externally = func->GetAttr<String>(tvm::attr::kGlobalSymbol).defined();
  if (is_exposed_externally) return func;

  auto inferable_expressions = CollectInferableExpressions(func);

  auto required_expressions = RequiredExpressionCollector::Collect(func, inferable_expressions);

  StructMap<PrimExpr, tir::Var> replacements;
  for (const auto& [expr, name] : inferable_expressions) {
    bool is_tir_var = expr->IsInstance<tir::VarNode>();

    auto expr_depends_on = tir::UndefinedVars(expr);
    bool internal_variable_is_required =
        std::any_of(expr_depends_on.begin(), expr_depends_on.end(),
                    [&](const tir::Var& subvar) { return required_expressions.count(subvar); });

    if (!is_tir_var && !internal_variable_is_required) {
      // For human-readability, use the location used to infer the
      // shape to name the variable.  (e.g. `A_dim0` for a parameter
      // inferred from parameter `A->shape[0]`.)
      replacements[expr] = tir::Var(name, expr->dtype);
    }
  }

  if (replacements.empty()) {
    return func;
  }

  SymbolicSubexprReplacer mutator(replacements);
  return Downcast<Function>(mutator(func));
}

namespace transform {
Pass RemoveSymbolicExpressionInSubroutine() {
  auto pass_func = [=](IRModule mod, PassContext pc) -> IRModule {
    IRModule updates;

    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto func = base_func.as<Function>()) {
        auto mutated = RemoveSymbolicExpressionInSubroutine(func.value());
        if (!mutated.same_as(base_func)) {
          updates->Add(gvar, mutated);
        }
      }
    }

    if (updates->functions.size()) {
      mod.CopyOnWrite()->Update(updates);
    }
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "RemoveSymbolicExpressionInSubroutine", {});
}

TVM_REGISTER_GLOBAL("relax.transform.RemoveSymbolicExpressionInSubroutine")
    .set_body_typed(RemoveSymbolicExpressionInSubroutine);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
