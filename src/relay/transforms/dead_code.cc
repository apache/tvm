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
 * \file src/relay/transforms/dead_code.cc
 * \brief Elides or inlines let-bindings.
 *
 * TODO(mbs): Track dead writes into references.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include <tvm/relay/transform.h>

#include "../op/call/call.h"

namespace tvm {
namespace relay {
namespace {

/*! \brief Maximum depth of calls to analyize. */
constexpr int kMaxCallDepth = 25;

/*!
 * \brief Captures (an approximation of) the purity for a Relay sub-expression. A pure
 * sub-expression is guaranteed never to access or mutate state. Thus the sub-expression
 * can safely be elided (if its result is never used), or inlined (which may change the
 * number of times and program order for the evaluation.)
 */
struct Purity {
  /*!
   * \brief True if evaling the sub-expression itself is pure.
   */
  bool pure_eval;
  /*!
   * \brief If the sub-expression is first-order then always true. Otherwise true only if evaling
   * a call to the sub-expression is pure. See [RULE A] below.
   */
  bool pure_call;
};

/*!
 * \brief Visits all the global functions in a module and records the purity of every let-bound
 * value.
 *
 * (See also inline.cc for function inlining.)
 *
 * Generally we track whether evaluation of a sub-expression is definitely pure. However for
 * sub-expressions f of higher-order type we also track the 'call purity' of evaling a call to f:
 *  - [RULE A] If f's result is itself higher-order then f is call-pure only if the result of f is
 *    also call-pure.
 *  - [RULE B] Higher-order function arguments are assumed call impure.
 *  - [RULE C] We assume functions extracted from tuples are call impure.
 *  - [RULE D] We assume functions extracted from references are call impure.
 *  - [RULE E] We assume functions extracted from ADTs are call impure.
 *  - [RULE F] We assume all external Functions and PrimFuncs are call impure.
 */
class PurityVisitor : ExprFunctor<Purity(const Expr&)> {
 public:
  explicit PurityVisitor(IRModule mod) : mod_(std::move(mod)), current_call_depth_(0) {}

  /*! \brief Visit all the functions in the module. */
  void VisitModule() {
    VLOG_CONTEXT << "PurityVisitor";
    // It is safe to visit the global functions in any order. Recursive global functions are
    // allowed.
    for (const auto& kv : mod_->functions) {
      if (const auto* function_node = kv.second.as<FunctionNode>()) {
        if (function_node->HasNonzeroAttr(attr::kPrimitive) ||
            function_node->HasNonzeroAttr(attr::kExtern)) {
          // Ignore primitive and external functions.
          continue;
        }
        // Everything of interest will be recorded in the purity maps so we ignore the result.
        (void)VisitGlobalFunction(kv.first, GetRef<Function>(function_node));
      }
    }
  }

  /*!
   * \brief Returns a map from every let-bound variable to whether its let-bound value is
   * definitely pure.
   */
  std::unordered_map<const VarNode*, bool> GetPurityMap() const {
    std::unordered_map<const VarNode*, bool> result;
    for (const auto& kv : var_to_purity_) {
      result.emplace(kv.first, kv.second.pure_eval);
    }
    return result;
  }

 private:
  Purity VisitExpr(const Expr& expr) final {
    auto it = memo_.find(expr.get());
    if (it != this->memo_.end()) {
      return it->second;
    } else {
      Purity result = ExprFunctor::VisitExpr(expr);
      memo_[expr.get()] = result;
      return result;
    }
  }

  Purity VisitExpr_(const ConstantNode*) final { return {/*pure_eval=*/true, /*pure_call=*/true}; }

  Purity VisitExpr_(const ConstructorNode*) final {
    return {/*pure_eval=*/true, /*pure_call=*/true};
  }

  Purity VisitExpr_(const OpNode* op_node) final {
    // Primitive operators are pure unless marked as 'stateful'.
    static OpAttrMap<bool> attr_map = Op::GetAttrMap<TOpIsStateful>("TOpIsStateful");
    bool is_stateful = attr_map.count(GetRef<Op>(op_node)) && attr_map[GetRef<Op>(op_node)];
    return {/*pure_eval=*/true, /*pure_call=*/!is_stateful};
  }

  Purity VisitExpr_(const GlobalVarNode* global_var_node) final {
    auto global_var = GetRef<GlobalVar>(global_var_node);
    ICHECK(mod_->ContainGlobalVar(global_var_node->name_hint))
        << "No definition for '" << global_var_node->name_hint << "'";
    auto func = mod_->Lookup(global_var);
    if (const auto* function_node = func.as<FunctionNode>()) {
      if (!function_node->HasNonzeroAttr(attr::kExtern)) {
        return VisitGlobalFunction(global_var, GetRef<Function>(function_node));
      }
    }
    // Assume externals and PrimFuncs are call-impure [RULE F].
    // (If they are pure then we should have dealt with them before lowering.)
    return {/*pure_eval==*/true, /*pure_call=*/false};
  }

  Purity VisitExpr_(const VarNode* var_node) final {
    // The var is bound to a value, but if that value is a function we need to propagate the
    // function body's purity.
    ICHECK(var_to_purity_.count(var_node)) << PrettyPrint(GetRef<Var>(var_node));
    return {/*pure_eval=*/true, /*pure_call=*/var_to_purity_[var_node].pure_call};
  }

  Purity VisitExpr_(const FunctionNode* function_node) final {
    for (const auto& param : function_node->params) {
      // Any higher-order parameters are assumed to be call-impure [RULE B]
      var_to_purity_[param.get()] = {/*pure_eval=*/true, /*pure_call=*/IsFirstOrder(param)};
    }
    Purity body_purity = VisitExpr(function_node->body);
    // The function itself is a value and thus pure. If the function returns
    // a function we'll fold its purity in here [RULE A]
    return {/*pure_eval=*/true, /*pure_call=*/body_purity.pure_eval && body_purity.pure_call};
  }

  Purity VisitExpr_(const LetNode* let_node) final {
    Expr expr = GetRef<Expr>(let_node);
    bool all_values_pure_eval = true;
    while (const auto* inner_let_node = expr.as<LetNode>()) {
      // In case the value is a recursive function assume the let-bound variable is call-pure.
      var_to_purity_[inner_let_node->var.get()] = {/*pure_eval=*/true, /*pure_call=*/true};
      Purity value_purity = VisitExpr(inner_let_node->value);
      // Now revise the variable to it's true purity.
      var_to_purity_[inner_let_node->var.get()] = value_purity;
      VLOG(2) << (value_purity.pure_eval ? "pure" : "impure") << " expression:" << std::endl
              << PrettyPrint(inner_let_node->value) << std::endl
              << "let-bound to variable:" << std::endl
              << PrettyPrint(inner_let_node->var);
      all_values_pure_eval = all_values_pure_eval && value_purity.pure_eval;
      expr = inner_let_node->body;
    }
    Purity body_purity = VisitExpr(expr);
    return {/*pure_eval=*/all_values_pure_eval && body_purity.pure_eval,
            /*pure_call=*/body_purity.pure_call};
  }

  Purity VisitExpr_(const CallNode* call_node) final {
    auto call = GetRef<Call>(call_node);
    if (current_call_depth_ >= kMaxCallDepth) {
      // Assume impure.
      VLOG(2) << "assuming call is impure since too deeply nested";
      return {/*pure_eval=*/false, /*pure_call*/ IsFirstOrder(call)};
    }

    ++current_call_depth_;

    // We can work with calls in both pre- and post-lowered form.
    Call vanilla_call = GetAnyCall(call_node);

    // Find purity for the callee and the args.
    Purity callee_purity = VisitExpr(vanilla_call->op);
    bool all_args_pure_eval = true;
    for (const auto& arg : vanilla_call->args) {
      Purity arg_purity = VisitExpr(arg);
      all_args_pure_eval = all_args_pure_eval && arg_purity.pure_eval;
    }

    VLOG(2) << (callee_purity.pure_call ? "pure" : "impure") << " call to:" << std::endl
            << PrettyPrint(vanilla_call->op);

    ICHECK_GT(current_call_depth_, 0);
    --current_call_depth_;

    // If the callee's result is itself a function then by [RULE A] its purity
    // is given by callee_purity.pure_call.
    return {/*pure_eval=*/all_args_pure_eval && callee_purity.pure_eval && callee_purity.pure_call,
            /*pure_call=*/IsFirstOrder(call) || callee_purity.pure_call};
  }

  Purity VisitExpr_(const IfNode* if_node) final {
    Purity cond_purity = VisitExpr(if_node->cond);
    ICHECK(cond_purity.pure_call);  // conditional is first-order
    Purity true_purity = VisitExpr(if_node->true_branch);
    Purity false_purity = VisitExpr(if_node->false_branch);
    return {/*pure_eval=*/cond_purity.pure_eval && true_purity.pure_eval && false_purity.pure_eval,
            /*pure_call=*/true_purity.pure_call && false_purity.pure_call};
  }

  Purity VisitExpr_(const TupleNode* tuple_node) final {
    bool all_fields_pure = true;
    for (const auto& field : tuple_node->fields) {
      // The call purity of each tuple field is lost [RULE C].
      Purity field_purity = VisitExpr(field);
      if (!field_purity.pure_eval) {
        all_fields_pure = false;
      }
    }
    return {/*pure_eval=*/all_fields_pure, /*pure_call=*/true};
  }

  Purity VisitExpr_(const TupleGetItemNode* tuple_get_item_node) final {
    Purity tuple_purity = VisitExpr(tuple_get_item_node->tuple);
    ICHECK(tuple_purity.pure_call);  // tuple is first-order
    // We don't track call purity through tuple fields, so if the result is a function type we
    // must assume it is call impure [RULE C].
    return {/*pure_eval=*/tuple_purity.pure_eval,
            /*pure_call=*/IsFirstOrder(GetRef<TupleGetItem>(tuple_get_item_node))};
  }

  Purity VisitExpr_(const RefCreateNode*) final {
    // The creation of the  ref itself is unobservable other than via the reads/writes into it.
    return {/*pure_eval=*/true, /*pure_call=*/true};
  }

  Purity VisitExpr_(const RefWriteNode* ref_write_node) final {
    Purity ref_purity = VisitExpr(ref_write_node->ref);
    ICHECK(ref_purity.pure_call);  // reference is first-order
    // The call purity of the written value is lost [RULE D].
    // (But we must still visit to accumulate purity for any let-bindings within in.)
    (void)VisitExpr(ref_write_node->value);
    return {/*pure_eval=*/false, /*pure_call=*/true};
  }

  Purity VisitExpr_(const RefReadNode* ref_read_node) final {
    Purity ref_purity = VisitExpr(ref_read_node->ref);
    ICHECK(ref_purity.pure_call);  // reference is first-order
    // We don't track call purity through reference values, so if the result is a function
    // type we must assume it is call impure [RULE D].
    return {/*pure_eval=*/false, /*pure_call=*/IsFirstOrder(GetRef<RefRead>(ref_read_node))};
  }

  class PurityPatternVisitor : public PatternVisitor {
   public:
    explicit PurityPatternVisitor(PurityVisitor* outer) : outer_(outer) {}

   private:
    void VisitPattern_(const PatternVarNode* pattern_var_node) final {
      // We don't track call purity through ADTs, so if var is a function type we must assume
      // it is call impure [RULE E].
      outer_->var_to_purity_[pattern_var_node->var.get()] = {
          /*pure_eval=*/true, /*pure_call=*/IsFirstOrder(pattern_var_node->var)};
    }

    /*! \brief (Mutable borrow of) the outer visitor. */
    PurityVisitor* outer_;
  };

  Purity VisitExpr_(const MatchNode* match_node) final {
    Purity data_purity = VisitExpr(match_node->data);
    ICHECK(data_purity.pure_call);  // ADT is first order
    bool all_clauses_pure_eval = true;
    bool all_clauses_pure_call = true;
    for (const auto& clause : match_node->clauses) {
      PurityPatternVisitor pattern_visitor(this);
      pattern_visitor.VisitPattern(clause->lhs);
      Purity rhs_purity = VisitExpr(clause->rhs);
      all_clauses_pure_eval = all_clauses_pure_eval && rhs_purity.pure_eval;
      all_clauses_pure_call = all_clauses_pure_call && rhs_purity.pure_call;
    }
    return {/*pure_eval=*/data_purity.pure_eval && all_clauses_pure_eval,
            /*pure_call=*/all_clauses_pure_call};
  }

  /*! \brief Visits \p func bound to global \p var and returns it's purity. */
  Purity VisitGlobalFunction(const GlobalVar& var, const Function& func) {
    VLOG_CONTEXT << "func " << var->name_hint;
    VLOG(2) << "visiting";
    auto itr = global_var_to_purity_.find(var.get());
    if (itr != global_var_to_purity_.end()) {
      // We've already visited the function body.
      return itr->second;
    }
    // We are entering the body of a possibly-recursive global function. Assume it's body is pure.
    global_var_to_purity_[var.get()] = {/*pure_eval=*/true, /*pure_call=*/true};
    // Visit the global function for the first time.
    Purity func_purity = VisitExpr(func);
    // Update with the true purity.
    global_var_to_purity_[var.get()] = func_purity;
    return func_purity;
  }

  static bool IsFirstOrder(const Expr& expr) {
    return expr->checked_type().as<FuncTypeNode>() == nullptr;
  }

  /*! \brief The module we're analyzing. */
  IRModule mod_;

  /*!
   * \brief Maps each let-bound and global variable to the purity of the value it is bound to.
   * If the variable is bound to a function then the purity of saturating that function is also
   * tracked.
   *
   * Note that global_var_to_purity_, and all the 'pure_call' fields, are only needed internally
   * during the analysis, andonly the var_to_purity_ 'pure_eval' fields are used downstream.
   */
  std::unordered_map<const VarNode*, Purity> var_to_purity_;
  std::unordered_map<const GlobalVarNode*, Purity> global_var_to_purity_;

  /*! \brief The current call depth. We'll just assume deeply nested calls are impure rather than
   * spending all that time to check for sure. A deeply nested call is almost certain to be needed
   * anyway.
   */

  int current_call_depth_;

  /*! \brief Internal map used for memoization. */
  std::unordered_map<const ExprNode*, Purity> memo_;
};

/*!
 * \brief Accumulate the bound values and usage count for each let-bound variable.
 *
 * We don't attempt to track the number of calls to local functions, and instead just assume they
 * are called at least twice.
 */
class UsageVisitor : public ExprVisitor {
 public:
  /*! \brief Accumulates the expression bound to every let-bound variable. */
  std::unordered_map<const VarNode*, Expr> let_bound_values_;
  /*! \brief Accumulates the usage count for every let-bound variable. */
  std::unordered_map<const VarNode*, size_t> use_map_;

  explicit UsageVisitor(const std::unordered_map<const VarNode*, bool>* var_to_purity,
                        bool default_purity)
      : var_to_purity_(var_to_purity), default_purity_(default_purity) {}

  void VisitExpr(const Expr& expr) final {
    // Once we've seen 2 usages of a variable we know it can be neither elided nor inlined,
    // so can stop visiting again.
    if (++visit_counter_[expr.get()] <= 2) {
      ExprFunctor<void(const Expr&)>::VisitExpr(expr);
    }
  }

  void VisitExpr_(const FunctionNode* function_node) final {
    ++current_scope_level_;
    ExprVisitor::VisitExpr_(function_node);
    ICHECK_GT(current_scope_level_, 0);
    --current_scope_level_;
  }

  void VisitExpr_(const LetNode* let_node) final {
    Expr expr = GetRef<Expr>(let_node);
    while (const auto* inner_let_node = expr.as<LetNode>()) {
      ++visit_counter_[inner_let_node];
      let_bound_values_[inner_let_node->var.get()] = inner_let_node->value;
      VLOG(2) << "seen let-binding for:" << std::endl << PrettyPrint(inner_let_node->var);
      use_map_[inner_let_node->var.get()] = 0;
      scope_level_map_[inner_let_node->var.get()] = current_scope_level_;
      if (is_pure(inner_let_node->var.get())) {
        // We'll defer visiting the let-bound value until we've seen the first use of the let-bound
        // variable and thus know it must be evaluated.
        // no-op.
      } else {
        // The let-bound value is impure so must always be evaluated. Visit now.
        VisitExpr(inner_let_node->value);
      }
      expr = inner_let_node->body;
    }
    VisitExpr(expr);
  }

  void VisitExpr_(const VarNode* var_node) final {
    if (let_bound_values_.count(var_node)) {
      size_t& n = use_map_[var_node];
      ++n;
      VLOG(2) << var_node->name_hint() << " = " << n;
      if (n == 1 && is_pure(var_node)) {
        // Now that we have at least one use of the let-bound var, we know the let-bound
        // value is necessary.
        VisitExpr(let_bound_values_[var_node]);
      }
      if (scope_level_map_[var_node] < current_scope_level_) {
        // Since the variable was bound outside of the current local function, assume the
        // function will be called at least twice.
        ++n;
        VLOG(2) << var_node->name_hint() << " = " << n << " (bound at level "
                << scope_level_map_[var_node] << " but used at level " << current_scope_level_
                << ")";
      }
    }
    // else: nothing to be done for function parameters or variable in match patterns.
  }

  bool is_pure(const VarNode* var_node) const {
    auto itr = var_to_purity_->find(var_node);
    return itr == var_to_purity_->end() ? default_purity_ : itr->second;
  }

  /*! \brief (Immutable borrow of) the already determined purity for every let-bound variable. */
  const std::unordered_map<const VarNode*, bool>* var_to_purity_;
  /*! \brief The default purity for variables which are not in the above map. */
  bool default_purity_;
  /*!
   * \brief The current scope level. 0 for global functions. Incremented by one within each
   * let-bound local function. Necessary so we can avoid inlining an expensive let-bound computation
   * into a function which could be called more than once.
   */
  int current_scope_level_ = 0;
  /*! \brief Accumulates the scope level for every let-bound variable. */
  std::unordered_map<const VarNode*, int> scope_level_map_;
};

/*! \brief Eliminate/inline let-bound values when sound to do so. */
class EliminatorMutator : public ExprMutator {
 public:
  EliminatorMutator(bool inline_once,
                    const std::unordered_map<const VarNode*, Expr>* let_bound_values,
                    const std::unordered_map<const VarNode*, size_t>* use_map,
                    const std::unordered_map<const VarNode*, bool>* var_to_purity,
                    bool default_purity)
      : inline_once_(inline_once),
        let_bound_values_(let_bound_values),
        use_map_(use_map),
        var_to_purity_(var_to_purity),
        default_purity_(default_purity) {}

 private:
  enum Action { kElide, kInline, kNoChange };

  /*! \brief What should we do with let-binding for \p var_node? */
  Action ActionFor(const VarNode* var_node) {
    if (let_bound_values_->count(var_node) == 0) {
      // Not let-bound var.
      return kNoChange;
    }
    if (!is_pure(var_node)) {
      // The let-bound value is impure -- we must leave it exactly where it is.
      return kNoChange;
    }
    switch (use_map_->count(var_node) ? use_map_->at(var_node) : 0) {
      case 0:
        return kElide;
      case 1:
        return inline_once_ ? kInline : kNoChange;
      default:
        return kNoChange;
    }
  }

  Expr VisitExpr_(const VarNode* var_node) final {
    if (ActionFor(var_node) == kInline) {
      VLOG(1) << "inlining let-bound variable:" << std::endl << PrettyPrint(GetRef<Var>(var_node));
      return VisitExpr(let_bound_values_->at(var_node));
    } else {
      return GetRef<Var>(var_node);
    }
  }

  Expr VisitExpr_(const LetNode* op) final {
    auto pre_visit = [this](const LetNode* op) {
      if (ActionFor(op->var.get()) != kElide) {
        (void)VisitExpr(op->value);
      }
    };
    auto post_visit = [this](const LetNode* op) {
      Expr body = VisitExpr(op->body);
      auto expr = GetRef<Expr>(op);
      switch (ActionFor(op->var.get())) {
        case kElide:
          VLOG(1) << "eliding let-bound variable:" << std::endl << PrettyPrint(op->var);
          memo_[expr] = body;
          break;
        case kInline:
          // Already inlined at use-side.
          memo_[expr] = body;
          break;
        case kNoChange:
          Expr value = VisitExpr(op->value);
          memo_[expr] = Let(op->var, value, body);
          break;
      }
    };
    ExpandANormalForm(op, pre_visit, post_visit);
    return memo_[GetRef<Expr>(op)];
  }

  bool is_pure(const VarNode* var_node) const {
    auto itr = var_to_purity_->find(var_node);
    return itr == var_to_purity_->end() ? default_purity_ : itr->second;
  }

  bool inline_once_;
  const std::unordered_map<const VarNode*, Expr>* let_bound_values_;
  const std::unordered_map<const VarNode*, size_t>* use_map_;
  const std::unordered_map<const VarNode*, bool>* var_to_purity_;
  bool default_purity_;
};

}  // namespace

namespace transform {

// Declared in relay/transform.h
Pass DeadCodeElimination(bool inline_once, bool ignore_impurity) {
  auto pass_func = [=](IRModule mod, PassContext pc) -> IRModule {
    VLOG(1) << "Before:" << std::endl << PrettyPrint(mod);
    // Which let bindings are pure and can be safely elided?
    std::unordered_map<const VarNode*, bool> var_to_purity;
    if (!ignore_impurity) {
      VLOG(1) << "determine purity";
      PurityVisitor purity_visitor(mod);
      purity_visitor.VisitModule();
      var_to_purity = purity_visitor.GetPurityMap();
    }

    IRModule result(/*functions=*/{}, mod->type_definitions, mod->Imports(), mod->source_map,
                    mod->attrs);
    for (const auto& kv : mod->functions) {
      if (auto opt = kv.second.as<Function>()) {
        auto function = opt.value();

        VLOG(1) << "processing " << PrettyPrint(kv.first);

        VLOG(2) << "count usage";
        UsageVisitor usage_visitor(&var_to_purity, /*default_purity=*/ignore_impurity);
        usage_visitor.VisitExpr(function);

        // Actually eliminate/inline the let-bindings.
        VLOG(2) << "eliminate";
        EliminatorMutator eliminator_mutator(inline_once, &usage_visitor.let_bound_values_,
                                             &usage_visitor.use_map_, &var_to_purity,
                                             /*default_purity=*/ignore_impurity);
        result->Add(kv.first, Downcast<Function>(eliminator_mutator.VisitExpr(function)));
      } else {
        // PrimFuncs come across unchanged.
        result->Add(kv.first, kv.second);
      }
    }
    VLOG(1) << "After:" << std::endl << PrettyPrint(result);

    return result;
  };
  return tvm::transform::CreateModulePass(pass_func, /*opt_level=*/1, "DeadCodeElimination",
                                          {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.DeadCodeElimination").set_body_typed(DeadCodeElimination);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
