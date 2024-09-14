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
 * \file src/relax/block_builder.cc
 */
#include <tvm/arith/analyzer.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/struct_info_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/relay/op.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/function.h>

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../node/ndarray_hash_equal.h"

// Block builder have three categories of logics that are interdependent with each other.
//
// The logics are somewhat interdependent with each other.
// To help us implement a block builder in two parts:
//
// - BlockBuilderImpl: implements ctx and scope management, with no normalization.
// - BlockBuilderImplWithNormalize: subclasses BlockBuilderImpl and implements normalization.
//
// The final blockbuilder create will be backed by BlockBuilderWithNormalize

namespace tvm {
namespace relax {

//---------------------------------------
// ctx and scope management.
//---------------------------------------
class BlockBuilderImpl : public BlockBuilderNode {
 public:
  explicit BlockBuilderImpl(IRModule context_mod) : context_mod_(std::move(context_mod)) {}

  ~BlockBuilderImpl() {
    if (!block_stack_.empty()) {
      LOG(WARNING) << "BlockBuilder destroyed with remaining blocks!";
    }
  }

  //-------------------------------
  // Global Context management
  //-------------------------------
  NameSupply name_supply() final { return name_supply_; }

  IRModule GetContextIRModule() const final { return context_mod_; }

  IRModule Finalize() final { return transform::NormalizeGlobalVar()(context_mod_); }

  GlobalVar AddFunction(const BaseFunc& func, String func_name_hint) final {
    LazyInitCtxFuncDedupMap();
    auto it = ctx_func_dedup_map_->find(func);
    if (it == ctx_func_dedup_map_->end()) {
      context_mod_.CopyOnWrite();

      String func_name = GetUniqueName(func_name_hint);
      while (context_mod_->ContainGlobalVar(func_name)) {
        func_name = GetUniqueName(func_name_hint);
      }
      GlobalVar gvar(func_name);

      StructInfo finfo;
      if (func->struct_info_.defined()) {
        finfo = GetStructInfo(func);
      } else if (auto* prim_func = func.as<tir::PrimFuncNode>()) {
        // NOTE: use a slightly different struct info than checked type
        // in PrimFunc so handle can turn into Tensor.
        // TODO(relax-team): add fine-grained PrimFunc struct info signature generation.
        finfo = FuncStructInfo::OpaqueFunc(StructInfoFromType(prim_func->ret_type));
      } else {
        finfo = StructInfoFromType(func->checked_type());
      }
      UpdateStructInfo(gvar, finfo);

      context_mod_->Add(gvar, func);

      (*ctx_func_dedup_map_)[func].insert(gvar);
      return gvar;
    } else {
      ICHECK(it->second.size()) << "Values contained in de-duplication map must be non-empty sets, "
                                << "but found an empty set for function of type "
                                << func->GetTypeKey();
      // To provide deterministic results, return the GlobalVar that
      // comes first in lexicographic order.
      return *std::min_element(
          it->second.begin(), it->second.end(),
          [](const GlobalVar& a, const GlobalVar& b) { return a->name_hint < b->name_hint; });
    }
  }

  void UpdateFunction(const GlobalVar& gv, BaseFunc function) final {
    context_mod_.CopyOnWrite();

    // Remove function from the de-duplication map.
    if (ctx_func_dedup_map_ != nullptr) {
      auto it = context_mod_->functions.find(gv);
      if (it != context_mod_->functions.end()) {
        BaseFunc old_func = (*it).second;
        auto ptr = ctx_func_dedup_map_->find(old_func);
        ICHECK(ptr != ctx_func_dedup_map_->end())
            << "BlockBuilder::UpdateFunction is updating " << gv
            << ", which appears in the BlockBuilder's context_mod_, "
            << "but does not appear in the de-duplication map";
        ICHECK(ptr->second.count(gv))
            << "BlockBuilder::UpdateFunction is updating " << gv
            << ", but the de-duplication map for the previous value of this function "
            << "does not include " << gv;
        ptr->second.erase(gv);
        if (ptr->second.empty()) {
          ctx_func_dedup_map_->erase(ptr);
        }
      }
    }

    context_mod_->Update(gv, function);

    // add new dedup map item.
    if (ctx_func_dedup_map_ != nullptr) {
      (*ctx_func_dedup_map_)[function].insert(gv);
    }
  }

  [[noreturn]] void ReportFatal(const Diagnostic& diagnostic) final {
    // TODO(relax-team): Print more context information by looking
    // into the diagnostic->loc and surrounding IRModule.
    // We do not materialzie DiagnosticContext to avoid double referencing to
    // the change IRModule in COW. Additionally, we need to be able to
    // continue use the builder after an error is thrown to avoid state building up.
    // in an interactive environment.
    LOG(FATAL) << diagnostic->message;
  }

  //-------------------------------
  // Scope management
  //-------------------------------
  Optional<Expr> LookupBinding(const Var& var) final {
    auto it = binding_table_.find(var->vid);
    if (it == binding_table_.end()) return NullOpt;
    return it->second;
  }

  void BeginDataflowBlock() final { block_stack_.emplace_back(BlockFrame{{}, true}); }

  void BeginBindingBlock() final { block_stack_.emplace_back(BlockFrame{{}, false}); }

  void BeginScope(Optional<Array<Var>> params) final {
    // The current implementation handles the collection of shape var
    // defined in parameter struct info annotations. The implementation
    // is correct (since we will simply erase all relax Vars in EraseToWellDefined),
    // but can be further improved.
    //
    // TODO(relax-team): Add support for relax Var in struct info annotations.

    scope_stack_.emplace_back(ScopeFrame());
    if (params.defined()) {
      for (const auto& param : params.value()) {
        AddDefinitionToScope(param);
      }
    }
  }

  void BeginInnerScope() final {
    if (scope_stack_.size()) {
      scope_stack_.emplace_back(scope_stack_.back());
    } else {
      scope_stack_.emplace_back(ScopeFrame());
    }
  }

  void AddDefinitionToScope(Var var) final {
    if (scope_stack_.empty()) {
      return;
    }

    auto& shape_var_map = CurrentScopeFrame()->shape_var_map;

    // The current implementation handles the collection of shape var
    // defined in parameter struct info annotations. The implementation
    // is correct (since we will simply erase all relax Vars in EraseToWellDefined),
    // but can be further improved.
    Map<tir::Var, PrimExpr> var_map = StructInfoVarCollector::Collect(GetStructInfo(var));
    for (const auto& kv : var_map) {
      const tir::Var& shape_var = kv.first;
      const PrimExpr& shape_expr = kv.second;
      auto it = shape_var_map.find(shape_var);
      if (it == shape_var_map.end()) {
        shape_var_map.Set(shape_var, shape_expr);
        // Expose the shape variable as non-negative, for purposes
        // of shape inference.  In many cases, knowning that the
        // shape variable is non-negative allows for simpler
        // expressions for dynamic shapes.
        analyzer_.MarkGlobalNonNegValue(shape_var);
      } else {
        const PrimExpr& old_shape_expr = (*it).second;
        CHECK(old_shape_expr.same_as(shape_expr) ||
              analyzer_.CanProveEqual(old_shape_expr, shape_expr))
            << "Inconsistent shape var " << shape_var << " in scope: " << old_shape_expr << " vs "
            << shape_expr;
      }
    }
  }

  void EndScope() final { scope_stack_.pop_back(); }

  BindingBlock EndBlock() final {
    BlockFrame* cur_frame = CurrentBlockFrame();
    BindingBlock ret = cur_frame->is_dataflow ? DataflowBlock(cur_frame->bindings)
                                              : BindingBlock(cur_frame->bindings);
    block_stack_.pop_back();
    return ret;
  }

  bool CurrentBlockIsDataFlow() final { return CurrentBlockFrame()->is_dataflow; }

  Var Emit(Expr expr, String name_hint) final {
    return this->Emit(expr, CurrentBlockFrame()->is_dataflow, name_hint);
  }

  Var EmitMatchCast(Expr value, StructInfo struct_info, String name_hint) final {
    value = this->Normalize(value);

    CHECK(StructInfoBaseCheck(GetStructInfo(value), struct_info) != BaseCheckResult::kFailL0)
        << "It is impossible to match cast any value into the target struct_info. "
           "But got value struct info: "
        << GetStructInfo(value) << ", given struct info: " << struct_info;

    // NOTE: do match cast checking later in a pass.
    BlockFrame* cur_frame = CurrentBlockFrame();
    Var var = CreateVar(cur_frame->is_dataflow, name_hint);
    UpdateStructInfo(var, struct_info);

    MatchCast match_cast(var, value, struct_info);
    cur_frame->bindings.push_back(match_cast);
    // NOTE match shape do not follow simple binding rule
    // as a result should not appear in binding table.

    AddDefinitionToScope(var);
    return var;
  }

  Var EmitOutput(Expr output, String name_hint) final {
    BlockFrame* cur_frame = CurrentBlockFrame();

    ICHECK(cur_frame->is_dataflow) << "EmitOutput has to be called inside dataflow block.";

    return Emit(output, false, name_hint);
  }

  void EmitNormalized(Binding binding) final {
    BlockFrame* cur_frame = CurrentBlockFrame();

    if (const auto* var_binding = binding.as<VarBindingNode>()) {
      if (!cur_frame->is_dataflow) {
        ICHECK(!var_binding->var.as<DataflowVarNode>())
            << "Cannot emit dataflow var in non-dataflow block";
      }
      // normalized check
      ICHECK(var_binding->var->struct_info_.defined());
      ICHECK(var_binding->value->struct_info_.defined());
      cur_frame->bindings.push_back(binding);
      binding_table_[var_binding->var->vid] = var_binding->value;
    } else if (const auto* match_cast = binding.as<MatchCastNode>()) {
      if (!cur_frame->is_dataflow) {
        ICHECK(!match_cast->var.as<DataflowVarNode>())
            << "Cannot emit dataflow var in non-dataflow block";
      }
      // normalized check
      ICHECK(match_cast->var->struct_info_.defined());
      ICHECK(match_cast->value->struct_info_.defined());
      // NOTE match shape do not follow simple binding rule
      // as a result should not appear in binding table.
      cur_frame->bindings.push_back(binding);
      AddDefinitionToScope(match_cast->var);
    } else {
      LOG(FATAL) << "Unsupported binding type: " << binding->GetTypeKey();
    }
  }

  arith::Analyzer* GetAnalyzer() final { return &analyzer_; }

 protected:
  /*!
   * \brief A representation of a block frame.
   *
   * A block frame is a record containing the bindings needed
   * to build a binding block, and a boolean to indicate if the
   * block being built is a DataflowBlock or not.
   */
  struct BlockFrame {
    /*!
     * \brief List of bindings
     */
    Array<Binding> bindings;
    /*! \brief Whether current block is dataflow block. */
    bool is_dataflow;
    /*!
     * \brief Binding map used by normalizer.
     *
     * \note The normalizer only caches reuse in the current block scope
     *       and will not cache bindings from parent scope.
     */
    std::unordered_map<Expr, Var, ObjectPtrHash, ObjectPtrEqual> normalize_binding_map;
  };
  /*!
   * \brief A representation of a scope frame.
   *
   * A scope frame records tracks the context of current scope.
   */
  struct ScopeFrame {
    // NOTE: for simplicity, only tracks symbolic var for now
    // the scope is only used for erasure, so less information means
    // more conservative analysis.
    // Consider impl alternative: merge with block frame if we have more frame kinds.
    //
    // TODO(relax-team) tracks the var defined also through match-cast.
    /*! \brief set of defined symbolic vars, value as themself. */
    Map<tir::Var, PrimExpr> shape_var_map;
  };

  /*! \brief A stack to store block frames. */
  std::vector<BlockFrame> block_stack_;

  /*! \brief A stack to store scope frames. */
  std::vector<ScopeFrame> scope_stack_;

  /*! \brief A binding table that maps var to value. */
  std::unordered_map<Id, Expr, ObjectPtrHash, ObjectPtrEqual> binding_table_;

  /*! \brief A name supply to get unique names for IR construction. */
  NameSupply name_supply_;

  /*! \brief The IRModule being built by the BlockBuilder. */
  IRModule context_mod_;

  /*! \brief Internal analzyer */
  arith::Analyzer analyzer_;

  /*!
   * \return The current frame.
   * \note Never hold the value of current frame between Normalize
   *       or other scope calls this value can change if the block stack get updated,
   *       then the block frame is no longer valid.
   */
  BlockFrame* CurrentBlockFrame() {
    ICHECK(!block_stack_.empty()) << "no block is being built";
    return &block_stack_.back();
  }

  /*!
   * \return The current scope frame.
   * \note only use this value
   */
  ScopeFrame* CurrentScopeFrame() {
    ICHECK(!scope_stack_.empty()) << "no scope is being opened";
    return &scope_stack_.back();
  }

  /*!
   * \brief Emits an Expr, and returns the variable it is bound to.
   * \param expr The Expr to be emitted.
   * \param is_dataflow Is the bound variable a DataflowVar or not(i.e. Var).
   * \param name_hint Name hint for the bound variable.
   * \note This Emit function normalizes the \p expr,
   *       and performs shape/type deductions by calling Normalize.
   * \return The new variable that \p expr is bound to.
   */
  Var Emit(Expr expr, bool is_dataflow, String name_hint) {
    expr = this->Normalize(expr);

    Var var = CreateVar(is_dataflow, name_hint);

    // set the values
    UpdateStructInfo(var, Downcast<StructInfo>(expr->struct_info_.value()));

    CurrentBlockFrame()->bindings.push_back(VarBinding(var, expr));

    // update the binding table
    binding_table_[var->vid] = expr;

    return var;
  }

  /*!
   * \brief Create var for bindings
   * \param is_dataflow Is the bound variable a DataflowVar or not(i.e. Var).
   * \param name_hint Name hint for the bound variable.
   * \return The created var.
   */
  Var CreateVar(bool is_dataflow, String name_hint) {
    if (name_hint.empty()) {
      name_hint = is_dataflow ? "lv" : "gv";
    }
    Id vid = Id(GetUniqueName(name_hint));
    return is_dataflow ? DataflowVar(vid, /*struct_info_annotation=*/NullOpt)
                       : Var(vid, /*struct_info_annotation=*/NullOpt);
  }

 private:
  std::string GetUniqueName(const std::string& prefix) {
    return name_supply_->FreshName(prefix, /*add_prefix*/ false, /*add_underscore*/ false);
  }

  /*! \brief A custom structural hashing that ignores NDArray raw data. */
  class StructuralHashIgnoreNDarray : public BaseValueHash {
   public:
    using BaseValueHash::operator();

    uint64_t operator()(const ObjectRef& key) const {
      return SHashHandlerIgnoreNDArray().Hash(key, false);
    }
  };

  /*!
   * \brief A hashmap to store the mapping of Relax functions and TIR PrimFuncs
   * in context_mod to their GlobalVar to avoid generating duplicated functions.
   * We use a custom hash to avoid hashing constants that may be bound to each BaseFunc.
   */
  std::unique_ptr<
      std::unordered_map<BaseFunc, std::unordered_set<GlobalVar, ObjectPtrHash, ObjectPtrEqual>,
                         StructuralHashIgnoreNDarray, StructuralEqual>>
      ctx_func_dedup_map_ = nullptr;

  /*!
   * \brief lazily initialize function dedeup map.
   */
  void LazyInitCtxFuncDedupMap() {
    if (ctx_func_dedup_map_ != nullptr) return;
    ctx_func_dedup_map_ = std::make_unique<
        std::unordered_map<BaseFunc, std::unordered_set<GlobalVar, ObjectPtrHash, ObjectPtrEqual>,
                           StructuralHashIgnoreNDarray, StructuralEqual>>();
    for (const auto& kv : context_mod_->functions) {
      const GlobalVar gv = kv.first;
      const BaseFunc func = kv.second;
      (*ctx_func_dedup_map_)[func].insert(gv);
    }
  }

  // Collect all the variables that a parameter var can define.
  // The collector is used to making sure that we record the
  // shape vars as defined when calling BeginScope(params)
  class StructInfoVarCollector : public StructInfoVisitor {
   public:
    static Map<tir::Var, PrimExpr> Collect(const StructInfo& struct_info) {
      StructInfoVarCollector collector;
      collector(struct_info);
      return collector.shape_var_map_;
    }

   private:
    void VisitStructInfo_(const TensorStructInfoNode* op) final {
      if (const auto* shape_expr = op->shape.as<ShapeExprNode>()) {
        for (const PrimExpr& s : shape_expr->values) {
          // Only collect single var defined shape. Ignore something like `R.Tensor((m + 1, n + 1))
          if (const auto* var = s.as<tir::VarNode>()) {
            shape_var_map_.Set(GetRef<tir::Var>(var), s);
          }
        }
      }
    }

    void VisitStructInfo_(const ShapeStructInfoNode* op) final {
      for (const PrimExpr& s : op->values.value_or(Array<PrimExpr>())) {
        // Only collect single var defined shape. Ignore something like `R.Shape((m + 1, n + 1))
        if (const auto* var = s.as<tir::VarNode>()) {
          shape_var_map_.Set(GetRef<tir::Var>(var), s);
        }
      }
    }

    void VisitStructInfo_(const PrimStructInfoNode* op) final {
      // Only collect single var defined shape. Ignore something like `R.Prim(value=m + 1)`
      if (op->value.defined()) {
        if (auto var = op->value.as<tir::Var>()) {
          shape_var_map_.Set(var.value(), op->value.value());
        }
      }
    }

   private:
    Map<tir::Var, PrimExpr> shape_var_map_;
  };
};

//---------------------------------------
// Normalization
//---------------------------------------
#define RELAX_EXPR_NORMALIZER_LEAF(OP) \
  Expr VisitExpr_(const OP* op) final { return GetRef<Expr>(op); }

// TODO(relax-team): Check normalize logic after struct info.

// Normalizer on struct info:
//
// We take benefit of the following invariants(that are checked in constructor):
// - If an expr appears in StructInfo, then it is already normalized.
//   As a result, we do not need to peek into StructInfo in Normalization.
// - Constant, ShapeExpr, already have their StructInfo populated in constructing time.
class Normalizer : public BlockBuilderImpl, private ExprFunctor<Expr(const Expr&)> {
 public:
  explicit Normalizer(IRModule context_mod) : BlockBuilderImpl(context_mod) {}
  explicit Normalizer(IRModule context_mod,
                      BlockBuilder::DisableOperatorSpecificNormalizationForTVMScript)
      : BlockBuilderImpl(context_mod), apply_f_normalize_(false) {}

  Expr Normalize(const Expr& expr) final {
    Expr normalized = this->VisitExpr(expr);
    // Invariant:
    // After Normalize: an Expr always have
    // struct_info (with the exception of Op).
    if (!normalized->IsInstance<OpNode>()) {
      ICHECK(normalized->struct_info_.defined())
          << "The struct_info_ of an Expr except OpNode after "
             "normalization must not be nullptr. However, this Expr does not have struct_info_: "
          << normalized;
    }

    return normalized;
  }

  /*!
   * \brief Normalize Argument values to call and other IR sub-fields.
   * \param arg The argument.
   * \return The normalized value.
   *
   * \note This function create a new binding for non-leaf expressions except for tuple.
   */
  Expr NormalizeArgument(const Expr& arg) final {
    if (!block_stack_.empty()) {
      // cache lookup
      BlockFrame* cur_frame = CurrentBlockFrame();
      auto it = cur_frame->normalize_binding_map.find(arg);
      if (it != cur_frame->normalize_binding_map.end()) {
        return it->second;
      }
    }
    // skip visit expr's cache, normalize arg
    Expr post = ExprFunctor::VisitExpr(arg);

    if (!IsLeafOrTuple(arg)) {
      ICHECK(!block_stack_.empty()) << "Cannot normalize non-leaf without a scope";
      Var var = this->Emit(post, "");
      // NOTE: current frame addr can change due to underlying vector
      // re-allocation, redo lookup
      CurrentBlockFrame()->normalize_binding_map[arg] = var;
      return var;
    } else {
      return post;
    }
  }

  RELAX_EXPR_NORMALIZER_LEAF(ExternFuncNode);
  RELAX_EXPR_NORMALIZER_LEAF(GlobalVarNode);
  RELAX_EXPR_NORMALIZER_LEAF(OpNode);
  RELAX_EXPR_NORMALIZER_LEAF(ConstantNode);
  RELAX_EXPR_NORMALIZER_LEAF(ShapeExprNode);
  RELAX_EXPR_NORMALIZER_LEAF(PrimValueNode);
  RELAX_EXPR_NORMALIZER_LEAF(StringImmNode);
  RELAX_EXPR_NORMALIZER_LEAF(DataTypeImmNode);

  template <typename T>
  Expr VisitVar_(const typename T::ContainerType* var) {
    // Parameters and free-vars must be present with struct info
    // Other vars must have already been normalized through binding
    ICHECK(var->struct_info_.defined())
        << "Var " << var->name_hint() << " does not have struct info.";
    return GetRef<Var>(var);
  }

  Expr VisitExpr_(const VarNode* var_ptr) final {
    auto var = VisitVar_<Var>(var_ptr);
    if (HasVoidStructInfo(var)) {
      return VisitExpr(Tuple(Array<Expr>{}));
    } else {
      return var;
    }
  }

  Expr VisitExpr_(const DataflowVarNode* var) final { return VisitVar_<DataflowVar>(var); }

  Expr VisitExpr(const Expr& expr) final {
    // lookup normalize map
    if (!block_stack_.empty()) {
      BlockFrame* cur_frame = CurrentBlockFrame();
      auto it = cur_frame->normalize_binding_map.find(expr);
      if (it != cur_frame->normalize_binding_map.end()) {
        return it->second;
      }
    }
    return ExprFunctor::VisitExpr(expr);
  }

  Expr VisitExpr_(const TupleNode* op) final {
    bool unchanged = true;
    Array<Expr> new_fields;

    for (const Expr& field : op->fields) {
      Expr new_field = this->NormalizeArgument(field);
      new_fields.push_back(new_field);
      unchanged &= new_field.same_as(field);
    }

    Tuple tuple = unchanged ? GetRef<Tuple>(op) : Tuple(new_fields, op->span);
    // Update tuple fields.
    if (!tuple->struct_info_.defined()) {
      Array<StructInfo> tuple_sinfo;
      for (Expr field : tuple->fields) {
        tuple_sinfo.push_back(GetStructInfo(field));
      }
      UpdateStructInfo(tuple, TupleStructInfo(tuple_sinfo, op->span));
    }
    return tuple;
  }

  Expr VisitExpr_(const FunctionNode* op) final {
    Expr new_body = this->VisitWithNewScope(op->body, op->params);

    if (new_body.same_as(op->body)) {
      return GetRef<Function>(op);
    } else {
      return Function(op->params, new_body, op->ret_struct_info, op->is_pure, op->attrs);
    }
  }

  Expr VisitExpr_(const CallNode* op) final {
    Expr new_op = this->NormalizeArgument(op->op);

    Array<Expr> new_args = op->args.Map([this](const Expr& arg) { return NormalizeArgument(arg); });

    Call call;
    if (new_op.same_as(op->op) && new_args.same_as(op->args)) {
      call = GetRef<Call>(op);
    } else {
      call = Call(new_op, new_args, op->attrs, op->sinfo_args);
    }

    if (!call->struct_info_.defined()) {
      auto inferred_sinfo = InferStructInfo(call);
      UpdateStructInfo(call, inferred_sinfo);
    }

    // If the operation has defined a custom normalization
    // function using the FNormalize attribute, apply it.  If the
    // normalization modified the expression, re-visit in case it
    // produced a nested expression.
    if (apply_f_normalize_) {
      if (auto func_normalize = op_map_normalize_.get(op->op, nullptr); func_normalize != nullptr) {
        Expr normalized = func_normalize(GetRef<BlockBuilder>(this), call);
        if (!normalized.same_as(call)) {
          return VisitExpr(normalized);
        }
      }
    }

    return call;
  }

  Expr VisitExpr_(const SeqExprNode* op) final {
    bool unchanged = true;
    Array<BindingBlock> new_blocks;
    for (BindingBlock block : op->blocks) {
      BindingBlock new_block = this->VisitBindingBlock(block);
      new_blocks.push_back(new_block);
      unchanged &= new_block.same_as(block);
    }

    // Because the input may not be normalized, the SeqExpr may occur
    // nested within another SeqExpr.  In that case, we want to use
    // whatever binding-block type the parent uses, so that we any
    // bindings collected into the prologue will be compatible with
    // the parent block.
    if (block_stack_.size() && CurrentBlockIsDataFlow()) {
      this->BeginDataflowBlock();
    } else {
      this->BeginBindingBlock();
    }

    // the body may not be a leaf expression, so check for that
    Expr new_body = this->NormalizeArgument(op->body);
    unchanged &= new_body.same_as(op->body);
    BindingBlock prologue = this->EndBlock();

    if (!prologue->bindings.empty()) {
      new_blocks.push_back(prologue);
      unchanged = false;
    }

    // Combine nearby blocks if possible
    Array<BindingBlock> normalized_blocks = NormalizeBlocks(new_blocks);
    unchanged &= normalized_blocks.same_as(new_blocks);

    SeqExpr seq_expr;
    if (unchanged) {
      seq_expr = GetRef<SeqExpr>(op);
    } else {
      seq_expr = SeqExpr(normalized_blocks, new_body, op->span);
    }

    // only do shape/type inference if the SeqExpr does not have shape/type
    if (!seq_expr->struct_info_.defined()) {
      UpdateStructInfo(seq_expr, EraseToWellDefinedInScope(GetStructInfo(seq_expr->body)));
    }
    return seq_expr;
  }

  Expr VisitExpr_(const IfNode* op) final {
    Expr new_cond = this->NormalizeArgument(op->cond);
    Expr new_true = this->VisitWithNewScope(op->true_branch);
    Expr new_false = this->VisitWithNewScope(op->false_branch);

    If if_node;
    if (new_cond.same_as(op->cond) && new_true.same_as(op->true_branch) &&
        new_false.same_as(op->false_branch)) {
      if_node = GetRef<If>(op);
    } else {
      if_node = If(new_cond, new_true, new_false, op->span);
    }
    if (!if_node->struct_info_.defined()) {
      auto true_info = EraseToWellDefinedInScope(GetStructInfo(new_true));
      auto false_info = EraseToWellDefinedInScope(GetStructInfo(new_false));
      UpdateStructInfo(if_node, StructInfoLCA(true_info, false_info));
    }
    return if_node;
  }

  Expr VisitExpr_(const TupleGetItemNode* op) final {
    Expr new_tuple = this->NormalizeArgument(op->tuple);

    TupleGetItem node = new_tuple.same_as(op->tuple) ? GetRef<TupleGetItem>(op)
                                                     : TupleGetItem(new_tuple, op->index);

    if (!node->struct_info_.defined()) {
      auto opt = MatchStructInfo<TupleStructInfo>(node->tuple);
      ICHECK(opt) << "The struct info of Tuple must be TupleStructInfo, "
                  << "but expression " << node->tuple << " has struct info "
                  << node->tuple->struct_info_;
      UpdateStructInfo(node, opt.value()->fields[node->index]);
    }

    return node;
  }

  Binding VisitBinding(const Binding& binding) {
    if (auto* var_binding = binding.as<VarBindingNode>()) {
      return this->VisitVarBinding(GetRef<VarBinding>(var_binding));
    } else {
      auto* match_cast = binding.as<MatchCastNode>();
      ICHECK(match_cast) << "Unsupported binding type: " << binding->GetTypeKey();
      return this->VisitMatchCast(GetRef<MatchCast>(match_cast));
    }
  }

  VarBinding VisitVarBinding(VarBinding binding) {
    Expr new_value = this->VisitExpr(binding->value);
    if (!new_value.same_as(binding->value)) {
      binding = VarBinding(binding->var, new_value, binding->span);
    }
    if (!binding->var->struct_info_.defined()) {
      UpdateStructInfo(binding->var, GetStructInfo(new_value));
    }
    return binding;
  }

  MatchCast VisitMatchCast(MatchCast binding) {
    Expr new_value = this->VisitExpr(binding->value);
    if (!new_value.same_as(binding->value)) {
      binding = MatchCast(binding->var, new_value, binding->struct_info, binding->span);
    }
    if (!binding->var->struct_info_.defined()) {
      UpdateStructInfo(binding->var, binding->struct_info);
    }
    return binding;
  }

  BindingBlock VisitBindingBlock(const BindingBlock& block) {
    if (block.as<DataflowBlockNode>()) {
      this->BeginDataflowBlock();
    } else {
      this->BeginBindingBlock();
    }

    bool unchanged = true;
    for (const Binding& binding : block->bindings) {
      Binding new_binding = this->VisitBinding(binding);
      unchanged &= new_binding.same_as(binding);

      this->EmitNormalized(new_binding);
    }
    BindingBlock new_block = this->EndBlock();
    unchanged &= new_block->bindings.size() == block->bindings.size();
    if (unchanged) {
      return block;
    }
    return new_block;
  }

 private:
  // Helper function to infer the type of a Call.
  StructInfo InferStructInfo(const Call& call) {
    if (auto* op_ptr = call->op.as<OpNode>()) {
      // Case 1: the op field is a primitive op, look up FInferStructInfo attribute
      Op op = GetRef<Op>(op_ptr);
      bool is_dist_op = false;
      for (const auto& arg : call->args) {
        if (arg->struct_info_.as<distributed::DTensorStructInfoNode>()) {
          is_dist_op = true;
          break;
        }
      }
      if (is_dist_op) {
        for (const auto& arg : call->args) {
          ICHECK(!arg->struct_info_.as<TensorStructInfoNode>())
              << "Distributed operator must take DTensor instead of Tensor as input";
        }
        ICHECK(op_map_dist_infer_struct_info_.count(op))
            << " Cannot find the dist.FInferStructInfo attribute registered to op: " << op->name;
        return op_map_dist_infer_struct_info_[op](call, GetRef<BlockBuilder>(this));
      }
      ICHECK(op_map_infer_struct_info_.count(op))
          << " Cannot find the FInferStructInfo attribute registered to op: " << op->name;
      return op_map_infer_struct_info_[op](call, GetRef<BlockBuilder>(this));
    } else {
      // derive using function parameters
      ICHECK(call->op->struct_info_.defined());
      auto opt = MatchStructInfo<FuncStructInfo>(call->op);
      ICHECK(opt) << "Call->op must contains a function struct info";
      FuncStructInfo finfo = opt.value();
      return DeriveCallRetStructInfo(finfo, call, GetRef<BlockBuilder>(this), &analyzer_);
    }
  }

  // erase to well defined within current scope.
  StructInfo EraseToWellDefinedInScope(StructInfo info) {
    if (scope_stack_.empty()) {
      // If no scopes are active, then this fragment does not require
      // any normalization.
      return info;
    }
    auto* curr_scope = CurrentScopeFrame();
    auto f_shape_var_map = [curr_scope](tir::Var var) -> Optional<PrimExpr> {
      auto it = curr_scope->shape_var_map.find(var);
      if (it != curr_scope->shape_var_map.end()) return (*it).second;
      return NullOpt;
    };
    return EraseToWellDefined(info, f_shape_var_map);
  }

  Expr VisitWithNewScope(const Expr& expr, Optional<Array<Var>> params = NullOpt) {
    if (params.defined()) {
      this->BeginScope(params.value());
    } else {
      this->BeginInnerScope();
    }

    Expr ret;

    // SeqExpr do not need to prepare for normalization.
    if (expr.as<SeqExprNode>()) {
      ret = this->VisitExpr(expr);
    } else {
      this->BeginBindingBlock();
      Expr post = this->NormalizeArgument(expr);
      BindingBlock prologue = this->EndBlock();
      // "New scopes" (function bodies, if/else clauses) must be wrapped in seq exprs.
      // Don't wrap if it's already a seq and there are no bindings to add
      if (post.as<SeqExprNode>() && prologue->bindings.empty()) {
        return post;
      }
      Array<BindingBlock> bindings;
      if (!prologue->bindings.empty()) {
        bindings.push_back(prologue);
      }

      SeqExpr seq(bindings, post);
      UpdateStructInfo(seq, EraseToWellDefinedInScope(GetStructInfo(seq->body)));

      ret = seq;
    }

    this->EndScope();
    return ret;
  }

  Array<BindingBlock> FlattenBlocks(const Array<BindingBlock>& blocks) {
    // If there is a binding that is a seq expr, split the current block,
    // add the nested blocks prior to the seq expr, and bind the seq expr body
    // to the var
    Array<BindingBlock> ret;
    bool changed = false;
    for (const BindingBlock& block : blocks) {
      bool is_dataflow = block->IsInstance<DataflowBlockNode>();
      Array<Binding> current;
      for (const Binding& binding : block->bindings) {
        Expr value;
        if (const auto* var_binding = binding.as<VarBindingNode>()) {
          value = var_binding->value;
        } else if (const auto* match_cast = binding.as<MatchCastNode>()) {
          value = match_cast->value;
        } else {
          LOG(FATAL) << "Unknown binding type: " << binding->GetTypeKey();
        }
        // if we encounter a nested seq, we have to flatten it:
        //   1. Append the binding block we've accumulated so far
        //   2. Reset the current block
        //   3. Append the inner blocks
        //   4. Add a binding of the current var to the seq expr's body to the current block
        // then continue
        if (auto seq = value.as<SeqExprNode>()) {
          changed = true;
          ret.push_back(is_dataflow ? DataflowBlock(current) : BindingBlock(current));
          current = {};
          // We do not need to flatten recursively because the normalizer will have normalized
          // and thus flattened the inner SeqExprs already
          for (const BindingBlock& block : seq->blocks) {
            if (is_dataflow && !block->IsInstance<DataflowBlockNode>()) {
              // A DataflowBlock occurring within a non-DataflowBlock
              // usually is an error, resulting from return of a
              // `BindingBlock`.  However, it may still be well-formed
              // if there are no relax::DataflowVar instances used by
              // the non-DataflowBlock.  This would result in multiple
              // dataflow sections, split by non-dataflow portions,
              // but would still be valid.
              //
              // Since the most common occurrence is due to mis-use,
              // explicitly check for it here rather than waiting for a
              // WellFormed check later on.

              auto free_vars = FreeVars(SeqExpr({block}, Tuple(Array<Expr>{})));
              Array<DataflowVar> free_dataflow_vars;
              for (const auto& var : free_vars) {
                if (auto opt = var.as<DataflowVar>()) {
                  free_dataflow_vars.push_back(opt.value());
                }
              }

              if (free_dataflow_vars.size()) {
                LOG(FATAL)
                    << "Malformed AST: "
                    << "A DataflowVar may only be used within a DataflowBlock.  "
                    << "The variable " << binding->var << " is defined within a DataflowBlock, "
                    << "but is bound to a SeqExpr that contains non-dataflow BindingBlocks.  "
                    << "These non-dataflow BindingBlocks use the DataflowVars "
                    << free_dataflow_vars << ", which is invalid.";
              }
            }
            ret.push_back(block);
          }

          if (const auto* var_binding = binding.as<VarBindingNode>()) {
            current.push_back(VarBinding(var_binding->var, seq->body));
          } else if (const auto* match_cast = binding.as<MatchCastNode>()) {
            current.push_back(MatchCast(match_cast->var, seq->body, match_cast->struct_info));
          } else {
            LOG(FATAL) << "Unknown binding type: " << binding->GetTypeKey();
          }
        } else {
          current.push_back(binding);
        }
      }
      ret.push_back(is_dataflow ? DataflowBlock(current) : BindingBlock(current));
    }
    return changed ? ret : blocks;
  }

  Array<BindingBlock> NormalizeBlocks(const Array<BindingBlock>& blocks) {
    bool changed = false;
    Array<BindingBlock> ret;
    auto flattened = FlattenBlocks(blocks);
    if (!flattened.same_as(blocks)) {
      changed = true;
    }
    for (const BindingBlock& block : flattened) {
      if (block->bindings.empty()) {
        // Case 1. Skip empty blocks
        changed = true;
      } else if (!ret.empty() && ret.back()->type_index() == block->type_index()) {
        // Case 2. Merge with previous block if possible
        BindingBlock merged;
        // NOTE: should check DataflowBlockNode first.
        if (const auto* dataflow_block = ret.back().as<DataflowBlockNode>()) {
          auto n = make_object<DataflowBlockNode>(*dataflow_block);
          n->bindings.insert(n->bindings.end(), block->bindings.begin(), block->bindings.end());
          merged = DataflowBlock(n);
        } else if (const auto* binding_block = ret.back().as<BindingBlockNode>()) {
          auto n = make_object<BindingBlockNode>(*binding_block);
          n->bindings.insert(n->bindings.end(), block->bindings.begin(), block->bindings.end());
          merged = BindingBlock(n);
        } else {
          LOG(FATAL) << "Unknown block type: " << ret.back()->GetTypeKey();
        }
        ret.pop_back();
        ret.push_back(merged);
        changed = true;
      } else {
        // Case 3. Add to the result
        ret.push_back(block);
      }
    }
    return changed ? ret : blocks;
  }

  /*! \brief Operator struct info inference map. */
  tvm::OpAttrMap<FInferStructInfo> op_map_infer_struct_info_ =
      Op::GetAttrMap<FInferStructInfo>("FInferStructInfo");
  tvm::OpAttrMap<FInferStructInfo> op_map_dist_infer_struct_info_ =
      Op::GetAttrMap<FInferStructInfo>("dist.FInferStructInfo");
  /*! \brief Operator normalization function */
  tvm::OpAttrMap<FNormalize> op_map_normalize_ = Op::GetAttrMap<FNormalize>("FNormalize");

  /*! \brief Whether the FNormalize function should be applied */
  bool apply_f_normalize_{true};
};

BlockBuilder BlockBuilder::Create(Optional<IRModule> mod) {
  ObjectPtr<BlockBuilderNode> n = make_object<Normalizer>(mod.value_or(IRModule()));
  return BlockBuilder(n);
}

BlockBuilder BlockBuilder::Create(Optional<IRModule> mod,
                                  BlockBuilder::DisableOperatorSpecificNormalizationForTVMScript) {
  ObjectPtr<BlockBuilderNode> n = make_object<Normalizer>(
      mod.value_or(IRModule()), BlockBuilder::DisableOperatorSpecificNormalizationForTVMScript());
  return BlockBuilder(n);
}

//---------------------------------------
// User facing function registration.
//---------------------------------------
TVM_REGISTER_OBJECT_TYPE(BlockBuilderNode);

TVM_REGISTER_GLOBAL("relax.BlockBuilderCreate").set_body_typed([](Optional<IRModule> mod) {
  return BlockBuilder::Create(mod);
});

TVM_REGISTER_GLOBAL("relax.BlockBuilderBeginDataflowBlock")
    .set_body_method<BlockBuilder>(&BlockBuilderNode::BeginDataflowBlock);

TVM_REGISTER_GLOBAL("relax.BlockBuilderBeginBindingBlock")
    .set_body_method<BlockBuilder>(&BlockBuilderNode::BeginBindingBlock);

TVM_REGISTER_GLOBAL("relax.BlockBuilderEndBlock")
    .set_body_method<BlockBuilder>(&BlockBuilderNode::EndBlock);

TVM_REGISTER_GLOBAL("relax.BlockBuilderNormalize")
    .set_body_method<BlockBuilder>(&BlockBuilderNode::Normalize);

TVM_REGISTER_GLOBAL("relax.BlockBuilderEmit")
    .set_body_typed([](BlockBuilder builder, Expr expr, String name_hint) {
      return builder->Emit(expr, name_hint);
    });

TVM_REGISTER_GLOBAL("relax.BlockBuilderEmitMatchCast")
    .set_body_typed([](BlockBuilder builder, Expr value, StructInfo struct_info, String name_hint) {
      return builder->EmitMatchCast(value, struct_info, name_hint);
    });

TVM_REGISTER_GLOBAL("relax.BlockBuilderEmitOutput")
    .set_body_typed([](BlockBuilder builder, const Expr& output, String name_hint) {
      return builder->EmitOutput(output, name_hint);
    });

TVM_REGISTER_GLOBAL("relax.BlockBuilderEmitNormalized")
    .set_body_typed([](BlockBuilder builder, Binding binding) {
      return builder->EmitNormalized(binding);
    });

TVM_REGISTER_GLOBAL("relax.BlockBuilderGetUniqueName")
    .set_body_typed([](BlockBuilder builder, String name_hint) {
      return builder->name_supply()->FreshName(name_hint, /*add_prefix*/ false,
                                               /*add_underscore*/ false);
    });

TVM_REGISTER_GLOBAL("relax.BlockBuilderAddFunction")
    .set_body_method<BlockBuilder>(&BlockBuilderNode::AddFunction);

TVM_REGISTER_GLOBAL("relax.BlockBuilderUpdateFunction")
    .set_body_method<BlockBuilder>(&BlockBuilderNode::UpdateFunction);

TVM_REGISTER_GLOBAL("relax.BlockBuilderGetContextIRModule")
    .set_body_method<BlockBuilder>(&BlockBuilderNode::GetContextIRModule);

TVM_REGISTER_GLOBAL("relax.BlockBuilderFinalize")
    .set_body_method<BlockBuilder>(&BlockBuilderNode::Finalize);

TVM_REGISTER_GLOBAL("relax.BlockBuilderCurrentBlockIsDataFlow")
    .set_body_method<BlockBuilder>(&BlockBuilderNode::CurrentBlockIsDataFlow);

TVM_REGISTER_GLOBAL("relax.BlockBuilderLookupBinding")
    .set_body_method<BlockBuilder>(&BlockBuilderNode::LookupBinding);

TVM_REGISTER_GLOBAL("relax.BlockBuilderBeginScope")
    .set_body_method<BlockBuilder>(&BlockBuilderNode::BeginScope);

TVM_REGISTER_GLOBAL("relax.BlockBuilderEndScope")
    .set_body_method<BlockBuilder>(&BlockBuilderNode::EndScope);
}  // namespace relax
}  // namespace tvm
