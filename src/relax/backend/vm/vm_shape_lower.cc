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
 * \file src/relax/backend/vm/vm_shape_lower.cc
 * \brief Lower the function boundary type checks and symbolic shape computations.
 */
#include <tvm/relax/analysis.h>
#include <tvm/relax/backend.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/struct_info_functor.h>
#include <tvm/runtime/relax_vm/builtin.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace relax {

/*! \brief A slot used in PrimExpr lowering. */
struct PrimExprSlot {
  /*! \brief The existing */
  PrimExpr expr;
  /*! \brief The slot index */
  int index;
  // The following three members are auxiliary data
  // to help shape rewriting.
  /*!
   * \brief List of slots whose PrimExpr uses this PrimExpr.
   * \note Users won't be empty only if PrimExpr is a Var and it does not include itself.
   */
  std::vector<PrimExprSlot*> user_slots;
  /*!
   * \brief Number of outstanding vars that are not defined in this PrimExpr.
   * \note This is a helper counter used in analysis to perform computations.
   */
  int outstanding_defs = 0;
  /*! \brief Whether we have computed the value. */
  bool value_computed = false;
};

/*!
 * \brief Helper dats structure to collect pairs of match shapes
 *        in a recursive matching process.
 */
struct MatchShapeTodoItem {
  Expr input;
  Array<PrimExpr> pattern;
  String err_ctx;
};

/*! \brief Slot map used for shape lowering. */
using PrimExprSlotMap =
    std::unordered_map<PrimExpr, PrimExprSlot*, StructuralHash, tir::ExprDeepEqual>;

// Collector to collect PrimExprSlotMap
class PrimExprSlotCollector : public ExprVisitor, public StructInfoVisitor {
 public:
  // collect the PrimExpr slot for a given function
  static void Collect(Function func, std::vector<std::unique_ptr<PrimExprSlot>>* slot_vec,
                      PrimExprSlotMap* slot_map) {
    PrimExprSlotCollector collector;
    collector.slot_vec_ = slot_vec;
    collector.slot_map_ = slot_map;
    // collect shape declaration in func params
    for (auto param : func->params) {
      collector.VisitStructInfo(GetStructInfo(param));
      collector.VisitExpr(param);
    }
    collector.VisitExpr(func->body);
    collector.VisitStructInfo(func->ret_struct_info);
  }

 private:
  void VisitPrimExpr(const PrimExpr& expr) final {
    if (expr->IsInstance<IntImmNode>()) return;
    if (slot_map_->count(expr) == 0) {
      auto slot = std::make_unique<PrimExprSlot>();
      slot->expr = expr;
      slot->index = static_cast<int>(slot_vec_->size());
      slot_map_->emplace(expr, slot.get());
      slot_vec_->emplace_back(std::move(slot));
    }
  }

  void VisitBinding_(const MatchCastNode* op) final {
    // Visit the match cast struct info so we can define
    // the symbolic variables here.
    this->VisitStructInfo(op->struct_info);
  }

  void VisitExpr_(const FunctionNode* op) final {
    // Do not recurse into function node as it is self-contained
  }

  void VisitStructInfo_(const FuncStructInfoNode* op) final {
    // Do not recurse into function struct info as it is self-contained
  }

  void VisitStructInfoExprField(const PrimExpr& expr) final { VisitPrimExpr(expr); }

  void VisitStructInfoExprField(const Expr& expr) final { ExprVisitor::VisitExpr(expr); }

  std::vector<std::unique_ptr<PrimExprSlot>>* slot_vec_;
  PrimExprSlotMap* slot_map_;
};

/*!
 * \brief Main logic to transform the shape lowered functions
 *
 * Consider the following input:
 *
 * \code
 *
 *  def f(x: R.Tuple(R.Tensor([m, n+1]), R.Tensor([n, 2])) -> R.Tensor:
 *     return x
 *
 * \endcode
 *
 * Overall flow of the algorithm:
 * - Preprocess: PrimExprSlot collection, we scan the function and allocate PrimExprSlot
 *   for each PrimExpr. In the above example, the result mapping from the slot index
 *   to expr would be {0:m, 1: n+1: 2: n}. Note that "n+1" also get a slot.
 *   PrimExprSlot also comes with auxiliary fields that track whether its value
 *   can be readily computed.
 *
 * Steps at each matching point:
 * - Step 0: We call CheckMatchCast,
 *   which will recursively unpack the StructInfo, and generate static information checks.
 *   Note that this step only generates functions for checking types and ndim info, but not
 *   the symbolic shape variables. The symbolic shape-matching results will be returned as
 *   vector<MatchShapeTodoItem>. This is because symbolic shape matching may not be completed
 *   in a single round. Importantly, CheckMatchCast also deals with tuple unpacking.
 *
 * - Step 1: We then call RunMatch to generate the statements for matching symbolic shapes.
 *   In the above example, the first round will store the value of m, n to their corresponding
 *   slot. RunMatch may return outstanding items. In the above example x.shape[1] == n+1 cannot
 *   be checked in the first round. RunMatch will populate new vars(this case n, m), these vars
 *   are added to a ready queue (ready_vars_)
 *
 * - Step 2: We EmitOutstandingPrimExprCompute to check if ready_vars will trigger new values
 *   to be computed. We eagerly compute all the outstanding values. The trigger is done through
 *   a ref counter which decreases when each outstanding def is satisfied.
 *   This step can also generate additional TIR functions to carry out shape computations.
 *
 * - Step 3: RunMatch again for given outstanding match todos. This time all invariants
 *   should be checked.
 *
 * The above step would populate each slot(which is backed by an element in shape_heap).
 * Each time we find a symbolic shape tuple, we call MakeShape for given slot indices
 * in the shape_heap.
 *
 *
 * Key functions in the flow:
 * - PrimExprSlotCollector: preprocessing and collecting the slots
 * - CheckMatchCast: recursively structinfo unpacking, generate checks and match items.
 * - RunMatch: generate symbolic shape matches
 * - EmitOutstandingPrimExprCompute: tracks the variables to be computed and emit shape computation
 * - VisitExpr_(ShapeExprNode*): makes symbolic shape tuple.
 *
 * The checks and symbolic shape all maps to runtime builtin functions. Please checkout
 * runtime/relax_vm/builtin.cc for their definitions.
 *
 * Shape computation are lowered to host-side TIR functions that load var from slot
 * and store computed results into the slot. For a given slot map: {0:m, 1: n+1: 2: n}
 * It will create the shape_func below that loads data from H[2](n's slot) run compute
 * and store back to H[1](n+1's slot).
 *
 * \code
 *
 * @T.prim_func
 * def shape_func(H: T.Buffer([3], "int64")):
 *     H[1] = H[2] + 1
 *
 * \endcode
 *
 * The current implementation will batch all shape computations at each match point.
 * For example, all the expressions that depend on n, m will be computed in a single
 * shape_func at the function boundary. If there are follow-up match_cast points,
 * that defines new variable, then we might we will generate new shape functions
 * to compute expressions that depend on these variables.
 */
class VMShapeLowerMutator
    : public ExprMutator,
      public StructInfoFunctor<void(const StructInfo&, Expr, bool, bool, const String&,
                                    std::vector<MatchShapeTodoItem>*)> {
 public:
  static IRModule Lower(IRModule mod, bool emit_err_ctx) {
    VMShapeLowerMutator mutator(mod, emit_err_ctx);

    for (auto& kv : mod->functions) {
      if (auto* func = kv.second.as<FunctionNode>()) {
        Function updated_func = mutator.Rewrite(kv.first, GetRef<Function>(func));
        mutator.builder_->UpdateFunction(kv.first, updated_func);
      }
    }
    return mutator.builder_->GetContextIRModule();
  }

 private:
  explicit VMShapeLowerMutator(IRModule mod, bool emit_err_ctx)
      : ExprMutator(mod), emit_err_ctx_(emit_err_ctx) {}

  using ExprMutator::VisitExpr_;

  // Unit rewrite function per function.
  Function Rewrite(GlobalVar gvar, Function func) {
    // prepare mapping and heap var
    slot_vec_.clear();
    slot_map_.clear();
    current_gvar_ = gvar;
    PrimExprSlotCollector::Collect(func, &slot_vec_, &slot_map_);
    heap_size_ = IntImm(ShapeDType(), static_cast<int64_t>(slot_vec_.size()));
    VarBinding shape_heap_binding = this->AllocShapeHeapBinding(heap_size_);
    shape_heap_ = shape_heap_binding->var;

    // prepare slot information
    this->PopulateSlotInfo();

    Array<BindingBlock> blocks;

    builder_->BeginScope(func->params);

    {
      // Check the parameter section.
      builder_->BeginBindingBlock();
      this->builder_->EmitNormalized(shape_heap_binding);
      std::vector<MatchShapeTodoItem> match_todos;
      size_t num_input = func->params.size();
      if (auto opt_num_input = func->attrs.GetAttr<Integer>(attr::kNumInput)) {
        // If the function has the attribute 'num_input', do shape checking on for the real inputs
        // and skip weights.
        num_input = static_cast<size_t>(opt_num_input.value()->value);
      }
      for (size_t i = 0; i < func->params.size(); ++i) {
        StructInfo sinfo = GetStructInfo(func->params[i]);
        std::ostringstream err_ctx;
        err_ctx << "ErrorContext(fn=" << gvar->name_hint << ", loc=param[" << i
                << "], param=" << func->params[i]->name_hint() << ", annotation=" << sinfo << ") ";
        this->CheckMatchCast(sinfo, func->params[i], true, i >= num_input, err_ctx.str(),
                             &match_todos);
      }
      // insert heap generation logic.
      match_todos = this->RunMatch(match_todos, false);
      this->EmitOutstandingPrimExprCompute();
      this->RunMatch(match_todos, true);

      BindingBlock pre_block = builder_->EndBlock();
      blocks.push_back(pre_block);
    }

    // new body.
    auto body_seq = Downcast<SeqExpr>(this->VisitWithNewScope(func->body, func->params));
    blocks.insert(blocks.end(), body_seq->blocks.begin(), body_seq->blocks.end());

    {
      // Insert the return value check
      builder_->BeginBindingBlock();
      std::ostringstream err_ctx;
      err_ctx << "ErrorContext(fn=" << gvar->name_hint
              << ", loc=return, annotation=" << func->ret_struct_info << ") ";
      std::vector<MatchShapeTodoItem> match_todos;
      // NOTE: the return value's shape computation must already be defined.
      this->CheckMatchCast(func->ret_struct_info, body_seq->body, false, false, err_ctx.str(),
                           &match_todos);
      // NOTE: the return value's shape computation must already be defined.
      this->RunMatch(match_todos, true);
      BindingBlock post_block = builder_->EndBlock();
      blocks.push_back(post_block);
    }

    auto new_body = builder_->Normalize(SeqExpr(blocks, body_seq->body));

    current_gvar_ = NullOpt;

    // create a new function
    return Function(func->params, new_body, func->ret_struct_info, func->is_pure, func->attrs);
  }

  //-------------------------------------------------------
  // PrimExpr slot handling
  //-------------------------------------------------------
  static DataType ShapeDType() { return DataType::Int(64); }

  /*! \brief populate additional information in the slot. */
  void PopulateSlotInfo() {
    for (auto& kv : slot_map_) {
      auto* slot = kv.second;
      if (!slot->expr.as<tir::VarNode>()) {
        Array<tir::Var> dep_vars = tir::UndefinedVars(slot->expr);
        for (auto var : dep_vars) {
          auto it = slot_map_.find(var);
          ICHECK(it != slot_map_.end())
              << "Var " << var << "is not defined in the function but is referenced by "
              << slot->expr;
          auto* var_slot = it->second;
          // populate the use slot.
          var_slot->user_slots.push_back(slot);
        }
        // set outstanding defs.
        slot->outstanding_defs += static_cast<int>(dep_vars.size());
      }
    }
  }
  //-------------------------------------------------------
  // Helper functions
  //-------------------------------------------------------
  StringImm GetErrContext(String err_ctx) const {
    return emit_err_ctx_ ? StringImm(err_ctx) : StringImm("");
  }

  VarBinding AllocShapeHeapBinding(IntImm heap_size) {
    if (heap_size->value > 0) {
      TensorStructInfo heap_sinfo(ShapeDType(), 1);
      Var var("shape_heap", heap_sinfo);
      // set up the builtin func.
      Call call(call_builtin_with_ctx_op_,
                {builtin_alloc_shape_heap_, Tuple({PrimValue(heap_size)})}, Attrs(), {heap_sinfo});
      UpdateStructInfo(call, heap_sinfo);
      return VarBinding(var, call);
    } else {
      Var var("shape_heap", ObjectStructInfo());
      Call call(null_value_op_, {});
      UpdateStructInfo(call, ObjectStructInfo());
      return VarBinding(var, call);
    }
  }

  //-------------------------------------------------------
  // Expr mutation overloading.
  //-------------------------------------------------------
  Expr VisitExpr_(const FunctionNode* op) final {
    LOG(FATAL) << "VMShapeLower do not work for local functions, make sure "
               << " to run it after LambdaLift";
    return GetRef<Expr>(op);
  }

  std::pair<Expr, Expr> MakeSymbolicShapeArg(const PrimExpr& expr) {
    using runtime::relax_vm::MakeShapeCode;

    if (auto* int_expr = expr.as<IntImmNode>()) {
      return {PrimValue::Int64(static_cast<int>(MakeShapeCode::kUseImm)),
              PrimValue::Int64(int_expr->value)};
    } else {
      auto it = slot_map_.find(expr);
      ICHECK(it != slot_map_.end());
      auto* slot = it->second;
      ICHECK(slot->value_computed)
          << "PrimExpr " << expr << " in function " << current_gvar_ << " has not been computed";
      return {PrimValue::Int64(static_cast<int>(MakeShapeCode::kLoadShape)),
              PrimValue::Int64(slot->index)};
    }
  }

  Expr VisitExpr_(const PrimValueNode* op) final {
    using runtime::relax_vm::MakeShapeCode;
    // Constant shape can be preserved.
    bool is_const_value =
        op->value->IsInstance<IntImmNode>() || op->value->IsInstance<FloatImmNode>();
    if (is_const_value) {
      return GetRef<Expr>(op);
    }

    Array<Expr> args = {shape_heap_};
    auto [code, value_or_index] = MakeSymbolicShapeArg(op->value);
    args.push_back(code);
    args.push_back(value_or_index);

    // make_shape(heap, n, c[0], r[0], c[1], r[1] ..., c[n], r[n])
    Call call(builtin_make_prim_value_, args, Attrs(), {Downcast<StructInfo>(op->struct_info_)});
    return call;
  }

  Expr VisitExpr_(const ShapeExprNode* op) final {
    using runtime::relax_vm::MakeShapeCode;
    // Constant shape can be preserved.
    bool is_const_shape = std::all_of(op->values.begin(), op->values.end(), [](const PrimExpr& e) {
      return e->IsInstance<IntImmNode>();
    });
    if (is_const_shape) {
      return GetRef<Expr>(op);
    }

    Array<Expr> args = {shape_heap_, PrimValue::Int64(static_cast<int64_t>(op->values.size()))};
    for (PrimExpr expr : op->values) {
      auto [code, value_or_index] = MakeSymbolicShapeArg(expr);
      args.push_back(code);
      args.push_back(value_or_index);
    }

    // make_shape(heap, n, c[0], r[0], c[1], r[1] ..., c[n], r[n])
    Call call(builtin_make_shape_, args, Attrs(),
              {ShapeStructInfo(static_cast<int>(op->values.size()))});
    return call;
  }

  void VisitBinding_(const MatchCastNode* binding) final {
    Expr value = ExprMutator::VisitExpr(binding->value);
    std::vector<MatchShapeTodoItem> match_todos;
    std::ostringstream err_ctx;
    err_ctx << "ErrorContext(match_cast, struct_info=" << binding->struct_info << ") ";
    // always_check=false
    this->CheckMatchCast(binding->struct_info, value, false, false, err_ctx.str(), &match_todos);

    match_todos = this->RunMatch(match_todos, false);
    this->EmitOutstandingPrimExprCompute();
    this->RunMatch(match_todos, true);

    // These checks are emitted as extra, in codegen
    // match-cast is simply ignored and treated as a normal binding.
    ExprMutator::VisitBinding_(binding);
  }

  // Do not override shape in struct info fields
  // We only override the shape that are already part of the normal function values
  // If future passes lift those values out into the values,
  // then codegen may not be able to handle symbolic values.
  // Place this pass as last pass before codegen.
  StructInfo VisitExprDepStructInfoField(const StructInfo& sinfo) final { return sinfo; }

  /* \brief Internal utility function used for RunMatch()
   *
   * \param expr The expression to be matched
   *
   * \param require_value_computed Whether we require all expr to be computed.
   *
   * \return The MatchShapeCode, and a relax expression specifying the
   *    argument used by that MatchShapeCode.
   */
  std::pair<runtime::relax_vm::MatchShapeCode, Expr> MakeMatchArgs(const PrimExpr& expr,
                                                                   bool require_value_computed) {
    using runtime::relax_vm::MatchShapeCode;

    if (auto* int_expr = expr.as<IntImmNode>()) {
      return {MatchShapeCode::kAssertEqualToImm, PrimValue::Int64(int_expr->value)};
    }

    auto it = slot_map_.find(expr);
    ICHECK(it != slot_map_.end());
    auto* slot = it->second;
    if (slot->value_computed) {
      return {MatchShapeCode::kAssertEqualToLoad, PrimValue::Int64(slot->index)};
    }

    // the value is not yet computed
    ICHECK(!require_value_computed) << "PrimExpr " << expr << " is not computed";
    if (expr.as<tir::VarNode>()) {
      // It is a var we will populate it in this round.

      slot->value_computed = true;
      ready_vars_.push_back(slot);

      return {MatchShapeCode::kStoreToHeap, PrimValue::Int64(slot->index)};
    }

    // otherwise, we skip and mark it as outstanding
    return {MatchShapeCode::kNoOp, PrimValue::Int64(0)};
  }

  //-------------------------------------------------------
  // Shape computations.
  //-------------------------------------------------------
  /*!
   * \brief Execute the match todo items.
   *
   * This function can populate vars in the match items when seeing it for the first time.
   * These new vars will be added to this->ready_vars_.
   *
   * If an item contains PrimExpr that are yet to be computed (but may be computable through
   * vars defined in this round), it will be returned to the caller.
   *
   * The caller should call EmitOutstandingPrimExprCompute, then call RunMatch again.
   *
   * \param match_todos The list of match items to be executed.
   * \param require_value_computed Whether we require all expr to be computed.
   * \return List of outstanding items that contains value that are yet to be computed.
   */
  std::vector<MatchShapeTodoItem> RunMatch(const std::vector<MatchShapeTodoItem>& match_todos,
                                           bool require_value_computed) {
    std::vector<MatchShapeTodoItem> outstanding_todos;

    using runtime::relax_vm::MatchShapeCode;
    for (const MatchShapeTodoItem& item : match_todos) {
      bool all_nop = true;
      bool any_nop = false;

      Array<Expr> args = {item.input, shape_heap_};

      Expr match_op;
      if (item.input->struct_info_.as<PrimStructInfoNode>()) {
        match_op = builtin_match_prim_value_;
        ICHECK_EQ(item.pattern.size(), 1);
      } else {
        match_op = builtin_match_shape_;
        args.push_back(PrimValue::Int64(item.pattern.size()));
      }

      for (PrimExpr expr : item.pattern) {
        auto [code, rvalue] = MakeMatchArgs(expr, require_value_computed);
        all_nop = all_nop && code == MatchShapeCode::kNoOp;
        any_nop = any_nop || code == MatchShapeCode::kNoOp;
        args.push_back(PrimValue::Int64(static_cast<int>(code)));
        args.push_back(rvalue);
      }
      if (any_nop) {
        outstanding_todos.push_back(item);
      }
      args.push_back(GetErrContext(item.err_ctx));
      if (!all_nop) {
        Call call(match_op, args, Attrs(), {void_sinfo_});
        builder_->Emit(call, "_");
      }
    }
    return std::move(outstanding_todos);
  }

  /*!
   * \brief Compute a list of prim expr that now be computed
   *        for given ready vars.
   */
  std::vector<PrimExprSlot*> GetReadyPrimExprSlots() {
    std::vector<PrimExprSlot*> to_compute;
    for (PrimExprSlot* slot : ready_vars_) {
      for (PrimExprSlot* user : slot->user_slots) {
        ICHECK_GT(user->outstanding_defs, 0);
        user->outstanding_defs -= 1;
        if (user->outstanding_defs == 0) {
          to_compute.push_back(user);
        }
      }
    }
    ready_vars_.clear();
    return to_compute;
  }

  /*!
   * \brief Check the dependent expressions of ready_vars_,
   *
   * If there are outstanding PrimExpr that can now be computed
   * we generate a PrimFunc that compute the extra shape values
   *
   * We will then clear the ready_vars.
   *
   * \return Number of PrimExpr computed.
   */
  size_t EmitOutstandingPrimExprCompute() {
    std::vector<PrimExprSlot*> to_compute = GetReadyPrimExprSlots();
    if (to_compute.size() == 0) return 0;
    ICHECK_GT(heap_size_->value, 0);
    // construct a PrimFunc that compute the shape.
    tir::Var heap("heap", DataType::Handle());
    Array<PrimExpr> buffer_shape{heap_size_};
    tir::Buffer buffer = tir::decl_buffer(buffer_shape, ShapeDType(), "H", "global");
    Map<tir::Var, tir::Buffer> buffer_map;
    buffer_map.Set(heap, buffer);

    auto var_map = [&](const tir::Var& var) -> Optional<PrimExpr> {
      auto it = slot_map_.find(var);
      ICHECK(it != slot_map_.end());
      return tir::BufferLoad(buffer, {IntImm(ShapeDType(), it->second->index)});
    };

    Array<tir::Stmt> seq;
    for (PrimExprSlot* slot : to_compute) {
      ICHECK(!slot->value_computed);
      slot->value_computed = true;
      PrimExpr value = tir::Substitute(slot->expr, var_map);
      seq.push_back(tir::BufferStore(buffer, value, {IntImm(ShapeDType(), slot->index)}));
    }

    tir::Stmt body = tir::SeqStmt::Flatten(seq);
    Array<tir::Var> params{heap};
    Type ret_type = VoidType();

    // TODO(relax-team): Consider attach the target attribute to
    // the shape_func to indicate that this is a host function
    // This could require us to attach target to the relax function here.
    tir::PrimFunc shape_func(params, body, ret_type, buffer_map);
    if (shape_func->attrs.GetAttr<tvm::Target>(tvm::attr::kTarget) == nullptr) {
      // kTarget and kIsHostFunc are mutually exclusive
      shape_func =
          WithAttr<tir::PrimFunc>(std::move(shape_func), tvm::tir::attr::kIsHostFunc, Integer(1));
    }
    GlobalVar shape_func_var = builder_->AddFunction(shape_func, "shape_func");
    builder_->Emit(Call(shape_func_var, {shape_heap_}), "_");
    return to_compute.size();
  }
  //-------------------------------------------------------
  // StructInfo value match logic
  //
  // CheckMatchCast is the only function needed by
  // other code sections
  //-------------------------------------------------------
  /*!
   * \brief Insert runtime check of the match cast condition(value, struct_info).
   *
   * \param struct_info The struct info to be matched.
   * \param value The input value.
   * \param always_check Whether we insert runtime check even if we can prove
   *        that value's struct info already satisfies the condition.
   *        This option is necessary for argument checking per our calling convention.
   * \param dynamic_only Whether we only check values with dynamic shapes.
   * \param err_ctx Extra error context to bring more informative error reporting.
   * \param match_todos List of match shape todo items collected when recursively
   *                    visit the match cast.
   */
  void CheckMatchCast(const StructInfo& struct_info, Expr value, bool always_check,
                      bool dynamic_only, const String& err_ctx,
                      std::vector<MatchShapeTodoItem>* match_todos) {
    return this->VisitStructInfo(struct_info, value, always_check, dynamic_only, err_ctx,
                                 match_todos);
  }

  void VisitStructInfo(const StructInfo& struct_info, Expr value, bool always_check,
                       bool dynamic_only, const String& err_ctx,
                       std::vector<MatchShapeTodoItem>* match_todos) final {
    // short-cut, if the struct info already satisfies the
    // constraint during match cast, we can skip matching
    if (!always_check && IsBaseOf(struct_info, GetStructInfo(value))) return;
    return StructInfoFunctor::VisitStructInfo(struct_info, value, always_check, dynamic_only,
                                              err_ctx, match_todos);
  }

  void VisitStructInfo_(const ObjectStructInfoNode* op, Expr value, bool always_check,
                        bool dynamic_only, const String& err_ctx,
                        std::vector<MatchShapeTodoItem>* match_todos) final {}

  void VisitStructInfo_(const PrimStructInfoNode* op, Expr value, bool always_check,
                        bool dynamic_only, const String& err_ctx,
                        std::vector<MatchShapeTodoItem>* match_todos) final {
    // emit runtime check of shape
    if (always_check || !IsBaseOf(PrimStructInfo(op->dtype), GetStructInfo(value))) {
      // check_shape_info(value, ndim, err_ctx)
      Call call(builtin_check_prim_value_info_,
                {value, DataTypeImm(op->dtype), GetErrContext(err_ctx)}, Attrs(), {void_sinfo_});
      builder_->Emit(call, "_");
    }
    if (op->value.defined()) {
      MatchShapeTodoItem item;
      item.input = value;
      item.pattern = {op->value.value()};
      item.err_ctx = err_ctx;
      match_todos->push_back(item);
    }
  }

  void VisitStructInfo_(const ShapeStructInfoNode* op, Expr value, bool always_check,
                        bool dynamic_only, const String& err_ctx,
                        std::vector<MatchShapeTodoItem>* match_todos) final {
    // emit runtime check of shape
    if (always_check || !IsBaseOf(ShapeStructInfo(op->ndim), GetStructInfo(value))) {
      // check_shape_info(value, ndim, err_ctx)
      Call call(builtin_check_shape_info_,
                {value, PrimValue::Int64(op->ndim), GetErrContext(err_ctx)}, Attrs(),
                {void_sinfo_});
      builder_->Emit(call, "_");
    }
    if (op->values.defined()) {
      MatchShapeTodoItem item;
      item.input = value;
      item.pattern = op->values.value();
      item.err_ctx = err_ctx;
      match_todos->push_back(item);
    }
  }

  void VisitStructInfo_(const TensorStructInfoNode* op, Expr value, bool always_check,
                        bool dynamic_only, const String& err_ctx,
                        std::vector<MatchShapeTodoItem>* match_todos) final {
    // emit runtime check of shape
    auto* shape_expr = op->shape.as<ShapeExprNode>();
    if (dynamic_only &&
        std::all_of(shape_expr->values.begin(), shape_expr->values.end(),
                    [](const PrimExpr& e) { return e->IsInstance<IntImmNode>(); })) {
      // if we only check dynamic shapes, and the shape is static, we can skip.
      return;
    }
    if (always_check || !IsBaseOf(TensorStructInfo(op->dtype, op->ndim), GetStructInfo(value))) {
      // check_tensor_info(value, ndim, dtype, err_ctx)
      Call call(builtin_check_tensor_info_,
                {value, PrimValue::Int64(op->ndim), DataTypeImm(op->dtype), GetErrContext(err_ctx)},
                Attrs(), {void_sinfo_});
      builder_->Emit(call, "_");
    }

    if (shape_expr != nullptr) {
      MatchShapeTodoItem item;
      item.input = value;
      item.pattern = shape_expr->values;
      item.err_ctx = err_ctx;
      match_todos->push_back(item);
    } else if (op->shape.as<VarNode>()) {
      // NOTE: This part of the logic is left empty for future support as it is less common.
      // Future implementors: we can emit a binding here and assert here.
      LOG(FATAL) << "Cannot handle Tensor shape pattern where a var appears multiple times";
    } else {
      ICHECK(!op->shape.defined()) << "Can only handle tensor shape pattern var";
    }
  }

  // Internal helper function to make tuple get item.
  // This function will try to simplify constant tuples
  // the return value **always** have struct info.
  Expr MakeTupleGetItem(Expr value, int64_t index) {
    if (auto* tuple_expr = value.as<TupleNode>()) {
      return tuple_expr->fields[index];
    } else if (GetStructInfoAs<TupleStructInfoNode>(value)) {
      // value is tuple type, it is OK to run tuple get item.
      return TupleGetItem(value, index);
    } else {
      // call runtime tuple get item, and return a object.
      Call call(builtin_tuple_getitem_, {value, PrimValue::Int64(index)}, Attrs(), {object_sinfo_});
      UpdateStructInfo(call, ObjectStructInfo());
      return call;
    }
  }

  void VisitStructInfo_(const TupleStructInfoNode* op, Expr value, bool always_check,
                        bool dynamic_only, const String& err_ctx,
                        std::vector<MatchShapeTodoItem>* match_todos) final {
    auto* value_tinfo = GetStructInfoAs<TupleStructInfoNode>(value);
    if (value_tinfo) {
      CHECK_EQ(value_tinfo->fields.size(), op->fields.size())
          << "TypeError: " << err_ctx << " during match-cast we find tuple size mismatch";
    }
    if (always_check || !value_tinfo) {
      // check_tuple_info(value, tuple_size)
      Call call(builtin_check_tuple_info_,
                {value, PrimValue::Int64(static_cast<int64_t>(op->fields.size())),
                 GetErrContext(err_ctx)},
                Attrs(), {void_sinfo_});
      builder_->Emit(call, "_");
    }
    // recursively visit each sub-field and run matching
    for (size_t i = 0; i < op->fields.size(); ++i) {
      this->VisitStructInfo(op->fields[i], MakeTupleGetItem(value, i), always_check, dynamic_only,
                            err_ctx, match_todos);
    }
  }

  void VisitStructInfo_(const FuncStructInfoNode* op, Expr value, bool always_check,
                        bool dynamic_only, const String& err_ctx,
                        std::vector<MatchShapeTodoItem>* match_todos) final {
    // we only check function is callable.
    if (!always_check && MatchStructInfo<FuncStructInfo>(value)) return;
    // check_func_info(value, err_ctx)
    Call call(builtin_check_func_info_, {value, GetErrContext(err_ctx)}, Attrs(), {void_sinfo_});
    builder_->Emit(call, "_");
  }

  //-------------------------------------------------------
  // Private member fields.
  //-------------------------------------------------------
  /*! \brief whether to emit error context, can be turned off for testing purposes. */
  bool emit_err_ctx_{true};
  /*! \brief heap ptr to store the PrimExpr slots. */
  Var shape_heap_;
  /*! \brief heap size. */
  IntImm heap_size_;
  /*! \brief index => slot. */
  std::vector<std::unique_ptr<PrimExprSlot>> slot_vec_;
  /*! \brief Expr => slot. */
  PrimExprSlotMap slot_map_;
  Optional<GlobalVar> current_gvar_ = NullOpt;
  /*!
   * \brief List of vars that are being defined but
   * have not go through outstanding shape compute check.
   */
  std::vector<PrimExprSlot*> ready_vars_;
  // call builtin cop
  const Op& call_builtin_with_ctx_op_ = Op::Get("relax.call_builtin_with_ctx");
  const Op& null_value_op_ = Op::Get("relax.null_value");
  // common struct info
  const StructInfo object_sinfo_ = ObjectStructInfo();
  const StructInfo void_sinfo_ = TupleStructInfo(Array<StructInfo>({}));
  // check function
  const ExternFunc builtin_alloc_shape_heap_{"vm.builtin.alloc_shape_heap"};
  const ExternFunc builtin_match_shape_{"vm.builtin.match_shape"};
  const ExternFunc builtin_make_shape_{"vm.builtin.make_shape"};
  const ExternFunc builtin_check_shape_info_{"vm.builtin.check_shape_info"};
  const ExternFunc builtin_match_prim_value_{"vm.builtin.match_prim_value"};
  const ExternFunc builtin_make_prim_value_{"vm.builtin.make_prim_value"};
  const ExternFunc builtin_check_prim_value_info_{"vm.builtin.check_prim_value_info"};
  const ExternFunc builtin_check_tensor_info_{"vm.builtin.check_tensor_info"};
  const ExternFunc builtin_check_tuple_info_{"vm.builtin.check_tuple_info"};
  const ExternFunc builtin_check_func_info_{"vm.builtin.check_func_info"};
  const ExternFunc builtin_tuple_getitem_{"vm.builtin.tuple_getitem"};
};

namespace transform {

Pass VMShapeLower(bool emit_err_ctx) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return VMShapeLowerMutator::Lower(mod, emit_err_ctx); };
  return CreateModulePass(pass_func, 0, "VMShapeLower", {});
}

TVM_REGISTER_GLOBAL("relax.transform.VMShapeLower").set_body_typed([](bool emit_err_ctx) {
  return VMShapeLower(emit_err_ctx);
});

}  // namespace transform
}  // namespace relax
}  // namespace tvm
