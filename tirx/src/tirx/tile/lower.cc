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
 * \file tirx/tile/lower.cc
 * \brief Lower tensor instructions and CUDA index calls using independent launch configuration.
 */

#include <tvm/backend/cuda/attr.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ir/function.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/sym/pattern.h>
#include <tvm/target/target.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/annotation.h>
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tile/dispatch.h>
#include <tvm/tirx/tile/thread_scope.h>
#include <tvm/tirx/transform.h>

#include <limits>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../backend/cuda/tile/exec_analysis.h"

namespace tvm {
namespace tirx {

namespace {

class NoOpCallVerifier : public StmtExprVisitor {
  friend class tvm::ObjectVisitor;
  using Result = ffi::Optional<VisitInterrupt>;

 public:
  NoOpCallVerifier() : StmtExprVisitor(GlobalVTable()) {}
  static bool Verify(const Stmt& stmt) {
    auto verifier = ffi::make_object<NoOpCallVerifier>();
    verifier->Visit(stmt);
    return verifier->valid_;
  }

 private:
  using StmtExprVisitor::Visit_;
  Result Visit(ffi::AnyView value) override {
    if (value.as<Stmt>() && !GlobalVTable()->CanDispatch(value)) {
      valid_ = false;
      return std::nullopt;
    }
    if (auto expr = value.as<Expr>(); expr && !GlobalVTable()->CanDispatch(value)) {
      if (value.as<OpaqueExprNode>() || value.as<VarNode>()) return std::nullopt;
      TVM_FFI_VISIT_THROW(TypeError, expr.value())
          << "Unsupported non-primitive TIR expression " << expr.value()->GetTypeKey();
    }
    return StmtExprVisitor::Visit(value);
  }

  Result Visit_(const CallNode* call) final {
    if (auto op = call->op.as<Op>(); op && Op::HasAttrMap(tile::op_attr::kTileOpDispatch)) {
      static const auto& dispatchers = Op::GetAttrMap(tile::op_attr::kTileOpDispatch);
      valid_ &= !dispatchers.count(op.value());
    }
    if (call->op.as<GlobalVar>() || call->op.as<OpaqueExprNode>()) this->Visit(call->op);
    return this->Visit(call->args);
  }

  Result Visit_(const BindNode* op) override {
    this->Visit(op->value);
    return this->Visit(op->var->ty);
  }

  Result Visit_(const RegionStmtNode* op) override {
    this->Visit(op->args);
    this->Visit(op->attrs);
    for (const auto& param : op->body_params) this->Visit(param->ty);
    this->Visit(op->body);
    for (const auto& var : op->result_vars) this->Visit(var->ty);
    return std::nullopt;
  }

  Result Visit_(const ForNode* op) override {
    this->Visit(op->min);
    this->Visit(op->extent);
    this->Visit(op->loop_var->ty);
    return this->Visit(op->body);
  }

  Result Visit_(const prim::LetNode* op) override {
    this->Visit(op->value);
    this->Visit(op->var->ty);
    return this->Visit(op->body);
  }

  Result Visit_(const LambdaExprNode* op) {
    this->Visit(op->ty);
    for (const auto& var : op->vars) this->Visit(var->ty);
    return this->Visit(op->body);
  }

  Result Visit_(const TensorLoadNode* op) override { return this->Visit(op->indices); }

  Result Visit_(const TensorStoreNode* op) override {
    this->Visit(op->value);
    return this->Visit(op->indices);
  }

  Result Visit_(const TensorRegionNode* op) override {
    if (!op->source.as<TensorVar>()) this->Visit(op->source);
    return this->Visit(op->region);
  }

  Result Visit_(const prim::RampNode* op) override {
    this->Visit(op->base);
    this->Visit(op->stride);
    return this->Visit(op->lanes);
  }

  Result Visit_(const prim::BroadcastNode* op) override {
    this->Visit(op->value);
    return this->Visit(op->lanes);
  }

  Result Visit_(const GenericConstNode* op) override { return std::nullopt; }

  Result Visit_(const AssertStmtNode* op) override {
    this->Visit(op->condition);
    this->Visit(op->error_kind);
    return this->Visit(op->message_parts);
  }

  static const VTable* GlobalVTable() {
    static const VTable table = [] {
      VTable table;
      StmtExprVisitor::InitVTable(&table);
      SetDispatch<NoOpCallVerifier, LambdaExprNode>(&table);
      table.Finalize();
      return table;
    }();
    return &table;
  }

  bool valid_{true};
};

class TileDispatcher : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit TileDispatcher(const Target& target) : target_(target) {}

  static Stmt LowerOpCalls(const Stmt& stmt, const Target& target) {
    return ffi::make_object<TileDispatcher>(target)
        ->Mutate(stmt, InplaceMode::kAllow)
        .ValueOrUnchanged(stmt);
  }

 private:
  class TensorRefRewriter : public StmtExprMutator {
   public:
    using StmtExprMutator::Mutate;
    using StmtExprMutator::Mutate_;
    static Stmt Rewrite(const Stmt& stmt, const TensorVar& src, const TensorVar& dst) {
      if (src.same_as(dst)) {
        return stmt;
      }
      return ffi::make_object<TensorRefRewriter>(src, dst)
          ->Mutate(stmt, InplaceMode::kAllow)
          .ValueOrUnchanged(stmt);
    }
    TensorRefRewriter(const TensorVar& src, const TensorVar& dst) { VarRemapSet(src, dst); }
  };

  class KernelReplacePointSearcher : public StmtExprMutator {
   public:
    using StmtExprMutator::Mutate;
    using StmtExprMutator::Mutate_;
    explicit KernelReplacePointSearcher(const Stmt& body) : body_(body) {}

    static Stmt Seek(const Stmt& stmt, const Stmt& body) {
      return ffi::make_object<KernelReplacePointSearcher>(body)
          ->Mutate(stmt, InplaceMode::kAllow)
          .ValueOrUnchanged(stmt);
    }

   private:
    UnchangedOr<Stmt> Mutate_(const EvaluateNode* op, InplaceMode inplace_mode) final {
      const auto* call = op->value.as<CallNode>();
      if (call != nullptr && call->op.same_as(tirx::kernel_replace_point_op())) {
        return body_;
      }
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }

    Stmt body_;
  };

  UnchangedOr<Stmt> Mutate_(const RegionStmtNode* op, InplaceMode inplace_mode) final {
    if (op->op.same_as(tirx::device_entry_op())) {
      return ProcessDeviceEntry(op);
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

  Stmt ProcessDeviceEntry(const RegionStmtNode* entry_node) {
    Stmt body_to_visit = entry_node->body;
    Target saved_target = target_;
    if (auto target = entry_node->attrs->dict.Get(tvm::attr::kTarget.name)) {
      target_ = target.value().cast<Target>();
    }
    With<Target> target_scope(target_);

    bool is_first_block = false;
    std::swap(is_first_block, is_first_block_);

    launch_params_.clear();
    cluster_bounds_.clear();
    variable_cluster_ = false;
    native_launch_ = entry_node->attrs->dict.count(tvm::backend::cuda::attr::kLaunchFields.name);
    TVM_FFI_CHECK(is_first_block || !native_launch_, ValueError)
        << "Nested CUDA device_entry regions are not supported";
    if (native_launch_) PrepareCudaLaunchParams(entry_node);
    auto saved_analysis = std::move(cuda_analysis_);
    if (target_->kind->name == "cuda") {
      cuda_analysis_ = std::make_unique<backend::cuda::ExecAnalysis>(
          launch_params_, cluster_bounds_, variable_cluster_);
    }

    auto body_result = Mutate(body_to_visit, InplaceMode::kDisallow);
    bool body_unchanged = body_result.UnchangedOrSameAs(body_to_visit);
    Stmt body = std::move(body_result).ValueOrUnchanged(body_to_visit);

    auto pop_exec_contexts = [&]() {
      cuda_analysis_ = std::move(saved_analysis);
      target_ = saved_target;
    };

    if (!is_first_block) {
      std::swap(is_first_block, is_first_block_);
      pop_exec_contexts();
      if (body_unchanged) {
        return ffi::GetRef<Stmt>(entry_node);
      }
      return RegionStmt(entry_node->op, entry_node->args, entry_node->body_params,
                        entry_node->attrs, body, entry_node->result_vars, entry_node->loc);
    }

    // Insert device init stmts into kernel body.
    for (auto it = device_init_stmts_.rbegin(); it != device_init_stmts_.rend(); ++it) {
      body = KernelReplacePointSearcher::Seek(*it, body);
    }
    // Insert alloc tensors at the beginning of the kernel body.
    if (!alloc_tensors_.empty()) {
      std::vector<Stmt> seq;
      seq.reserve(alloc_tensors_.size() + 1);
      for (const auto& allocation : alloc_tensors_) {
        seq.push_back(allocation);
      }
      seq.push_back(std::move(body));
      body = SeqStmt(seq);
    }
    alloc_tensors_.clear();

    Stmt res = body;
    if (native_launch_) res = backend::cuda::LowerIndices(res, launch_params_);
    if (native_launch_ || entry_node->attrs->dict.count("backend_config")) {
      res = RegionStmt(Op::Get("tirx.device_scope"), entry_node->args, {}, entry_node->attrs, res);
    }

    // Insert host init stmts outside the outermost thread binding or block.
    if (is_first_thread_attr_) {
      for (const auto& stmt : host_init_stmts_) {
        // These statements leave the kernel region for host scope, where a
        // ``tensor_data_ptr`` projection of a device-local view cannot be
        // resolved.  Rewrite each projection onto its storage root, which is
        // a Function parameter and therefore visible on the host.
        res = KernelReplacePointSearcher::Seek(StorageRootResolver::Apply(stmt, tensor_root_),
                                               std::move(res));
      }
      host_init_stmts_.clear();
    }
    std::swap(is_first_block, is_first_block_);
    pop_exec_contexts();
    return res;
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtNode* op, InplaceMode inplace_mode) final {
    Stmt stmt = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    if (post_tensor_def_stmts_.empty()) {
      return stmt;
    }
    const auto& seq = stmt.as_or_throw<SeqStmt>();

    std::vector<Stmt> rebuilt;
    rebuilt.reserve(seq->seq.size() + post_tensor_def_stmts_.size());
    bool changed = false;
    for (const Stmt& s : seq->seq) {
      rebuilt.push_back(s);
      if (const auto* bind = s.as<BindNode>()) {
        if (const auto* call = bind->value.as<CallNode>();
            call && (call->op.same_as(tirx::alloc_tensor_op()) ||
                     (call->op.same_as(tirx::decl_tensor_op()) ||
                      call->op.same_as(Op::Get("tirx.cuda.decl_tmem"))))) {
          changed |= AppendPostTensorDefStmts(&rebuilt, bind->var.as_or_throw<TensorVar>(),
                                              bind->var.as_or_throw<TensorVar>());
        }
      }
    }
    if (!changed) {
      return stmt;
    }
    return SeqStmt(rebuilt, seq->loc);
  }

  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) final {
    if (const auto* call = op->value.as<CallNode>(); call) {
      if (call->op.same_as(tirx::alloc_tensor_op())) return MutateAllocTensor(op, inplace_mode);
      if ((call->op.same_as(tirx::decl_tensor_op()) ||
           call->op.same_as(Op::Get("tirx.cuda.decl_tmem"))))
        return MutateDeclTensor(op, inplace_mode);
    }
    Stmt stmt = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    const auto* bind = stmt.as<BindNode>();
    TVM_FFI_ICHECK(bind);
    if (auto value = bind->value.as<PrimExpr>()) {
      if (cuda_analysis_) cuda_analysis_->Bind(bind->var, value.value());
      // Bind is flat: the definition is visible to subsequent statements in
      // its enclosing scope.  Under SSA, an inner-scope Var cannot be
      // referenced after leaving that scope or rebound elsewhere, so stale
      // entries are never consulted and no scope-based cleanup is needed.
      var_range_map_.Set(bind->var, Range::FromMinExtent(value.value(), 1));
    }
    return stmt;
  }

  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) final {
    // Collect the loop variables
    auto loop_var = op->loop_var.as_or_throw<Var>();
    TVM_FFI_ICHECK(!var_range_map_.count(loop_var)) << "Internal Error: Duplicate loop variable";
    var_range_map_.Set(loop_var, Range::FromMinExtent(op->min, op->extent));
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

  /*!
   * \brief Track the storage root of a tensor variable.
   *
   * A ``DeclTensor`` whose data is ``tensor_data_ptr(src)`` is a view over
   * ``src``'s storage, so it inherits ``src``'s root; anything else owns its
   * storage.  Tensors with no definition in the body (Function parameters)
   * are absent from the map and are their own root.
   */
  void RegisterStorageRoot(const Var& old_var, const Var& new_var,
                           const ffi::Optional<Expr>& data) {
    Var root = new_var;
    if (data.has_value()) {
      if (const auto* call = data.value().as<CallNode>();
          call && call->op.same_as(tirx::tensor_data_ptr_op()) && call->args.size() == 1) {
        if (auto src = call->args[0].as<Var>();
            src.has_value() && src.value()->ty.as<TensorTypeNode>()) {
          root = StorageRootOf(src.value());
        }
      }
    }
    tensor_root_.insert_or_assign(new_var, root);
    if (!old_var.same_as(new_var)) {
      tensor_root_.insert_or_assign(old_var, root);
    }
  }

  Var StorageRootOf(const Var& var) const {
    auto it = tensor_root_.find(var);
    return it == tensor_root_.end() ? var : it->second;
  }

  /*! \brief Rewrite ``tensor_data_ptr(view)`` onto ``tensor_data_ptr(storage root)``. */
  class StorageRootResolver : public StmtExprMutator {
   public:
    using StmtExprMutator::Mutate;
    using StmtExprMutator::Mutate_;
    static Stmt Apply(
        Stmt stmt,
        const std::unordered_map<Var, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>& tensor_root) {
      auto resolver = ffi::make_object<StorageRootResolver>(tensor_root);
      return resolver->Mutate(stmt, InplaceMode::kAllow).ValueOrUnchanged(stmt);
    }
    explicit StorageRootResolver(
        const std::unordered_map<Var, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>& tensor_root)
        : tensor_root_(tensor_root) {}

    UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) final {
      if (op->op.same_as(tirx::tensor_data_ptr_op()) && op->args.size() == 1) {
        if (auto var = op->args[0].as<Var>();
            var.has_value() && var.value()->ty.as<TensorTypeNode>()) {
          auto it = tensor_root_.find(var.value());
          if (it != tensor_root_.end() && !it->second.same_as(var.value())) {
            return it->second.as_or_throw<TensorVar>().data();
          }
        }
      }
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }

    const std::unordered_map<Var, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>& tensor_root_;
  };

  UnchangedOr<Stmt> MutateAllocTensor(const BindNode* op, InplaceMode inplace_mode) {
    TensorVar old_tensor = op->var.as_or_throw<TensorVar>();
    Stmt stmt = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    op = stmt.as<BindNode>();
    TVM_FFI_ICHECK(op);
    RegisterStorageRoot(old_tensor, op->var, std::nullopt);

    std::vector<Stmt> seq{stmt};
    AppendPostTensorDefStmts(&seq, old_tensor, op->var.as_or_throw<TensorVar>());
    return SeqStmt(seq);
  }

  UnchangedOr<Stmt> MutateDeclTensor(const BindNode* op, InplaceMode inplace_mode) {
    TensorVar old_tensor = op->var.as_or_throw<TensorVar>();
    Stmt stmt = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    op = stmt.as<BindNode>();
    TVM_FFI_ICHECK(op);
    const auto* tensor_call = op->value.as<CallNode>();
    RegisterStorageRoot(old_tensor, op->var, tensor_call->args[0]);

    std::vector<Stmt> seq{stmt};
    AppendPostTensorDefStmts(&seq, old_tensor, op->var.as_or_throw<TensorVar>());
    return SeqStmt(seq);
  }

  UnchangedOr<Stmt> Mutate_(const IfNode* op, InplaceMode inplace_mode) final {
    int pushed_ctx = cuda_analysis_ ? cuda_analysis_->PushPredicate(op->condition) : 0;
    PrimExpr new_cond =
        cuda_analysis_ ? cuda_analysis_->RewritePredicate(op->condition) : op->condition;
    auto then_case_result = Mutate(op->then_case, inplace_mode);
    bool then_case_unchanged = then_case_result.UnchangedOrSameAs(op->then_case);
    Stmt then_case = std::move(then_case_result).ValueOrUnchanged(op->then_case);
    if (cuda_analysis_) cuda_analysis_->Pop(pushed_ctx);
    ffi::Optional<SeqStmt> else_case;
    if (op->else_case.has_value()) {
      else_case =
          Mutate(op->else_case.value(), inplace_mode).ValueOrUnchanged(op->else_case.value());
    }
    bool unchanged = new_cond.same_as(op->condition) && then_case_unchanged &&
                     ((!op->else_case.has_value() && !else_case.has_value()) ||
                      (op->else_case.has_value() && else_case.has_value() &&
                       else_case.value().same_as(op->else_case.value())));
    if (unchanged) return ffi::Unchanged();
    return If(new_cond, then_case, else_case);
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateNode* stmt, InplaceMode inplace_mode) final {
    const auto* op = stmt->value.as<CallNode>();
    if (!op || !op->op.as<Op>() || !Op::HasAttrMap(tile::op_attr::kTileOpDispatch)) {
      return StmtExprMutator::Mutate_(stmt, inplace_mode);
    }
    static const auto& dispatchers = Op::GetAttrMap(tile::op_attr::kTileOpDispatch);
    if (!dispatchers.count(op->op.as_or_throw<Op>())) {
      return StmtExprMutator::Mutate_(stmt, inplace_mode);
    }
    op->op.as_or_throw<Op>().Validate(op);
    TVM_FFI_CHECK(op->attrs.has_value(), ValueError)
        << "A tile instruction must have scope attributes";
    auto scope = ffi::reflection::FieldGetter(op->attrs->GetTypeKey(), "scope")(op->attrs.value())
                     .cast<ThreadScope>();
    TVM_FFI_CHECK(scope->backend() == target_->kind->name, ValueError)
        << "Thread scope " << scope.name() << " does not match target " << target_->kind->name;
    ffi::Map<ffi::String, ffi::Array<PrimExpr>> inter_map, intra_map;
    ffi::String scope_kind = scope->local_name();
    if (cuda_analysis_) {
      auto split = cuda_analysis_->Split(scope);
      inter_map = std::move(split.first);
      intra_map = std::move(split.second);
    }
    tile::DispatchContext sctx(target_, scope, launch_params_, var_range_map_,
                               /*alloc_only=*/false, /*callbacks=*/{}, shared_state_, inter_map,
                               intra_map, scope_kind);
    Stmt res = tile::Dispatch(ffi::GetRef<Call>(op), sctx);
    // Implementation found, handle callbacks
    if (auto bufs = sctx->callbacks.Get(tirx::tile::callback::kPrivateAlloc.name)) {
      auto buf_list = bufs.value().as<Array<Bind>>().value();
      alloc_tensors_.insert(alloc_tensors_.end(), buf_list.begin(), buf_list.end());
    }
    if (auto stmts = sctx->callbacks.Get(tirx::tile::callback::kDeviceInitStmt.name)) {
      auto stmt_list = stmts.value().as<Array<Stmt>>().value();
      device_init_stmts_.insert(device_init_stmts_.end(), stmt_list.begin(), stmt_list.end());
    }
    if (auto stmts = sctx->callbacks.Get(tirx::tile::callback::kHostInitStmt.name)) {
      auto stmt_list = stmts.value().as<Array<Stmt>>().value();
      host_init_stmts_.insert(host_init_stmts_.end(), stmt_list.begin(), stmt_list.end());
    }
    if (auto mapping = sctx->callbacks.Get(tirx::tile::callback::kPostTensorDefStmt.name)) {
      auto map = mapping.value().as_or_throw<ffi::Map<TensorVar, Array<Stmt>>>();
      for (const auto& [tensor, stmts] : map) {
        auto& vec = post_tensor_def_stmts_[tensor];
        vec.insert(vec.end(), stmts.begin(), stmts.end());
      }
    }
    // Propagate shared_state changes back (Map uses COW semantics)
    shared_state_ = sctx->shared_state;
    return Mutate(res, inplace_mode).ValueOrUnchanged(res);
  }

  // --- Scope-id resolution at kernel scope ----------------------------------

  void PrepareCudaLaunchParams(const RegionStmtNode* entry) {
    TVM_FFI_CHECK(target_->kind->name == "cuda", ValueError)
        << "CUDA LaunchConfig requires a CUDA target";
    auto fields = entry->attrs->dict.at(tvm::backend::cuda::attr::kLaunchFields.name)
                      .as_or_throw<ffi::Array<ffi::String>>();
    TVM_FFI_ICHECK_EQ(fields.size(), entry->args.size());
    for (size_t i = 0; i < fields.size(); ++i) {
      std::string field(fields[i]);
      std::string tag;
      if (field.rfind("grid.", 0) == 0) tag = "blockIdx." + field.substr(5);
      if (field.rfind("block.", 0) == 0) tag = "threadIdx." + field.substr(6);
      if (field.rfind("cluster.", 0) == 0) tag = "clusterCtaIdx." + field.substr(8);
      if (!tag.empty()) {
        auto extent = analyzer_->Simplify(entry->args[i].as_or_throw<PrimExpr>());
        launch_params_.insert(
            {tag, ffi::Tuple<PrimVar, PrimExpr>(PrimVar(tag, PrimType::Int(32)), extent)});
      }
    }
    TVM_FFI_CHECK(launch_params_.count("blockIdx.x") && launch_params_.count("threadIdx.x"),
                  ValueError)
        << "LaunchConfig requires grid and block dimensions";
    variable_cluster_ = false;
    cluster_bounds_.clear();
    for (char axis : {'x', 'y', 'z'}) {
      auto it = launch_params_.find(std::string("clusterCtaIdx.") + axis);
      if (it == launch_params_.end()) continue;
      PrimExpr extent = it->second.get<1>();
      variable_cluster_ |= !extent.as<IntImmNode>();
      for (size_t i = 0; i < fields.size(); ++i) {
        if (fields[i] == std::string("preferred_cluster.") + axis) {
          auto preferred = entry->args[i].as_or_throw<PrimExpr>();
          variable_cluster_ |= !analyzer_->CanProveEqual(extent, preferred);
          extent = analyzer_->Simplify(prim::Max(extent, preferred));
        }
      }
      cluster_bounds_.emplace(std::string("clusterCtaIdx.") + axis, extent);
    }
  }

  bool native_launch_{false};
  bool variable_cluster_{false};
  std::unordered_map<std::string, PrimExpr> cluster_bounds_;
  ffi::Map<Var, Range> var_range_map_;
  sym::Analyzer analyzer_;
  Target target_;
  std::unique_ptr<backend::cuda::ExecAnalysis> cuda_analysis_;
  std::unordered_map<ffi::String, ffi::Tuple<PrimVar, PrimExpr>> launch_params_;
  std::vector<Bind> alloc_tensors_;
  std::vector<Stmt> device_init_stmts_;
  std::vector<Stmt> host_init_stmts_;
  /*! \brief Storage root of each tensor variable defined in the body. */
  std::unordered_map<Var, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> tensor_root_;
  std::unordered_map<TensorVar, std::vector<Stmt>, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      post_tensor_def_stmts_;
  ffi::Map<ffi::String, ffi::ObjectRef> shared_state_;

  bool is_first_block_{true};
  bool is_first_thread_attr_{true};

  bool AppendPostTensorDefStmts(std::vector<Stmt>* seq, const TensorVar& old_tensor,
                                const TensorVar& new_tensor) {
    auto append_with_remap = [this, seq, &new_tensor](auto it) -> bool {
      TensorVar src = it->first;
      for (const auto& stmt : it->second) {
        Stmt remapped = TensorRefRewriter::Rewrite(stmt, src, new_tensor);
        seq->push_back(KernelReplacePointSearcher::Seek(remapped, Evaluate(0)));
      }
      post_tensor_def_stmts_.erase(it);
      return true;
    };

    bool changed = false;
    if (auto it = post_tensor_def_stmts_.find(old_tensor); it != post_tensor_def_stmts_.end()) {
      changed |= append_with_remap(it);
    }
    if (!new_tensor.same_as(old_tensor)) {
      if (auto it = post_tensor_def_stmts_.find(new_tensor); it != post_tensor_def_stmts_.end()) {
        changed |= append_with_remap(it);
      }
    }
    return changed;
  }

  // No failure aggregation; pass surfaces per-op exceptions
};

}  // namespace

namespace {
Target ResolveTarget(const Function& f) {
  auto target = f->GetAttr(tvm::attr::kTarget);
  if (!target.has_value()) {
    target = Target::Current(false).value();
  }
  return target.value();
}
}  // namespace

namespace transform {

Pass TileDispatch() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    if (!f->body.has_value()) return f;
    Target target = ResolveTarget(f);
    auto* n = f.CopyOnWrite();
    n->body = TileDispatcher::LowerOpCalls(n->body.value(), target);
    if (!NoOpCallVerifier::Verify(n->body.value())) {
      LOG(FATAL) << "Failed to lower the TIRx program: " << f;
    }
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "tirx.TileDispatch");
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
