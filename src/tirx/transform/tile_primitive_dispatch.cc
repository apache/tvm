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
 * \file tile_primitive_dispatch.cc
 * \brief Lower TilePrimitiveCall nodes via registered dispatchers (also resolves ScopeIdDef
 * declarations and emits launch params).
 */

#include <tvm/arith/analyzer.h>
#include <tvm/arith/pattern.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/exec_context.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/target_builtin/cuda.h>
#include <tvm/tirx/tirx_op.h>
#include <tvm/tirx/transform.h>

#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../analysis/filter_canonical.h"
#include "../ir/functor_common.h"
#include "../ir/tir_visitor_with_path.h"

namespace tvm {
namespace tirx {

namespace {

// Gather every ScopeIdDef declared anywhere under a given Stmt, paired with
// the source stmt node that declared it (for implicit-eval routing). The
// source is the AttrStmt(kDeviceEntry) marker.
struct ScopeIdDefWithSource {
  ScopeIdDef def;
  const StmtNode* source_stmt;
};

class ScopeIdDefGather : public StmtExprVisitor {
 public:
  static std::vector<ScopeIdDefWithSource> Gather(const Stmt& stmt) {
    ScopeIdDefGather gather;
    gather(stmt);
    return std::move(gather.out_);
  }

  void VisitStmt_(const AttrStmtNode* op) override {
    if (op->attr_key == tvm::tirx::attr::kDeviceEntry) {
      EnterSourceAndPartition(op, [&]() { StmtExprVisitor::VisitStmt_(op); });
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ScopeIdDefStmtNode* op) override {
    out_.push_back({op->def, source_stmt_});
    StmtExprVisitor::VisitStmt_(op);
  }

 private:
  // Visit body with ``src`` as the source-stmt context, then re-order
  // newly-added defs so direct-children defs come after nested ones —
  // preserves LIFO order required by ExtractKernelLaunchParams.
  template <typename F>
  void EnterSourceAndPartition(const StmtNode* src, F&& visit_body) {
    const StmtNode* prev_source = source_stmt_;
    size_t baseline = out_.size();
    source_stmt_ = src;
    visit_body();
    source_stmt_ = prev_source;

    std::vector<ScopeIdDefWithSource> direct;
    std::vector<ScopeIdDefWithSource> nested;
    direct.reserve(out_.size() - baseline);
    for (size_t i = baseline; i < out_.size(); ++i) {
      if (out_[i].source_stmt == src) {
        direct.push_back(out_[i]);
      } else {
        nested.push_back(out_[i]);
      }
    }
    out_.resize(baseline);
    out_.insert(out_.end(), nested.begin(), nested.end());
    out_.insert(out_.end(), direct.begin(), direct.end());
  }

  std::vector<ScopeIdDefWithSource> out_;
  const StmtNode* source_stmt_ = nullptr;
};

class ElectSyncFinder : public StmtExprVisitor {
 public:
  static bool Contains(const PrimExpr& expr) {
    ElectSyncFinder finder;
    finder(expr);
    return finder.found_;
  }

 private:
  using StmtExprVisitor::VisitStmt_;

  void VisitExpr_(const CallNode* op) final {
    auto is_canonical_elect_sync = [&]() {
      if (op->op.same_as(tirx::builtin::ptx_elect_sync())) return true;
      if (auto call_op = op->op.as<Op>()) {
        return call_op.value()->name == "tirx.ptx.elect_sync";
      }
      return false;
    };
    if (is_canonical_elect_sync()) {
      found_ = true;
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  bool found_{false};
};

class ScopeIdVarFinder : public StmtExprVisitor {
 public:
  static bool Contains(const PrimExpr& expr, const std::vector<Var>& vars) {
    ScopeIdVarFinder finder(vars);
    finder(expr);
    return finder.found_;
  }

 private:
  explicit ScopeIdVarFinder(const std::vector<Var>& vars) : vars_(vars) {}

  using StmtExprVisitor::VisitStmt_;

  void VisitExpr_(const VarNode* op) final {
    Var var = ffi::GetRef<Var>(op);
    for (const auto& candidate : vars_) {
      if (candidate.same_as(var)) {
        found_ = true;
        return;
      }
    }
  }

  const std::vector<Var>& vars_;
  bool found_{false};
};

// Remove any standalone ``ScopeIdDefStmt`` nodes; the resolved values are
// bound at kernel scope via Bind statements emitted separately.
class ScopeIdDefRemover : public StmtExprMutator {
 public:
  static Stmt Remove(const Stmt& stmt) { return ScopeIdDefRemover()(stmt); }

  Stmt VisitStmt_(const ScopeIdDefStmtNode* op) override {
    // Drop the def stmt by replacing with a no-op Evaluate(0). It will be
    // flattened away by SeqStmt::Flatten elsewhere or stay as a benign
    // no-op for downstream passes.
    return Evaluate(IntImm(DataType::Int(32), 0));
  }
};

// For implicitly-named ScopeIdDefs (parser-emitted Var("")), inject an
// Evaluate(var) at the source stmt's body so the binding stays observably
// live in the IR even if user code never references it. Routing uses source
// stmt-node identity to match against the device-entry marker.
class ImplicitScopeIdEvalInjector : public StmtExprMutator {
 public:
  static Stmt Inject(const Stmt& stmt,
                     const std::vector<std::pair<Var, const StmtNode*>>& eval_specs) {
    ImplicitScopeIdEvalInjector injector(eval_specs);
    return injector(stmt);
  }

 private:
  explicit ImplicitScopeIdEvalInjector(
      const std::vector<std::pair<Var, const StmtNode*>>& eval_specs) {
    for (const auto& [var, src] : eval_specs) {
      eval_map_[src].push_back(var);
    }
  }

  ffi::Array<Stmt> ConsumeEvalsFor(const StmtNode* src) {
    ffi::Array<Stmt> evals;
    auto it = eval_map_.find(src);
    if (it != eval_map_.end() && !it->second.empty()) {
      evals.reserve(it->second.size());
      for (const Var& var : it->second) {
        evals.push_back(Evaluate(var));
      }
      eval_map_.erase(it);
    }
    return evals;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    Stmt body = VisitStmt(op->body);
    if (op->attr_key == tvm::tirx::attr::kDeviceEntry) {
      auto evals = ConsumeEvalsFor(op);
      if (!evals.empty()) {
        body = SeqStmt::Flatten(evals, body);
      }
    }
    if (body.same_as(op->body)) return ffi::GetRef<Stmt>(op);
    return AttrStmt(op->node, op->attr_key, op->value, body, op->span);
  }

  std::unordered_map<const StmtNode*, std::vector<Var>> eval_map_;
};

}  // namespace

class NoOpCallVerifier : public Verifier<NoOpCallVerifier> {
 public:
  using Verifier::Verifier;

 private:
  using Verifier::Visit;

  void VisitStmt_(const tirx::TilePrimitiveCallNode* obj, ffi::reflection::AccessPath path) final {
    Verify(false) << "TIRxError: TilePrimitiveCall at " << path
                  << " is not allowed in TIRx before lowering";
  }
};

class TilePrimitiveDispatcher : public StmtExprMutator {
 public:
  explicit TilePrimitiveDispatcher(const Target& target) : target_(target) {}

  static Stmt LowerOpCalls(const Stmt& stmt, const Target& target) {
    return TilePrimitiveDispatcher(target)(stmt);
  }

 private:
  class BufferRefRewriter : public StmtExprMutator {
   public:
    static Stmt Rewrite(const Stmt& stmt, const Buffer& src, const Buffer& dst) {
      if (src.same_as(dst)) {
        return stmt;
      }
      return BufferRefRewriter(src, dst)(stmt);
    }

   private:
    BufferRefRewriter(Buffer src, Buffer dst) : src_(std::move(src)), dst_(std::move(dst)) {}

    Buffer VisitBufferDef(const Buffer& buffer, bool alloc_data) final {
      Buffer new_buffer = StmtExprMutator::VisitBufferDef(buffer, alloc_data);
      if (new_buffer.same_as(src_)) {
        return dst_;
      }
      return new_buffer;
    }

    Buffer VisitBufferUse(const Buffer& buffer) final {
      if (buffer.same_as(src_)) {
        return dst_;
      }
      return StmtExprMutator::VisitBufferUse(buffer);
    }

    Buffer src_;
    Buffer dst_;
  };

  class KernelReplacePointSearcher : public StmtExprMutator {
   public:
    explicit KernelReplacePointSearcher(const Stmt& body) : body_(body) {}

    static Stmt Seek(const Stmt& stmt, const Stmt& body) {
      return KernelReplacePointSearcher(body)(stmt);
    }

   private:
    Stmt VisitStmt_(const EvaluateNode* op) final {
      const auto* call = op->value.as<CallNode>();
      if (call != nullptr && call->op.same_as(tirx::builtin::tvm_kernel_replace_point())) {
        return body_;
      }
      return StmtExprMutator::VisitStmt_(op);
    }

    Stmt body_;
  };

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tirx::attr::kDeviceEntry) {
      return ProcessDeviceEntry(op);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt ProcessDeviceEntry(const AttrStmtNode* entry_node) {
    Stmt body_to_visit = entry_node->body;

    bool is_first_block = false;
    std::swap(is_first_block, is_first_block_);

    std::vector<std::pair<Var, PrimExpr>> scope_binds;
    std::vector<std::pair<Var, const StmtNode*>> implicit_scope_id_evals;

    launch_params_.clear();
    // Pre-dispatch: only populate ``launch_params_`` + synthesize
    // ``warp_id_in_cta``. The dispatch impls (run via ``VisitStmt`` below)
    // read ``launch_params_`` through ``sctx``, so this much must happen
    // first. The per-def Bind resolution is deferred to AFTER dispatch so
    // it can pick up any ``ScopeIdDef`` declared inside dispatched impls.
    PrepareLaunchParams(entry_node, body_to_visit, &scope_binds);
    bool pushed_base_ctx = PushKernelEntryCtx();

    // Direct ScopeIdDefStmt children of the device-entry marker live here.
    scope_id_defs_at_level_.push_back({});
    Stmt body = VisitStmt(body_to_visit);
    scope_id_defs_at_level_.pop_back();

    // Post-dispatch: re-gather the now-inlined body and resolve every
    // ``ScopeIdDef`` (kernel-side + dispatch-introduced) into ``scope_binds``.
    ResolveAllScopeBinds(entry_node, body, &scope_binds, &implicit_scope_id_evals);

    auto pop_exec_contexts = [&]() {
      if (pushed_base_ctx) ctx_stack_.pop_back();
    };

    if (!is_first_block) {
      std::swap(is_first_block, is_first_block_);
      pop_exec_contexts();
      if (body.same_as(body_to_visit)) {
        return ffi::GetRef<Stmt>(entry_node);
      }
      return AttrStmt(entry_node->node, entry_node->attr_key, entry_node->value, body,
                      entry_node->span);
    }

    // Insert device init stmts into kernel body.
    for (auto it = device_init_stmts_.rbegin(); it != device_init_stmts_.rend(); ++it) {
      body = KernelReplacePointSearcher::Seek(*it, body);
    }
    // Insert alloc buffers at the beginning of the kernel body.
    if (!alloc_buffers_.empty()) {
      std::vector<Stmt> seq;
      seq.reserve(alloc_buffers_.size() + 1);
      for (const auto& buffer : alloc_buffers_) {
        seq.push_back(tvm::tirx::AllocBuffer(buffer));
      }
      seq.push_back(std::move(body));
      body = SeqStmt::Flatten(seq);
    }
    alloc_buffers_.clear();

    // Partition implicit evals: evals sourced from the device-entry marker
    // are prepended directly to ``body``. The entry-marker wrapper is
    // stripped below, so the injector (which matches by source node identity)
    // can't reach into the stripped node — handle these inline. Evals
    // sourced from inner ExecScopes (which survive lowering) are still
    // routed via the injector.
    {
      ffi::Array<Stmt> prepend_evals;
      std::vector<std::pair<Var, const StmtNode*>> remaining;
      const StmtNode* entry_stmt = static_cast<const StmtNode*>(entry_node);
      for (const auto& [var, src] : implicit_scope_id_evals) {
        if (src == entry_stmt) {
          prepend_evals.push_back(Evaluate(var));
        } else {
          remaining.push_back({var, src});
        }
      }
      if (!prepend_evals.empty()) {
        body = SeqStmt::Flatten(prepend_evals, body);
      }
      implicit_scope_id_evals = std::move(remaining);
    }

    // Strip the device-entry marker; its only role was to scope this
    // processing. Downstream passes consume the bound launch params and
    // alloc buffers wrapping ``body`` directly.
    Stmt res = body;

    // Inject implicit scope-id evals sourced from the device-entry marker.
    // Must run before ScopeIdDefRemover, which rebuilds nodes and
    // invalidates source identities.
    res = ImplicitScopeIdEvalInjector::Inject(res, implicit_scope_id_evals);

    // Strip standalone ScopeIdDefStmt nodes -- their values are now bound at
    // kernel scope via the Bind statements below.
    res = ScopeIdDefRemover::Remove(res);

    // Prepend Bind(var, value) for every resolved scope id (and the derived
    // warp_id_in_cta var when threadIdx is present).
    ffi::Array<Stmt> bind_stmts;
    bind_stmts.reserve(scope_binds.size());
    for (const auto& [var, value] : scope_binds) {
      bind_stmts.push_back(Bind(var, value));
    }
    res = SeqStmt::Flatten(bind_stmts, res);

    // Wrap with thread_extent attrs (consumed by downstream codegen passes
    // that expect TVM-standard thread launch annotations).
    for (const auto& [tag, iv] : launch_params_) {
      if (tag == "warp_id_in_cta") continue;
      res = AttrStmt(iv, tirx::attr::thread_extent, iv->dom->extent, res);
    }

    // Insert host init stmts outside the outermost thread binding or block.
    if (is_first_thread_attr_) {
      for (const auto& stmt : host_init_stmts_) {
        res = KernelReplacePointSearcher::Seek(stmt, std::move(res));
      }
      host_init_stmts_.clear();
    }
    std::swap(is_first_block, is_first_block_);
    pop_exec_contexts();
    return res;
  }

  Stmt VisitStmt_(const ScopeIdDefStmtNode* op) final {
    // Register the def at the current (innermost) ExecScope's level so
    // ResolveScopeIdTarget / ScopeIdTargets can find it. The def remains
    // visible to subsequent sibling stmts within this scope.
    if (!scope_id_defs_at_level_.empty()) {
      scope_id_defs_at_level_.back().push_back(op->def);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const SeqStmtNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    if (post_buffer_def_stmts_.empty()) {
      return stmt;
    }
    const auto* seq = stmt.as<SeqStmtNode>();
    if (seq == nullptr) {
      return stmt;
    }

    std::vector<Stmt> rebuilt;
    rebuilt.reserve(seq->seq.size() + post_buffer_def_stmts_.size());
    bool changed = false;
    for (const Stmt& s : seq->seq) {
      rebuilt.push_back(s);
      if (const auto* alloc = s.as<AllocBufferNode>()) {
        changed |= AppendPostBufferDefStmts(&rebuilt, alloc->buffer, alloc->buffer);
      } else if (const auto* decl = s.as<DeclBufferNode>()) {
        changed |= AppendPostBufferDefStmts(&rebuilt, decl->buffer, decl->buffer);
      }
    }
    if (!changed) {
      return stmt;
    }
    return SeqStmt::Flatten(rebuilt);
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // Collect the loop variables
    auto loop_var = Downcast<Var>(op->loop_var);
    TVM_FFI_ICHECK(!var_range_map_.count(loop_var)) << "Internal Error: Duplicate loop variable";
    var_range_map_.Set(loop_var, Range::FromMinExtent(op->min, op->extent));
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocBufferNode* op) final {
    Buffer old_buffer = op->buffer;
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<AllocBufferNode>();
    TVM_FFI_ICHECK(op);

    std::vector<Stmt> seq{stmt};
    AppendPostBufferDefStmts(&seq, old_buffer, op->buffer);
    return SeqStmt::Flatten(seq);
  }

  Stmt VisitStmt_(const DeclBufferNode* op) final {
    Buffer old_buffer = op->buffer;
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<DeclBufferNode>();
    TVM_FFI_ICHECK(op);

    std::vector<Stmt> seq{stmt};
    AppendPostBufferDefStmts(&seq, old_buffer, op->buffer);
    return SeqStmt::Flatten(seq);
  }

  Stmt VisitStmt_(const IfThenElseNode* op) final {
    // Narrow ExecContext for structurally recognized predicates on the
    // then-branch. The canonical-form classifier (filter_canonical.h)
    // recognizes the dominant shapes: pure conjunctions of `scopeid_var op
    // const` comparisons plus bare `ptx_elect_sync()` calls. Predicates
    // outside that grammar (e.g. linear shifts like `v - 1 < 5`, modulo
    // equality like `v % 2 == 0`, or the legacy `tirx.filter` wrapper) fall
    // back to the existing dispatcher, which has more permissive matching
    // paths.
    int pushed_ctx = TryPushCanonicalCtx(op->condition);
    if (pushed_ctx < 0) {
      pushed_ctx = PushPredicateCtx(op->condition);
    }
    PrimExpr new_cond = RewriteFilterCalls(op->condition);
    Stmt then_case = VisitStmt(op->then_case);
    while (pushed_ctx-- > 0) ctx_stack_.pop_back();
    ffi::Optional<Stmt> else_case;
    if (op->else_case.defined()) {
      else_case = VisitStmt(op->else_case.value());
    }
    bool unchanged = new_cond.same_as(op->condition) && then_case.same_as(op->then_case) &&
                     ((!op->else_case.defined() && !else_case.defined()) ||
                      (op->else_case.defined() && else_case.defined() &&
                       else_case.value().same_as(op->else_case.value())));
    if (unchanged) return ffi::GetRef<Stmt>(op);
    return IfThenElse(new_cond, then_case, else_case);
  }

  Stmt VisitStmt_(const tirx::TilePrimitiveCallNode* op) final {
    // Scope is a per-call field on the node. Derive the (inter, intra) split
    // on the spot from the current active set ``A`` (tracked through control
    // flow on ``ctx_stack_``) under this call's own ``op->scope``.
    ffi::Map<ffi::String, ffi::Array<PrimExpr>> inter_map, intra_map;
    ffi::String scope_kind = ScopeKindToString(op->scope->kind);
    if (!ctx_stack_.empty()) {
      ExecSplit split;
      std::string err;
      if (ScopeSwitch(ctx_stack_.back().A, op->scope->kind, &split, &err)) {
        inter_map = EncodeSplitSide(split.inter);
        intra_map = EncodeSplitSide(split.intra);
      } else {
        // Factoring failure (e.g. warpgroup with a lane that crosses a
        // warpgroup boundary unaligned). Leave the split empty; dispatchers
        // fall back to scope_kind. This is not validated earlier, so an
        // incompatible per-call scope only warns here and yields a degenerate
        // split rather than a hard error.
        LOG(WARNING) << "ExecContext scope_switch failed: " << err;
      }
    }
    tirx::DispatchContext sctx(target_, op->scope, launch_params_, var_range_map_,
                               /*alloc_only=*/false, /*callbacks=*/{}, shared_state_, inter_map,
                               intra_map, scope_kind);
    static auto f_op_dispatcher_ = ffi::Function::GetGlobal("tirx.f_op_dispatcher");
    TVM_FFI_ICHECK(f_op_dispatcher_.has_value())
        << "Internal Error: tirx.f_op_dispatcher is not registered";
    PrimFunc res =
        f_op_dispatcher_.value()(ffi::GetRef<tirx::TilePrimitiveCall>(op), sctx).cast<PrimFunc>();
    TVM_FFI_ICHECK(res.defined()) << "TIRx dispatcher did not return a PrimFunc";
    // Implementation found, handle callbacks
    if (auto bufs = sctx->callbacks.Get(tirx::callback::kPrivateAlloc)) {
      auto buf_list = bufs.value().as<Array<Buffer>>().value();
      alloc_buffers_.insert(alloc_buffers_.end(), buf_list.begin(), buf_list.end());
    }
    if (auto stmts = sctx->callbacks.Get(tirx::callback::kDeviceInitStmt)) {
      auto stmt_list = stmts.value().as<Array<Stmt>>().value();
      device_init_stmts_.insert(device_init_stmts_.end(), stmt_list.begin(), stmt_list.end());
    }
    if (auto stmts = sctx->callbacks.Get(tirx::callback::kHostInitStmt)) {
      auto stmt_list = stmts.value().as<Array<Stmt>>().value();
      host_init_stmts_.insert(host_init_stmts_.end(), stmt_list.begin(), stmt_list.end());
    }
    if (auto mapping = sctx->callbacks.Get(tirx::callback::kPostBufferDefStmt)) {
      auto map = Downcast<ffi::Map<Buffer, Array<Stmt>>>(mapping.value());
      for (const auto& [buffer, stmts] : map) {
        auto& vec = post_buffer_def_stmts_[buffer];
        vec.insert(vec.end(), stmts.begin(), stmts.end());
      }
    }
    // Propagate shared_state changes back (Map uses COW semantics)
    shared_state_ = sctx->shared_state;
    return res->body;
  }

  // --- Scope-id resolution at kernel scope ----------------------------------

  // PRE-DISPATCH step: gather + verify ScopeIdDefs on the original kernel
  // body, populate ``launch_params_`` from the canonical bindings, and
  // synthesize the ``warp_id_in_cta`` helper bind. The per-def Bind
  // resolution that used to live here is now in ``ResolveAllScopeBinds``,
  // which runs AFTER dispatch so it sees ScopeIdDefs introduced by
  // dispatched impls too.
  void PrepareLaunchParams(const AttrStmtNode* entry_node, Stmt body,
                           std::vector<std::pair<Var, PrimExpr>>* scope_binds) {
    Stmt gather_target = AttrStmt(IntImm(DataType::Int(32), 0), tvm::tirx::attr::kDeviceEntry,
                                  IntImm(DataType::Bool(), 1), body);
    std::vector<ScopeIdDefWithSource> gathered = ScopeIdDefGather::Gather(gather_target);
    Array<ScopeIdDef> defs;
    defs.reserve(gathered.size());
    for (const auto& g : gathered) defs.push_back(g.def);
    ScopeIdDefVerifier verifier;
    TVM_FFI_ICHECK(verifier.Verify(defs)) << "Inconsistent ScopeIdDef";

    ExtractKernelLaunchParams(verifier.id_set);

    // Synthesize the warp_id_in_cta helper (CUDA only) when threadIdx is set.
    if (launch_params_.count("threadIdx.x") > 0) {
      PrimExpr shuffled = ScopeIdResolve::ComputeWarpIdInCta(launch_params_);
      Var warp_id_in_cta_var("warp_id_in_cta", shuffled.dtype());
      scope_binds->push_back({warp_id_in_cta_var, shuffled});
      IterVar warp_iv(Range::FromMinExtent(0, 1), warp_id_in_cta_var, kThreadIndex,
                      "warp_id_in_cta");
      launch_params_.insert({"warp_id_in_cta", warp_iv});
    }
  }

  // POST-DISPATCH step: re-gather the now-inlined body (which includes any
  // ScopeIdDefs introduced inside dispatched impls), verify against the
  // current launch_params, resolve each def, and push (Var, value) pairs
  // into ``*scope_binds``. Implicit (unnamed) scope-id Vars are recorded
  // for later evaluate-injection.
  void ResolveAllScopeBinds(const AttrStmtNode* entry_node, Stmt body,
                            std::vector<std::pair<Var, PrimExpr>>* scope_binds,
                            std::vector<std::pair<Var, const StmtNode*>>* implicit_scope_id_evals) {
    // Gather from a temporary stmt synthesized as the device-entry marker
    // so direct ScopeIdDefStmt children are attributed back to entry_node.
    Stmt gather_target = AttrStmt(IntImm(DataType::Int(32), 0), tvm::tirx::attr::kDeviceEntry,
                                  IntImm(DataType::Bool(), 1), body);
    std::vector<ScopeIdDefWithSource> gathered = ScopeIdDefGather::Gather(gather_target);
    // Remap the synthetic source pointer back to the real entry_node so the
    // injector matches against the actual node present in the post-processed
    // IR.
    const StmtNode* synth_src = static_cast<const StmtNode*>(gather_target.get());
    for (auto& g : gathered) {
      if (g.source_stmt == synth_src) {
        g.source_stmt = static_cast<const StmtNode*>(entry_node);
      }
    }
    Array<ScopeIdDef> defs;
    defs.reserve(gathered.size());
    for (const auto& g : gathered) defs.push_back(g.def);

    ScopeIdDefVerifier verifier;
    TVM_FFI_ICHECK(verifier.Verify(defs)) << "Inconsistent ScopeIdDef";

    auto is_implicit = [](const Var& v) { return v->name_hint.empty(); };
    for (const auto& g : gathered) {
      ScopeIdDef def = g.def;
      // Deferred extents: resolved via closure into verifier.id_set.
      if (def.is_deferred()) {
        auto it = verifier.id_set.find(def->scope);
        TVM_FFI_ICHECK(it != verifier.id_set.end() && !(*it).second.is_deferred())
            << "Internal Error: deferred def not resolved";
        def = ScopeIdDef(def->def_ids, (*it).second->extents, def->scope, def->preferred_extents);
      }
      const auto& extents = def->extents.value();
      auto resolved = ScopeIdResolve::Resolve(def->scope, def->extents, extents.size(),
                                              target_->kind->name, launch_params_);
      TVM_FFI_ICHECK_EQ(resolved.size(), extents.size())
          << "Internal Error: Inconsistent resolved size";
      for (size_t i = 0; i < def->def_ids.size(); i++) {
        // Reuse the original Var as the bind target -- no rename, no
        // substitution. The IR already references this Var directly, and
        // dispatch's filter resolution walks ScopeIdDefStmt::def
        // to map Vars back to their ScopeBinding.
        Var bind_var = def->def_ids[i];
        PrimExpr value = resolved[i];
        if (bind_var->dtype != value.dtype()) {
          value = Cast(bind_var->dtype, value);
        }
        scope_binds->push_back({bind_var, value});
        if (is_implicit(bind_var)) {
          implicit_scope_id_evals->push_back({bind_var, g.source_stmt});
        }
      }
    }
  }

  // Translate the canonical ScopeBinding -> launch param IterVars
  // (blockIdx.{x,y,z}, clusterCtaIdx.*, threadIdx.{x,y,z}, etc.).
  void ExtractKernelLaunchParams(const ScopeIdDefVerifier::ScopeIdSet& id_set) {
    auto add_launch_param = [&](ScopeBinding binding, const std::string& prefix) {
      auto it = id_set.find(binding);
      if (it == id_set.end()) return;
      const auto& def = (*it).second;
      TVM_FFI_ICHECK(!def.is_deferred()) << "Internal Error: launch param built from deferred def";
      const auto& extents = def->extents.value();
      TVM_FFI_ICHECK_LE(extents.size(), 3) << "ValueError: Only up to 3 extents are supported";
      for (size_t i = 0; i < extents.size(); i++) {
        std::string thread_tag = prefix + static_cast<char>('x' + i);
        IterVar iv(Range::FromMinExtent(0, extents[i]), Var(thread_tag), IterVarType::kThreadIndex,
                   thread_tag);
        launch_params_.insert({ffi::String(thread_tag), iv});
      }
    };
    auto cluster_cta_it = id_set.find(ScopeBinding::kClusterCta);
    if (cluster_cta_it == id_set.end() || is_one((*cluster_cta_it).second.fused_extent())) {
      // no cluster
      add_launch_param(ScopeBinding::kKernelCta, "blockIdx.");
    } else {
      // use cluster
      TVM_FFI_ICHECK(target_->kind->name == "cuda")
          << "ValueError: cluster is only supported in CUDA";
      TVM_FFI_ICHECK_EQ(target_->kind->default_device_type, kDLCUDA)
          << "ValueError: cluster is only supported in CUDA";
      add_launch_param(ScopeBinding::kClusterCta, "clusterCtaIdx.");
      // Preferred cluster size (CUDA 12.8+)
      const auto& cta_def = (*cluster_cta_it).second;
      if (cta_def->preferred_extents.defined()) {
        const auto& pref = cta_def->preferred_extents.value();
        for (size_t i = 0; i < pref.size(); i++) {
          std::string tag = "preferredClusterCtaIdx." + std::string(1, 'x' + i);
          IterVar iv(Range::FromMinExtent(0, pref[i]), Var(tag), IterVarType::kThreadIndex, tag);
          launch_params_.insert({ffi::String(tag), iv});
        }
      }
      add_launch_param(ScopeBinding::kKernelCta, "blockIdx.");
    }
    add_launch_param(ScopeBinding::kCtaThread, "threadIdx.");
    if (!id_set.empty()) {
      TVM_FFI_ICHECK(launch_params_.count("threadIdx.x") > 0)
          << "ValueError: kernel has no thread launch parameters. "
          << "At minimum, declare cta->thread extent (e.g., Tx.thread_id([128]))";
    }
  }

  // --- ExecContext tracking helpers -----------------------------------------

  bool PushKernelEntryCtx() {
    auto prod_extent = [&](std::initializer_list<const char*> keys) -> int64_t {
      int64_t n = 1;
      for (const char* k : keys) {
        auto it = launch_params_.find(ffi::String(k));
        if (it == launch_params_.end()) continue;
        const auto* imm = it->second->dom->extent.as<IntImmNode>();
        if (imm == nullptr) return 0;  // symbolic
        n *= imm->value;
      }
      return n;
    };
    auto collect_extents = [&](std::initializer_list<std::pair<const char*, const char*>> keys) {
      std::vector<std::pair<std::string, int64_t>> out;
      for (const auto& [thread_key, axis_name] : keys) {
        auto it = launch_params_.find(ffi::String(thread_key));
        if (it == launch_params_.end()) continue;
        const auto* imm = it->second->dom->extent.as<IntImmNode>();
        if (imm == nullptr) return std::vector<std::pair<std::string, int64_t>>();
        out.push_back({axis_name, imm->value});
      }
      return out;
    };
    int64_t thread_ext = prod_extent({"threadIdx.x", "threadIdx.y", "threadIdx.z"});
    if (thread_ext <= 0) {
      // launch params missing or symbolic; ExecContext tracking is not
      // available for this kernel. Dispatchers fall back to scope_kind only.
      LOG(WARNING) << "ExecContext tracking disabled: missing/symbolic threadIdx extents";
      return false;
    }
    int64_t warp_ext = thread_ext / 32;
    auto cluster_cta_axes = collect_extents(
        {{"clusterCtaIdx.x", "cbx"}, {"clusterCtaIdx.y", "cby"}, {"clusterCtaIdx.z", "cbz"}});
    cluster_cta_axis_extents_ = cluster_cta_axes;
    auto cta_axes = cluster_cta_axes;
    if (cta_axes.empty()) {
      cta_axes =
          collect_extents({{"blockIdx.x", "bx"}, {"blockIdx.y", "by"}, {"blockIdx.z", "bz"}});
      cluster_cta_axis_extents_.clear();
    }
    int64_t cta_ext = 1;
    for (const auto& axis : cta_axes) {
      cta_ext *= axis.second;
    }
    // Preserve the old flattened cta_id split for 0-D/1-D declarations. Multi-dimensional
    // CTA ids keep their concrete factor axes (bx/by/bz or cbx/cby/cbz).
    if (cta_axes.size() <= 1) cta_axes.clear();
    ctx_stack_.push_back(ExecContext::AtKernelEntry(/*lane_ext=*/32, warp_ext, cta_ext, cta_axes));
    return true;
  }

  struct ScopeIdTarget {
    ScopeBinding binding;
    int dim = 0;
    int ndim = 1;
  };

  struct ScopeIdRange {
    ScopeIdTarget target;
    int64_t lo = arith::ConstIntBound::kNegInf;
    int64_t hi = arith::ConstIntBound::kPosInf;
  };

  struct PendingRangeGroup {
    ScopeIdTarget target;
    int64_t lo = arith::ConstIntBound::kNegInf;
    int64_t hi = arith::ConstIntBound::kPosInf;
    std::vector<size_t> indices;
  };

  static bool SameScopeIdTarget(const ScopeIdTarget& lhs, const ScopeIdTarget& rhs) {
    return lhs.binding == rhs.binding && lhs.dim == rhs.dim && lhs.ndim == rhs.ndim;
  }

  bool KernelCtaPredicateOverlapsClusterCta(const ScopeIdTarget& target) const {
    return target.binding == ScopeBinding::kKernelCta && !cluster_cta_axis_extents_.empty();
  }

  std::optional<ScopeIdTarget> ResolveScopeIdTarget(const PrimExpr& expr) const {
    const auto* var_node = expr.as<VarNode>();
    if (var_node == nullptr) return std::nullopt;
    Var var = ffi::GetRef<Var>(var_node);
    // Walk the parallel ScopeIdDef stack (defs visible at each nesting
    // level) innermost-first.
    for (auto it = scope_id_defs_at_level_.rbegin(); it != scope_id_defs_at_level_.rend(); ++it) {
      for (const auto& def : *it) {
        for (size_t i = 0; i < def->def_ids.size(); ++i) {
          if (def->def_ids[i].same_as(var)) {
            return ScopeIdTarget{def->scope, static_cast<int>(i),
                                 static_cast<int>(def->def_ids.size())};
          }
        }
      }
    }
    return std::nullopt;
  }

  bool TryPushRangeForTarget(const ScopeIdTarget& target, int64_t lo, int64_t hi) {
    if (ctx_stack_.empty()) return false;
    if (target.binding == ScopeBinding::kClusterCtaPair) {
      if (hi != lo + 1 || lo < 0 || lo > 1) return false;
      return TryPushCtaPairValue(lo);
    }
    if (KernelCtaPredicateOverlapsClusterCta(target)) return false;
    ExecContext new_ctx;
    std::string err;
    if (target.ndim != 1) {
      auto cta_axis = CtaAxisName(target);
      if (!cta_axis) return false;
      if (!ctx_stack_.back().WithCtaAxisFilter(*cta_axis, lo, hi, &new_ctx, &err)) return false;
      ctx_stack_.push_back(new_ctx);
      return true;
    }
    if (!ctx_stack_.back().WithFilter(target.binding, lo, hi, &new_ctx, &err)) return false;
    ctx_stack_.push_back(new_ctx);
    return true;
  }

  bool TryPushModuloForTarget(const ScopeIdTarget& target, int64_t modulus, int64_t residue) {
    if (ctx_stack_.empty()) return false;
    if (target.binding == ScopeBinding::kClusterCtaPair) return false;
    if (KernelCtaPredicateOverlapsClusterCta(target)) return false;
    ExecContext new_ctx;
    std::string err;
    if (target.ndim != 1) {
      auto cta_axis = CtaAxisName(target);
      if (!cta_axis) return false;
      if (!ctx_stack_.back().WithCtaAxisModulo(*cta_axis, modulus, residue, &new_ctx, &err)) {
        return false;
      }
      ctx_stack_.push_back(new_ctx);
      return true;
    }
    if (target.binding == ScopeBinding::kKernelCta || target.binding == ScopeBinding::kClusterCta) {
      if (!ctx_stack_.back().WithCtaAxisModulo("cta_id", modulus, residue, &new_ctx, &err)) {
        return false;
      }
      ctx_stack_.push_back(new_ctx);
      return true;
    }
    return false;
  }

  bool TryPushCtaPairValue(int64_t value) {
    if (ctx_stack_.empty()) return false;
    if (cluster_cta_axis_extents_.empty()) return false;
    if (cluster_cta_axis_extents_.size() <= 1) {
      ExecContext new_ctx;
      std::string err;
      if (!ctx_stack_.back().WithCtaAxisModulo("cta_id", 2, value, &new_ctx, &err)) return false;
      ctx_stack_.push_back(new_ctx);
      return true;
    }

    std::optional<std::string> parity_axis;
    int64_t coeff = 1;
    int64_t fixed = 0;
    for (const auto& [axis, extent] : cluster_cta_axis_extents_) {
      AxisRange range;
      if (!ctx_stack_.back().A.GetAxis(axis, &range)) return false;
      int64_t active_extent = 0;
      int64_t active_offset = 0;
      int64_t active_stride = 0;
      if (!TryExtractIntImm(range.extent, &active_extent) ||
          !TryExtractIntImm(range.offset, &active_offset) ||
          !TryExtractIntImm(range.stride, &active_stride)) {
        return false;
      }
      fixed += coeff * active_offset;
      if (active_extent > 1 && (coeff * active_stride) % 2 != 0) {
        if (parity_axis) return false;
        parity_axis = axis;
      }
      coeff *= extent;
    }
    int64_t residue = (value - fixed) % 2;
    if (residue < 0) residue += 2;
    if (!parity_axis) {
      if (residue != 0) return false;
      ctx_stack_.push_back(ctx_stack_.back());
      return true;
    }

    ExecContext new_ctx;
    std::string err;
    if (!ctx_stack_.back().WithCtaAxisModulo(*parity_axis, 2, residue, &new_ctx, &err)) {
      return false;
    }
    ctx_stack_.push_back(new_ctx);
    return true;
  }

  static std::optional<std::string> CtaAxisName(const ScopeIdTarget& target) {
    static constexpr const char* kKernelCtaAxes[] = {"bx", "by", "bz"};
    static constexpr const char* kClusterCtaAxes[] = {"cbx", "cby", "cbz"};
    if (target.dim < 0 || target.dim >= 3) return std::nullopt;
    if (target.binding == ScopeBinding::kKernelCta) {
      return std::string(kKernelCtaAxes[target.dim]);
    }
    if (target.binding == ScopeBinding::kClusterCta) {
      return std::string(kClusterCtaAxes[target.dim]);
    }
    return std::nullopt;
  }

  bool TryPushSelectorForTarget(const ScopeIdTarget& target, PrimExpr selector) {
    if (ctx_stack_.empty()) return false;
    if (target.ndim != 1) return false;
    if (KernelCtaPredicateOverlapsClusterCta(target)) return false;
    ExecContext new_ctx;
    std::string err;
    if (!ctx_stack_.back().WithSelector(target.binding, selector, &new_ctx, &err)) return false;
    ctx_stack_.push_back(new_ctx);
    return true;
  }

  static bool TryExtractIntImm(const PrimExpr& expr, int64_t* value) {
    if (const auto* imm = expr.as<IntImmNode>()) {
      *value = imm->value;
      return true;
    }
    return false;
  }

  std::vector<std::pair<Var, ScopeIdTarget>> ScopeIdTargets() const {
    std::vector<std::pair<Var, ScopeIdTarget>> out;
    for (auto it = scope_id_defs_at_level_.rbegin(); it != scope_id_defs_at_level_.rend(); ++it) {
      for (const auto& def : *it) {
        for (size_t i = 0; i < def->def_ids.size(); ++i) {
          out.push_back({def->def_ids[i], ScopeIdTarget{def->scope, static_cast<int>(i),
                                                        static_cast<int>(def->def_ids.size())}});
        }
      }
    }
    return out;
  }

  std::vector<Var> ScopeIdVars() const {
    std::vector<Var> vars;
    for (const auto& [var, _] : ScopeIdTargets()) {
      vars.push_back(var);
    }
    return vars;
  }

  bool ContainsScopeIdVar(const PrimExpr& pred) const {
    return ScopeIdVarFinder::Contains(pred, ScopeIdVars());
  }

  bool TryExtractLinearScopeDiff(const PrimExpr& diff, ScopeIdTarget* target, int64_t* coeff,
                                 int64_t* base) {
    PrimExpr simplified = analyzer_->Simplify(diff);
    for (const auto& [var, candidate] : ScopeIdTargets()) {
      ffi::Array<PrimExpr> linear = arith::DetectLinearEquation(simplified, {var});
      if (linear.size() != 2) continue;
      int64_t c = 0;
      int64_t b = 0;
      if (!TryExtractIntImm(analyzer_->Simplify(linear[0]), &c) ||
          !TryExtractIntImm(analyzer_->Simplify(linear[1]), &b)) {
        continue;
      }
      if (c != 1 && c != -1) continue;
      *target = candidate;
      *coeff = c;
      *base = b;
      return true;
    }
    return false;
  }

  bool TryExtractLinearCompareRange(const PrimExpr& lhs, const PrimExpr& rhs, bool inclusive,
                                    bool lhs_less_rhs, ScopeIdRange* range) {
    ScopeIdTarget target;
    int64_t coeff = 0;
    int64_t base = 0;
    if (!TryExtractLinearScopeDiff(lhs - rhs, &target, &coeff, &base)) return false;

    // Interpret `coeff * v + base <op> 0` where coeff is +/- 1.
    int64_t lo = arith::ConstIntBound::kNegInf;
    int64_t hi = arith::ConstIntBound::kPosInf;
    if (lhs_less_rhs) {
      if (coeff == 1) {
        // v + base < 0  -> v < -base
        // v + base <= 0 -> v <= -base
        hi = inclusive ? -base + 1 : -base;
      } else {
        // -v + base < 0  -> v > base
        // -v + base <= 0 -> v >= base
        lo = inclusive ? base : base + 1;
      }
    } else {
      if (coeff == 1) {
        // v + base > 0  -> v > -base
        // v + base >= 0 -> v >= -base
        lo = inclusive ? -base : -base + 1;
      } else {
        // -v + base > 0  -> v < base
        // -v + base >= 0 -> v <= base
        hi = inclusive ? base + 1 : base;
      }
    }
    *range = ScopeIdRange{target, lo, hi};
    return true;
  }

  bool TryPushLinearCompare(const PrimExpr& lhs, const PrimExpr& rhs, bool inclusive,
                            bool lhs_less_rhs) {
    ScopeIdRange range;
    if (!TryExtractLinearCompareRange(lhs, rhs, inclusive, lhs_less_rhs, &range)) return false;
    return TryPushRangeForTarget(range.target, range.lo, range.hi);
  }

  bool TryExtractLinearEqualityRange(const PrimExpr& lhs, const PrimExpr& rhs,
                                     ScopeIdRange* range) {
    ScopeIdTarget target;
    int64_t coeff = 0;
    int64_t base = 0;
    if (!TryExtractLinearScopeDiff(lhs - rhs, &target, &coeff, &base)) return false;
    int64_t value = (coeff == 1) ? -base : base;
    *range = ScopeIdRange{target, value, value + 1};
    return true;
  }

  bool TryPushLinearEquality(const PrimExpr& lhs, const PrimExpr& rhs) {
    ScopeIdRange range;
    if (!TryExtractLinearEqualityRange(lhs, rhs, &range)) return false;
    return TryPushRangeForTarget(range.target, range.lo, range.hi);
  }

  bool TryExtractModuloTarget(const PrimExpr& expr, ScopeIdTarget* target, int64_t* modulus) {
    PrimExpr lhs;
    PrimExpr rhs;
    if (const auto* mod = expr.as<ModNode>()) {
      lhs = mod->a;
      rhs = mod->b;
    } else if (const auto* floormod = expr.as<FloorModNode>()) {
      lhs = floormod->a;
      rhs = floormod->b;
    } else {
      return false;
    }
    auto maybe_target = ResolveScopeIdTarget(lhs);
    if (!maybe_target) return false;
    int64_t mod_value = 0;
    if (!TryExtractIntImm(analyzer_->Simplify(rhs), &mod_value) || mod_value <= 0) return false;
    *target = *maybe_target;
    *modulus = mod_value;
    return true;
  }

  bool TryPushModuloEquality(const PrimExpr& lhs, const PrimExpr& rhs) {
    ScopeIdTarget target;
    int64_t modulus = 0;
    int64_t residue = 0;
    if (TryExtractModuloTarget(lhs, &target, &modulus) &&
        TryExtractIntImm(analyzer_->Simplify(rhs), &residue)) {
      return TryPushModuloForTarget(target, modulus, residue);
    }
    if (TryExtractModuloTarget(rhs, &target, &modulus) &&
        TryExtractIntImm(analyzer_->Simplify(lhs), &residue)) {
      return TryPushModuloForTarget(target, modulus, residue);
    }
    return false;
  }

  bool TryPushComparisonPredicate(const PrimExpr& pred) {
    if (const auto* eq = pred.as<EQNode>()) {
      return TryPushLinearEquality(eq->a, eq->b) || TryPushModuloEquality(eq->a, eq->b);
    }
    if (const auto* lt = pred.as<LTNode>()) {
      return TryPushLinearCompare(lt->a, lt->b, /*inclusive=*/false, /*lhs_less_rhs=*/true);
    }
    if (const auto* le = pred.as<LENode>()) {
      return TryPushLinearCompare(le->a, le->b, /*inclusive=*/true, /*lhs_less_rhs=*/true);
    }
    if (const auto* gt = pred.as<GTNode>()) {
      return TryPushLinearCompare(gt->a, gt->b, /*inclusive=*/false, /*lhs_less_rhs=*/false);
    }
    if (const auto* ge = pred.as<GENode>()) {
      return TryPushLinearCompare(ge->a, ge->b, /*inclusive=*/true, /*lhs_less_rhs=*/false);
    }
    return false;
  }

  bool TryExtractComparisonRange(const PrimExpr& pred, ScopeIdRange* range) {
    if (const auto* eq = pred.as<EQNode>()) {
      return TryExtractLinearEqualityRange(eq->a, eq->b, range);
    }
    if (const auto* lt = pred.as<LTNode>()) {
      return TryExtractLinearCompareRange(lt->a, lt->b, /*inclusive=*/false,
                                          /*lhs_less_rhs=*/true, range);
    }
    if (const auto* le = pred.as<LENode>()) {
      return TryExtractLinearCompareRange(le->a, le->b, /*inclusive=*/true,
                                          /*lhs_less_rhs=*/true, range);
    }
    if (const auto* gt = pred.as<GTNode>()) {
      return TryExtractLinearCompareRange(gt->a, gt->b, /*inclusive=*/false,
                                          /*lhs_less_rhs=*/false, range);
    }
    if (const auto* ge = pred.as<GENode>()) {
      return TryExtractLinearCompareRange(ge->a, ge->b, /*inclusive=*/true,
                                          /*lhs_less_rhs=*/false, range);
    }
    return false;
  }

  static bool IsBitwiseAndCall(const CallNode* call) {
    return call->op.same_as(tirx::builtin::bitwise_and()) && call->args.size() == 2;
  }

  void FlattenConjuncts(const PrimExpr& pred, std::vector<PrimExpr>* out) const {
    if (const auto* and_node = pred.as<AndNode>()) {
      FlattenConjuncts(and_node->a, out);
      FlattenConjuncts(and_node->b, out);
      return;
    }
    if (const auto* call = pred.as<CallNode>()) {
      if (IsBitwiseAndCall(call)) {
        FlattenConjuncts(call->args[0], out);
        FlattenConjuncts(call->args[1], out);
        return;
      }
    }
    out->push_back(pred);
  }

  int PushFilterPredicateCtx(const CallNode* call) {
    TVM_FFI_ICHECK_EQ(call->args.size(), 2)
        << "TIRxError: tirx.filter expects (var, cond); got " << call->args.size() << " args";
    auto target = ResolveScopeIdTarget(call->args[0]);
    if (target && ElectSyncFinder::Contains(call->args[1])) {
      PrimExpr selector = tirx::Call(call->args[0].dtype(), tirx::builtin::selector(),
                                     {call->args[0], call->args[1]});
      int pushed = TryPushSelectorForTarget(*target, selector) ? 1 : 0;
      return pushed + PushPredicateCtx(call->args[1]);
    }
    return PushPredicateCtx(call->args[1]);
  }

  int PushConjunctivePredicateCtx(const PrimExpr& pred) {
    std::vector<PrimExpr> terms;
    FlattenConjuncts(pred, &terms);
    std::vector<bool> consumed(terms.size(), false);
    std::vector<PendingRangeGroup> groups;
    std::vector<int> term_to_group(terms.size(), -1);

    for (size_t i = 0; i < terms.size(); ++i) {
      ScopeIdRange range;
      if (!TryExtractComparisonRange(terms[i], &range)) continue;
      bool found = false;
      for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
        PendingRangeGroup& group = groups[group_index];
        if (!SameScopeIdTarget(group.target, range.target)) continue;
        group.lo = std::max(group.lo, range.lo);
        group.hi = std::min(group.hi, range.hi);
        group.indices.push_back(i);
        term_to_group[i] = static_cast<int>(group_index);
        found = true;
        break;
      }
      if (!found) {
        groups.push_back(PendingRangeGroup{range.target, range.lo, range.hi, {i}});
        term_to_group[i] = static_cast<int>(groups.size() - 1);
      }
    }

    int pushed = 0;
    bool progress = true;
    while (progress) {
      progress = false;
      for (size_t i = 0; i < terms.size(); ++i) {
        if (consumed[i]) continue;
        int group_index = term_to_group[i];
        if (group_index >= 0) {
          const PendingRangeGroup& group = groups[group_index];
          if (group.indices.size() > 1 && group.indices.front() != i) continue;
          if (group.lo >= group.hi) continue;
          if (TryPushRangeForTarget(group.target, group.lo, group.hi)) {
            for (size_t index : group.indices) {
              consumed[index] = true;
            }
            ++pushed;
            progress = true;
          }
          continue;
        }
        if (TryPushComparisonPredicate(terms[i])) {
          consumed[i] = true;
          ++pushed;
          progress = true;
        }
      }
    }

    for (size_t i = 0; i < terms.size(); ++i) {
      if (consumed[i]) continue;
      int group_index = term_to_group[i];
      if (group_index >= 0) {
        consumed[i] = true;
        continue;
      }
      pushed += PushPredicateCtx(terms[i]);
    }
    return pushed;
  }

  // Try to classify `cond` as a canonical thread-filter predicate
  // (see filter_canonical.h) and narrow the ExecContext on each atom.
  //
  // Range atoms that share a ScopeIdTarget are intersected into a single
  // merged range before being pushed (this mirrors PushConjunctivePredicateCtx
  // and matters for multi-axis targets like kCtaThread, where pushing the two
  // half-bounded ranges of e.g. `0 <= tid AND tid < 128` separately would
  // overflow inside NarrowFlatProductRange).
  //
  // If the predicate is not canonical but contains a `ptx_elect_sync()` call,
  // it is treated as a lane-scope thread filter with the whole predicate
  // preserved verbatim as the selector argument -- mirroring the legacy
  // PushFilterPredicateCtx behavior for forms like `elect_sync() != 0` or
  // `not elect_sync()`.
  //
  // Returns:
  //   -1   `cond` is not canonical and does not contain elect_sync -- caller
  //        should fall back to the legacy PushPredicateCtx dispatch (which
  //        handles tirx.filter wrappers, linear shifts, modulo equality).
  //   >= 0 number of context frames pushed on `ctx_stack_` (may be 0 if all
  //        atoms were recognized but none could be narrowed -- e.g. a range
  //        target that overlaps a fixed CTA pair axis).
  int TryPushCanonicalCtx(const PrimExpr& cond) {
    if (ctx_stack_.empty()) return -1;
    ScopeIdPredicate is_scope_id = [this](const Var& v) {
      return ResolveScopeIdTarget(v).has_value();
    };
    auto canonical = TryClassifyCanonical(cond, is_scope_id);
    if (!canonical) {
      // Non-canonical fallback: any predicate containing elect_sync is
      // still a lane-scope thread filter. Push the predicate as an opaque
      // selector so downstream code-gen can reuse the existing selector
      // narrowing logic.
      if (ElectSyncFinder::Contains(cond)) {
        auto lane = FindLaneScopeVar();
        if (!lane) return -1;
        ScopeIdTarget target{ScopeBinding::kWarpThread, 0, 1};
        PrimExpr selector = tirx::Call(lane->dtype(), tirx::builtin::selector(), {*lane, cond});
        return TryPushSelectorForTarget(target, selector) ? 1 : 0;
      }
      return -1;
    }

    struct RangeGroup {
      ScopeIdTarget target;
      int64_t lo;
      int64_t hi;
    };
    std::vector<RangeGroup> groups;
    std::vector<const FilterAtom*> elect_atoms;
    for (const FilterAtom& atom : canonical->atoms) {
      if (atom.kind == FilterAtomKind::kElectSync) {
        elect_atoms.push_back(&atom);
        continue;
      }
      auto target = ResolveScopeIdTarget(atom.scopeid_var);
      if (!target) continue;  // atom recognized but target not in scope
      bool merged = false;
      for (auto& g : groups) {
        if (!SameScopeIdTarget(g.target, *target)) continue;
        g.lo = std::max(g.lo, atom.lo);
        g.hi = std::min(g.hi, atom.hi);
        merged = true;
        break;
      }
      if (!merged) groups.push_back({*target, atom.lo, atom.hi});
    }

    // Iterative push with progress: some pushes depend on a prior push
    // (e.g. a flat warpgroup-thread range can only narrow once wgid has
    // collapsed to a single warpgroup via an equality push). Mirrors the
    // progress loop in PushConjunctivePredicateCtx.
    std::vector<bool> consumed(groups.size(), false);
    int pushed = 0;
    bool progress = true;
    while (progress) {
      progress = false;
      for (size_t i = 0; i < groups.size(); ++i) {
        if (consumed[i]) continue;
        const auto& g = groups[i];
        if (g.lo >= g.hi) {
          consumed[i] = true;  // unsatisfiable; skip
          continue;
        }
        if (TryPushRangeForTarget(g.target, g.lo, g.hi)) {
          consumed[i] = true;
          ++pushed;
          progress = true;
        }
      }
    }
    for (const FilterAtom* atom : elect_atoms) {
      if (PushElectSyncAtom(*atom)) ++pushed;
    }
    return pushed;
  }

  bool PushElectSyncAtom(const FilterAtom& atom) {
    // Bind to lane-in-warp scope. The selector wraps the call with the lane
    // Var so downstream code generation can reuse the selector(var, pred)
    // shape produced by PushFilterPredicateCtx.
    auto lane = FindLaneScopeVar();
    if (!lane) return false;
    ScopeIdTarget target{ScopeBinding::kWarpThread, 0, 1};
    PrimExpr selector =
        tirx::Call(lane->dtype(), tirx::builtin::selector(), {*lane, atom.elect_sync_call});
    return TryPushSelectorForTarget(target, selector);
  }

  std::optional<Var> FindLaneScopeVar() const {
    // Walk innermost-first; the first single-axis kWarpThread def wins.
    for (auto it = scope_id_defs_at_level_.rbegin(); it != scope_id_defs_at_level_.rend(); ++it) {
      for (const auto& def : *it) {
        if (def->scope != ScopeBinding::kWarpThread) continue;
        if (def->def_ids.size() != 1) continue;
        return def->def_ids[0];
      }
    }
    return std::nullopt;
  }

  int PushPredicateCtx(const PrimExpr& pred) {
    if (ctx_stack_.empty()) return 0;
    if (const auto* and_node = pred.as<AndNode>()) {
      (void)and_node;
      return PushConjunctivePredicateCtx(pred);
    }
    if (const auto* call = pred.as<CallNode>()) {
      if (call->op.same_as(tirx::builtin::filter())) {
        return PushFilterPredicateCtx(call);
      }
      if (IsBitwiseAndCall(call)) {
        return PushConjunctivePredicateCtx(pred);
      }
    }
    if (TryPushComparisonPredicate(pred)) return 1;
    return 0;
  }

  PrimExpr RewriteFilterCall(const CallNode* call) const {
    TVM_FFI_ICHECK_EQ(call->args.size(), 2)
        << "TIRxError: tirx.filter expects (var, cond); got " << call->args.size() << " args";
    return AsBool(call->args[1]);
  }

  PrimExpr RewriteFilterCalls(const PrimExpr& pred) const {
    if (const auto* and_node = pred.as<AndNode>()) {
      PrimExpr a = RewriteFilterCalls(and_node->a);
      PrimExpr b = RewriteFilterCalls(and_node->b);
      if (a.same_as(and_node->a) && b.same_as(and_node->b)) {
        return pred;
      }
      return PrimExpr(a && b);
    }
    if (const auto* call = pred.as<CallNode>()) {
      if (call->op.same_as(tirx::builtin::filter())) {
        return RewriteFilterCalls(RewriteFilterCall(call));
      }
      bool changed = false;
      ffi::Array<PrimExpr> args;
      args.reserve(call->args.size());
      for (const auto& arg : call->args) {
        PrimExpr new_arg = RewriteFilterCalls(arg);
        changed = changed || !new_arg.same_as(arg);
        args.push_back(new_arg);
      }
      if (changed) {
        return tirx::Call(call->dtype, call->op, args, call->attrs, call->span);
      }
    }
    return pred;
  }

  PrimExpr AsBool(PrimExpr pred) const {
    if (pred.dtype().is_bool()) {
      return pred;
    }
    return pred != make_zero(pred.dtype());
  }

  ffi::Map<Var, Range> var_range_map_;
  arith::Analyzer analyzer_;
  const Target& target_;
  // List of ScopeIdDefs visible at each nesting level (one entry for the
  // device-entry body itself, plus one per ScopeIdDefStmt-bearing region).
  // Grows as ScopeIdDefStmt nodes are visited.
  std::vector<std::vector<ScopeIdDef>> scope_id_defs_at_level_;
  std::vector<ExecContext> ctx_stack_;
  std::unordered_map<ffi::String, IterVar> launch_params_;
  std::vector<Buffer> alloc_buffers_;
  std::vector<Stmt> device_init_stmts_;
  std::vector<Stmt> host_init_stmts_;
  std::unordered_map<Buffer, std::vector<Stmt>, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      post_buffer_def_stmts_;
  ffi::Map<ffi::String, ffi::ObjectRef> shared_state_;
  std::vector<std::pair<std::string, int64_t>> cluster_cta_axis_extents_;

  bool is_first_block_{true};
  bool is_first_thread_attr_{true};

  bool AppendPostBufferDefStmts(std::vector<Stmt>* seq, const Buffer& old_buffer,
                                const Buffer& new_buffer) {
    auto append_with_remap = [this, seq, &new_buffer](auto it) -> bool {
      Buffer src = it->first;
      for (const auto& stmt : it->second) {
        Stmt remapped = BufferRefRewriter::Rewrite(stmt, src, new_buffer);
        seq->push_back(KernelReplacePointSearcher::Seek(remapped, Evaluate(0)));
      }
      post_buffer_def_stmts_.erase(it);
      return true;
    };

    bool changed = false;
    if (auto it = post_buffer_def_stmts_.find(old_buffer); it != post_buffer_def_stmts_.end()) {
      changed |= append_with_remap(it);
    }
    if (!new_buffer.same_as(old_buffer)) {
      if (auto it = post_buffer_def_stmts_.find(new_buffer); it != post_buffer_def_stmts_.end()) {
        changed |= append_with_remap(it);
      }
    }
    return changed;
  }

  // No failure aggregation; pass surfaces per-op exceptions
};

namespace {
Target ResolveTarget(const PrimFunc& f) {
  auto target = f->GetAttr<Target>(tvm::attr::kTarget);
  if (!target.defined()) {
    target = Target::Current(false);
  }
  return target.value();
}
}  // namespace

namespace transform {

Pass TilePrimitiveDispatch() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    Target target = ResolveTarget(f);
    auto* n = f.CopyOnWrite();
    n->body = TilePrimitiveDispatcher::LowerOpCalls(n->body, target);
    if (!NoOpCallVerifier::Verify(n->body, false)) {
      LOG(FATAL) << "Failed to lower the TIRx program: " << f;
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.TilePrimitiveDispatch", {});
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
