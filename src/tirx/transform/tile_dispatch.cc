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
 * \file tile_dispatch.cc
 * \brief Lower tensor instructions and CUDA index calls using independent launch configuration.
 */

#include <tvm/ir/function.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/sym/pattern.h>
#include <tvm/target/target.h>
#include <tvm/tirx/exec_context.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/annotation.h>
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tile_dispatch.h>
#include <tvm/tirx/transform.h>

#include <limits>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../analysis/filter_canonical.h"
#include "../ir/tir_visitor_with_path.h"

namespace tvm {
namespace tirx {

namespace {

class ElectSyncFinder : public StmtExprVisitor {
 public:
  static bool Contains(const PrimExpr& expr) {
    auto finder = ffi::make_object<ElectSyncFinder>();
    finder->Visit(expr);
    return finder->found_;
  }

 private:
  using StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit_(const CallNode* op) final {
    auto is_canonical_elect_sync = [&]() {
      static const Op ptx_elect_sync_op = Op::Get("tirx.cuda.elect_sync");
      return op->op.same_as(ptx_elect_sync_op);
    };
    if (is_canonical_elect_sync()) {
      found_ = true;
      return std::nullopt;
    }
    return StmtExprVisitor::Visit_(op);
  }

  bool found_{false};
};

class ScopeIdVarFinder : public StmtExprVisitor {
 public:
  static bool Contains(const PrimExpr& expr, const std::vector<PrimVar>& vars) {
    auto finder = ffi::make_object<ScopeIdVarFinder>(vars);
    finder->Visit(expr);
    return finder->found_;
  }

  explicit ScopeIdVarFinder(const std::vector<PrimVar>& vars) : vars_(vars) {}

 private:
  using StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit_(const VarNode* op) final {
    for (const PrimVar& candidate : vars_) {
      if (candidate.get() == op) {
        found_ = true;
        return std::nullopt;
      }
    }
    return std::nullopt;
  }

  const std::vector<PrimVar>& vars_;
  bool found_{false};
};

using CudaLaunchParams = std::unordered_map<ffi::String, ffi::Tuple<PrimVar, PrimExpr>>;

// CUDA indices are ordinary pure calls. Lower them after instruction dispatch so
// implementations can introduce indices without changing the launch configuration.
class CudaIndexLowerer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit CudaIndexLowerer(CudaLaunchParams params) : params_(std::move(params)) {}

  static PrimExpr Register(const std::string& name) {
    std::string operation;
    if (name.rfind("ctaid.", 0) == 0) operation = "block_idx";
    if (name.rfind("tid.", 0) == 0) operation = "thread_idx";
    if (name.rfind("ntid.", 0) == 0) operation = "block_dim";
    if (name.rfind("nctaid.", 0) == 0) operation = "grid_dim";
    if (!operation.empty()) {
      return Call(PrimType::Int(32), Op::Get("tirx.cuda." + operation),
                  {StringImm(name.substr(name.size() - 1))})
          .as_or_throw<PrimExpr>();
    }
    return Call(PrimType::Int(32), Op::Get("tirx.cuda.mov_sreg"),
                {IntImm::Int32(32), StringImm(name)})
        .as_or_throw<PrimExpr>();
  }

  static Stmt Lower(const Stmt& body, const CudaLaunchParams& params) {
    auto lowerer = ffi::make_object<CudaIndexLowerer>(params);
    for (const auto& [tag, iv] : params) {
      std::string name(tag);
      std::string reg;
      if (name.rfind("blockIdx.", 0) == 0) reg = "ctaid." + name.substr(9);
      if (name.rfind("threadIdx.", 0) == 0) reg = "tid." + name.substr(10);
      if (name.rfind("clusterCtaIdx.", 0) == 0) reg = "cluster_ctaid." + name.substr(14);
      if (!reg.empty()) lowerer->VarRemapSet(iv.get<0>(), Register(reg));
    }
    auto result = lowerer->Mutate(body).ValueOrUnchanged(body);
    ffi::Array<Stmt> prefix;
    if (lowerer->needs_warp_) {
      int64_t threads = 1;
      for (int axis = 0; axis < 3; ++axis) {
        auto extent = lowerer->BlockDimension(axis).as<IntImmNode>();
        TVM_FFI_CHECK(extent, ValueError) << "Warp indices require a static block size";
        auto dimension = extent->value.as<int64_t>();
        TVM_FFI_CHECK(dimension && *dimension > 0 && *dimension <= 1024 / threads, ValueError)
            << "Block size must be positive and contain at most 1024 threads";
        threads *= *dimension;
      }
      TVM_FFI_CHECK(threads % 32 == 0, ValueError)
          << "Warp indices require a block size divisible by 32";
      prefix.push_back(Bind(
          lowerer->warp_,
          Call(PrimType::Int(32), tirx::gpu_warp_shuffle_op(),
               {IntImm(PrimType::UInt(32), 0xffffffff), prim::FloorDiv(lowerer->LinearThread(), 32),
                IntImm::Int32(0), IntImm::Int32(32), IntImm::Int32(32)})));
    }
    prefix.push_back(result);
    return SeqStmt(prefix);
  }

 private:
  PrimExpr Axis(const std::string& tag, int axis) {
    auto it = params_.find(tag + std::string(1, 'x' + axis));
    if (it == params_.end()) return IntImm::Int32(0);
    // An explicit singleton cluster still needs a hardware index when preferred
    // clusters are enabled, so do not constant-fold cluster coordinates.
    if (tag != "clusterCtaIdx." && prim::IsOne(it->second.get<1>())) return IntImm::Int32(0);
    std::string reg = tag == "blockIdx."    ? "ctaid."
                      : tag == "threadIdx." ? "tid."
                                            : "cluster_ctaid.";
    return Register(reg + std::string(1, 'x' + axis));
  }
  PrimExpr BlockDimension(int axis) {
    auto it = params_.find("threadIdx." + std::string(1, 'x' + axis));
    if (it != params_.end() && it->second.get<1>().as<IntImmNode>()) return it->second.get<1>();
    return Register("ntid." + std::string(1, 'x' + axis));
  }
  PrimExpr LinearThread() {
    sym::Analyzer analyzer;
    return analyzer->Simplify(
        Axis("threadIdx.", 0) +
        BlockDimension(0) * (Axis("threadIdx.", 1) + BlockDimension(1) * Axis("threadIdx.", 2)));
  }
  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) final {
    const auto* name_op = op->op.as<OpNode>();
    if (!name_op) return StmtExprMutator::Mutate_(op, inplace_mode);
    std::string name(name_op->name);
    if (name.rfind("tirx.cuda.", 0) != 0) return StmtExprMutator::Mutate_(op, inplace_mode);
    name = name.substr(10);
    auto axis = [&]() { return op->args[0].as_or_throw<StringImm>()->value.c_str()[0] - 'x'; };
    if (name == "block_idx") return Axis("blockIdx.", axis());
    if (name == "thread_idx") return Axis("threadIdx.", axis());
    if (name == "cluster_cta_id") {
      TVM_FFI_CHECK(params_.count("clusterCtaIdx.x"), ValueError)
          << "cluster_cta_id requires an explicit launch cluster";
      return Axis("clusterCtaIdx.", axis());
    }
    std::string reg;
    if (name == "cluster_id") reg = "clusterid.";
    if (name == "grid_dim") reg = "nctaid.";
    if (name == "block_dim") reg = "ntid.";
    if (name == "cluster_dim") reg = "cluster_nctaid.";
    if (!reg.empty()) {
      if (name == "cluster_id" || name == "cluster_dim") {
        TVM_FFI_CHECK(params_.count("clusterCtaIdx.x"), ValueError)
            << name << " requires an explicit launch cluster";
      }
      return Register(reg + std::string(1, 'x' + axis()));
    }
    if (name == "linear_thread_id") return LinearThread();
    if (name == "lane_id") return prim::FloorMod(LinearThread(), 32);
    if (name == "thread_in_warpgroup") return prim::FloorMod(LinearThread(), 128);
    if (name == "warp_id" || name == "warpgroup_id" || name == "warp_in_warpgroup") {
      needs_warp_ = true;
      if (name == "warp_id") return warp_;
      if (name == "warpgroup_id") return prim::FloorDiv(warp_, 4);
      return prim::FloorMod(warp_, 4);
    }
    if (name == "cta_pair_id") {
      TVM_FFI_CHECK(params_.count("clusterCtaIdx.x"), ValueError)
          << "cta_pair_id requires an explicit launch cluster";
      return prim::FloorMod(Axis("clusterCtaIdx.", 0) +
                                Register("cluster_nctaid.x") *
                                    (Axis("clusterCtaIdx.", 1) +
                                     Register("cluster_nctaid.y") * Axis("clusterCtaIdx.", 2)),
                            2);
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }
  CudaLaunchParams params_;
  PrimVar warp_{"warp_id_in_cta", PrimType::Int(32)};
  bool needs_warp_{false};
};

}  // namespace

class NoOpCallVerifier : public Verifier<NoOpCallVerifier> {
 public:
  using Verifier::Verifier;

 private:
  using Verifier::Visit;

  void Dispatch_(const CallNode* call, ffi::reflection::AccessPath path) final {
    if (auto op = call->op.as<Op>()) {
      static const auto& categories =
          Op::GetAttrMap<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory);
      auto category = categories.get(op.value(), ffi::String(""));
      Verify(category != "tile_primitive" && category != "tile_composite")
          << "Unlowered tensor instruction " << op.value()->name << " at " << path;
    }
    Verifier::Dispatch_(call, path);
  }
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
  class BufferRefRewriter : public StmtExprMutator {
   public:
    using StmtExprMutator::Mutate;
    using StmtExprMutator::Mutate_;
    static Stmt Rewrite(const Stmt& stmt, const TensorVar& src, const TensorVar& dst) {
      if (src.same_as(dst)) {
        return stmt;
      }
      return ffi::make_object<BufferRefRewriter>(src, dst)
          ->Mutate(stmt, InplaceMode::kAllow)
          .ValueOrUnchanged(stmt);
    }
    BufferRefRewriter(const TensorVar& src, const TensorVar& dst) { VarRemapSet(src, dst); }
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

    bool is_first_block = false;
    std::swap(is_first_block, is_first_block_);

    launch_params_.clear();
    scope_aliases_.clear();
    cluster_bounds_.clear();
    variable_cluster_ = false;
    native_launch_ = entry_node->attrs->dict.count("cuda.launch_fields");
    TVM_FFI_CHECK(is_first_block || !native_launch_, ValueError)
        << "Nested CUDA device_entry regions are not supported";
    if (native_launch_) PrepareCudaLaunchParams(entry_node);
    bool pushed_base_ctx = PushKernelEntryCtx();

    auto body_result = Mutate(body_to_visit, InplaceMode::kDisallow);
    bool body_unchanged = body_result.UnchangedOrSameAs(body_to_visit);
    Stmt body = std::move(body_result).ValueOrUnchanged(body_to_visit);

    auto pop_exec_contexts = [&]() {
      if (pushed_base_ctx) ctx_stack_.pop_back();
    };

    if (!is_first_block) {
      std::swap(is_first_block, is_first_block_);
      pop_exec_contexts();
      if (body_unchanged) {
        return ffi::GetRef<Stmt>(entry_node);
      }
      return RegionStmt(entry_node->op, entry_node->args, entry_node->body_params,
                        entry_node->attrs, body, entry_node->result_vars, entry_node->span);
    }

    // Insert device init stmts into kernel body.
    for (auto it = device_init_stmts_.rbegin(); it != device_init_stmts_.rend(); ++it) {
      body = KernelReplacePointSearcher::Seek(*it, body);
    }
    // Insert alloc buffers at the beginning of the kernel body.
    if (!alloc_buffers_.empty()) {
      std::vector<Stmt> seq;
      seq.reserve(alloc_buffers_.size() + 1);
      for (const auto& allocation : alloc_buffers_) {
        seq.push_back(allocation);
      }
      seq.push_back(std::move(body));
      body = SeqStmt(seq);
    }
    alloc_buffers_.clear();

    Stmt res = body;
    if (native_launch_) {
      res = CudaIndexLowerer::Lower(res, launch_params_);
      res = RegionStmt(Op::Get("tirx.device_scope"), entry_node->args, {}, entry_node->attrs, res);
    }

    // Insert host init stmts outside the outermost thread binding or block.
    if (is_first_thread_attr_) {
      for (const auto& stmt : host_init_stmts_) {
        // These statements leave the kernel region for host scope, where a
        // ``tensor_data_ptr`` projection of a device-local view cannot be
        // resolved.  Rewrite each projection onto its storage root, which is
        // a Function parameter and therefore visible on the host.
        res = KernelReplacePointSearcher::Seek(StorageRootResolver::Apply(stmt, buffer_root_),
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
    if (post_buffer_def_stmts_.empty()) {
      return stmt;
    }
    const auto& seq = stmt.as_or_throw<SeqStmt>();

    std::vector<Stmt> rebuilt;
    rebuilt.reserve(seq->seq.size() + post_buffer_def_stmts_.size());
    bool changed = false;
    for (const Stmt& s : seq->seq) {
      rebuilt.push_back(s);
      if (const auto* bind = s.as<BindNode>()) {
        if (const auto* call = bind->value.as<CallNode>();
            call && (call->op.same_as(tirx::alloc_tensor_op()) ||
                     (call->op.same_as(tirx::decl_tensor_op()) ||
                      call->op.same_as(Op::Get("tirx.cuda.decl_tmem"))))) {
          changed |= AppendPostBufferDefStmts(&rebuilt, bind->var.as_or_throw<TensorVar>(),
                                              bind->var.as_or_throw<TensorVar>());
        }
      }
    }
    if (!changed) {
      return stmt;
    }
    return SeqStmt(rebuilt, seq->span);
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
      if (auto target = ResolveScopeIdTarget(value.value())) {
        scope_aliases_.insert_or_assign(bind->var, *target);
      }
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
   * \brief Track the storage root of a buffer variable.
   *
   * A ``DeclTensor`` whose data is ``tensor_data_ptr(src)`` is a view over
   * ``src``'s storage, so it inherits ``src``'s root; anything else owns its
   * storage.  Buffers with no definition in the body (Function parameters)
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
    buffer_root_.insert_or_assign(new_var, root);
    if (!old_var.same_as(new_var)) {
      buffer_root_.insert_or_assign(old_var, root);
    }
  }

  Var StorageRootOf(const Var& var) const {
    auto it = buffer_root_.find(var);
    return it == buffer_root_.end() ? var : it->second;
  }

  /*! \brief Rewrite ``tensor_data_ptr(view)`` onto ``tensor_data_ptr(storage root)``. */
  class StorageRootResolver : public StmtExprMutator {
   public:
    using StmtExprMutator::Mutate;
    using StmtExprMutator::Mutate_;
    static Stmt Apply(
        Stmt stmt,
        const std::unordered_map<Var, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>& buffer_root) {
      auto resolver = ffi::make_object<StorageRootResolver>(buffer_root);
      return resolver->Mutate(stmt, InplaceMode::kAllow).ValueOrUnchanged(stmt);
    }
    explicit StorageRootResolver(
        const std::unordered_map<Var, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>& buffer_root)
        : buffer_root_(buffer_root) {}

    UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) final {
      if (op->op.same_as(tirx::tensor_data_ptr_op()) && op->args.size() == 1) {
        if (auto var = op->args[0].as<Var>();
            var.has_value() && var.value()->ty.as<TensorTypeNode>()) {
          auto it = buffer_root_.find(var.value());
          if (it != buffer_root_.end() && !it->second.same_as(var.value())) {
            return it->second.as_or_throw<TensorVar>().data();
          }
        }
      }
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }

    const std::unordered_map<Var, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>& buffer_root_;
  };

  UnchangedOr<Stmt> MutateAllocTensor(const BindNode* op, InplaceMode inplace_mode) {
    TensorVar old_buffer = op->var.as_or_throw<TensorVar>();
    Stmt stmt = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    op = stmt.as<BindNode>();
    TVM_FFI_ICHECK(op);
    RegisterStorageRoot(old_buffer.var(), op->var, std::nullopt);

    std::vector<Stmt> seq{stmt};
    AppendPostBufferDefStmts(&seq, old_buffer, op->var.as_or_throw<TensorVar>());
    return SeqStmt(seq);
  }

  UnchangedOr<Stmt> MutateDeclTensor(const BindNode* op, InplaceMode inplace_mode) {
    TensorVar old_buffer = op->var.as_or_throw<TensorVar>();
    Stmt stmt = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    op = stmt.as<BindNode>();
    TVM_FFI_ICHECK(op);
    const auto* buffer_call = op->value.as<CallNode>();
    RegisterStorageRoot(old_buffer.var(), op->var, buffer_call->args[0]);

    std::vector<Stmt> seq{stmt};
    AppendPostBufferDefStmts(&seq, old_buffer, op->var.as_or_throw<TensorVar>());
    return SeqStmt(seq);
  }

  UnchangedOr<Stmt> Mutate_(const IfNode* op, InplaceMode inplace_mode) final {
    // Narrow ExecContext for structurally recognized predicates on the
    // then-branch. The canonical-form classifier (filter_canonical.h)
    // recognizes the dominant shapes: pure conjunctions of `scopeid_var op
    // const` comparisons plus bare `ptx_elect_sync()` calls. Predicates
    // outside that grammar (e.g. linear shifts like `v - 1 < 5`, modulo
    // equality like `v % 2 == 0`, or the legacy `tirx.gpu_thread_filter` wrapper) fall
    // back to the existing dispatcher, which has more permissive matching
    // paths.
    int pushed_ctx = TryPushCanonicalCtx(op->condition);
    if (pushed_ctx < 0) {
      pushed_ctx = PushPredicateCtx(op->condition);
    }
    PrimExpr new_cond = RewriteFilterCalls(op->condition);
    auto then_case_result = Mutate(op->then_case, inplace_mode);
    bool then_case_unchanged = then_case_result.UnchangedOrSameAs(op->then_case);
    Stmt then_case = std::move(then_case_result).ValueOrUnchanged(op->then_case);
    while (pushed_ctx-- > 0) ctx_stack_.pop_back();
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
    if (!op || !op->op.as<Op>()) return StmtExprMutator::Mutate_(stmt, inplace_mode);
    static const auto& categories = Op::GetAttrMap<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory);
    auto category = categories.get(op->op.as_or_throw<Op>(), ffi::String(""));
    if (category != "tile_primitive" && category != "tile_composite") {
      return StmtExprMutator::Mutate_(stmt, inplace_mode);
    }
    op->op.as_or_throw<Op>().Validate(op);
    static auto get_scope = ffi::Function::GetGlobalRequired("tirx.TensorCallScope");
    ExecScope scope = get_scope(ffi::GetRef<Call>(op)).cast<ExecScope>();
    TVM_FFI_CHECK(!variable_cluster_ || scope->kind != ScopeKind::kCluster, ValueError)
        << "Cluster tensor instructions require one static cluster shape; "
           "use explicit CUDA/PTX instructions for variable or preferred cluster shapes";
    // Scope is a static attribute of this Call. Derive the (inter, intra) split
    // on the spot from the current active set ``A`` (tracked through control
    // flow on ``ctx_stack_``) under this call's own scope.
    ffi::Map<ffi::String, ffi::Array<PrimExpr>> inter_map, intra_map;
    ffi::String scope_kind = ScopeKindToString(scope->kind);
    if (!ctx_stack_.empty()) {
      ExecSplit split;
      std::string err;
      if (ScopeSwitch(ctx_stack_.back().A, scope->kind, &split, &err)) {
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
    tirx::DispatchContext sctx(target_, scope, launch_params_, var_range_map_,
                               /*alloc_only=*/false, /*callbacks=*/{}, shared_state_, inter_map,
                               intra_map, scope_kind);
    static auto f_op_dispatcher_ = ffi::Function::GetGlobal("tirx.f_op_dispatcher");
    TVM_FFI_ICHECK(f_op_dispatcher_.has_value())
        << "Internal Error: tirx.f_op_dispatcher is not registered";
    Function res = f_op_dispatcher_.value()(ffi::GetRef<Call>(op), sctx).cast<Function>();
    TVM_FFI_ICHECK(res.defined()) << "TIRx dispatcher did not return a Function";
    // Implementation found, handle callbacks
    if (auto bufs = sctx->callbacks.Get(tirx::callback::kPrivateAlloc)) {
      auto buf_list = bufs.value().as<Array<Bind>>().value();
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
      auto map = mapping.value().as_or_throw<ffi::Map<TensorVar, Array<Stmt>>>();
      for (const auto& [buffer, stmts] : map) {
        auto& vec = post_buffer_def_stmts_[buffer];
        vec.insert(vec.end(), stmts.begin(), stmts.end());
      }
    }
    // Propagate shared_state changes back (Map uses COW semantics)
    shared_state_ = sctx->shared_state;
    TVM_FFI_CHECK(res->body.has_value(), ValueError)
        << "A tile primitive implementation must have a body";
    return Mutate(res->body.value(), inplace_mode).ValueOrUnchanged(res->body.value());
  }

  // --- Scope-id resolution at kernel scope ----------------------------------

  void PrepareCudaLaunchParams(const RegionStmtNode* entry) {
    TVM_FFI_CHECK(target_->kind->name == "cuda", ValueError)
        << "CUDA LaunchConfig requires a CUDA target";
    auto fields =
        entry->attrs->dict.at("cuda.launch_fields").as_or_throw<ffi::Array<ffi::String>>();
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

  // --- ExecContext tracking helpers -----------------------------------------

  bool PushKernelEntryCtx() {
    auto prod_extent = [&](std::initializer_list<const char*> keys) -> int64_t {
      int64_t n = 1;
      for (const char* k : keys) {
        auto it = launch_params_.find(ffi::String(k));
        if (it == launch_params_.end()) continue;
        const auto* imm = it->second.get<1>().as<IntImmNode>();
        if (imm == nullptr) return 0;  // symbolic
        auto product = (n * imm->value).as<int64_t>();
        if (!product.has_value()) return 0;
        n = *product;
      }
      return n;
    };
    auto collect_extents = [&](std::initializer_list<std::pair<const char*, const char*>> keys) {
      std::vector<std::pair<std::string, int64_t>> out;
      for (const auto& [thread_key, axis_name] : keys) {
        auto it = launch_params_.find(ffi::String(thread_key));
        if (it == launch_params_.end()) continue;
        PrimExpr extent = it->second.get<1>();
        if (auto bound = cluster_bounds_.find(thread_key); bound != cluster_bounds_.end())
          extent = bound->second;
        const auto* imm = extent.as<IntImmNode>();
        if (imm == nullptr) return std::vector<std::pair<std::string, int64_t>>();
        auto value = imm->value.as<int64_t>();
        if (!value.has_value()) return std::vector<std::pair<std::string, int64_t>>();
        out.push_back({axis_name, *value});
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
    if (!cluster_bounds_.empty() && cluster_cta_axes.empty()) return false;
    while (cluster_cta_axes.size() > 1 && cluster_cta_axes.back().second == 1)
      cluster_cta_axes.pop_back();
    cluster_cta_axis_extents_ = cluster_cta_axes;
    auto cta_axes = cluster_cta_axes;
    if (cta_axes.empty()) {
      cta_axes =
          collect_extents({{"blockIdx.x", "bx"}, {"blockIdx.y", "by"}, {"blockIdx.z", "bz"}});
      cluster_cta_axis_extents_.clear();
    }
    while (cta_axes.size() > 1 && cta_axes.back().second == 1) cta_axes.pop_back();
    int64_t cta_ext = 1;
    for (const auto& axis : cta_axes) {
      if (axis.second <= 0 || axis.second > std::numeric_limits<int64_t>::max() / cta_ext)
        return false;
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
    int64_t lo = sym::ConstIntBound::kNegInf;
    int64_t hi = sym::ConstIntBound::kPosInf;
  };

  struct PendingRangeGroup {
    ScopeIdTarget target;
    int64_t lo = sym::ConstIntBound::kNegInf;
    int64_t hi = sym::ConstIntBound::kPosInf;
    std::vector<size_t> indices;
  };

  static bool SameScopeIdTarget(const ScopeIdTarget& lhs, const ScopeIdTarget& rhs) {
    return lhs.binding == rhs.binding && lhs.dim == rhs.dim && lhs.ndim == rhs.ndim;
  }

  bool KernelCtaPredicateOverlapsClusterCta(const ScopeIdTarget& target) const {
    return target.binding == ScopeBinding::kKernelCta && !cluster_cta_axis_extents_.empty();
  }

  std::optional<ScopeIdTarget> ResolveScopeIdTarget(const PrimExpr& expr) const {
    if (auto cast = expr.as<prim::CastNode>()) return ResolveScopeIdTarget(cast->value);
    if (auto call = expr.as<CallNode>()) {
      const auto* op = call->op.as<OpNode>();
      if (!op) return std::nullopt;
      std::string name(op->name);
      ScopeBinding binding;
      std::string tag;
      if (name == "tirx.cuda.block_idx") {
        binding = ScopeBinding::kKernelCta;
        tag = "blockIdx.";
      } else if (name == "tirx.cuda.cluster_cta_id") {
        binding = ScopeBinding::kClusterCta;
        tag = "clusterCtaIdx.";
      } else if (name == "tirx.cuda.thread_idx") {
        binding = ScopeBinding::kCtaThread;
        tag = "threadIdx.";
      } else if (name == "tirx.cuda.cluster_id")
        return ScopeIdTarget{ScopeBinding::kKernelCluster, 0, 3};
      else if (name == "tirx.cuda.warp_id")
        return ScopeIdTarget{ScopeBinding::kCtaWarp};
      else if (name == "tirx.cuda.warpgroup_id")
        return ScopeIdTarget{ScopeBinding::kCtaWarpgroup};
      else if (name == "tirx.cuda.warp_in_warpgroup")
        return ScopeIdTarget{ScopeBinding::kWarpgroupWarp};
      else if (name == "tirx.cuda.lane_id")
        return ScopeIdTarget{ScopeBinding::kWarpThread};
      else if (name == "tirx.cuda.thread_in_warpgroup")
        return ScopeIdTarget{ScopeBinding::kWarpgroupThread};
      else if (name == "tirx.cuda.cta_pair_id")
        return ScopeIdTarget{ScopeBinding::kClusterCtaPair};
      else if (name == "tirx.cuda.linear_thread_id")
        return ScopeIdTarget{ScopeBinding::kCtaThread};
      else
        return std::nullopt;
      int dim = call->args[0].as_or_throw<StringImm>()->value.c_str()[0] - 'x';
      int ndim = 1;
      for (int i = 1; i < 3; ++i) {
        auto key = tag + std::string(1, 'x' + i);
        auto it = launch_params_.find(key);
        if (it != launch_params_.end()) {
          auto bound = cluster_bounds_.find(key);
          const auto& extent = bound == cluster_bounds_.end() ? it->second.get<1>() : bound->second;
          if (!prim::IsOne(extent)) ndim = i + 1;
        }
      }
      return ScopeIdTarget{binding, dim, std::max(ndim, dim + 1)};
    }
    const auto* var_node = expr.as<VarNode>();
    if (var_node == nullptr) return std::nullopt;
    Var var = ffi::GetRef<Var>(var_node);
    if (auto it = scope_aliases_.find(var); it != scope_aliases_.end()) return it->second;
    return std::nullopt;
  }

  bool TryPushRangeForTarget(const ScopeIdTarget& target, int64_t lo, int64_t hi) {
    if (ctx_stack_.empty()) return false;
    if (target.binding == ScopeBinding::kClusterCtaPair) {
      if (variable_cluster_) return false;
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
      if (auto value_i64 = imm->value.as<int64_t>(); value_i64.has_value()) {
        *value = *value_i64;
        return true;
      }
    }
    return false;
  }

  std::vector<std::pair<PrimVar, ScopeIdTarget>> ScopeIdTargets() const {
    std::vector<std::pair<PrimVar, ScopeIdTarget>> out;
    for (const auto& [var, target] : scope_aliases_)
      out.push_back({var.as_or_throw<PrimVar>(), target});
    return out;
  }

  std::vector<PrimVar> ScopeIdVars() const {
    std::vector<PrimVar> vars;
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
      ffi::Array<PrimExpr> linear = sym::DetectLinearEquation(simplified, {var});
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
    int64_t lo = sym::ConstIntBound::kNegInf;
    int64_t hi = sym::ConstIntBound::kPosInf;
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
    PrimExpr lhs{ffi::UnsafeInit{}};
    PrimExpr rhs{ffi::UnsafeInit{}};
    if (const auto* mod = expr.as<prim::ModNode>()) {
      lhs = mod->a;
      rhs = mod->b;
    } else if (const auto* floormod = expr.as<prim::FloorModNode>()) {
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
    if (const auto* eq = pred.as<prim::EQNode>()) {
      return TryPushLinearEquality(eq->a, eq->b) || TryPushModuloEquality(eq->a, eq->b);
    }
    if (const auto* lt = pred.as<prim::LTNode>()) {
      return TryPushLinearCompare(lt->a, lt->b, /*inclusive=*/false, /*lhs_less_rhs=*/true);
    }
    if (const auto* le = pred.as<prim::LENode>()) {
      return TryPushLinearCompare(le->a, le->b, /*inclusive=*/true, /*lhs_less_rhs=*/true);
    }
    if (const auto* gt = pred.as<prim::GTNode>()) {
      return TryPushLinearCompare(gt->a, gt->b, /*inclusive=*/false, /*lhs_less_rhs=*/false);
    }
    if (const auto* ge = pred.as<prim::GENode>()) {
      return TryPushLinearCompare(ge->a, ge->b, /*inclusive=*/true, /*lhs_less_rhs=*/false);
    }
    return false;
  }

  bool TryExtractComparisonRange(const PrimExpr& pred, ScopeIdRange* range) {
    if (const auto* eq = pred.as<prim::EQNode>()) {
      return TryExtractLinearEqualityRange(eq->a, eq->b, range);
    }
    if (const auto* lt = pred.as<prim::LTNode>()) {
      return TryExtractLinearCompareRange(lt->a, lt->b, /*inclusive=*/false,
                                          /*lhs_less_rhs=*/true, range);
    }
    if (const auto* le = pred.as<prim::LENode>()) {
      return TryExtractLinearCompareRange(le->a, le->b, /*inclusive=*/true,
                                          /*lhs_less_rhs=*/true, range);
    }
    if (const auto* gt = pred.as<prim::GTNode>()) {
      return TryExtractLinearCompareRange(gt->a, gt->b, /*inclusive=*/false,
                                          /*lhs_less_rhs=*/false, range);
    }
    if (const auto* ge = pred.as<prim::GENode>()) {
      return TryExtractLinearCompareRange(ge->a, ge->b, /*inclusive=*/true,
                                          /*lhs_less_rhs=*/false, range);
    }
    return false;
  }

  void FlattenConjuncts(const PrimExpr& pred, std::vector<PrimExpr>* out) const {
    if (const auto* and_node = pred.as<prim::AndNode>()) {
      FlattenConjuncts(and_node->a, out);
      FlattenConjuncts(and_node->b, out);
      return;
    }
    if (const auto* and_node = pred.as<prim::BitwiseAndNode>()) {
      FlattenConjuncts(and_node->a, out);
      FlattenConjuncts(and_node->b, out);
      return;
    }
    out->push_back(pred);
  }

  int PushFilterPredicateCtx(const CallNode* call) {
    TVM_FFI_ICHECK_EQ(call->args.size(), 2)
        << "TIRxError: tirx.gpu_thread_filter expects (var, cond); got " << call->args.size()
        << " args";
    PrimExpr var = call->args[0].as_or_throw<PrimExpr>();
    PrimExpr cond = call->args[1].as_or_throw<PrimExpr>();
    auto target = ResolveScopeIdTarget(var);
    if (target && ElectSyncFinder::Contains(cond)) {
      PrimExpr selector = Call(var.ty(), tirx::gpu_active_thread_selector_op(), {var, cond})
                              .as_or_throw<PrimExpr>();
      int pushed = TryPushSelectorForTarget(*target, selector) ? 1 : 0;
      return pushed + PushPredicateCtx(cond);
    }
    return PushPredicateCtx(cond);
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
  //        handles tirx.gpu_thread_filter wrappers, linear shifts, modulo equality).
  //   >= 0 number of context frames pushed on `ctx_stack_` (may be 0 if all
  //        atoms were recognized but none could be narrowed -- e.g. a range
  //        target that overlaps a fixed CTA pair axis).
  int TryPushCanonicalCtx(const PrimExpr& cond) {
    if (ctx_stack_.empty()) return -1;
    ScopeIdPredicate is_scope_id = [this](const Var& v) {
      return ResolveScopeIdTarget(v.as_or_throw<PrimExpr>()).has_value();
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
        PrimExpr selector =
            Call((*lane)->ty, tirx::gpu_active_thread_selector_op(), ffi::Array<Expr>{*lane, cond})
                .as_or_throw<PrimExpr>();
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
      auto target = ResolveScopeIdTarget(atom.scopeid_var.value().as_or_throw<PrimExpr>());
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
    PrimExpr selector = Call((*lane)->ty, tirx::gpu_active_thread_selector_op(),
                             ffi::Array<Expr>{*lane, atom.elect_sync_call.value()})
                            .as_or_throw<PrimExpr>();
    return TryPushSelectorForTarget(target, selector);
  }

  std::optional<PrimVar> FindLaneScopeVar() const {
    for (const auto& [var, target] : scope_aliases_) {
      if (target.binding == ScopeBinding::kWarpThread) return var.as_or_throw<PrimVar>();
    }
    return std::nullopt;
  }

  int PushPredicateCtx(const PrimExpr& pred) {
    if (ctx_stack_.empty()) return 0;
    if (pred.as<prim::AndNode>() || pred.as<prim::BitwiseAndNode>()) {
      return PushConjunctivePredicateCtx(pred);
    }
    if (const auto* call = pred.as<CallNode>()) {
      if (call->op.same_as(tirx::gpu_thread_filter_op())) {
        return PushFilterPredicateCtx(call);
      }
    }
    if (TryPushComparisonPredicate(pred)) return 1;
    return 0;
  }

  PrimExpr RewriteFilterCall(const CallNode* call) const {
    TVM_FFI_ICHECK_EQ(call->args.size(), 2)
        << "TIRxError: tirx.gpu_thread_filter expects (var, cond); got " << call->args.size()
        << " args";
    return AsBool(call->args[1].as_or_throw<PrimExpr>());
  }

  PrimExpr RewriteFilterCalls(const PrimExpr& pred) const {
    if (const auto* and_node = pred.as<prim::AndNode>()) {
      PrimExpr a = RewriteFilterCalls(and_node->a);
      PrimExpr b = RewriteFilterCalls(and_node->b);
      if (a.same_as(and_node->a) && b.same_as(and_node->b)) {
        return pred;
      }
      return PrimExpr(a && b);
    }
    if (const auto* op = pred.as<prim::LShiftNode>()) {
      PrimExpr a = RewriteFilterCalls(op->a);
      PrimExpr b = RewriteFilterCalls(op->b);
      if (a.same_as(op->a) && b.same_as(op->b)) return pred;
      return tvm::left_shift(a, b, op->span);
    }
    if (const auto* op = pred.as<prim::RShiftNode>()) {
      PrimExpr a = RewriteFilterCalls(op->a);
      PrimExpr b = RewriteFilterCalls(op->b);
      if (a.same_as(op->a) && b.same_as(op->b)) return pred;
      return tvm::right_shift(a, b, op->span);
    }
    if (const auto* op = pred.as<prim::BitwiseAndNode>()) {
      PrimExpr a = RewriteFilterCalls(op->a);
      PrimExpr b = RewriteFilterCalls(op->b);
      if (a.same_as(op->a) && b.same_as(op->b)) return pred;
      return tvm::bitwise_and(a, b, op->span);
    }
    if (const auto* op = pred.as<prim::BitwiseOrNode>()) {
      PrimExpr a = RewriteFilterCalls(op->a);
      PrimExpr b = RewriteFilterCalls(op->b);
      if (a.same_as(op->a) && b.same_as(op->b)) return pred;
      return tvm::bitwise_or(a, b, op->span);
    }
    if (const auto* op = pred.as<prim::BitwiseXorNode>()) {
      PrimExpr a = RewriteFilterCalls(op->a);
      PrimExpr b = RewriteFilterCalls(op->b);
      if (a.same_as(op->a) && b.same_as(op->b)) return pred;
      return tvm::bitwise_xor(a, b, op->span);
    }
    if (const auto* op = pred.as<prim::BitwiseNotNode>()) {
      PrimExpr a = RewriteFilterCalls(op->a);
      if (a.same_as(op->a)) return pred;
      return prim::BitwiseNot(a, op->span);
    }
    if (const auto* call = pred.as<CallNode>()) {
      if (call->op.same_as(tirx::gpu_thread_filter_op())) {
        return RewriteFilterCalls(RewriteFilterCall(call));
      }
      bool changed = false;
      ffi::Array<Expr> args;
      args.reserve(call->args.size());
      for (const Expr& arg : call->args) {
        Expr new_arg = arg;
        if (auto prim_arg = arg.as<PrimExpr>()) {
          new_arg = RewriteFilterCalls(prim_arg.value());
        }
        changed = changed || !new_arg.same_as(arg);
        args.push_back(new_arg);
      }
      if (changed) {
        return Call(call->ty, call->op, args, call->attrs, {}, call->span).as_or_throw<PrimExpr>();
      }
    }
    return pred;
  }

  PrimExpr AsBool(PrimExpr pred) const {
    PrimType pred_ty = pred.ty();
    if (pred_ty.MatchesCode(DLDataTypeCode::kDLBool)) {
      return pred;
    }
    return pred != IntImm(pred.ty(), 0);
  }

  bool native_launch_{false};
  bool variable_cluster_{false};
  std::unordered_map<std::string, PrimExpr> cluster_bounds_;
  std::unordered_map<Var, ScopeIdTarget, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> scope_aliases_;
  ffi::Map<Var, Range> var_range_map_;
  sym::Analyzer analyzer_;
  const Target& target_;
  std::vector<ExecContext> ctx_stack_;
  std::unordered_map<ffi::String, ffi::Tuple<PrimVar, PrimExpr>> launch_params_;
  std::vector<Bind> alloc_buffers_;
  std::vector<Stmt> device_init_stmts_;
  std::vector<Stmt> host_init_stmts_;
  /*! \brief Storage root of each buffer variable defined in the body. */
  std::unordered_map<Var, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> buffer_root_;
  std::unordered_map<TensorVar, std::vector<Stmt>, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      post_buffer_def_stmts_;
  ffi::Map<ffi::String, ffi::ObjectRef> shared_state_;
  std::vector<std::pair<std::string, int64_t>> cluster_cta_axis_extents_;

  bool is_first_block_{true};
  bool is_first_thread_attr_{true};

  bool AppendPostBufferDefStmts(std::vector<Stmt>* seq, const TensorVar& old_buffer,
                                const TensorVar& new_buffer) {
    auto append_with_remap = [this, seq, &new_buffer](auto it) -> bool {
      TensorVar src = it->first;
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
Target ResolveTarget(const Function& f) {
  auto target = f->GetAttr<Target>(tvm::attr::kTarget);
  if (!target.has_value()) {
    target = Target::Current(false);
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
    if (!NoOpCallVerifier::Verify(n->body.value(), false)) {
      LOG(FATAL) << "Failed to lower the TIRx program: " << f;
    }
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "tirx.TileDispatch");
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
