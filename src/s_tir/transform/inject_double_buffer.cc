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
 * \brief Inject double buffering optimization for data fetch.
 * \file inject_double_buffer.cc
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include "../../tirx/transform/ir_utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

namespace {

ffi::Optional<Var> GetBufferDataVar(const ffi::Any& data) {
  if (auto var = data.as<Var>()) {
    return var;
  }
  if (const auto* call = data.as<CallNode>();
      call && call->op.same_as(tirx::builtin::buffer_data()) && call->args.size() == 1) {
    return call->args[0].as<Var>();
  }
  return std::nullopt;
}

}  // namespace

struct InjectDoubleBufferConfigNode : public ffi::Object {
  int split_loop;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<InjectDoubleBufferConfigNode>().def_ro(
        "split_loop", &InjectDoubleBufferConfigNode::split_loop, "Split loop factors",
        refl::DefaultValue(1));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.transform.InjectDoubleBufferConfig",
                                    InjectDoubleBufferConfigNode, ffi::Object);
};

class InjectDoubleBufferConfig : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(InjectDoubleBufferConfig, ffi::ObjectRef,
                                                InjectDoubleBufferConfigNode);
};

TVM_FFI_STATIC_INIT_BLOCK() { InjectDoubleBufferConfigNode::RegisterReflection(); }

TVM_REGISTER_PASS_CONFIG_OPTION("s_tir.InjectDoubleBuffer", InjectDoubleBufferConfig);

// Detect double buffer variables.
class DoubleBufferDetector : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;
  ffi::Optional<VisitInterrupt> Visit_(const AttrStmtNode* op) final {
    if (op->attr_key == s_tir::attr::double_buffer_scope) {
      if (auto buffer = GetBufferDataVar(op->node)) {
        touched_.insert(buffer.value().get());
      }
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    } else {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    }
    return std::nullopt;
  }

  // Known loads and stores are not opaque escapes of the buffer variable.
  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadNode* op) final {
    for (const auto& index : op->indices) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(index));
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const BufferStoreNode* op) final {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->value));
    for (const auto& index : op->indices) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(index));
    }
    return std::nullopt;
  }

  // Declared regions carry bounds, not opaque runtime accesses.
  ffi::Optional<VisitInterrupt> Visit_(const BufferRegionNode* op) final {
    for (const Range& range : op->region) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(range->min));
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(range->extent));
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const VarNode* op) final {
    if (def_region_kind() != kTVMFFIDefRegionKindNone) return std::nullopt;
    if (touched_.count(op)) {
      touched_.erase(op);
    }
    return std::nullopt;
  }
  // The set of touched variable.
  std::unordered_set<const VarNode*> touched_;
};

class StripDoubleBufferWrite : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override {
    if (value.as<ExprNode>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const AttrStmtNode* op, InplaceMode inplace_mode) final {
    if (op->attr_key == s_tir::attr::double_buffer_write) {
      return Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
    } else {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
  }
};

class DoubleBufferInjector : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  explicit DoubleBufferInjector(int split_loop) : split_loop_(split_loop) {}

  Stmt Inject(Stmt stmt) {
    auto detector = ffi::make_object<DoubleBufferDetector>();
    detector->Visit(stmt);
    if (detector->touched_.empty()) return stmt;
    for (const VarNode* v : detector->touched_) {
      dbuffer_info_[v] = StorageEntry();
    }
    return ConvertSSA(Mutate(stmt, InplaceMode::kAllow).ValueOrUnchanged(std::move(stmt)));
  }

  UnchangedOr<Stmt> Mutate_(const AttrStmtNode* op, InplaceMode inplace_mode) final {
    if (op->attr_key == s_tir::attr::double_buffer_scope) {
      return MakeProducer(op, inplace_mode);
    } else {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
  }

  UnchangedOr<Stmt> Mutate_(const AllocBufferNode* op, InplaceMode inplace_mode) final {
    const VarNode* buf = op->buffer.get();
    auto it = dbuffer_info_.find(buf);
    if (it != dbuffer_info_.end()) {
      StorageEntry& entry = it->second;
      entry.scope = op->buffer.scope();

      TVM_FFI_ICHECK_EQ(op->buffer->shape.size(), 1)
          << "InjectDoubleBuffer expects flat 1-d buffers.  "
          << "Has FlattenBuffer been run?";
      entry.stride = op->buffer->shape[0];

      // In flat IR, AllocBuffer appears before its usage in the SeqStmt,
      // so entry.loop may not be set yet. Defer double-buffer allocation
      // processing to be handled in VisitStmt_(ForNode*).
      pending_dbuffer_allocs_[buf] = ffi::GetRef<AllocBuffer>(op);
      // Remove the original AllocBuffer (will be re-emitted in ForNode visitor)
      return Evaluate(0);
    } else {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
  }

  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) final {
    loop_nest_.push_back(op);
    Stmt stmt = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    // Process any pending double-buffer allocations that were deferred
    // from VisitStmt_(AllocBufferNode*) -- now entry.loop should be set.
    for (auto pend_it = pending_dbuffer_allocs_.begin();
         pend_it != pending_dbuffer_allocs_.end();) {
      auto db_it = dbuffer_info_.find(pend_it->first);
      if (db_it != dbuffer_info_.end() && db_it->second.loop != nullptr) {
        StorageEntry& entry = db_it->second;
        const AllocBuffer& alloc = pend_it->second;
        auto new_buf = GetRemappedBuffer(alloc->buffer, entry.stride);
        auto& alloc_nest = loop_allocs_[entry.loop];
        alloc_nest.emplace_back(AllocBuffer(new_buf, alloc->annotations));
        pend_it = pending_dbuffer_allocs_.erase(pend_it);
      } else {
        ++pend_it;
      }
    }
    auto it = loop_pre_.find(op);
    if (it != loop_pre_.end()) {
      const ForNode* old_loop = stmt.as<ForNode>();
      if (split_loop_ != 0) {
        // Explicitly unroll the loop
        TVM_FFI_ICHECK(split_loop_ % 2 == 0 || split_loop_ == 1)
            << "It is better to split with multiple of 2";
        TVM_FFI_ICHECK(is_zero(old_loop->min));
        PrimExpr zero = old_loop->min;
        PrimExpr new_ext = old_loop->extent - IntImm(old_loop->loop_var.ty(), 1);
        PrimExpr factor = IntImm(new_ext.ty(), split_loop_);
        PrimExpr outer_ext = new_ext / factor;
        PrimExpr tail_base = outer_ext * factor;
        Var outer_var(old_loop->loop_var->name + ".outer", old_loop->loop_var.ty());
        std::unordered_map<const VarNode*, PrimExpr> vmap;
        auto map_var = [&vmap](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
          if (auto it = vmap.find(var.get()); it != vmap.end()) return ffi::Any(it->second);
          return ffi::Unchanged();
        };
        std::vector<Stmt> loop_seq;
        for (int32_t i = 0; i < split_loop_; ++i) {
          vmap[old_loop->loop_var.get()] =
              outer_var.as_or_throw<PrimExpr>() * factor + IntImm(factor.ty(), i);
          loop_seq.emplace_back(
              ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(old_loop->body, map_var)
                  .as_or_throw<Stmt>());
        }
        Stmt loop = For(outer_var.as_or_throw<PrimVar>(), zero, outer_ext, old_loop->kind,
                        SeqStmt::Flatten(loop_seq));
        // tail
        std::vector<Stmt> tail_seq;
        Stmt tail_body = ffi::make_object<StripDoubleBufferWrite>()
                             ->Mutate(old_loop->body)
                             .ValueOrUnchanged(old_loop->body);
        for (int32_t i = 0; i < split_loop_; ++i) {
          PrimExpr idx = tail_base + IntImm(tail_base.ty(), i);
          vmap[old_loop->loop_var.get()] = idx;
          tail_seq.emplace_back(
              IfThenElse(idx < old_loop->extent,
                         ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(tail_body, map_var)
                             .as_or_throw<Stmt>()));
        }
        stmt = SeqStmt::Flatten(loop, tail_seq);
      }
      stmt = SeqStmt::Flatten(it->second, stmt);
    }
    it = loop_allocs_.find(op);
    if (it != loop_allocs_.end()) {
      stmt = MergeNest(it->second, stmt);
    }
    loop_nest_.pop_back();
    return stmt;
  }

  UnchangedOr<Stmt> Mutate_(const BufferStoreNode* op, InplaceMode inplace_mode) final {
    auto value = Mutate(op->value);
    auto indices = Mutate(op->indices).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    BufferStore node = ffi::GetRef<BufferStore>(op);
    if (!value.UnchangedOrSameAs(op->value) || !indices.UnchangedOrSameAs(op->indices)) {
      auto* n = node.CopyOnWrite();
      n->value = std::move(value).ValueOrUnchanged(op->value);
      n->indices = std::move(indices).ValueOrUnchanged(op->indices);
    }

    auto it = dbuffer_info_.find(node->buffer.get());
    if (it != dbuffer_info_.end()) {
      const StorageEntry& e = it->second;
      TVM_FFI_ICHECK(in_double_buffer_scope_);
      TVM_FFI_ICHECK(e.switch_write_var.defined());

      TVM_FFI_ICHECK_EQ(node->indices.size(), 1) << "InjectDoubleBuffer expects flat 1-d buffers.  "
                                                 << "Has FlattenBuffer been run?";

      auto writer = node.CopyOnWrite();
      writer->buffer = GetRemappedBuffer(node->buffer, e.stride);
      writer->indices = {e.switch_write_var.as_or_throw<PrimExpr>() * e.stride + node->indices[0]};
    }

    return node;
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) final {
    auto indices = Mutate(op->indices).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorLoad node = ffi::GetRef<TensorLoad>(op);
    if (!indices.UnchangedOrSameAs(op->indices)) {
      node.CopyOnWrite()->indices = std::move(indices).ValueUnchecked();
    }
    BufferVar buffer = node->source.as_or_throw<tvm::tirx::BufferVar>();

    auto it = dbuffer_info_.find(buffer.get());
    if (it != dbuffer_info_.end()) {
      const StorageEntry& e = it->second;
      TVM_FFI_ICHECK(e.switch_read_var.defined());

      TVM_FFI_ICHECK_EQ(node->indices.size(), 1) << "InjectDoubleBuffer expects flat 1-d buffers.  "
                                                 << "Has FlattenBuffer been run?";

      auto* writer = node.CopyOnWrite();
      writer->source = GetRemappedBuffer(buffer, e.stride);
      writer->indices = {e.switch_read_var * e.stride + node->indices[0]};
      return node;
    }

    return node;
  }

  BufferVar GetRemappedBuffer(BufferVar buf, PrimExpr stride) {
    BufferVar original = buf;
    if (auto replacement = VarRemapGet(buf).as<BufferVar>()) return replacement.value();

    TVM_FFI_ICHECK(stride.defined());
    // TODO(Lunderberg): Move this pass to before
    // FlattenBuffer.  That will simplify the
    // implementation, to be the insertion of a new dimension for the
    // buffer, rather than adjusting the other indices.
    TVM_FFI_ICHECK_EQ(buf->shape.size(), 1) << "InjectDoubleBuffer expects flat 1-d buffers.  "
                                            << "Has FlattenBuffer been run?";

    // Stride gives the distance between the two halves of the
    // double-buffer, not the stride of the buffer's index.
    auto type = CopyBufferType(buf);
    type->shape = {buf->shape[0] + stride};
    buf = RebuildBufferVar(buf, std::move(type));

    VarRemapSet(original, buf);
    return buf;
  }

  UnchangedOr<Expr> Mutate_(const BufferRegionNode* op, InplaceMode inplace_mode) final {
    auto region = Mutate(op->region).as_or_throw<UnchangedOr<ffi::Array<Range>>>();
    if (region.UnchangedOrSameAs(op->region)) return ffi::Unchanged();
    BufferRegion node = ffi::GetRef<BufferRegion>(op);
    node.CopyOnWrite()->region = std::move(region).ValueUnchecked();
    return node;
  }

  UnchangedOr<Expr> Mutate_(const VarNode* op, InplaceMode inplace_mode) final {
    if (def_region_kind() == kTVMFFIDefRegionKindNone) {
      TVM_FFI_ICHECK(!dbuffer_info_.count(op));
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

 private:
  Stmt MakeProducer(const AttrStmtNode* op, InplaceMode inplace_mode) {
    const Var buffer = GetBufferDataVar(op->node).value();
    TVM_FFI_ICHECK_NE(loop_nest_.size(), 0U) << "Double buffer scope must be inside a loop";
    auto it = dbuffer_info_.find(buffer.get());
    if (it == dbuffer_info_.end()) {
      LOG(WARNING) << "Skip double buffer scope " << op->node;
      return this->Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
    }
    StorageEntry& e = it->second;
    e.loop = loop_nest_.back();
    PrimExpr zero = IntImm(e.loop->loop_var.ty(), 0);
    PrimExpr one = IntImm(e.loop->loop_var.ty(), 1);
    PrimExpr two = IntImm(e.loop->loop_var.ty(), 2);
    PrimExpr loop_shift = e.loop->loop_var + one;
    e.switch_write_var = Var(e.loop->loop_var->name + ".db", e.loop->loop_var.ty());
    e.switch_read_var = indexmod(e.loop->loop_var, two);
    in_double_buffer_scope_ = true;
    Stmt body = this->Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
    in_double_buffer_scope_ = false;
    std::unordered_map<const VarNode*, PrimExpr> vmap;
    auto map_var = [&vmap](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto it = vmap.find(var.get()); it != vmap.end()) return ffi::Any(it->second);
      return ffi::Unchanged();
    };
    vmap[e.switch_write_var.get()] = zero;
    vmap[e.loop->loop_var.get()] = zero;
    loop_pre_[e.loop].emplace_back(
        ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(body, map_var).as_or_throw<Stmt>());
    vmap[e.loop->loop_var.get()] = loop_shift;
    vmap[e.switch_write_var.get()] = indexmod(loop_shift, two);
    body = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(body, map_var).as_or_throw<Stmt>();
    body = AttrStmt(GetRemappedBuffer(BufferVar(buffer), e.stride).data(),
                    s_tir::attr::double_buffer_write, 1, body);
    body = IfThenElse(loop_shift < e.loop->extent, body);
    return body;
  }
  // Storage entry for those who need double buffering.
  struct StorageEntry {
    // The size of the buffer
    PrimExpr stride;
    // The loop we need
    const ForNode* loop{nullptr};
    // The switch variable.
    Var switch_write_var;
    // The switch variable for reading.
    PrimExpr switch_read_var;
    // The storage scope.
    std::string scope;
  };
  // Whether split loop
  int32_t split_loop_;
  // Whether we are inside double buffer scope.
  bool in_double_buffer_scope_{false};
  // The current loop next
  std::vector<const ForNode*> loop_nest_;
  // The allocs to be appended before the loop
  std::unordered_map<const ForNode*, std::vector<Stmt>> loop_allocs_;
  // The stmt to be appended before the loop
  std::unordered_map<const ForNode*, std::vector<Stmt>> loop_pre_;
  // The allocation size of the buffer
  std::unordered_map<const VarNode*, StorageEntry> dbuffer_info_;
  // The updated BufferVar objects
  // Pending double-buffer AllocBuffer nodes (deferred from flat AllocBuffer visit)
  std::unordered_map<const VarNode*, AllocBuffer> pending_dbuffer_allocs_;
};

namespace transform {

Pass InjectDoubleBuffer() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    auto cfg = ctx->GetConfig<InjectDoubleBufferConfig>("s_tir.InjectDoubleBuffer");
    if (!cfg.has_value()) {
      cfg = tvm::transform::PassConfigWithDefaults<InjectDoubleBufferConfig>();
    }
    n->body =
        ffi::make_object<DoubleBufferInjector>(cfg.value()->split_loop)->Inject(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "s_tir.InjectDoubleBuffer", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.InjectDoubleBuffer", InjectDoubleBuffer);
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
