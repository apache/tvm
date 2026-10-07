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
 * Lower warp memory to use local memory
 * and shuffle intrinsics.
 *
 * \file lower_warp_memory.cc
 */
// Thanks to Andrew Adams and Vinod Grover for
// explaining the concept of warp shuffle.
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/sym/analyzer.h>
#include <tvm/sym/pattern.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>

#include "../../runtime/thread_storage_scope.h"
#include "../../sym/pattern_match.h"
#include "ir_utils.h"
#include "update_pointer_storage_scope.h"

namespace tvm {
namespace tirx {
using namespace tvm::prim;

// Rewrite Rule
//
// There is no special warp memory in most GPUs.
// Instead, we can stripe the data into threads
// and store the data into local memory.
//
// This requires us to do the following rewriting:
// - Rewrite allocation to use local memory.
// - Rewrite store of warp memory to local store.
// - Rewrite load of warp memory to local plus a shuffle.
//
// Define a generic shuffle intrinsic warp_shuffle(data, warp_index).
// We can use the following rewriting rule
//
// Before rewrite,
//
//   alloc warp warp_mem[n * width * m]
//   store warp_mem[m * warp_index + (width * m) * y + x]
//   load warp_mem[m * z + (width * m) * y + x]
//   subject to x \in [0, m), y \in [0, n)
//
// where width equals to the extent of threadIdx.x, which should
// be no larger than the warp size
//
// After rewrite:
//
//   alloc local local_mem[n * m]
//   store warp_mem[m * y + x]
//   warp_shuffle(load warp_mem[m * y + x], z)
//   subject to (m * y + x) is invariant to warp_index
//
// If width == warp size, we are shuffling on full warps.
// Otherwise, we are virtually shuffling on sub-warps,
// whose size equals to width. In this case, you can imagine
// a warp only consists of `width` threads. Width is passed
// as an argument to the shuffle primitive, and will be
// lowered to the device code if the target supports.
//
// A limitation of this sub-warp approach is that users
// cannot shuffle across the sub-warp boundary (i.e. shuffle
// with threadIdx.y or threadIdx.z indices). It can be solved
// via fusing threadIdx.x to the warp size, or improving the
// analyzer to detect both 3 thread axes, which is left for
// future improvements.

// Algorithm
//
// To implement this rewrite rule, we can do the follow step:
// For each warp memory alloc
// - Use linear pattern detector on load index to find m
// - Deduce n given width and alloc size
// - Now that we have m, n, width, we can proceed with the rewrite

// Visitor to find m in pattern
// store warp_mem[m * warp_index + (width * m) * y + x]
const VarNode* GetTensorVar(const Expr& expr) {
  if (const auto* var = expr.as<VarNode>()) {
    return var;
  }
  if (const auto* call = expr.as<CallNode>();
      call && call->op.same_as(tirx::buffer_data_op()) && call->args.size() == 1) {
    return call->args[0].as<VarNode>();
  }
  return nullptr;
}

// Hardware axis identity belongs to a lexical definition, not to a reused Var.
// Each pass instance processes one device Function after SplitHostDevice.
struct WarpThreadBinding {
  ffi::String tag;
  PrimExpr extent;
};
using WarpThreadBindings = std::unordered_map<const VarNode*, WarpThreadBinding>;
using WarpIndexAliases = std::unordered_map<const VarNode*, PrimExpr>;

PrimExpr ExpandWarpIndexAliases(const PrimExpr& index, const WarpIndexAliases& aliases) {
  return SubstituteWithDataTypeLegalization(index, [&](const Var& var) -> ffi::Optional<PrimExpr> {
    auto it = aliases.find(var.get());
    if (it != aliases.end()) return it->second;
    return std::nullopt;
  });
}

// Normalize equivalent hardware bindings only for index analysis. The selected
// representative is the innermost live definition, so emitted indexes stay in scope.
PrimExpr NormalizeWarpIndex(const PrimExpr& index, const WarpThreadBindings& active_bindings,
                            const Var& warp_index, const WarpIndexAliases& aliases) {
  PrimExpr expanded = ExpandWarpIndexAliases(index, aliases);
  return SubstituteWithDataTypeLegalization(
      expanded, [&](const Var& var) -> ffi::Optional<PrimExpr> {
        auto it = active_bindings.find(var.get());
        if (it != active_bindings.end() && it->second.tag == "threadIdx.x" &&
            !var.same_as(warp_index)) {
          return prim::cast(var->ty.as_or_throw<PrimType>(), warp_index.as_or_throw<PrimVar>());
        }
        return std::nullopt;
      });
}

class WarpStoreCoeffFinder : public StmtExprVisitor {
 public:
  WarpStoreCoeffFinder(const VarNode* buffer, const WarpThreadBindings& bindings,
                       WarpThreadBindings active_bindings, Var warp_index,
                       sym::AnalyzerObj* analyzer, const WarpIndexAliases& aliases)
      : buffer_(buffer),
        bindings_(bindings),
        active_bindings_(std::move(active_bindings)),
        warp_index_(warp_index),
        analyzer_(analyzer),
        aliases_(aliases) {}
  // find the warp co-efficient in the statement given the warp size
  int Find(const Stmt& stmt) {
    this->Visit(stmt);
    return warp_coeff_;
  }

 private:
  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtNode* op) final {
    // Operands are evaluated before the new launch binding comes into scope.
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->args));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->attrs));
    Var previous_index = warp_index_;
    auto previous_bindings = active_bindings_;
    if (op->op.same_as(tirx::launch_thread_op())) {
      PrimVar var = op->body_params[0].as_or_throw<PrimVar>();
      const auto& binding = bindings_.at(var.get());
      active_bindings_.insert_or_assign(var.get(), binding);
      if (binding.tag == "threadIdx.x") warp_index_ = var;
    }
    auto result = Visit(op->body);
    active_bindings_ = std::move(previous_bindings);
    warp_index_ = previous_index;
    return result;
  }

  void UpdateCoefficient(int64_t coefficient) {
    TVM_FFI_ICHECK(warp_index_.defined())
        << "Warp memory store must be inside a threadIdx.x launch";
    TVM_FFI_ICHECK_GT(coefficient, 0);
    if (warp_coeff_ != 0) {
      TVM_FFI_ICHECK_EQ(warp_coeff_, coefficient)
          << "LowerWarpMemory requires compatible store coefficients across launch bindings";
    } else {
      warp_coeff_ = coefficient;
    }
  }

  /// Visitor implementation
  ffi::Optional<VisitInterrupt> Visit_(const CallNode* op) final {
    static const Op mma_fill_op = Op::Get("tirx.mma_fill");
    static const Op ptx_ldmatrix_legacy_op = Op::Get("tirx.ptx_legacy.ldmatrix");
    static const Op mma_fill_legacy_op = Op::Get("tirx.mma_fill_legacy");
    if (op->op.same_as(mma_fill_op) && GetTensorVar(op->args[1]) == buffer_) {
      auto* local_size = op->args[0].as<IntImmNode>();
      TVM_FFI_ICHECK(local_size) << "Integer expected for the first argument of mma_fill";
      UpdateCoefficient(local_size->value.as<int>().value());
    } else if (op->op.same_as(ptx_ldmatrix_legacy_op) && GetTensorVar(op->args[3]) == buffer_) {
      // ldmatrix writes the warp buffer; its local_offset carries
      // ``... + lift(local_size) * tx`` from which the warp coefficient
      // is derived.
      UpdatePattern(op->args[4].as_or_throw<PrimExpr>());
    } else if (op->op.same_as(mma_fill_legacy_op) && GetTensorVar(op->args[1]) == buffer_) {
      auto* local_size = op->args[0].as<IntImmNode>();
      TVM_FFI_ICHECK(local_size) << "Integer expected for the first argument of mma_fill_legacy";
      UpdateCoefficient(local_size->value.as<int>().value());
    }
    // mma_store_legacy/ptx_mma_legacy only *use* the warp buffer
    // (read+rewrite); WarpStoreCoeffFinder relies on ldmatrix/mma_fill
    // (the actual stores) for the warp coefficient.

    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreNode* op) final {
    if (op->buffer.get() != buffer_) {
      return StmtExprVisitor::Visit_(op);
    }

    TVM_FFI_ICHECK_EQ(op->indices.size(), 1) << "Expected flat memory to use as warp memory.  "
                                             << "Has FlattenBuffer been run?";

    PrimExpr index = op->indices[0];
    PrimType value_ty = op->value.ty();
    if (value_ty.lanes() != 1) {
      sym::PVar<PrimExpr> base;
      TVM_FFI_ICHECK(sym::ramp(base, 1, value_ty.lanes()).Match(index))
          << "LowerWarpMemory failed due to store index=" << index
          << ", can only handle continuous store";
      UpdatePattern(base.Eval());

      index = base.Eval();
    }

    UpdatePattern(index);
    return std::nullopt;
  }

  void UpdatePattern(const PrimExpr& index) {
    TVM_FFI_ICHECK(warp_index_.defined())
        << "Warp memory store must be inside a threadIdx.x launch";
    PrimExpr normalized_index = NormalizeWarpIndex(index, active_bindings_, warp_index_, aliases_);
    ffi::Array<PrimExpr> m =
        sym::DetectLinearEquation(normalized_index, {warp_index_.as_or_throw<PrimVar>()});
    TVM_FFI_ICHECK_EQ(m.size(), 2U)
        << "LowerWarpMemory failed. Could not simplify the store index `" << index
        << "` into the form ax + by + cz + ... Warp memory is approximated by storing values in "
           "thread local registers and shuffling values between these registers. Currently only "
           "linear equation indices are supported.";
    PrimExpr mcoeff = analyzer_->canonical_simplify(m[0]);
    const auto* mcoeff_as_int = mcoeff.as<IntImmNode>();
    TVM_FFI_ICHECK(mcoeff_as_int && mcoeff_as_int->value > 0)
        << "LowerWarpMemory failed due to store index=" << index
        << ", require positive constant coefficient on warp index " << warp_index_ << " but get "
        << mcoeff;

    UpdateCoefficient(mcoeff_as_int->value.as<int>().value());
  }

  // The buffer variable
  const VarNode* buffer_;
  const WarpThreadBindings& bindings_;
  WarpThreadBindings active_bindings_;
  // The active lexical warp index.
  Var warp_index_{ffi::UnsafeInit{}};
  // the coefficient
  int64_t warp_coeff_{0};
  // analyzer.
  sym::AnalyzerObj* analyzer_;
  const WarpIndexAliases& aliases_;
};

// Collect each lexical launch definition and validate compatible hardware widths.
class WarpIndexFinder : public StmtExprVisitor {
 public:
  explicit WarpIndexFinder(int warp_size, WarpThreadBindings enclosing_bindings)
      : warp_size_(warp_size), bindings_(std::move(enclosing_bindings)) {
    for (const auto& [var, binding] : bindings_) CheckWidth(binding);
  }

  std::pair<WarpThreadBindings, int> Find(const Stmt& stmt) {
    this->Visit(stmt);
    TVM_FFI_ICHECK_GT(width_, 0) << "Cannot find threadIdx.x within the scope of warp memory";
    return {std::move(bindings_), width_};
  }

 private:
  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtNode* op) final {
    if (op->op.same_as(tirx::launch_thread_op())) {
      WarpThreadBinding binding{op->args[0].as_or_throw<StringImm>()->value,
                                op->args[1].as_or_throw<PrimExpr>()};
      CheckWidth(binding);
      bindings_.insert_or_assign(op->body_params[0].as_or_throw<PrimVar>().get(),
                                 std::move(binding));
    }
    return StmtExprVisitor::Visit_(op);
  }

  void CheckWidth(const WarpThreadBinding& binding) {
    if (binding.tag != "threadIdx.x") return;
    const auto* extent = binding.extent.as<IntImmNode>();
    TVM_FFI_ICHECK(extent && extent->value > 0 && extent->value <= warp_size_ &&
                   warp_size_ % extent->value == 0)
        << "Expect threadIdx.x size to be a positive factor of warp size (" << warp_size_
        << "), but got " << binding.extent;
    int width = extent->value.as<int>().value();
    TVM_FFI_ICHECK(width_ == 0 || width_ == width)
        << "LowerWarpMemory requires compatible threadIdx.x extents across launch bindings";
    width_ = width;
  }

  int warp_size_;
  int width_{0};
  WarpThreadBindings bindings_;
};
// Mutator to change the read pattern
class WarpAccessRewriter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit WarpAccessRewriter(int warp_size, sym::AnalyzerObj* analyzer,
                              WarpThreadBindings enclosing_bindings, Var warp_index,
                              const WarpIndexAliases& aliases)
      : warp_size_(warp_size),
        warp_index_(warp_index),
        analyzer_(analyzer),
        bindings_(enclosing_bindings),
        active_bindings_(std::move(enclosing_bindings)),
        aliases_(aliases) {}
  // Rewrite the AllocTensor statement which transforms
  // warp memory to local memory.
  // \param op The allocation binding for warp memory.
  // \param buffer_call The matched allocation Call.
  // \param body The remaining statements (siblings) that use this buffer.
  Stmt Rewrite(const BindNode* op, const CallNode* buffer_call, Stmt body) {
    tvm::Tuple shape = buffer_call->args[0].as_or_throw<tvm::Tuple>();
    DLDataType dtype = buffer_call->args[1].as_or_throw<DataTypeImm>()->value;
    PrimType element_type(dtype);
    buffer_ = op->var.get();
    int64_t alloc_size = 1;
    for (const auto& dim : shape->fields) {
      if (const IntImmNode* int_size = dim.as<IntImmNode>()) {
        alloc_size = static_cast<int64_t>(alloc_size * int_size->value);
      } else {
        alloc_size = 0;
      }
    }
    TVM_FFI_ICHECK_GT(alloc_size, 0) << "warp memory only support constant alloc size";
    alloc_size *= element_type.lanes();
    std::tie(bindings_, width_) =
        ffi::make_object<WarpIndexFinder>(warp_size_, bindings_)->Find(body);
    warp_coeff_ = ffi::make_object<WarpStoreCoeffFinder>(buffer_, bindings_, active_bindings_,
                                                         warp_index_, analyzer_, aliases_)
                      ->Find(body);

    // Align the local memory size. The number of elements may not
    // be a multiple of width_ * warp_coeff_; round it up.
    int factor = width_ * warp_coeff_;
    TVM_FFI_ICHECK_NE(factor, 0) << "Divide by zero";
    warp_group_ = (alloc_size + (factor - 1)) / factor;
    alloc_size = warp_group_ * factor;

    auto type = CopyTensorType(op->var.as_or_throw<TensorVar>());
    type->storage_scope = "local";
    type->shape = {IntImm::Int32(alloc_size / width_)};
    type->strides = {};
    type->elem_offset = IntImm(op->var.as_or_throw<TensorVar>()->elem_offset.ty(), 0);
    TensorVar new_buf = RebuildTensorVar(op->var.as_or_throw<TensorVar>(), std::move(type));
    new_buffer_ = new_buf;
    Stmt rewritten_body = this->Mutate(body, InplaceMode::kDisallow).ValueOrUnchanged(body);
    return SeqStmt::Flatten(
        Bind(new_buf.var(),
             Call(new_buf.type(), tirx::alloc_tensor_op(),
                  {tvm::Tuple(new_buf->shape, buffer_call->args[0]->span),
                   DataTypeImm(new_buf->dtype->dtype, buffer_call->args[1]->span),
                   StringImm(new_buf.scope(), buffer_call->args[2]->span)},
                  buffer_call->attrs, buffer_call->ty_args, buffer_call->span),
             op->span),
        rewritten_body);
  }

 protected:
  UnchangedOr<Stmt> Mutate_(const RegionStmtNode* op, InplaceMode inplace_mode) final {
    if (!op->op.same_as(tirx::launch_thread_op()))
      return StmtExprMutator::Mutate_(op, inplace_mode);
    PrimExpr old_extent = op->args[1].as_or_throw<PrimExpr>();
    PrimExpr extent = Mutate(old_extent, inplace_mode).ValueOrUnchanged(old_extent);
    PrimVar var = op->body_params[0].as_or_throw<PrimVar>();
    Var previous_index = warp_index_;
    auto previous_bindings = active_bindings_;
    const auto& binding = bindings_.at(var.get());
    active_bindings_.insert_or_assign(var.get(), binding);
    if (binding.tag == "threadIdx.x") warp_index_ = var;
    Stmt body = Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
    active_bindings_ = std::move(previous_bindings);
    warp_index_ = previous_index;
    return RegionStmt(op->op, {op->args[0], extent}, op->body_params, op->attrs, body,
                      op->result_vars, op->span);
  }

  Expr RewriteIndicesAt(const CallNode* op, const std::vector<int>& indices) {
    ffi::Array<Expr> new_args = op->args;
    for (int i : indices) {
      // Preserve the pointer operand as an Expr and narrow only its scalar index.
      if (GetTensorVar(op->args[i]) == buffer_) {
        PrimExpr local_index = SplitIndexByGroup(op->args[i + 1].as_or_throw<PrimExpr>()).first;
        new_args.Set(i, new_buffer_.data());
        new_args.Set(i + 1, local_index);
      }
    }
    return Call(op->ty, op->op, new_args, op->attrs, {}, op->span);
  }

  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) override {
    static const Op mma_store_op = Op::Get("tirx.mma_store");
    static const Op mma_fill_op = Op::Get("tirx.mma_fill");
    static const Op ptx_mma_legacy_op = Op::Get("tirx.ptx_legacy.mma");
    static const Op ptx_ldmatrix_legacy_op = Op::Get("tirx.ptx_legacy.ldmatrix");
    static const Op mma_store_legacy_op = Op::Get("tirx.mma_store_legacy");
    static const Op mma_fill_legacy_op = Op::Get("tirx.mma_fill_legacy");
    if (op->op.same_as(mma_store_op)) {
      return RewriteIndicesAt(op, {3});
    }

    if (op->op.same_as(mma_fill_op)) {
      return RewriteIndicesAt(op, {1});
    }

    // Legacy variants: (ptr_var, offset) pairs in apache positions.
    if (op->op.same_as(ptx_mma_legacy_op)) {
      return RewriteIndicesAt(op, {6, 8, 10});
    }
    if (op->op.same_as(ptx_ldmatrix_legacy_op)) {
      // args: trans, num, type, local_ptr, local_offset, smem_ptr_call, smem_offset
      // Only local_ptr is a raw warp buffer Var; smem_ptr is an
      // access_ptr Call wrapping a shared-scope var.
      return RewriteIndicesAt(op, {3});
    }
    if (op->op.same_as(mma_store_legacy_op)) {
      // args: m, n, dst_ptr, src_ptr, src_offset, dst_stride
      return RewriteIndicesAt(op, {3});
    }
    if (op->op.same_as(mma_fill_legacy_op)) {
      // args: local_size, local_ptr, offset
      return RewriteIndicesAt(op, {1});
    }

    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Expr> Mutate_(const VarNode* op, InplaceMode inplace_mode) override {
    if (def_region_kind() == kTVMFFIDefRegionKindNone) {
      TVM_FFI_ICHECK(op != buffer_) << "Cannot access address of warp memory directly";
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode* op, InplaceMode inplace_mode) override {
    // The source is a memory access, not a direct address use checked by the Var hook.
    auto value = Mutate(op->value, inplace_mode);
    auto indices =
        Mutate(op->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorStore store = ffi::GetRef<TensorStore>(op);
    if (!value.UnchangedOrSameAs(op->value) || !indices.UnchangedOrSameAs(op->indices)) {
      auto* n = store.CopyOnWrite();
      n->value = std::move(value).ValueOrUnchanged(op->value);
      n->indices = std::move(indices).ValueOrUnchanged(op->indices);
    }

    if (store->buffer.get() == buffer_) {
      TVM_FFI_ICHECK_EQ(store->indices.size(), 1) << "Expected flat memory to use as warp memory.  "
                                                  << "Has FlattenBuffer been run?";

      auto [local_index, group] = SplitIndexByGroup(store->indices[0]);
      (void)group;  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81767

      auto writer = store.CopyOnWrite();
      writer->buffer = new_buffer_;
      writer->indices = {local_index};
    }

    return store;
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) override {
    // Keep the memory source opaque to the direct-address check in the Var hook.
    auto indices =
        Mutate(op->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorLoad load = ffi::GetRef<TensorLoad>(op);
    if (!indices.UnchangedOrSameAs(op->indices)) {
      load.CopyOnWrite()->indices = std::move(indices).ValueUnchecked();
    }

    if (load->source.as_or_throw<tvm::tirx::TensorVar>().get() != buffer_) {
      return load;
    }

    TVM_FFI_ICHECK_EQ(op->indices.size(), 1) << "Expected flat memory to use as warp memory.  "
                                             << "Has FlattenBuffer been run?";

    auto [local_index, group] = SplitIndexByGroup(op->indices[0]);
    // invariance: local index must do not contain warp id
    auto walkfn = [this](const Var& var) -> ffi::Expected<ffi::WalkResult> {
      return var.get() == warp_index_.get() ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var))
                                            : ffi::WalkResult::Advance();
    };
    TVM_FFI_ICHECK(!ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(local_index, walkfn).has_value())
        << "LowerWarpMemory failed to rewrite load to shuffle for index " << op->indices[0]
        << " local_index=" << local_index;

    load = MakeTensorLoad(new_buffer_, {local_index}, load->span);

    if (analyzer_->CanProveEqual(group, warp_index_.as_or_throw<PrimExpr>())) {
      return load;
    }

    PrimExpr mask =
        Call(PrimType::UInt(32), tirx::tvm_warp_activemask_op(), {}).as_or_throw<PrimExpr>();
    return Call(load.ty(), tirx::tvm_warp_shuffle_op(),
                ffi::Array<PrimExpr>{mask, load, group, width_, warp_size_})
        .as_or_throw<PrimExpr>();
  }

  // Split the index to the two component
  // <local_index, source_index>
  // local index is the index in the local
  // source index is the corresponding source index
  // in this access pattern.
  std::pair<PrimExpr, PrimExpr> SplitIndexByGroup(const PrimExpr& original_index) {
    TVM_FFI_ICHECK(warp_index_.defined())
        << "Warp memory access must be inside a threadIdx.x launch";
    PrimExpr index = NormalizeWarpIndex(original_index, active_bindings_, warp_index_, aliases_);
    PrimType index_ty = index.ty();
    if (index_ty.lanes() != 1) {
      sym::PVar<PrimExpr> base;
      TVM_FFI_ICHECK(sym::ramp(base, 1, index_ty.lanes()).Match(index));

      auto [local_index, group] = SplitIndexByGroup(base.Eval());
      local_index = prim::Ramp(local_index, IntImm(local_index.ty(), 1), index_ty.lanes());
      return std::make_pair(local_index, group);
    }
    PrimExpr m = IntImm(index_ty, warp_coeff_);

    // simple case, warp index is on the highest.
    if (warp_group_ == 1) {
      PrimExpr x = analyzer_->canonical_simplify(indexmod(index, m));
      PrimExpr z = analyzer_->canonical_simplify(indexdiv(index, m));
      return std::make_pair(x, z);
    } else {
      PrimExpr x = analyzer_->canonical_simplify(indexmod(index, m));
      PrimExpr y = index / MakeConst(index_ty, warp_coeff_ * width_);
      y = y * m + x;
      PrimExpr z = indexdiv(indexmod(index, IntImm(index_ty, warp_coeff_ * width_)), m);
      return std::make_pair(analyzer_->canonical_simplify(y), analyzer_->canonical_simplify(z));
    }
  }

 private:
  // the warp size
  int warp_size_{0};
  // The buffer variable
  const VarNode* buffer_;
  // The fresh local buffer replacing the warp-scoped definition.
  TensorVar new_buffer_{ffi::UnsafeInit{}};
  // number of threads involved in one shuffle
  int width_{0};
  // Warp index
  Var warp_index_{ffi::UnsafeInit{}};
  // the coefficient m
  int warp_coeff_{0};
  // the coefficient n
  int warp_group_{0};
  // Internal analyzer
  sym::AnalyzerObj* analyzer_;
  WarpThreadBindings bindings_;
  WarpThreadBindings active_bindings_;
  const WarpIndexAliases& aliases_;
};

// Bind bound information of variables to make analyzer more effective
// TODO(tqchen): consider a pass to inline the bound info into the expr
// so analysis can be context independent.
class BindVarBoundInfo : public StmtExprVisitor {
 public:
  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
    if (value.as<ExprNode>()) return std::nullopt;
    return StmtExprVisitor::Visit(value);
  }
  explicit BindVarBoundInfo(sym::AnalyzerObj* analyzer, WarpIndexAliases* aliases)
      : analyzer_(analyzer), aliases_(aliases) {}

  ffi::Optional<VisitInterrupt> Visit_(const BindNode* op) final {
    if (auto value = op->value.as<PrimExpr>();
        value && op->var.as<PrimVar>() && value->ty().IsScalar() &&
        value->ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) &&
        SideEffect(*value) <= CallEffectKind::kPure) {
      // Preserve thread dependence even for unit extents. Analyzer range
      // simplification would erase that dependence before coefficient matching.
      aliases_->insert_or_assign(op->var.get(), ExpandWarpIndexAliases(*value, *aliases_));
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const ForNode* op) final {
    const Var& loop_var = op->loop_var;
    analyzer_->Bind(loop_var, Range::FromMinExtent(op->min, op->extent));
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtNode* op) final {
    if (op->op.same_as(tirx::launch_thread_op())) {
      PrimVar var = op->body_params[0].as_or_throw<PrimVar>();
      PrimExpr extent = op->args[1].as_or_throw<PrimExpr>();
      Range dom = Range::FromMinExtent(IntImm(extent.ty(), 0), extent);
      analyzer_->Bind(var, dom);
    }
    return StmtExprVisitor::Visit_(op);
  }

 protected:
  // internal analyzer.
  sym::AnalyzerObj* analyzer_;
  WarpIndexAliases* aliases_;
};

// Mutator to change the read pattern
class WarpMemoryRewriter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView input, InplaceMode inplace_mode) override {
    if (input.as<ExprNode>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(input, inplace_mode);
  }
  explicit WarpMemoryRewriter(int warp_size) : warp_size_(warp_size) {}

  Stmt Rewrite(Stmt stmt) {
    if (warp_size_ == 1) return stmt;
    auto binder = ffi::make_object<BindVarBoundInfo>(analyzer_.get(), &aliases_);
    binder->Visit(stmt);
    stmt = Mutate(stmt, InplaceMode::kAllow).ValueOrUnchanged(stmt);
    return stmt;
  }

  // Keep the old variables alive until UpdatePointerStorageScope reads their types.
  std::unordered_map<Var, ffi::String, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> new_storage_scopes_;

 private:
  UnchangedOr<Stmt> Mutate_(const RegionStmtNode* op, InplaceMode inplace_mode) final {
    if (!op->op.same_as(tirx::launch_thread_op()))
      return StmtExprMutator::Mutate_(op, inplace_mode);
    PrimVar var = op->body_params[0].as_or_throw<PrimVar>();
    auto previous_bindings = active_bindings_;
    Var previous_index = warp_index_;
    ffi::String tag = op->args[0].as_or_throw<StringImm>()->value;
    active_bindings_.insert_or_assign(var.get(),
                                      WarpThreadBinding{tag, op->args[1].as_or_throw<PrimExpr>()});
    if (tag == "threadIdx.x") warp_index_ = var;
    Stmt body = Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
    active_bindings_ = std::move(previous_bindings);
    warp_index_ = previous_index;
    return RegionStmt(op->op, op->args, op->body_params, op->attrs, body, op->result_vars,
                      op->span);
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtNode* op, InplaceMode inplace_mode) {
    // Process SeqStmt to find warp AllocTensor and gather remaining siblings as body.
    ffi::Array<Stmt> new_seq;
    bool changed = false;
    for (size_t i = 0; i < op->seq.size(); ++i) {
      const auto* alloc = op->seq[i].as<BindNode>();
      if (const auto* call = alloc ? alloc->value.as<CallNode>() : nullptr;
          call && call->op.same_as(tirx::alloc_tensor_op()) &&
          call->args[2].as_or_throw<StringImm>()->value == "warp") {
        new_storage_scopes_[alloc->var] = "local";
        // Gather remaining siblings as the "body" for rewriting.
        ffi::Array<Stmt> remaining;
        for (size_t j = i + 1; j < op->seq.size(); ++j) {
          remaining.push_back(op->seq[j]);
        }
        Stmt body = remaining.empty() ? Stmt(Evaluate(0)) : SeqStmt::Flatten(remaining);
        auto rewriter = ffi::make_object<WarpAccessRewriter>(
            warp_size_, analyzer_.get(), active_bindings_, warp_index_, aliases_);
        Stmt rewritten = rewriter->Rewrite(alloc, call, body);
        // Continue through the remaining allocation bindings in the same scope.
        new_seq.push_back(Mutate(rewritten, inplace_mode).ValueOrUnchanged(rewritten));
        changed = true;
        break;
      } else {
        auto result = this->Mutate(op->seq[i]);
        changed |= !result.UnchangedOrSameAs(op->seq[i]);
        new_seq.push_back(std::move(result).ValueOrUnchanged(op->seq[i]).as_or_throw<Stmt>());
      }
    }
    if (!changed) return ffi::Unchanged();
    return SeqStmt::Flatten(new_seq);
  }

  int warp_size_{0};
  sym::Analyzer analyzer_;
  WarpThreadBindings active_bindings_;
  Var warp_index_{ffi::UnsafeInit{}};
  WarpIndexAliases aliases_;
};

namespace transform {

Pass LowerWarpMemory() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    if (!f->body.has_value()) return f;
    auto* n = f.CopyOnWrite();
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    TVM_FFI_ICHECK(target.has_value()) << "LowerWarpMemory: Require the target attribute";
    int warp_size = target.value()->GetAttr<int64_t>("thread_warp_size", 1).value();
    auto warp_memory_rewriter = ffi::make_object<WarpMemoryRewriter>(warp_size);
    auto stmt = warp_memory_rewriter->Rewrite(std::move(n->body).value());
    n->body = ffi::make_object<UpdatePointerStorageScope>(warp_memory_rewriter->new_storage_scopes_)
                  ->Mutate(stmt, InplaceMode::kAllow)
                  .ValueOrUnchanged(stmt);
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "tirx.LowerWarpMemory", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.LowerWarpMemory", LowerWarpMemory);
}

}  // namespace transform

}  // namespace tirx
}  // namespace tvm
