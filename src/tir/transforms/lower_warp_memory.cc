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
#include <tvm/arith/analyzer.h>
#include <tvm/arith/pattern.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

#include "../../arith/pattern_match.h"
#include "../../runtime/thread_storage_scope.h"
#include "ir_utils.h"
#include "update_pointer_storage_scope.h"

namespace tvm {
namespace tir {

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
class WarpStoreCoeffFinder : private StmtExprVisitor {
 public:
  WarpStoreCoeffFinder(const VarNode* buffer, Var warp_index, arith::Analyzer* analyzer)
      : buffer_(buffer), warp_index_(warp_index), analyzer_(analyzer) {}
  // find the warp co-efficient in the statement given the warp size
  int Find(const Stmt& stmt) {
    this->VisitStmt(stmt);
    return warp_coeff_;
  }

 private:
  /// Visitor implementation
  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::ptx_ldmatrix()) && op->args[3].as<VarNode>() == buffer_) {
      UpdatePattern(op->args[4]);
    } else if (op->op.same_as(builtin::mma_fill()) && op->args[1].as<VarNode>() == buffer_) {
      auto* local_size = op->args[0].as<IntImmNode>();
      ICHECK(local_size) << "Integer expected for the first argument of mma_fill";
      warp_coeff_ = local_size->value;
    }

    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    if (op->buffer->data.get() != buffer_) {
      StmtVisitor::VisitStmt_(op);
      return;
    }

    ICHECK_EQ(op->indices.size(), 1) << "Expected flat memory to use as warp memory.  "
                                     << "Has StorageFlatten (TE-based schedule) or "
                                     << "FlattenBuffer (TIR-based schedules) been run?";

    PrimExpr index = op->indices[0];
    if (op->value.dtype().lanes() != 1) {
      arith::PVar<PrimExpr> base;
      ICHECK(arith::ramp(base, 1, op->value.dtype().lanes()).Match(index))
          << "LowerWarpMemory failed due to store index=" << index
          << ", can only handle continuous store";
      UpdatePattern(base.Eval());

      index = base.Eval();
    }

    UpdatePattern(index);
  }

  void UpdatePattern(const PrimExpr& index) {
    Array<PrimExpr> m = arith::DetectLinearEquation(index, {warp_index_});
    ICHECK_EQ(m.size(), 2U)
        << "LowerWarpMemory failed. Could not simplify the store index `" << index
        << "` into the form ax + by + cz + ... Warp memory is approximated by storing values in "
           "thread local registers and shuffling values between these registers. Currently only "
           "linear equation indices are supported.";
    PrimExpr mcoeff = analyzer_->canonical_simplify(m[0]);
    const auto* mcoeff_as_int = mcoeff.as<IntImmNode>();
    ICHECK(mcoeff_as_int && mcoeff_as_int->value > 0)
        << "LowerWarpMemory failed due to store index=" << index
        << ", require positive constant coefficient on warp index " << warp_index_ << " but get "
        << mcoeff;

    if (warp_coeff_ != 0) {
      ICHECK_EQ(warp_coeff_, mcoeff_as_int->value)
          << "LowerWarpMemory failed due to two different store coefficient to warp index";
    } else {
      warp_coeff_ = mcoeff_as_int->value;
    }
  }

  // The buffer variable
  const VarNode* buffer_;
  // the warp index
  Var warp_index_;
  // the coefficient
  int64_t warp_coeff_{0};
  // analyzer.
  arith::Analyzer* analyzer_;
};

// Visitor to find the warp index
class WarpIndexFinder : private StmtVisitor {
 public:
  explicit WarpIndexFinder(int warp_size) : warp_size_(warp_size) {}
  // find the warp co-efficient and the shuffle width in the statement
  std::pair<Var, int> Find(const Stmt& stmt) {
    this->VisitStmt(stmt);
    ICHECK(warp_index_.defined())
        << "Cannot find warp index(threadIdx.x) within the scope of warp memory";
    return std::make_pair(warp_index_->var, width_);
  }

 private:
  /// Visitor implementation
  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        auto* value_as_int = op->value.as<IntImmNode>();
        ICHECK(value_as_int && value_as_int->value <= warp_size_ &&
               warp_size_ % value_as_int->value == 0)
            << "Expect threadIdx.x 's size to be no larger than, and a factor of"
            << " warp size(" << warp_size_ << ")"
            << " to enable warp memory"
            << " but get " << op->value << " instead";
        if (warp_index_.defined()) {
          ICHECK(warp_index_.same_as(iv))
              << "Find two instance of " << warp_index_->thread_tag << " in the same kernel. "
              << "Please create it using thread_axis once and reuse the axis "
              << "across multiple binds in the same kernel";
        } else {
          width_ = value_as_int->value;
          warp_index_ = iv;
        }
      }
    }
    StmtVisitor::VisitStmt_(op);
  }
  // warp size
  int warp_size_{0};
  // number of threads involved in one shuffle
  int width_{0};
  // the warp index
  IterVar warp_index_{nullptr};
};
// Mutator to change the read pattern
class WarpAccessRewriter : protected StmtExprMutator {
 public:
  explicit WarpAccessRewriter(int warp_size, arith::Analyzer* analyzer)
      : warp_size_(warp_size), analyzer_(analyzer) {}
  // Rewrite the allocate statement which transforms
  // warp memory to local memory.
  Stmt Rewrite(const AllocateNode* op) {
    buffer_ = op->buffer_var.get();
    int alloc_size = op->ConstantAllocationSize();
    ICHECK_GT(alloc_size, 0) << "warp memory only support constant alloc size";
    alloc_size *= op->dtype.lanes();
    std::tie(warp_index_, width_) = WarpIndexFinder(warp_size_).Find(op->body);
    warp_coeff_ = WarpStoreCoeffFinder(buffer_, warp_index_, analyzer_).Find(op->body);

    // Align the local memory size. The number of elements may not
    // be a multiple of width_ * warp_coeff_; round it up.
    int factor = width_ * warp_coeff_;
    ICHECK_NE(factor, 0) << "Divide by zero";
    warp_group_ = (alloc_size + (factor - 1)) / factor;
    alloc_size = warp_group_ * factor;

    return Allocate(op->buffer_var, op->dtype, {make_const(DataType::Int(32), alloc_size / width_)},
                    op->condition, this->VisitStmt(op->body), op->annotations);
  }

 protected:
  PrimExpr RewriteIndicesAt(const CallNode* op, const std::vector<int>& indices) {
    Array<PrimExpr> new_args = op->args;
    for (int i : indices) {
      if (op->args[i].get() == buffer_) {
        PrimExpr local_index = SplitIndexByGroup(op->args[i + 1]).first;
        new_args.Set(i + 1, local_index);
      }
    }
    return Call(op->dtype, op->op, new_args);
  }

  PrimExpr VisitExpr_(const CallNode* op) override {
    if (op->op.same_as(builtin::ptx_mma())) {
      return RewriteIndicesAt(op, {6, 8, 10});
    }

    if (op->op.same_as(builtin::ptx_ldmatrix())) {
      return RewriteIndicesAt(op, {3});
    }

    if (op->op.same_as(builtin::mma_store())) {
      return RewriteIndicesAt(op, {3});
    }

    if (op->op.same_as(builtin::mma_fill())) {
      return RewriteIndicesAt(op, {1});
    }

    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const VarNode* op) override {
    ICHECK(op != buffer_) << "Cannot access address of warp memory directly";
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) override {
    auto store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));

    if (store->buffer->data.get() == buffer_) {
      ICHECK_EQ(store->indices.size(), 1) << "Expected flat memory to use as warp memory.  "
                                          << "Has StorageFlatten (TE-based schedule) or "
                                          << "FlattenBuffer (TIR-based schedules) been run?";

      auto [local_index, group] = SplitIndexByGroup(store->indices[0]);
      (void)group;  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81767

      auto writer = store.CopyOnWrite();
      writer->indices = {local_index};
    }

    return std::move(store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) override {
    auto load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));

    if (load->buffer->data.get() != buffer_) {
      return std::move(load);
    }

    ICHECK_EQ(op->indices.size(), 1) << "Expected flat memory to use as warp memory.  "
                                     << "Has StorageFlatten (TE-based schedule) or "
                                     << "FlattenBuffer (TIR-based schedules) been run?";

    auto [local_index, group] = SplitIndexByGroup(op->indices[0]);
    // invariance: local index must do not contain warp id
    ICHECK(!UsesVar(local_index, [this](const VarNode* var) { return var == warp_index_.get(); }))
        << "LowerWarpMemory failed to rewrite load to shuffle for index " << op->indices[0]
        << " local_index=" << local_index;

    auto writer = load.CopyOnWrite();
    writer->indices = {local_index};

    if (analyzer_->CanProveEqual(group, warp_index_)) {
      return std::move(load);
    }

    PrimExpr mask = Call(DataType::UInt(32), builtin::tvm_warp_activemask(), {});
    return Call(load.dtype(), builtin::tvm_warp_shuffle(), {mask, load, group, width_, warp_size_});
  }

  // Split the index to the two component
  // <local_index, source_index>
  // local index is the index in the local
  // source index is the corresponding source index
  // in this access pattern.
  std::pair<PrimExpr, PrimExpr> SplitIndexByGroup(const PrimExpr& index) {
    if (index.dtype().lanes() != 1) {
      arith::PVar<PrimExpr> base;
      ICHECK(arith::ramp(base, 1, index.dtype().lanes()).Match(index));

      auto [local_index, group] = SplitIndexByGroup(base.Eval());
      local_index = Ramp(local_index, make_const(local_index.dtype(), 1), index.dtype().lanes());
      return std::make_pair(local_index, group);
    }
    PrimExpr m = make_const(index.dtype(), warp_coeff_);

    // simple case, warp index is on the highest.
    if (warp_group_ == 1) {
      PrimExpr x = analyzer_->canonical_simplify(indexmod(index, m));
      PrimExpr z = analyzer_->canonical_simplify(indexdiv(index, m));
      return std::make_pair(x, z);
    } else {
      PrimExpr x = analyzer_->canonical_simplify(indexmod(index, m));
      PrimExpr y = index / make_const(index.dtype(), warp_coeff_ * width_);
      y = y * m + x;
      PrimExpr z = indexdiv(indexmod(index, make_const(index.dtype(), warp_coeff_ * width_)), m);
      return std::make_pair(analyzer_->canonical_simplify(y), analyzer_->canonical_simplify(z));
    }
  }

 private:
  // the warp size
  int warp_size_{0};
  // The buffer variable
  const VarNode* buffer_;
  // number of threads involved in one shuffle
  int width_{0};
  // Warp index
  Var warp_index_;
  // the coefficient m
  int warp_coeff_{0};
  // the coefficient n
  int warp_group_{0};
  // Internal analyzer
  arith::Analyzer* analyzer_;
};

// Bind bound information of variables to make analyzer more effective
// TODO(tqchen): consider a pass to inline the bound info into the expr
// so analysis can be context independent.
class BindVarBoundInfo : public StmtVisitor {
 public:
  explicit BindVarBoundInfo(arith::Analyzer* analyzer) : analyzer_(analyzer) {}

  void VisitStmt_(const ForNode* op) final {
    const Var& loop_var = op->loop_var;
    analyzer_->Bind(loop_var, Range::FromMinExtent(op->min, op->extent));
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode* op) {
    if (op->attr_key == attr::thread_extent || op->attr_key == attr::virtual_thread) {
      IterVar iv = Downcast<IterVar>(op->node);
      ICHECK_NE(iv->thread_tag.length(), 0U);
      if (!var_dom_.count(iv->var.get())) {
        Range dom = Range::FromMinExtent(0, op->value);
        var_dom_[iv->var.get()] = dom;
        analyzer_->Bind(iv->var, dom);
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

 protected:
  // internal analyzer.
  arith::Analyzer* analyzer_;
  // variable domain
  std::unordered_map<const VarNode*, Range> var_dom_;
};

// Mutator to change the read pattern
class WarpMemoryRewriter : private StmtMutator {
 public:
  explicit WarpMemoryRewriter(int warp_size) : warp_size_(warp_size) {}

  Stmt Rewrite(Stmt stmt) {
    if (warp_size_ == 1) return stmt;
    BindVarBoundInfo binder(&analyzer_);
    binder(stmt);
    stmt = operator()(std::move(stmt));
    return stmt;
  }

  std::unordered_map<const VarNode*, String> new_storage_scopes_;

 private:
  Stmt VisitStmt_(const AllocateNode* op) {
    auto ret = StmtMutator::VisitStmt_(op);
    op = ret.as<AllocateNode>();
    if (GetPtrStorageScope(op->buffer_var) == "warp") {
      new_storage_scopes_[op->buffer_var.get()] = "local";
      WarpAccessRewriter rewriter(warp_size_, &analyzer_);
      ret = rewriter.Rewrite(op);
    }
    return ret;
  }

  int warp_size_{0};
  arith::Analyzer analyzer_;
  // variable domain
  std::unordered_map<const VarNode*, Range> var_dom_;
};

namespace transform {

Pass LowerWarpMemory() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined()) << "LowerWarpMemory: Require the target attribute";
    int warp_size = target.value()->GetAttr<Integer>("thread_warp_size", 1).value().IntValue();
    WarpMemoryRewriter warp_memory_rewriter(warp_size);
    auto stmt = warp_memory_rewriter.Rewrite(std::move(n->body));
    n->body = UpdatePointerStorageScope(warp_memory_rewriter.new_storage_scopes_)(stmt);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerWarpMemory", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerWarpMemory").set_body_typed(LowerWarpMemory);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
