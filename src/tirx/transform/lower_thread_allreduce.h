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

#ifndef TVM_TIRX_TRANSFORM_LOWER_THREAD_ALLREDUCE_H_
#define TVM_TIRX_TRANSFORM_LOWER_THREAD_ALLREDUCE_H_

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/sym/analyzer.h>
#include <tvm/target/target.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <map>
#include <unordered_set>

#include "../../runtime/thread_storage_scope.h"
#include "ir_utils.h"
#include "update_pointer_storage_scope.h"

namespace tvm {
namespace tirx {
namespace detail {
using namespace tvm::prim;
using namespace tvm::tirx;

inline ffi::Optional<Var> GetBufferDataVar(const ffi::Any& data) {
  if (auto var = data.as<Var>()) {
    return var;
  }
  if (const auto* call = data.as<CallNode>();
      call && call->op.same_as(tirx::builtin::buffer_data()) && call->args.size() == 1) {
    return call->args[0].as<Var>();
  }
  return std::nullopt;
}

template <typename DialectMutator>
class ThreadAllreduceBuilder final : public DialectMutator {
 public:
  using DialectMutator::Mutate;
  using DialectMutator::Mutate_;

  explicit ThreadAllreduceBuilder(const TargetNode* target, const ffi::Array<Var>& params)
      : target_(target),
        warp_size_(target->GetAttr<int64_t>("thread_warp_size", 1).value()),
        max_num_threads_(target->GetAttr<int64_t>("max_num_threads", -1).value()) {
    for (const Var& param : params) {
      if (param->ty.as<TensorTypeNode>()) {
        buffer_aliases_.Set(param, param);
      }
    }
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtNode* op, InplaceMode inplace_mode) final {
    if (!op->op.same_as(tirx::builtin::launch_thread()) ||
        std::string(op->args[0].as_or_throw<StringImm>()->value).rfind("vthread", 0) == 0) {
      return DialectMutator::Mutate_(op, inplace_mode);
    }
    thread_extents_.push_back(op);
    auto result = DialectMutator::Mutate_(op, inplace_mode);
    thread_extents_.pop_back();
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateNode* op, InplaceMode inplace_mode) final {
    Stmt stmt = DialectMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    op = stmt.as<EvaluateNode>();
    const CallNode* call = op->value.as<CallNode>();
    if (call && call->op.same_as(tirx::builtin::tvm_thread_allreduce())) {
      return MakeAllreduce(call);
    } else {
      return stmt;
    }
  }
  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) final {
    if (const auto* call = op->value.as<CallNode>(); call) {
      if (call->op.same_as(builtin::alloc_tensor())) return MutateAllocTensor(op, inplace_mode);
      if (call->op.same_as(builtin::decl_tensor())) return MutateDeclTensor(op, call, inplace_mode);
    }
    return DialectMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> MutateAllocTensor(const BindNode* op, InplaceMode inplace_mode) {
    buffer_aliases_.Set(op->var, op->var);
    // In flat IR, alloc_remap_ may not yet be populated when this AllocTensor is visited
    // (the remap is set up by MakeAllreduce which runs during Evaluate visit
    // that appears later in the sequence). We record the original data pointer and
    // attempt the remap; if it's not ready, the post-processing pass will handle it.
    const VarNode* orig_data_ptr = op->var.get();
    auto node = DialectMutator::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                    .template as_or_throw<Bind>();

    if (auto it = alloc_remap_.find(orig_data_ptr); it != alloc_remap_.end()) {
      return RemapAllocTensor(node, it->second);
    }
    // Record for deferred remapping (flat IR case)
    pending_alloc_buffers_.emplace_back(orig_data_ptr);
    return node;
  }

  /*!
   * \brief Remap an AllocTensor node to use the replacement buffer.
   * \param node The original AllocTensor node.
   * \param replacement The replacement buffer.
   * \return The remapped statement(s).
   */
  Stmt RemapAllocTensor(Bind node, const TensorVar& replacement) {
    const CallNode* call = node->value.template as<CallNode>();
    DictAttrs annotations = call->attrs.as_or_throw<DictAttrs>();
    if (replacement.scope() == "shared") {
      annotations.CopyOnWrite()->dict.Set(tirx::attr::kVolatile, true);
    }
    return Bind(replacement.var(),
                Call(replacement.type(), tirx::builtin::alloc_tensor(),
                     {tvm::Tuple(replacement->shape, call->args[0]->span),
                      DataTypeImm(replacement->dtype->dtype, call->args[1]->span),
                      StringImm(replacement.scope(), call->args[2]->span)},
                     annotations, call->ty_args, call->span),
                node->span);
  }

  ffi::Optional<TensorVar> GetRemappedBuffer(const TensorVar& buf) {
    Var root = buffer_aliases_.Get(buf.var()).value_or(buf.var());
    if (auto it = allreduce_var_remap_.find(root.get()); it != allreduce_var_remap_.end()) {
      return it->second.template as_or_throw<TensorVar>();
    }

    return std::nullopt;
  }

  UnchangedOr<Stmt> MutateDeclTensor(const BindNode* op, const CallNode* buffer_call,
                                     InplaceMode inplace_mode) {
    RegisterBufferAlias(op->var.as_or_throw<TensorVar>(), buffer_call->args[0]);
    // Remap declarations only after the complete traversal has populated the
    // physical-root maps.  Eagerly replacing an alias declared after its
    // allreduce would retain the old source pointer on the new buffer.
    return DialectMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) final {
    const VarNode* allocation =
        GetAllocationKey(op->source.as_or_throw<tvm::tirx::TensorVar>().get());
    if (auto it = load_remap_.find(allocation); it != load_remap_.end()) {
      for (const auto& index : op->indices) {
        TVM_FFI_ICHECK(IsZero(index));
      }
      return it->second;
    }

    TensorLoad load = DialectMutator::Mutate_(op, inplace_mode)
                          .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op))
                          .template as_or_throw<TensorLoad>();
    op = load.get();

    if (auto opt = GetRemappedBuffer(load->source.as_or_throw<tvm::tirx::TensorVar>())) {
      return MakeTensorLoad(opt.value(), load->indices, load->span);
    }
    return load;
  }

  UnchangedOr<Stmt> Mutate_(const BufferStoreNode* op, InplaceMode inplace_mode) final {
    const VarNode* allocation = GetAllocationKey(op->buffer.get());
    BufferStore store = DialectMutator::Mutate_(op, inplace_mode)
                            .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                            .template as_or_throw<BufferStore>();

    if (auto it = load_remap_.find(allocation); it != load_remap_.end()) {
      const auto* replacement = it->second.template as<TensorLoadNode>();
      TVM_FFI_ICHECK(replacement);
      for (const auto& index : store->indices) {
        TVM_FFI_ICHECK(IsZero(index));
      }
      auto* writer = store.CopyOnWrite();
      writer->buffer = replacement->source.template as_or_throw<tvm::tirx::TensorVar>();
      writer->indices = replacement->indices;
    } else if (auto opt = GetRemappedBuffer(store->buffer)) {
      store.CopyOnWrite()->buffer = opt.value();
    }
    return store;
  }

 private:
  // Thread entry
  struct ThreadEntry {
    runtime::ThreadScope scope;
    ffi::Optional<PrimVar> var;
    int extent;
    // comparator
    bool operator<(const ThreadEntry& other) const {
      return scope.dim_index < other.scope.dim_index;
    }
  };

  static ffi::Array<PrimExpr> ApplyCombiner(const LambdaExpr& combiner,
                                            const ffi::Array<PrimExpr>& lhs,
                                            const ffi::Array<PrimExpr>& rhs) {
    ffi::Array<Expr> arguments;
    for (const PrimExpr& value : lhs) arguments.push_back(value);
    for (const PrimExpr& value : rhs) arguments.push_back(value);
    return builtin::GetAllreduceFields(combiner->Apply(arguments)).Map([](const Expr& value) {
      return value.as_or_throw<PrimExpr>();
    });
  }

  // make allreduce.
  Stmt MakeAllreduce(const CallNode* call) {
    LambdaExpr combiner = call->args[0].as_or_throw<LambdaExpr>();
    ffi::Array<Expr> inits = builtin::GetAllreduceFields(call->args[1]);
    ffi::Array<Expr> inputs = builtin::GetAllreduceFields(call->args[2]);
    ffi::Array<Expr> destinations = builtin::GetAllreduceFields(call->args[4]);
    ffi::Array<Expr> thread_axes = builtin::GetAllreduceFields(call->args[5]);
    size_t size = inputs.size();
    std::vector<PrimExpr> values;
    values.reserve(size);
    std::vector<PrimType> dtypes;
    dtypes.reserve(size);
    PrimExpr cond = call->args[3].as_or_throw<PrimExpr>();
    for (size_t idx = 0; idx < size; ++idx) {
      values.push_back(inputs[idx].as_or_throw<PrimExpr>());
      if (!IsOne(cond)) {
        values[idx] = Select(cond, values[idx], inits[idx].as_or_throw<PrimExpr>());
      }
      dtypes.push_back(values[idx].ty());
    }
    std::vector<TensorVar> buffers;
    buffers.reserve(size);
    for (size_t idx = 0; idx < size; ++idx) {
      PrimExpr arg = destinations[idx].as_or_throw<PrimExpr>();
      // Loads from boolean buffers may have cast nodes inserted by
      // earlier passes.
      if (auto cast = arg.as<CastNode>()) {
        arg = cast->value;
      }
      buffers.push_back(arg.as_or_throw<TensorLoad>()->source.as_or_throw<tvm::tirx::TensorVar>());
    }

    std::unordered_set<const VarNode*> reduce_set;
    for (const Expr& axis : thread_axes) {
      auto var = axis.as<PrimVar>();
      const VarNode* v = var.has_value() ? var.value().get() : nullptr;
      // The simply optimization replace a iteration variable with a constant
      // when extent of the iteration is 1. As thread indexes always start from 0,
      // we can just ignore this variable in this case.
      if (v) {
        reduce_set.insert(v);
      } else {
        TVM_FFI_ICHECK(axis.as<IntImmNode>() && axis.as<IntImmNode>()->value == 0)
            << "Reduction thread axis should be a VarNode or zero IntImmNode";
      }
    }

    size_t nmatch = 0;
    std::vector<ThreadEntry> vred, vpar;
    std::map<int, std::pair<ThreadEntry, bool>> thread_axes_by_dim;
    for (const RegionStmtNode* launch : thread_extents_) {
      ThreadEntry e;
      PrimVar var = launch->body_params[0].as_or_throw<PrimVar>();
      ffi::String tag = launch->args[0].as_or_throw<StringImm>()->value;
      e.scope = runtime::ThreadScope::Create(tag);
      e.var = var;
      TVM_FFI_ICHECK_LE(e.scope.rank, 1);
      TVM_FFI_ICHECK_GE(e.scope.dim_index, 0) << "vthread do not work with cross thread reduction";
      if (e.scope.rank == 1) {
        const auto* ptr = launch->args[1].as_or_throw<PrimExpr>().as<IntImmNode>();
        TVM_FFI_ICHECK(ptr) << "Need constant extent for reduce set " << var;
        e.extent = ptr->value.as<int>().value();
        bool is_reduce = reduce_set.count(var.get());
        nmatch += is_reduce;
        auto [it, inserted] =
            thread_axes_by_dim.emplace(e.scope.dim_index, std::make_pair(e, is_reduce));
        if (!inserted) {
          TVM_FFI_ICHECK_EQ(it->second.first.extent, e.extent)
              << "Incompatible extents for nested bindings of " << tag;
          // Fresh lexical bindings of one hardware axis are aliases. Use the
          // innermost binding for generated indexes and reduce the axis once
          // when any of its live aliases appears in the reduction operands.
          it->second.first = e;
          it->second.second |= is_reduce;
        }
      }
    }
    for (const auto& [dim, entry] : thread_axes_by_dim) {
      if (entry.first.extent != 1) (entry.second ? vred : vpar).push_back(entry.first);
    }
    TVM_FFI_ICHECK_EQ(nmatch, reduce_set.size())
        << "Not all reduce index are presented in the context";
    std::sort(vred.begin(), vred.end());
    std::sort(vpar.begin(), vpar.end());
    // the size of each index.
    int reduce_extent, group_extent;
    PrimExpr reduce_index = FlattenThread(vred, &reduce_extent);
    PrimExpr group_index = FlattenThread(vpar, &group_extent);

    // the longest contiguous reduce extent after flattening
    int contiguous_reduce_extent = 1;
    std::vector<std::tuple<int, int, bool>> block_threads;  // tuple(dim_index, extent, is_reduce)
    for (const ThreadEntry& thr : vred) {
      if (thr.scope.rank == 1) {  // threadIdx
        block_threads.emplace_back(thr.scope.dim_index, thr.extent, true);
      }
    }
    for (const ThreadEntry& thr : vpar) {
      if (thr.scope.rank == 1) {  // threadIdx
        block_threads.emplace_back(thr.scope.dim_index, thr.extent, false);
      }
    }
    // sort according to dim_index
    std::sort(block_threads.begin(), block_threads.end());
    for (auto&& thr_attr : block_threads) {
      auto [dim_index, extent, is_reduce] = thr_attr;
      (void)dim_index;  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81767
      if (is_reduce) {
        contiguous_reduce_extent *= extent;
      } else {
        break;
      }
    }

    std::vector<Stmt> seq;
    std::vector<TensorVar> new_alloc_bufs;
    //
    // This is an optimization. For small reduction sizes, it may be beneficial
    // for a single warp to performance the entire reduction. No trips to shared
    // memory and no cross warp synchronizations are required.
    // The following code emits the reduction as follows:
    //
    // Allocate reduction vars v[i], i = 0..size-1
    //
    // for offset from WARP_SIZE to 1 by 2
    //
    //   a    <- load(v[i])
    //   b    <- shuffle_down(load(v[i], offset))
    //   v[i] <- reduction(a, b)
    //
    // broadcast results from lane 0 to all other lanes and store
    // the final reduction result to the proper location.
    //
    // When the thread extent is multiple of warp size, we can use a two-stage
    // warp-level reduction to optimize. This is implemented by applying the
    // algorithm above twice.
    //
    // For example, suppose we want to use 512 threads to reduce 512 elements
    // and the warp size is 32. In this case there are (512 / 32) = 16 warps.
    // In the first stage, each of the 16 warps reduces 32 elements. So after
    // the stage, we have 16 remaining elements to be reduced, one for each warp.
    // We store the 16 elements in shared memory, and start the second stage.
    // In the second stage we use the first 16 lanes of the first warp to reduce
    // the remaining elements, and this reduction can also be optimized by
    // shuffle_down warp-level primitives.
    PrimExpr zero_index = IntImm(reduce_index.ty(), 0);
    if (IsWarpReduction(dtypes, group_extent, reduce_extent, contiguous_reduce_extent)) {
      std::vector<PrimExpr> reduce_results;
      PrimExpr mask = Call(PrimType::UInt(32), tirx::builtin::tvm_warp_activemask(), {})
                          .as_or_throw<PrimExpr>();

      if (reduce_extent <= warp_size_) {
        std::tie(reduce_results, new_alloc_bufs) =
            MakeWarpAllreduce(values, dtypes, combiner, reduce_index, reduce_extent, group_index,
                              mask, std::nullopt, &seq);

        // Broadcast the reduction result from lane 0 to all other lanes.
        // This avoids to emit predicated stores, as all threads are
        // uniformly writing the same result.
        for (size_t i = 0; i < size; ++i) {
          TensorVar buf = reduce_results[i]
                              .as_or_throw<TensorLoad>()
                              ->source.as_or_throw<tvm::tirx::TensorVar>();
          PrimExpr val = MakeTensorLoad(buf, {zero_index});
          TVM_FFI_ICHECK_EQ(val.ty(), dtypes[i]);
          PrimExpr splat = WarpShuffle(tirx::builtin::tvm_warp_shuffle(), new_alloc_bufs.back(),
                                       val, reduce_extent * group_index);
          seq.push_back(BufferStore(buf, splat, {zero_index}));
        }
      } else {
        int n_warps = reduce_extent / warp_size_;
        std::vector<TensorVar> local_bufs;

        // 1. Create the staging buffer in shared memory.
        std::vector<TensorVar> staging_shared_bufs;
        staging_shared_bufs.reserve(size);
        for (size_t i = 0; i < size; ++i) {
          TensorVar staging_shared_buf = decl_tensor(
              /*shape=*/{IntImm(reduce_index.ty(), n_warps * group_extent)},
              /*dtype=*/buffers[i]->dtype, /*name=*/"red_buf_staging", /*storage_scope=*/"shared");
          staging_shared_bufs.push_back(staging_shared_buf);
          new_alloc_bufs.push_back(staging_shared_buf);
        }

        // 2. First round of allreduce.
        std::tie(reduce_results, local_bufs) =
            MakeWarpAllreduce(values, dtypes, combiner, reduce_index, warp_size_, group_index, mask,
                              std::nullopt, &seq);
        new_alloc_bufs.insert(new_alloc_bufs.end(), local_bufs.begin(), local_bufs.end());

        // 3. Write allreduce results to staging buffer.
        std::vector<Stmt> write_staging_buf;
        write_staging_buf.reserve(size);
        for (size_t i = 0; i < size; ++i) {
          new_alloc_bufs.push_back(reduce_results[i]
                                       .as_or_throw<TensorLoad>()
                                       ->source.as_or_throw<tvm::tirx::TensorVar>());
          write_staging_buf.push_back(BufferStore(
              /*buffer=*/staging_shared_bufs[i],
              /*value=*/reduce_results[i],
              /*indices=*/{group_index * n_warps + floordiv(reduce_index, warp_size_)}));
        }
        PrimExpr cond = floormod(reduce_index, warp_size_) == zero_index;
        seq.push_back(IfThenElse(cond, SeqStmt::Flatten(write_staging_buf)));
        seq.push_back(SyncThread("shared"));

        // 4. Load staging buffer.
        //    Second round of allreduce.
        for (size_t i = 0; i < size; ++i) {
          values[i] = MakeTensorLoad(/*buffer=*/staging_shared_bufs[i],
                                     /*indices=*/{group_index * n_warps + reduce_index});
        }
        std::tie(reduce_results, local_bufs) = MakeWarpAllreduce(
            values, dtypes, combiner, reduce_index, n_warps, group_index, mask,
            /*predicate=*/reduce_index < IntImm(reduce_index.ty(), n_warps), &seq);
        new_alloc_bufs.insert(new_alloc_bufs.end(), local_bufs.begin(), local_bufs.end());

        // 5. Create shared memory buffer(s) of `group_extent` elements, storing
        // the allreduce results so each thread can access.
        std::vector<Stmt> write_result;
        write_result.reserve(size);
        for (size_t i = 0; i < size; ++i) {
          new_alloc_bufs.push_back(reduce_results[i]
                                       .as_or_throw<TensorLoad>()
                                       ->source.as_or_throw<tvm::tirx::TensorVar>());
          TensorVar broadcast_shared_buf = decl_tensor(
              /*shape=*/{IntImm(reduce_index.ty(), group_extent)},
              /*dtype=*/buffers[i]->dtype, /*name=*/"red_result", /*storage_scope=*/"shared");
          write_result.push_back(
              BufferStore(broadcast_shared_buf, reduce_results[i], {group_index}));
          // Update `reduce_results`, pointing to the value loaded from the shared memory buffer.
          reduce_results[i] = MakeTensorLoad(broadcast_shared_buf, {group_index});
        }
        seq.push_back(IfThenElse(reduce_index == zero_index, SeqStmt::Flatten(write_result)));
        seq.push_back(SyncThread("shared"));
      }

      // Write back allreduce results and update existing allocations.
      for (size_t i = 0; i < size; ++i) {
        const VarNode* alloc_key = GetAllocationKey(buffers[i].get());
        TVM_FFI_ICHECK(!load_remap_.count(alloc_key));
        TensorVar buf =
            reduce_results[i].as_or_throw<TensorLoad>()->source.as_or_throw<tvm::tirx::TensorVar>();
        TVM_FFI_ICHECK_EQ(reduce_results[i].ty(), dtypes[i]);
        load_remap_.insert_or_assign(alloc_key, reduce_results[i]);

        // The AllocTensor doesn't need to be emitted here since alloc_remap_
        // will cause the existing allocation to be rewritten in MutateAllocTensor.
        alloc_remap_.insert_or_assign(alloc_key, buf);
        allreduce_var_remap_.insert_or_assign(alloc_key, buf.var());
        allreduce_var_remap_.insert_or_assign(buffers[i].get(), buf.var());
      }
    } else {
      std::vector<TensorVar> shared_bufs;
      shared_bufs.reserve(size);
      if (reduce_extent == 1) {
        // special case, no reduction is needed.
        std::vector<Stmt> stores;
        for (size_t i = 0; i < size; ++i) {
          stores.push_back(BufferStore(buffers[i], values[i], {0}));
        }
        return SeqStmt::Flatten(stores);
      }
      // This sync is necessary because there might be incomplete read of
      // previous iteration on the same buffer.
      seq.emplace_back(SyncThread("shared"));
      for (size_t idx = 0; idx < size; ++idx) {
        shared_bufs.push_back(decl_tensor({IntImm(group_index.ty(), group_extent * reduce_extent)},
                                          dtypes[idx], "red_buf" + std::to_string(idx), "shared"));
        seq.emplace_back(BufferStore(shared_bufs[idx], values[idx],
                                     {BufIndex(reduce_index, group_index, reduce_extent)}));
      }
      seq.emplace_back(SyncThread("shared"));
      seq.emplace_back(MakeBufAllreduce(combiner, dtypes, shared_bufs, reduce_index, group_index,
                                        reduce_extent, group_extent, contiguous_reduce_extent));
      for (size_t idx = 0; idx < size; ++idx) {
        const VarNode* alloc_key = GetAllocationKey(buffers[idx].get());
        TVM_FFI_ICHECK(!load_remap_.count(alloc_key));
        PrimExpr pred =
            prim::MakeConst(PrimType::Bool(static_cast<int16_t>(dtypes[idx].lanes())), true);
        TensorLoad load = MakeTensorLoad(
            shared_bufs[idx], {BufIndex(IntImm(reduce_index.ty(), 0), group_index, reduce_extent)});
        TVM_FFI_ICHECK_EQ(load.ty(), dtypes[idx]);
        load_remap_.insert_or_assign(alloc_key, load);
        alloc_remap_.insert_or_assign(alloc_key, shared_bufs[idx]);
        allreduce_var_remap_.insert_or_assign(alloc_key, shared_bufs[idx].var());
        allreduce_var_remap_.insert_or_assign(buffers[idx].get(), shared_bufs[idx].var());
      }
    }

    // Fix all local allocations as all statements are built.
    ffi::Array<Stmt> alloc_stmts;
    for (TensorVar buf : new_alloc_bufs) {
      alloc_stmts.push_back(Bind(
          buf.var(),
          Call(buf.type(), tirx::builtin::alloc_tensor(),
               {tvm::Tuple(buf->shape), DataTypeImm(buf->dtype->dtype), StringImm(buf.scope())},
               DictAttrs())));
    }
    // Prepend allocations before the sequence
    for (const auto& s : seq) {
      alloc_stmts.push_back(s);
    }
    Stmt body = SeqStmt::Flatten(alloc_stmts);

    return body;
  }

  std::pair<std::vector<PrimExpr>, std::vector<TensorVar>> MakeWarpAllreduce(
      std::vector<PrimExpr> src_values,                  //
      std::vector<PrimType> dtypes,                      //
      const LambdaExpr& combiner,                        //
      PrimExpr reduce_index, int reduce_extent,          //
      PrimExpr group_index,                              //
      PrimExpr mask, ffi::Optional<PrimExpr> predicate,  //
      std::vector<Stmt>* seq) {
    int n_buffers = src_values.size();

    std::vector<TensorVar> shared_bufs;
    std::vector<TensorVar> local_bufs;
    shared_bufs.reserve(n_buffers);

    // This is the index to the reduction variable, one reduction
    // variable per warp. Local scope seems easier to reason without
    // relying on a pattern match pass to fix it later.
    ffi::Array<PrimExpr> zero_indices = {0};
    ffi::Array<PrimExpr> shape = {1};

    std::vector<Stmt> load_values;
    load_values.reserve(n_buffers);
    for (int idx = 0; idx < n_buffers; ++idx) {
      shared_bufs.push_back(
          decl_tensor(shape, dtypes[idx], "red_buf" + std::to_string(idx), "local"));
      load_values.push_back(BufferStore(shared_bufs[idx], src_values[idx], zero_indices));

      // Uses a local variable to store the shuffled data.  Later
      // on, an allocation will be built for this local variable.
      local_bufs.push_back(decl_tensor(shape, dtypes[idx], "t" + std::to_string(idx), "local"));
    }

    if (predicate.has_value()) {
      seq->push_back(IfThenElse(predicate.value(), SeqStmt::Flatten(load_values)));
    } else {
      seq->insert(seq->end(), load_values.begin(), load_values.end());
    }

    // The mask for this reducer, as this reducer may sit inside
    // a divergent control flow. Here it uses a variable to cache the current
    // active channels.
    ffi::Optional<TensorVar> mask_buffer;
    if (need_warp_shuffle_mask_) {
      mask_buffer = decl_tensor(shape, mask.ty(), "mask", "local");
      seq->emplace_back(BufferStore(mask_buffer.value(), mask, zero_indices));
      // Push the buffer description.  Later this will have an
      // allocation built for it.
      local_bufs.push_back(mask_buffer.value());
    }

    // Emit reductions within a warp.
    int start_offset = 1;
    while (start_offset * 2 < reduce_extent) {
      start_offset *= 2;
    }
    for (int offset = start_offset; offset > 0; offset /= 2) {
      // Load reduction values, no synchronization needed.
      ffi::Array<PrimExpr> a, b;
      for (int i = 0; i < n_buffers; ++i) {
        TensorVar shared_buf = shared_bufs[i];
        TensorLoad val = MakeTensorLoad(shared_buf, zero_indices);
        TVM_FFI_ICHECK_EQ(val.ty(), dtypes[i]);
        a.push_back(val);

        // __shfl_*sync calls shall not appear in if_then_else expressions
        // as this is causing extra divergency. E.g.
        //
        // v1 = (v2 < v3) ? v3 : __shfl_sync(mask, v1, 0);
        //
        // behaves differently from
        //
        // int t = __shfl_sync(mask, v1, 0);
        // v1 = (v2 < v3) ? v3 : t;
        //
        // The former may cause dead lock as there is a divergent
        // branch with a warp sync call inside.
        PrimExpr other =
            WarpShuffle(tirx::builtin::tvm_warp_shuffle_down(), mask_buffer, val, offset);
        TensorVar local_buf = local_bufs[i];
        Stmt s = BufferStore(local_buf, other, zero_indices);
        seq->push_back(s);

        TensorLoad load = MakeTensorLoad(local_buf, zero_indices);
        TVM_FFI_ICHECK_EQ(load.ty(), dtypes[i]);
        b.push_back(load);
      }

      // Do reductions.
      ffi::Array<PrimExpr> ret = ApplyCombiner(combiner, a, b);

      // Store the reduction result to itself.
      std::vector<Stmt> stores;
      stores.reserve(n_buffers);
      for (int i = 0; i < n_buffers; ++i) {
        TensorVar buf = shared_bufs[i];
        stores.push_back(BufferStore(buf, ret[i], zero_indices));
      }

      // During the sub-warp reduction, values from inactive threads could be read,
      // which is an undefined behavior according to the cuda document.
      //
      // In practice, the return value are usually 0, which does no harm to sum reduction.
      // However, the result can be incorrect in max or prod reduction.
      // Therefore an additional range check has to be performed to ensure the correctness.
      if (offset * 2 > reduce_extent) {
        PrimExpr cond = reduce_index + offset < reduce_extent;
        seq->push_back(IfThenElse(cond, SeqStmt::Flatten(stores)));
      } else {
        seq->push_back(SeqStmt::Flatten(stores));
      }
    }

    std::vector<PrimExpr> reduce_results;
    reduce_results.reserve(n_buffers);
    for (int i = 0; i < n_buffers; ++i) {
      reduce_results.push_back(MakeTensorLoad(shared_bufs[i], zero_indices));
    }

    return {reduce_results, local_bufs};
  }

  // make allreduce.
  Stmt MakeBufAllreduce(const LambdaExpr& combiner, const std::vector<PrimType>& dtypes,
                        const ffi::Array<TensorVar>& shared_bufs, PrimExpr reduce_index,
                        PrimExpr group_index, int reduce_extent, int group_extent,
                        int contiguous_reduce_extent) {
    // Get next power of two
    int reduce_align = 1;
    while (reduce_extent > reduce_align) {
      reduce_align = reduce_align << 1;
    }
    TVM_FFI_ICHECK_GT(reduce_align, 1);
    std::vector<Stmt> seq;

    size_t size = shared_bufs.size();
    PrimExpr buf_index = BufIndex(reduce_index, group_index, reduce_extent);
    // make reduction
    auto fload = [&](int offset) {
      ffi::Array<PrimExpr> a, b;
      for (size_t i = 0; i < size; ++i) {
        TensorLoad b_load = MakeTensorLoad(
            shared_bufs[i], {BufIndex(reduce_index + offset, group_index, reduce_extent)});
        TVM_FFI_ICHECK_EQ(b_load.ty(), dtypes[i]);
        b.push_back(b_load);

        TensorLoad a_load = MakeTensorLoad(shared_bufs[i], {buf_index});
        TVM_FFI_ICHECK_EQ(a_load.ty(), dtypes[i]);
        a.push_back(a_load);
      }
      ffi::Array<PrimExpr> ret = ApplyCombiner(combiner, a, b);
      return ret;
    };
    auto fstore = [&](const ffi::Array<PrimExpr>& ret) {
      std::vector<Stmt> stores;
      stores.reserve(size);
      for (size_t i = 0; i < size; ++i) {
        stores.push_back(BufferStore(shared_bufs[i], ret[i], {buf_index}));
      }
      return SeqStmt::Flatten(stores);
    };
    auto freduce = [&](int offset) {
      auto ret = fload(offset);
      return fstore(ret);
    };
    // Step one, check for
    if (reduce_align > reduce_extent) {
      // reduction with the boundary condition
      reduce_align = reduce_align >> 1;
      PrimExpr cond = reduce_index < (reduce_extent - reduce_align);
      seq.emplace_back(IfThenElse(cond, freduce(reduce_align)));
      seq.emplace_back(SyncThread("shared"));
    }

    // normal synchronization
    bool warp_align = group_extent == 1 || contiguous_reduce_extent % warp_size_ == 0;
    while (reduce_align > contiguous_reduce_extent || reduce_align > warp_size_ || !warp_align) {
      if (reduce_align == 1) {
        break;
      }
      reduce_align = reduce_align >> 1;
      PrimExpr cond = reduce_index < reduce_align;
      seq.emplace_back(IfThenElse(cond, freduce(reduce_align)));
      seq.emplace_back(SyncThread("shared"));
    }
    // in warp synchronization.
    if (reduce_align > 1) {
      PrimExpr in_warp_cond = reduce_index < (reduce_align >> 1);

      std::vector<Stmt> in_warp_seq;

      while (reduce_align > 1) {
        reduce_align = reduce_align >> 1;

        // freduce can read/write to the same memory location.  For
        // example, with reduce_align of 4, threadIdx 3 reads from
        // memory location 7 as threadIdx 7 is writing to it.
        // Therefore, we need to separate out the load from the store
        // with a memory barrier in-between.  This isn't necessary for
        // the earlier normal synchronization, because those are each
        // protected by an if-statement.  The if-statement is avoided
        // here to reduce thread divergence.
        auto loads = fload(reduce_align);

        ffi::Array<Var> in_warp_local_vars;
        for (auto expr : loads) {
          Var var(
              "w_" + std::to_string(reduce_align) + "_" + std::to_string(in_warp_local_vars.size()),
              expr.ty());
          in_warp_local_vars.push_back(var);
        }

        std::vector<Stmt> in_let_statement;
        in_let_statement.emplace_back(SyncThread("warp"));
        ffi::Array<PrimExpr> prim_in_warp_local_vars =
            in_warp_local_vars.Map([](const Var& var) { return var.as_or_throw<PrimExpr>(); });
        in_let_statement.emplace_back(fstore(prim_in_warp_local_vars));
        in_let_statement.emplace_back(SyncThread("warp"));

        ffi::Array<Stmt> bind_stmts;
        for (size_t i = 0; i < size; i++) {
          bind_stmts.push_back(Bind(in_warp_local_vars[i], loads[i]));
        }
        for (const auto& s : in_let_statement) {
          bind_stmts.push_back(s);
        }
        Stmt body = SeqStmt::Flatten(bind_stmts);
        in_warp_seq.push_back(body);
      }

      Stmt warp_body = SeqStmt::Flatten(in_warp_seq);

      seq.emplace_back(IfThenElse(in_warp_cond, warp_body));
      seq.emplace_back(SyncThread("shared"));
    }
    return SeqStmt::Flatten(seq);
  }
  // Flatten the thread index.
  // Also return a warp number,
  PrimExpr FlattenThread(const std::vector<ThreadEntry>& tvec, int* out_total_extent) {
    int& total_extent = *out_total_extent;
    total_extent = 1;
    if (tvec.size() == 0) {
      return IntImm::Int32(0);
    }

    PrimExpr ret = tvec.front().var.value();
    total_extent = tvec.front().extent;
    for (size_t i = 1; i < tvec.size(); ++i) {
      const ThreadEntry& e = tvec[i];
      ret = ret + e.var.value() * total_extent;
      total_extent *= e.extent;
    }
    return ret;
  }
  // The local buffer index.
  PrimExpr BufIndex(PrimExpr reduce_index, PrimExpr group_index, int reduce_extent) {
    if (!IsZero(group_index)) {
      return analyzer_->Simplify(group_index * reduce_extent + reduce_index);
    } else {
      return reduce_index;
    }
  }
  // sync thread op.
  static Stmt SyncThread(const std::string& sync) {
    return Evaluate(Call(PrimType::Int(32), tirx::builtin::tvm_storage_sync(), {StringImm(sync)})
                        .as_or_throw<PrimExpr>());
  }

  // Emit warp shuffle  calls.
  PrimExpr WarpShuffle(const Op& op, ffi::Optional<TensorVar> mask_buffer, PrimExpr val,
                       PrimExpr delta_or_lane) {
    ffi::Array<PrimExpr> indices = {0};
    PrimExpr mask{ffi::UnsafeInit{}};
    if (need_warp_shuffle_mask_ && mask_buffer.has_value()) {
      mask = MakeTensorLoad(mask_buffer.value(), indices);
    } else {
      mask = IntImm::Int32(0);
    }
    PrimExpr width = IntImm::Int32(warp_size_);
    ffi::Array<PrimExpr> args{mask, val, delta_or_lane, width, width};
    return Call(val.ty(), op, args).as_or_throw<PrimExpr>();
  }

  // Check if we can use warp level reduction.
  //
  // Note: The ROCm backend will only have warp reductions for now.
  // Also, the warp/wavefront size differs (64 on rocm, 32 on cuda and metal).
  bool IsWarpReduction(const std::vector<PrimType>& dtypes, int group_extent, int reduce_extent,
                       int contiguous_reduce_extent) {
    if ((target_->kind->name != "cuda") && (target_->kind->name != "rocm") &&
        (target_->kind->name != "metal") && (target_->kind->name != "webgpu")) {
      return false;
    }

    need_warp_shuffle_mask_ = target_->kind->name != "metal" && target_->kind->name != "webgpu";

    // rocm only supports 32 bit operands for shuffling at the moment
    if ((target_->kind->name == "rocm") &&
        (std::any_of(dtypes.begin(), dtypes.end(), [](const PrimType& dtype) {
          int16_t lanes = static_cast<int16_t>(dtype.lanes());
          if (lanes > 1) return dtype.bits() * lanes != 32;
          return dtype.bits() != 32;
        }))) {
      return false;
    }

    // Supported types:
    // {u}int, {u}long, {u}long long, float, double, half/half2
    if (std::any_of(dtypes.begin(), dtypes.end(), [](const PrimType& dtype) {
          int16_t lanes = static_cast<int16_t>(dtype.lanes());
          if (dtype.MatchesCode(kDLFloat) && dtype.bits() == 16) return lanes > 2;
          if (lanes > 1) return true;
          int bytes = dtype.StorageBytes();
          return bytes < 4 || bytes > 8;
        })) {
      return false;
    }
    if (thread_extents_.empty()) {
      return false;
    }

    // reduce region must be contiguous.
    if (contiguous_reduce_extent != reduce_extent) {
      return false;
    }

    // whether reduce_extent and group_extent are valid for warp reduction.
    if (target_->kind->name == "rocm") {
      return reduce_extent == warp_size_;
    } else {
      if (reduce_extent == 1) {
        return false;  // no need to warp reduce
      } else {
        bool is_subwarp_reduction = warp_size_ % reduce_extent == 0;
        bool is_multiwarp_reduction = max_num_threads_ != -1 &&
                                      max_num_threads_ <= warp_size_ * warp_size_ &&
                                      reduce_extent % warp_size_ == 0;
        if (is_subwarp_reduction || is_multiwarp_reduction) {
          return true;
        } else {
          return group_extent == 1 && reduce_extent <= warp_size_;
        }
      }
    }
  }

  void RegisterBufferAlias(TensorVar buffer, const Expr& data) {
    Var root = buffer.var();
    if (auto source = GetBufferDataVar(data);
        source.has_value() && source.value()->ty.as<TensorTypeNode>()) {
      auto source_root = buffer_aliases_.Get(source.value());
      TVM_FFI_ICHECK(source_root.has_value()) << "Buffer alias source " << source.value()->name
                                              << " must be registered before its DeclTensor alias";
      root = source_root.value();
    }
    buffer_aliases_.Set(buffer.var(), root);
  }

  // The target.
  const TargetNode* target_ = nullptr;

  // The warp size of the device.
  int warp_size_{1};
  // The maximum number of threads of the device. "-1" denotes unknown.
  int max_num_threads_{-1};
  // A boolean indicating if the target supports warp-level masking.
  bool need_warp_shuffle_mask_;

  // surrounding scope of thread extent.
  std::vector<const RegionStmtNode*> thread_extents_;
  // The load remap
  std::unordered_map<const VarNode*, PrimExpr> load_remap_;
  // Internal analyzer
  sym::Analyzer analyzer_;

 public:
  const VarNode* GetAllocationKey(const VarNode* buffer) const {
    if (buffer->ty.as<TensorTypeNode>()) {
      Var var = ffi::GetRef<Var>(buffer);
      return buffer_aliases_.Get(var).value_or(var).get();
    }
    return buffer;
  }

  // These members are public for post-processing by DeferredRemapper.
  // Allocate remap
  std::unordered_map<const VarNode*, TensorVar> alloc_remap_;
  // TensorVar remap
  std::unordered_map<const VarNode*, Var> allreduce_var_remap_;
  // Pending AllocTensor original data pointers (for flat IR deferred remapping)
  std::vector<const VarNode*> pending_alloc_buffers_;
  // Physical roots of buffer aliases, flattened at each declaration.
  ffi::Map<Var, Var> buffer_aliases_;
};

/*!
 * \brief Post-processing pass to apply deferred remappings for flat IR.
 *
 * In flat IR, AllocTensor nodes may be visited before the alloc_remap_ is populated
 * (since MakeAllreduce runs when Evaluate is visited, which is later in the flat sequence).
 * Handles AllocTensor, DeclTensor, and TensorLoad nodes whose remappings
 * were not available during the main traversal.
 */
template <typename DialectMutator>
class DeferredRemapper : public DialectMutator {
 public:
  using DialectMutator::Mutate;
  using DialectMutator::Mutate_;

  DeferredRemapper(const std::unordered_map<const VarNode*, TensorVar>& alloc_remap,
                   const std::unordered_map<const VarNode*, Var>& var_remap,
                   const ffi::Map<Var, Var>& buffer_aliases,
                   const std::vector<const VarNode*>& pending)
      : alloc_remap_(alloc_remap),
        allreduce_var_remap_(var_remap),
        buffer_aliases_(buffer_aliases) {
    for (const VarNode* ptr : pending) {
      pending_set_.insert(ptr);
    }
  }

  bool HasPendingRemaps() const {
    for (const VarNode* ptr : pending_set_) {
      if (alloc_remap_.count(ptr)) return true;
    }
    return false;
  }

  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) final {
    if (const auto* call = op->value.as<CallNode>(); call) {
      if (call->op.same_as(builtin::alloc_tensor())) return MutateAllocTensor(op, inplace_mode);
      if (call->op.same_as(builtin::decl_tensor())) return MutateDeclTensor(op, inplace_mode);
    }
    return DialectMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> MutateAllocTensor(const BindNode* op, InplaceMode inplace_mode) {
    auto node = DialectMutator::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                    .template as_or_throw<Bind>();
    const VarNode* data_ptr = op->var.get();
    if (pending_set_.count(data_ptr)) {
      if (auto it = alloc_remap_.find(data_ptr); it != alloc_remap_.end()) {
        const TensorVar& replacement = it->second;
        const CallNode* call = node->value.template as<CallNode>();
        DictAttrs annotations = call->attrs.as_or_throw<DictAttrs>();
        if (replacement.scope() == "shared") {
          annotations.CopyOnWrite()->dict.Set(tirx::attr::kVolatile, true);
        }
        return Bind(replacement.var(),
                    Call(replacement.type(), tirx::builtin::alloc_tensor(),
                         {tvm::Tuple(replacement->shape, call->args[0]->span),
                          DataTypeImm(replacement->dtype->dtype, call->args[1]->span),
                          StringImm(replacement.scope(), call->args[2]->span)},
                         annotations, call->ty_args, call->span),
                    node->span);
      }
    }
    return node;
  }

  UnchangedOr<Stmt> MutateDeclTensor(const BindNode* op, InplaceMode inplace_mode) {
    const VarNode* root = buffer_aliases_.Get(op->var).value_or(op->var).get();
    if (pending_set_.count(root) && alloc_remap_.count(root)) {
      return Evaluate(0);
    }
    auto node = DialectMutator::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                    .template as_or_throw<Bind>();
    if (auto new_buf = GetRemappedBuffer(node->var.template as_or_throw<TensorVar>())) {
      const CallNode* call = node->value.template as<CallNode>();
      return Bind(
          new_buf.value(),
          Call(new_buf.value().type(), builtin::decl_tensor(),
               {call->args[0], tvm::Tuple(new_buf.value()->shape),
                DataTypeImm(new_buf.value()->dtype->dtype), StringImm(new_buf.value().scope())},
               call->attrs, call->ty_args, call->span),
          node->span);
    }
    return node;
  }

 private:
  ffi::Optional<TensorVar> GetRemappedBuffer(const TensorVar& buf) {
    Var root = buffer_aliases_.Get(buf.var()).value_or(buf.var());
    if (auto it = allreduce_var_remap_.find(root.get()); it != allreduce_var_remap_.end()) {
      return it->second.template as_or_throw<TensorVar>();
    }
    return std::nullopt;
  }

  const std::unordered_map<const VarNode*, TensorVar>& alloc_remap_;
  const std::unordered_map<const VarNode*, Var>& allreduce_var_remap_;
  const ffi::Map<Var, Var>& buffer_aliases_;
  std::unordered_set<const VarNode*> pending_set_;
};

template <typename DialectMutator>
PrimFunc LowerThreadAllreduce(PrimFunc f) {
  auto* n = f.CopyOnWrite();
  auto target = f->GetAttr<Target>(tvm::attr::kTarget);
  TVM_FFI_ICHECK(target.has_value()) << "LowerThreadAllreduce: Require the target attribute";
  const TargetNode* target_node = target.as<TargetNode>();
  auto thread_all_reduce =
      ffi::make_object<ThreadAllreduceBuilder<DialectMutator>>(target_node, f->params);
  n->body = thread_all_reduce->Mutate(n->body).ValueOrUnchanged(n->body);
  // Post-process: apply deferred remappings for flat IR
  auto remapper = ffi::make_object<DeferredRemapper<DialectMutator>>(
      thread_all_reduce->alloc_remap_, thread_all_reduce->allreduce_var_remap_,
      thread_all_reduce->buffer_aliases_, thread_all_reduce->pending_alloc_buffers_);
  if (remapper->HasPendingRemaps()) {
    n->body = remapper->Mutate(n->body).ValueOrUnchanged(n->body);
  }
  return f;
}

}  // namespace detail
}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_TRANSFORM_LOWER_THREAD_ALLREDUCE_H_
