/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file lower_sparse_tir.cc
 */

#include <tvm/arith/analyzer.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <utility>

#include "ir_utils.h"

namespace tvm {
namespace tir {

Map<Var, Buffer> UpdateBufferMap(PrimFunc f) {
  struct BufferMapUpdater : public StmtVisitor {
    explicit BufferMapUpdater(Map<Var, Buffer> buffer_map) : buffer_map_(std::move(buffer_map)) {}

    void VisitStmt_(const SparseBlockNode* sp_block) {
      for (const auto& it : sp_block->sp_struct2param_map) {
        if (const auto* dv_axis = it.first.as<DenseVariableAxisNode>()) {
          ICHECK_EQ(it.second.size(), 1);
          buffer_map_.Set(it.second[0], dv_axis->indptr);
        } else if (const auto* sf_axis = it.first.as<SparseFixedAxisNode>()) {
          ICHECK_EQ(it.second.size(), 1);
          buffer_map_.Set(it.second[0], sf_axis->indices);
        } else if (const auto* sv_axis = it.first.as<SparseVariableAxisNode>()) {
          ICHECK_EQ(it.second.size(), 2);
          buffer_map_.Set(it.second[0], sv_axis->indptr);
          buffer_map_.Set(it.second[1], sv_axis->indices);
        } else if (const auto* sp_buffer = it.first.as<SparseBufferNode>()) {
          ICHECK_EQ(it.second.size(), 1);
          buffer_map_.Set(it.second[0], sp_buffer->data);
        }
      }
      return;
    }

    Map<Var, Buffer> buffer_map_;
  };

  BufferMapUpdater updater(f->buffer_map);
  updater(f->body);
  return std::move(updater.buffer_map_);
}

/*!
 * \brief Check whether a given SparseBuffer contains the given axis.
 * \brief buffer The SparseBuffer to be checked
 * \brief axis The axis to be checked
 * \return A boolean indicating whether the given SparseBuffer contains the given axis
 */
bool BufferContainsAxis(const SparseBuffer& buffer, const Axis& axis) {
  for (int i = 0; i < static_cast<int>(buffer->axes.size()); ++i) {
    if (buffer->axes[i].same_as(axis)) {
      return true;
    }
  }
  return false;
}

using BufferAccessMap = Map<SparseBuffer, Array<SpIterVar>>;
using DependencyMap =
    std::unordered_map<SpIterVar, std::pair<SparseBuffer, int>, ObjectPtrHash, ObjectPtrEqual>;

/*!
 * \brief For each sparse-fixed or sparse-variable iterator, collect the iterators that it depends
 * on.
 */
class AccessAndDependencyCollector : public StmtExprVisitor {
 public:
  void Collect(Stmt stmt) {
    VisitStmt(std::move(stmt));

    for (const std::pair<SparseBuffer, Array<SpIterVar>>& kv_pair : buffer_access_map_) {
      const SparseBuffer& buffer = kv_pair.first;
      int ndim = static_cast<int>(kv_pair.second.size());
      for (int k = 0; k < ndim; ++k) {
        const SpIterVar& sp_iter = kv_pair.second[k];
        if (sp_iter->kind == SpIterKind::kDenseFixed ||
            !BufferContainsAxis(buffer, sp_iter->axis)) {
          continue;
        }

        auto it = dependency_map_.find(sp_iter);
        if (it == dependency_map_.end()) {
          dependency_map_[sp_iter] = std::make_pair(buffer, k);
        } else {
          const Array<SpIterVar>& dependent_iters = buffer_access_map_[it->second.first];
          for (int i = 0; i < k; ++i) {
            CHECK(kv_pair.second[i].same_as(dependent_iters[i]))
                << "ValueError: A SpIterVar can only depend on a fixed set of iterators";
          }
        }
      }
    }
  }

  void GetIteratedBufferAndDependentIters(const SpIterVar& sp_iter, SparseBuffer* iterated_buffer,
                                          Array<PrimExpr>* dependent_iters) {
    std::pair<SparseBuffer, int> dependent_pair = dependency_map_[sp_iter];
    Array<SpIterVar> buffer_access_iters = buffer_access_map_[dependent_pair.first];
    int n_dependent = dependent_pair.second;

    *iterated_buffer = std::move(dependent_pair.first);
    *dependent_iters = Array<PrimExpr>();
    dependent_iters->reserve(n_dependent);
    for (int i = 0; i < n_dependent; ++i) {
      dependent_iters->push_back(buffer_access_iters[i]->var);
    }
  }

  SpIterVar GetSpIterFromIndex(PrimExpr index) {
    auto it = var2sp_iter_map_.find(index.as<VarNode>());
    CHECK(it != var2sp_iter_map_.end())
        << "ValueError: Currently an index is only allowed to be SpIterVar";
    return it->second;
  }

 private:
  void AddAccessPattern(const SparseBuffer& buffer, const Array<PrimExpr>& indices) {
    int ndim = buffer->ndim();
    CHECK_EQ(static_cast<int>(indices.size()), ndim);

    Array<SpIterVar> iters;
    iters.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      iters.push_back(GetSpIterFromIndex(indices[i]));
    }

    BufferAccessMap::iterator it = buffer_access_map_.find(buffer);
    if (it == buffer_access_map_.end()) {
      buffer_access_map_.Set(buffer, iters);
    } else {
      ICHECK_EQ(static_cast<int>((*it).second.size()), ndim);
      for (int i = 0; i < ndim; ++i) {
        CHECK((*it).second[i].same_as(iters[i]))
            << "ValueError: Currently all accesses to a same buffer are required to be the same";
      }
    }
  }

  void VisitStmt_(const SparseBlockNode* sp_block) final {
    for (const SpIterVar& sp_iter : sp_block->sp_iter_vars) {
      var2sp_iter_map_[sp_iter->var.get()] = sp_iter;
    }
    StmtVisitor::VisitStmt_(sp_block);
  }

  void VisitStmt_(const SparseBufferStoreNode* store) final {
    ExprVisitor::VisitExpr(store->value);
    AddAccessPattern(store->buffer, store->indices);
  }

  void VisitExpr_(const SparseBufferLoadNode* load) final {
    AddAccessPattern(load->buffer, load->indices);
  }

  BufferAccessMap buffer_access_map_;
  DependencyMap dependency_map_;
  std::unordered_map<const VarNode*, SpIterVar> var2sp_iter_map_;
};

class IndexTransformer : public StmtExprMutator {
 public:
  explicit IndexTransformer(AccessAndDependencyCollector collector)
      : collector_(std::move(collector)) {}

 private:
  PrimExpr LowerIndices(SparseBuffer sp_buffer, const Array<PrimExpr>& indices) {
    int ndim = sp_buffer->ndim();
    int n_lower = static_cast<int>(indices.size());
    ICHECK_LE(n_lower, ndim);

    PrimExpr lowered_index = Integer(0);

    for (int i = 0; i < n_lower; ++i) {
      const Axis& axis = sp_buffer->axes[i];
      const PrimExpr& index = indices[i];

      // Stage 1. Get the sparse index.
      SpIterVar sp_iter = collector_.GetSpIterFromIndex(index);
      PrimExpr sp_index{nullptr};

      PrimExpr l = AccumulateLowerIndex(lowered_index, sp_buffer, i, 0);
      PrimExpr r = AccumulateLowerIndex(add(lowered_index, 1), sp_buffer, i, 0);

      SpIterKind kind = sp_iter->kind;
      if (kind == SpIterKind::kDenseFixed) {
        CHECK(!axis->IsInstance<DenseVariableAxisNode>());
        if (const auto* df_axis = axis.as<DenseFixedAxisNode>()) {
          CHECK(ana_.CanProveEqual(sp_iter->max_extent, df_axis->length));
          sp_index = sp_iter;
        } else {
          Var buffer_var;
          if (const auto* sf_axis = axis.as<SparseFixedAxisNode>()) {
            CHECK(ana_.CanProveEqual(sp_iter->max_extent, sf_axis->length));
            buffer_var = sf_axis->indices->data;
          } else if (const auto* sv_axis = axis.as<SparseVariableAxisNode>()) {
            CHECK(ana_.CanProveEqual(sp_iter->max_extent, sv_axis->length));
            buffer_var = sv_axis->indices->data;
          } else {
            LOG(FATAL) << "Cannot reach here";
          }
          sp_index = lower_bound(buffer_var, index, std::move(l), std::move(r));
        }
      } else if (kind == SpIterKind::kDenseVariable) {
        const auto* dv_axis = axis.as<DenseVariableAxisNode>();
        CHECK(dv_axis != nullptr);
        CHECK(sp_iter->axis.defined());
        sp_index = sp_iter;
      } else if (kind == SpIterKind::kSparseFixed) {
        CHECK(!axis->IsInstance<DenseVariableAxisNode>());
        CHECK(sp_iter->axis.defined());
        const Axis& iterated_axis = sp_iter->axis;
        if (axis->IsInstance<DenseFixedAxisNode>()) {
          sp_index = GetDenseValue(sp_iter);
        } else if (const auto* sf_axis = axis.as<SparseFixedAxisNode>()) {
          if (iterated_axis.get() == sf_axis) {
            sp_index = sp_iter;
          } else {
            sp_index = lower_bound(sf_axis->indices->data, GetDenseValue(sp_iter), std::move(l),
                                   std::move(r));
          }
        } else if (const auto* sv_axis = axis.as<SparseVariableAxisNode>()) {
          sp_index = lower_bound(sv_axis->indices->data, GetDenseValue(sp_iter), std::move(l),
                                 std::move(r));
        } else {
          LOG(FATAL) << "Cannot reach here";
        }
      } else {
        CHECK(kind == SpIterKind::kSparseVariable);
        CHECK(!axis->IsInstance<DenseVariableAxisNode>());
        CHECK(sp_iter->axis.defined());
        const Axis& iterated_axis = sp_iter->axis;
        if (const auto* df_axis = axis.as<DenseFixedAxisNode>()) {
          CHECK(ana_.CanProveEqual(sp_iter->max_extent, df_axis->length));
          sp_index = GetDenseValue(sp_iter);
        } else if (const auto* sf_axis = axis.as<SparseFixedAxisNode>()) {
          CHECK(ana_.CanProveEqual(sp_iter->max_extent, sf_axis->length));
          sp_index = lower_bound(sf_axis->indices->data, GetDenseValue(sp_iter), std::move(l),
                                 std::move(r));
        } else if (const auto* sv_axis = axis.as<SparseVariableAxisNode>()) {
          CHECK(ana_.CanProveEqual(sp_iter->max_extent, sv_axis->length));
          if (iterated_axis.get() == sv_axis) {
            sp_index = sp_iter;
          } else {
            sp_index = lower_bound(sv_axis->indices->data, GetDenseValue(sp_iter), std::move(l),
                                   std::move(r));
          }
        } else {
          LOG(FATAL) << "Cannot reach here";
        }
      }

      // Stage 2. Accumulate the lowered index.
      lowered_index =
          AccumulateLowerIndex(std::move(lowered_index), sp_buffer, i, std::move(sp_index));
    }

    return lowered_index;
  }

  PrimExpr AccumulateLowerIndex(PrimExpr prev_lowered_index, const SparseBuffer& sp_buffer, int dim,
                                PrimExpr index) {
    const Axis& axis = sp_buffer->axes[dim];
    if (axis->IsInstance<DenseFixedAxisNode>()) {
      return ana_.Simplify(std::move(prev_lowered_index) * axis->length + std::move(index));
    } else if (const auto* sf_axis = axis.as<SparseFixedAxisNode>()) {
      return ana_.Simplify(std::move(prev_lowered_index) * sf_axis->num_cols + std::move(index));
    } else if (const auto* dv_axis = axis.as<DenseVariableAxisNode>()) {
      return ana_.Simplify(
          add(BufferLoad(dv_axis->indptr, {std::move(prev_lowered_index)}), std::move(index)));
    } else if (const auto* sv_axis = axis.as<SparseVariableAxisNode>()) {
      return ana_.Simplify(
          add(BufferLoad(sv_axis->indptr, {std::move(prev_lowered_index)}), std::move(index)));
    }
    LOG(FATAL) << "Cannot reach here";
    throw;
  }

  PrimExpr GetDenseValue(SpIterVar sp_iter) {
    SpIterKind kind = sp_iter->kind;
    CHECK(kind == SpIterKind::kSparseFixed || kind == SpIterKind::kSparseVariable);
    Axis iterated_axis = sp_iter->axis;

    SparseBuffer iterated_buffer{nullptr};
    Array<PrimExpr> iters{nullptr};

    collector_.GetIteratedBufferAndDependentIters(sp_iter, &iterated_buffer, &iters);
    iters.push_back(sp_iter);
    PrimExpr lowered_indices = LowerIndices(std::move(iterated_buffer), iters);

    if (kind == SpIterKind::kSparseFixed) {
      return BufferLoad(Downcast<SparseFixedAxis>(iterated_axis)->indices,
                        {std::move(lowered_indices)});
    } else {
      return BufferLoad(Downcast<SparseVariableAxis>(iterated_axis)->indices,
                        {std::move(lowered_indices)});
    }
  }

  PrimExpr VisitExpr_(const SparseBufferLoadNode* load) final {
    buffer_read_.insert(load->buffer.get());
    PrimExpr lowered_indices = LowerIndices(load->buffer, load->indices);
    return BufferLoad(load->buffer->data, {std::move(lowered_indices)});
  }

  Stmt VisitStmt_(const SparseBufferStoreNode* store) final {
    buffer_write_.insert(store->buffer.get());
    PrimExpr value = ExprMutator::VisitExpr(store->value);
    PrimExpr lowered_indices = LowerIndices(store->buffer, store->indices);
    return BufferStore(store->buffer->data, std::move(value), {std::move(lowered_indices)});
  }

  Stmt VisitStmt_(const SparseBlockNode* sp_block) {
    int n_iter = static_cast<int>(sp_block->sp_iter_vars.size());
    buffer_read_.clear();
    buffer_write_.clear();

    // Step 1. Recursively mutate the `init` field and the block body.
    Optional<Stmt> init =
        sp_block->init.defined() ? VisitStmt(sp_block->init.value()) : Optional<Stmt>(NullOpt);
    Stmt body = VisitStmt(sp_block->body);

    // Step 2. Create the new outer loop vars.
    Array<Var> loop_vars;
    std::unordered_map<const VarNode*, PrimExpr> var_map;
    loop_vars.reserve(n_iter);
    var_map.reserve(n_iter);
    for (const SpIterVar& sp_iter : sp_block->sp_iter_vars) {
      Var loop_var("v_" + sp_iter->var->name_hint);
      loop_vars.push_back(loop_var);
      var_map[sp_iter->var.get()] = loop_var;
    }

    // Step 3. Create block iters and iter bindings.
    Array<IterVar> block_iters;
    Array<PrimExpr> iter_bindings;
    block_iters.reserve(n_iter);
    iter_bindings.reserve(n_iter);
    for (int i = 0; i < n_iter; ++i) {
      block_iters.push_back(SpIterVar2IterVar(sp_block->sp_iter_vars[i], var_map));
      iter_bindings.push_back(loop_vars[i]);
    }

    // Step 4. Generate the read-region and write-retion of the block.
    Array<BufferRegion> reads{nullptr};
    Array<BufferRegion> writes{nullptr};
    GenerateReadWriteRegions(sp_block, &reads, &writes);

    // Step 5. Create the block and block-realize
    Block block(block_iters, std::move(reads), std::move(writes), sp_block->name, std::move(body),
                std::move(init));
    BlockRealize block_realize(std::move(iter_bindings), const_true(), std::move(block));

    // Step 6. Create outer loops and the block binding.
    Stmt loop = GenerateLoops(std::move(block_realize), block_iters, loop_vars);

    return loop;
  }

  IterVar SpIterVar2IterVar(const SpIterVar& sp_iter,
                            const std::unordered_map<const VarNode*, PrimExpr>& var_map) {
    PrimExpr extent{nullptr};

    SpIterKind kind = sp_iter->kind;
    if (kind == SpIterKind::kDenseFixed || kind == SpIterKind::kSparseFixed) {
      extent = sp_iter->max_extent;
    } else {
      SparseBuffer iterated_buffer{nullptr};
      Array<PrimExpr> dependent_iters{nullptr};
      collector_.GetIteratedBufferAndDependentIters(sp_iter, &iterated_buffer, &dependent_iters);
      PrimExpr lowered_indices = LowerIndices(std::move(iterated_buffer), dependent_iters);

      Buffer indptr{kind == SpIterKind::kDenseVariable
                        ? Downcast<DenseVariableAxis>(sp_iter->axis)->indptr
                        : Downcast<SparseVariableAxis>(sp_iter->axis)->indptr};
      PrimExpr l = BufferLoad(indptr, {lowered_indices});
      PrimExpr r = BufferLoad(indptr, {add(lowered_indices, 1)});
      extent = sub(r, l);
    }

    // Substitute the iteration vars in the expression with the loop vars.
    return IterVar(Range::FromMinExtent(0, Substitute(std::move(extent), var_map)), sp_iter->var,
                   sp_iter->is_reduction ? kCommReduce : kDataPar);
  }

  void GenerateReadWriteRegions(const SparseBlockNode* sp_block, Array<BufferRegion>* reads,
                                Array<BufferRegion>* writes) {
    for (const ObjectRef& obj : sp_block->sp_structs) {
      if (const auto* dv_axis = obj.as<DenseVariableAxisNode>()) {
        reads->push_back(BufferRegion::FullRegion(dv_axis->indptr));
      } else if (const auto* sf_axis = obj.as<SparseFixedAxisNode>()) {
        reads->push_back(BufferRegion::FullRegion(sf_axis->indices));
      } else if (const auto* sv_axis = obj.as<SparseVariableAxisNode>()) {
        reads->push_back(BufferRegion::FullRegion(sv_axis->indptr));
        reads->push_back(BufferRegion::FullRegion(sv_axis->indices));
      } else if (const auto* sp_buffer = obj.as<SparseBufferNode>()) {
        if (buffer_read_.count(sp_buffer)) {
          reads->push_back(BufferRegion::FullRegion(sp_buffer->data));
        }
        if (buffer_write_.count(sp_buffer)) {
          writes->push_back(BufferRegion::FullRegion(sp_buffer->data));
        }
      }
    }
  }

  Stmt GenerateLoops(Stmt body, const Array<IterVar>& block_iters, const Array<Var>& loop_vars) {
    int n_iter = static_cast<int>(block_iters.size());
    for (int i = n_iter - 1; i >= 0; --i) {
      const Range& dom = block_iters[i]->dom;
      body = For(loop_vars[i], dom->min, dom->extent, ForKind::kSerial, std::move(body));
    }
    return body;
  }

  AccessAndDependencyCollector collector_;
  arith::Analyzer ana_;
  std::unordered_set<const SparseBufferNode*> buffer_read_;
  std::unordered_set<const SparseBufferNode*> buffer_write_;
};

Stmt WrapWithRootBlock(Stmt body) {
  Block root_block({}, {}, {}, "root", std::move(body));
  body = BlockRealize({}, const_true(), std::move(root_block));
  return Stmt(body);
}

PrimFunc LowerSparseTIR(PrimFunc f) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    // Step 1. Update the PrimFunc's buffer map.
    fptr->buffer_map = UpdateBufferMap(f);
    // Step 2. Collect buffer access information and dependency.
    AccessAndDependencyCollector collector;
    collector.Collect(f->body);
    // Step 3. Lower indices.
    fptr->body = IndexTransformer(collector)(std::move(f->body));
    // Step 4. Wrap the function body with a root block.
    fptr->body = WrapWithRootBlock(std::move(fptr->body));
    return f;
  } else {
    return f;
  }
}

namespace transform {

Pass LowerSparseTIR() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LowerSparseTIR(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerSparseTIR", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerSparseTIR").set_body_typed(LowerSparseTIR);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
