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
            sp_iter->kind == SpIterKind::kDenseVariable ||
            !BufferContainsAxis(buffer, sp_iter->axis)) {
          continue;
        }

        ICHECK(dependency_map_.count(sp_iter) == 0);
        dependency_map_[sp_iter] = std::make_pair(buffer, k);
      }
    }
  }

  BufferAccessMap buffer_access_map_;
  DependencyMap dependency_map_;

 private:
  void AddAccessPattern(const SparseBuffer& buffer, const Array<PrimExpr>& indices) {
    int ndim = buffer->ndim();
    CHECK_EQ(static_cast<int>(indices.size()), ndim);

    Array<SpIterVar> iters;
    iters.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      const SpIterVarNode* sp_iter = indices[i].as<SpIterVarNode>();
      CHECK(sp_iter) << "ValueError: Currently an index is only allowed to be SpIterVar";
      iters.push_back(GetRef<SpIterVar>(sp_iter));
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

  void VisitStmt_(const SparseBufferStoreNode* store) final {
    ExprVisitor::VisitExpr(store->value);
    AddAccessPattern(store->buffer, store->indices);
  }

  void VisitExpr_(const SparseBufferLoadNode* load) final {
    AddAccessPattern(load->buffer, load->indices);
  }
};

class IndexTransformer : public StmtExprMutator {
 public:
  explicit IndexTransformer(BufferAccessMap buffer_access_map, DependencyMap dependency_map)
      : buffer_access_map_(std::move(buffer_access_map)),
        dependency_map_(std::move(dependency_map)) {}

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
      const auto* sp_iter = index.as<SpIterVarNode>();
      PrimExpr sp_index{nullptr};
      CHECK(sp_iter) << "ValueError: Currently an index is only allowed to be SpIterVar";

      PrimExpr l = AccumulateLowerIndex(lowered_index, sp_buffer, i, 0);
      PrimExpr r = AccumulateLowerIndex(add(lowered_index, 1), sp_buffer, i, 0);

      SpIterKind kind = sp_iter->kind;
      if (kind == SpIterKind::kDenseFixed) {
        CHECK(!axis->IsInstance<DenseVariableAxisNode>());
        if (const auto* df_axis = axis.as<DenseFixedAxisNode>()) {
          CHECK(ana_.CanProveEqual(sp_iter->max_extent, df_axis->length));
          sp_index = GetRef<SpIterVar>(sp_iter);
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
        sp_index = GetRef<SpIterVar>(sp_iter);
      } else if (kind == SpIterKind::kSparseFixed) {
        CHECK(!axis->IsInstance<DenseVariableAxisNode>());
        CHECK(sp_iter->axis.defined());
        const Axis& iterated_axis = sp_iter->axis;
        if (const auto* df_axis = axis.as<DenseFixedAxisNode>()) {
          CHECK(ana_.CanProveEqual(sp_iter->max_extent, df_axis->length));
          sp_index = GetDenseValue(sp_iter);
        } else if (const auto* sf_axis = axis.as<SparseFixedAxisNode>()) {
          CHECK(ana_.CanProveEqual(sp_iter->max_extent, sf_axis->length));
          if (iterated_axis.get() == sf_axis) {
            sp_index = GetRef<SpIterVar>(sp_iter);
          } else {
            sp_index = lower_bound(sf_axis->indices->data, GetDenseValue(sp_iter), std::move(l),
                                   std::move(r));
          }
        } else if (const auto* sv_axis = axis.as<SparseVariableAxisNode>()) {
          CHECK(ana_.CanProveEqual(sp_iter->max_extent, sv_axis->length));
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
            sp_index = GetRef<SpIterVar>(sp_iter);
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
    if (axis->IsInstance<DenseFixedAxisNode>() || axis->IsInstance<SparseFixedAxisNode>()) {
      return ana_.Simplify(std::move(prev_lowered_index) * axis->length + std::move(index));
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

  PrimExpr GetDenseValue(const SpIterVarNode* sp_iter) {
    SpIterKind kind = sp_iter->kind;
    CHECK(kind == SpIterKind::kSparseFixed || kind == SpIterKind::kSparseVariable);
    Axis iterated_axis = sp_iter->axis;

    std::pair<SparseBuffer, int> dependent_pair = dependency_map_[GetRef<SpIterVar>(sp_iter)];
    Array<SpIterVar> buffer_access_iters = buffer_access_map_[dependent_pair.first];
    int n_dependent = dependent_pair.second;

    Array<PrimExpr> dependent_iters{buffer_access_iters.begin(),
                                    buffer_access_iters.begin() + n_dependent};
    PrimExpr lowered_indices = LowerIndices(dependent_pair.first, dependent_iters);

    if (kind == SpIterKind::kSparseFixed) {
      return BufferLoad(Downcast<SparseFixedAxis>(iterated_axis)->indices,
                        {std::move(lowered_indices)});
    } else {
      return BufferLoad(Downcast<SparseVariableAxis>(iterated_axis)->indices,
                        {std::move(lowered_indices)});
    }
  }

  PrimExpr VisitExpr_(const SparseBufferLoadNode* load) final {
    PrimExpr lowered_indices = LowerIndices(load->buffer, load->indices);
    return BufferLoad(load->buffer->data, {std::move(lowered_indices)});
  }

  Stmt VisitStmt_(const SparseBufferStoreNode* store) final {
    PrimExpr value = ExprMutator::VisitExpr(store->value);
    PrimExpr lowered_indices = LowerIndices(store->buffer, store->indices);
    return BufferStore(store->buffer->data, std::move(value), {std::move(lowered_indices)});
  }

  BufferAccessMap buffer_access_map_;
  DependencyMap dependency_map_;
  arith::Analyzer ana_;
};

PrimFunc LowerSparseTIR(PrimFunc f) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    AccessAndDependencyCollector collector;
    collector.Collect(f->body);
    fptr->body = IndexTransformer(collector.buffer_access_map_,
                                  collector.dependency_map_)(std::move(f->body));
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
