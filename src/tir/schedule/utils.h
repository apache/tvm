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
#ifndef TVM_TIR_SCHEDULE_UTILS_H_
#define TVM_TIR_SCHEDULE_UTILS_H_

#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_set.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/schedule/instruction.h>
#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/schedule/state.h>
#include <tvm/tir/schedule/trace.h>
#include <tvm/tir/stmt_functor.h>

#include <random>
#include <unordered_map>
#include <utility>

#include "../../node/attr_registry.h"
#include "../../printer/text_printer.h"
#include "../../runtime/thread_storage_scope.h"
#include "../../support/array.h"
#include "./analysis.h"
#include "./error.h"
#include "./instruction_traits.h"
#include "./primitive.h"

namespace tvm {
namespace tir {

/*!
 * \brief A helper macro to convert an sref to the statement it points to,
 * then check if the downcasting succeeded.
 * \param Result The result variable, used for checking
 * \param SRef The SRef to be casted
 * \param Type The type to be casted to, can be Block or For
 */
#define TVM_SREF_AS_OR_ERR(Result, SRef, Type) \
  SRef->StmtAs<Type>();                        \
  ICHECK(Result)

/*!
 * \brief A helper macro to convert an sref to the block it points to,
 * throwing an internal error if downcasting fails
 * \param Result The result variable, used for checking
 * \param SRef The SRef to be casted
 */
#define TVM_SREF_TO_BLOCK(Result, SRef)                   \
  TVM_SREF_AS_OR_ERR(Result, SRef, ::tvm::tir::BlockNode) \
      << "TypeError: Expects StmtSRef `" << #SRef         \
      << "` points to `Block`, but gets: " << (SRef->stmt ? SRef->stmt->GetTypeKey() : "None")

/*!
 * \brief A helper macro to convert an sref to the for-loop it points to,
 * throwing an internal error if downcasting fails
 * \param Result The name of the result variable, used for checking
 * \param SRef The SRef to be casted
 */
#define TVM_SREF_TO_FOR(Result, SRef)                   \
  TVM_SREF_AS_OR_ERR(Result, SRef, ::tvm::tir::ForNode) \
      << "TypeError: Expects StmtSRef `" << #SRef       \
      << "` points to `Loop`, but gets: " << (SRef->stmt ? SRef->stmt->GetTypeKey() : "None")

/*!
 * \brief Downcast a TVM ObjectRef to its corresponding container using `ObjectRef::as<Type>`,
 * then check if the downcasting succeeded.
 * \param Result The result variable, used for checking
 * \param From The ObjectRef to be downcasted
 * \param Type The type to be downcasted to
 */
#define TVM_TYPE_AS_OR_ERR(Result, From, Type) \
  From.as<Type>();                             \
  ICHECK(Result)

/*!
 * \brief Downcast a TVM ObjectRef to its corresponding container using `ObjectRef::as<Type>`,
 * throwing an internal error if downcast fails.
 * \param Result The result variable, used for checking
 * \param From The ObjectRef to be downcasted
 * \param Type The type to be downcasted to
 */
#define TVM_TYPE_AS(Result, From, Type)                                           \
  TVM_TYPE_AS_OR_ERR(Result, From, Type)                                          \
      << "TypeError: Expects `" << #From << "` to have type `" << Type::_type_key \
      << "`, but gets: " << (From.defined() ? From->GetTypeKey() : "None")

/*!
 * \brief Convert an array of loop StmtSRefs to an array of loops
 * \param loop_srefs The loop StmtSRefs to be converted
 * \return The conversion result loops
 */
inline Array<For> LoopSRefs2Loops(const Array<StmtSRef>& loop_srefs) {
  Array<For> loops;
  loops.reserve(loop_srefs.size());
  for (StmtSRef loop_sref : loop_srefs) {
    const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
    loops.push_back(GetRef<For>(loop));
  }
  return loops;
}

/*!
 * \brief Convert a tvm::runtime::Array to std::vector
 * \tparam TSrc The type of elements in the source Array
 * \tparam TDst The type of elements in the result vector
 * \return The result vector
 */
template <class TSrc, class TDst>
std::vector<TDst> AsVector(const Array<TSrc>& vec);

/******** Storage scope ********/

/*!
 * \brief Determine if iterators of a storage scope should be relaxed
 * under a specific thread scope
 * \param storage_scope The storage scope that the iterators are on
 * \param thread_scope The thread scope to be relaxed
 * \return A boolean indicating the result
 */
inline bool CanRelaxStorageUndereThread(const runtime::StorageScope& storage_scope,
                                        const runtime::ThreadScope& thread_scope) {
  if (storage_scope.rank == runtime::StorageRank::kWarp) {
    // for warp memory, we only relax threadIdx.x
    return thread_scope.rank == 1 && thread_scope.dim_index == 0;
  }
  return static_cast<int>(storage_scope.rank) <= static_cast<int>(thread_scope.rank);
}

/******** SeqStmt ********/

/*!
 * \brief Remove a specific Stmt from a SeqStmt. If a SeqStmt contains a BlockRealize,
 * whose block is the Stmt to be removed, then remove that BlockRealize too.
 * \param seq The SeqStmt to be removed from
 * \param to_remove The Stmt to be removed
 * \return The removal result
 */
inline Stmt RemoveFromSeqStmt(const SeqStmt& seq, const Stmt& to_remove) {
  ICHECK_GT(seq->size(), 1);
  Array<Stmt> new_stmts;
  new_stmts.reserve(seq->size());
  for (const Stmt& stmt : seq->seq) {
    if (to_remove.same_as(stmt)) {
      continue;
    }
    if (const auto* realize = stmt.as<BlockRealizeNode>()) {
      if (to_remove.same_as(realize->block)) {
        continue;
      }
    }
    new_stmts.push_back(stmt);
  }
  return SeqStmt::Flatten(new_stmts);
}

/*!
 * \brief Create a new IterVar for the input For loop, with specified name and type
 * \param loop The loop to be created from
 * \param name The name of the new IterVar
 * \param iter_var_type The type of the new IterVar
 * \return The newly created IterVar
 */
inline IterVar IterVarFromLoop(const For& loop, String name, IterVarType iter_var_type) {
  return IterVar(Range::FromMinExtent(loop->min, loop->extent),
                 Var(std::move(name), loop->loop_var.dtype()), iter_var_type);
}

/******** Integer set ********/

/*!
 * \brief Converts the Ranges to IntSets
 * \param var_dom The ranges of variables
 * \return The integer sets of the variables
 */
inline Map<Var, arith::IntSet> AsIntSet(const Map<Var, Range>& var_dom) {
  std::unordered_map<Var, arith::IntSet, ObjectPtrHash, ObjectPtrEqual> result;
  result.reserve(var_dom.size());
  for (auto kv : var_dom) {
    Var& var = kv.first;
    Range& range = kv.second;
    result.emplace(std::move(var), arith::IntSet::FromRange(std::move(range)));
  }
  return {result.begin(), result.end()};
}

/**************** AsVector<TSrc, TDst> ****************/

namespace details {

template <class TSrc, class TDst>
struct AsVectorImpl {};

template <class TSrc>
struct AsVectorImpl<TSrc, TSrc> {
  inline std::vector<TSrc> operator()(const Array<TSrc>& vec) const {
    return std::vector<TSrc>(vec.begin(), vec.end());
  }
};

template <class TSrcObjectRef>
struct AsVectorImpl<TSrcObjectRef, int> {
  inline std::vector<int> operator()(const Array<TSrcObjectRef>& vec) const {
    std::vector<int> results;
    for (const TSrcObjectRef& x : vec) {
      const auto* n = x.template as<IntImmNode>();
      ICHECK(n) << "TypeError: Expects IntImm, but gets: " << x->GetTypeKey();
      results.push_back(n->value);
    }
    return results;
  }
};

template <class TSrcObjectRef>
struct AsVectorImpl<TSrcObjectRef, int64_t> {
  inline std::vector<int64_t> operator()(const Array<TSrcObjectRef>& vec) const {
    std::vector<int64_t> results;
    for (const TSrcObjectRef& x : vec) {
      const auto* n = x.template as<IntImmNode>();
      ICHECK(n) << "TypeError: Expects IntImm, but gets: " << x->GetTypeKey();
      results.push_back(n->value);
    }
    return results;
  }
};

template <class TSrcObjectRef>
struct AsVectorImpl<TSrcObjectRef, double> {
  inline std::vector<double> operator()(const Array<TSrcObjectRef>& array) const {
    std::vector<double> results;
    for (const TSrcObjectRef& x : array) {
      const auto* n = x.template as<FloatImmNode>();
      ICHECK(n) << "TypeError: Expects FloatImm, but gets: " << x->GetTypeKey();
      results.push_back(n->value);
    }
    return results;
  }
};
}  // namespace details

template <class TSrc, class TDst>
inline std::vector<TDst> AsVector(const Array<TSrc>& vec) {
  return details::AsVectorImpl<TSrc, TDst>()(vec);
}

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_UTILS_H_
