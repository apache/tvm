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

#ifndef TVM_TIRX_IR_SEQ_STMT_MUTATE_H_
#define TVM_TIRX_IR_SEQ_STMT_MUTATE_H_

#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/tirx/stmt.h>

#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace tvm {
namespace tirx {
namespace detail {

inline bool IsSeqStmtNoOp(ffi::AnyView stmt) {
  if (stmt.type_index() == ffi::TypeIndex::kTVMFFINone) return true;
  const auto* evaluate = stmt.as<EvaluateNode>();
  const auto* value = evaluate == nullptr ? nullptr : evaluate->value.as<IntImmNode>();
  return value != nullptr && value->value == 0;
}

inline size_t SeqStmtFlattenedSize(ffi::AnyView stmt) {
  if (IsSeqStmtNoOp(stmt)) return 0;
  if (const auto* nested = stmt.as<SeqStmtNode>()) {
    size_t size = 0;
    for (const Stmt& child : nested->seq) size += SeqStmtFlattenedSize(child);
    return size;
  }
  return 1;
}

inline void SeqStmtAppendFlattened(Stmt stmt, std::vector<Stmt>* output) {
  if (IsSeqStmtNoOp(stmt)) return;
  if (const auto* nested = stmt.as<SeqStmtNode>()) {
    for (const Stmt& child : nested->seq) SeqStmtAppendFlattened(child, output);
  } else {
    output->push_back(std::move(stmt));
  }
}

// Delete markers are None; compaction preserves the remaining slot order.
inline size_t SeqStmtCompactNop(ffi::Any* slots, size_t size) {
  size_t write = 0;
  for (size_t read = 0; read < size; ++read) {
    if (slots[read].type_index() == ffi::TypeIndex::kTVMFFINone) continue;
    if (write != read) slots[write] = std::move(slots[read]);
    ++write;
  }
  return write;
}

inline void SeqStmtExpandOne(Stmt stmt, ffi::Any* slots, size_t* write) {
  if (IsSeqStmtNoOp(stmt)) return;
  if (const auto* nested = stmt.as<SeqStmtNode>()) {
    // Keep stmt alive when an expanded child overwrites its original owning slot.
    for (size_t i = nested->seq.size(); i-- > 0;) {
      SeqStmtExpandOne(nested->seq[i], slots, write);
    }
  } else {
    slots[--*write] = ffi::Any(std::move(stmt));
  }
}

inline void SeqStmtExpandNested(ffi::Any* slots, size_t size, size_t final_size) {
  size_t write = final_size;
  for (size_t read = size; read-- > 0;) {
    SeqStmtExpandOne(slots[read].cast<Stmt>(), slots, &write);
  }
}

// The callable returns UnchangedOr<Stmt> on native frames, or its Expected form for
// structural hooks. Only the latter instantiation contains Expected error handling.
template <typename MutateElement>
auto MutateSeqStmt(const SeqStmtNode* self, ffi::InplaceMode inplace_mode,
                   MutateElement mutate_element)
    -> decltype(mutate_element(ffi::AnyView(), ffi::InplaceMode::kDisallow)) {
  using Result = decltype(mutate_element(ffi::AnyView(), ffi::InplaceMode::kDisallow));
  constexpr bool kExpected = std::is_same_v<Result, ffi::Expected<ffi::UnchangedOr<Stmt>>>;
  const bool inplace_array = inplace_mode == ffi::InplaceMode::kAllow && self->seq.unique();
  auto* array = self->seq.GetArrayObj();
  auto* slots = const_cast<ffi::Any*>(array->begin());
  const size_t size = array->size();
  std::vector<Stmt> flattened;
  if (!inplace_array) flattened.reserve(size);
  bool changed = false;
  bool normalize = size <= 1;
  size_t final_size = 0;

  for (size_t i = 0; i < size; ++i) {
    auto pending = mutate_element(
        slots[i], inplace_array ? ffi::InplaceMode::kAllow : ffi::InplaceMode::kDisallow);
    ffi::UnchangedOr<Stmt> mapped = ffi::Unchanged();
    if constexpr (kExpected) {
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, result, std::move(pending));
      mapped = std::move(result);
    } else {
      mapped = std::move(pending);
    }
    bool same = mapped.UnchangedOrSameAs(slots[i].cast<Stmt>());
    changed |= !same;
    if (inplace_array) {
      if (!same) slots[i] = ffi::Any(std::move(mapped).ValueUnchecked());
      size_t count = SeqStmtFlattenedSize(slots[i]);
      normalize |= count == 0 || slots[i].as<SeqStmtNode>() != nullptr;
      final_size += count;
      if (count == 0) slots[i] = ffi::Any();
    } else {
      Stmt stmt = std::move(mapped).ValueOrUnchanged(slots[i].cast<Stmt>());
      normalize |= IsSeqStmtNoOp(stmt) || stmt.as<SeqStmtNode>() != nullptr;
      SeqStmtAppendFlattened(std::move(stmt), &flattened);
    }
  }

  if (!normalize && (!changed || inplace_array)) return ffi::Unchanged();
  if (inplace_array) {
    if (final_size <= array->SeqBaseObj::capacity()) {
      size_t compacted = SeqStmtCompactNop(slots, size);
      array->resize(final_size);
      SeqStmtExpandNested(slots, compacted, final_size);
    } else {
      flattened.reserve(final_size);
      for (size_t i = 0; i < size; ++i) {
        if (slots[i].type_index() != ffi::TypeIndex::kTVMFFINone)
          SeqStmtAppendFlattened(slots[i].cast<Stmt>(), &flattened);
      }
    }
  } else {
    final_size = flattened.size();
  }

  if (final_size == 0) return Evaluate(0);
  if (final_size == 1) {
    return flattened.empty() ? slots[0].cast<Stmt>() : std::move(flattened[0]);
  }
  if (inplace_array && flattened.empty()) return ffi::Unchanged();
  ffi::Array<Stmt> seq(std::make_move_iterator(flattened.begin()),
                       std::make_move_iterator(flattened.end()));
  if (inplace_mode == ffi::InplaceMode::kAllow) {
    const_cast<SeqStmtNode*>(self)->seq = std::move(seq);
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<SeqStmtNode>(*self);
  copy->seq = std::move(seq);
  return Stmt(std::move(copy));
}

}  // namespace detail
}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_IR_SEQ_STMT_MUTATE_H_
