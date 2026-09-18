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
#include <tvm/ir/prim/expr.h>
#include <tvm/tirx/stmt.h>

#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace tvm {
namespace tirx {
namespace detail {

inline bool IsSeqStmtNoOp(ffi::AnyView stmt) {
  const auto* evaluate = stmt.as<EvaluateNode>();
  const auto* value = evaluate == nullptr ? nullptr : evaluate->value.as<prim::IntImmNode>();
  return value != nullptr && value->value == 0;
}

// Move the non-None slots to the front, preserving order; return the compacted size.
inline size_t SeqStmtCompactNop(ffi::Any* begin, size_t current_size) {
  size_t write = 0;
  for (size_t read = 0; read < current_size; ++read) {
    if (begin[read].type_index() == ffi::TypeIndex::kTVMFFINone) {
      continue;
    }
    if (write != read) {
      begin[write] = std::move(begin[read]);
    }
    ++write;
  }
  return write;
}

// Expand each nested SeqStmt in [0, current_size) into its final position in [0, target_size).
// Walk backward so a destination is never below its source; requires every nested SeqStmt to be
// non-empty and target_size slots to exist.
inline void SeqStmtExpandNested(ffi::Any* begin, size_t current_size, size_t target_size) {
  size_t write = target_size;
  for (size_t read = current_size; read-- > 0;) {
    if (const auto* nested = begin[read].as<SeqStmtNode>()) {
      for (size_t i = nested->seq.size(); i-- > 0;) {
        begin[--write] = ffi::Any(nested->seq[i]);
      }
    } else {
      --write;
      if (write != read) {
        begin[write] = std::move(begin[read]);
      }
    }
  }
}

template <typename MutateElement>
auto MutateSeqStmtChanged(MutateElement& mutate_element, const SeqStmtNode* self, size_t index,
                          Stmt mapped, ffi::InplaceMode inplace_mode)
    -> decltype(mutate_element(ffi::AnyView(), ffi::InplaceMode::kDisallow)) {
  const size_t size = self->seq.size();
  std::vector<Stmt> results;
  results.reserve(size);
  results.assign(self->seq.begin(), self->seq.begin() + index);
  auto append = [&](Stmt stmt) {
    if (IsSeqStmtNoOp(stmt)) {
      return;
    }
    if (const auto* nested = stmt.as<SeqStmtNode>()) {
      for (const Stmt& child : nested->seq) {
        results.emplace_back(child);
      }
    } else {
      results.emplace_back(std::move(stmt));
    }
  };
  append(std::move(mapped));
  for (size_t i = index + 1; i < size; ++i) {
    auto pending = mutate_element(self->seq[i], ffi::InplaceMode::kDisallow);
    ffi::UnchangedOr<Stmt> element = ffi::Unchanged();
    if constexpr (std::is_same_v<decltype(pending), ffi::Expected<ffi::UnchangedOr<Stmt>>>) {
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, value, std::move(pending));
      element = std::move(value);
    } else {
      element = std::move(pending);
    }
    append(element.IsUnchanged() ? self->seq[i] : std::move(element).ValueUnchecked());
  }
  if (results.empty()) {
    return Evaluate(0);
  }
  if (results.size() == 1) {
    return std::move(results[0]);
  }
  ffi::Array<Stmt> seq(std::make_move_iterator(results.begin()),
                       std::make_move_iterator(results.end()));
  if (inplace_mode == ffi::InplaceMode::kAllow) {
    const_cast<SeqStmtNode*>(self)->seq = std::move(seq);
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<SeqStmtNode> copy = ffi::make_object<SeqStmtNode>(*self);
  copy->seq = std::move(seq);
  return Stmt(std::move(copy));
}

template <typename MutateElement>
auto MutateSeqStmtRaw(MutateElement& mutate_element, const SeqStmtNode* self,
                      ffi::InplaceMode inplace_mode)
    -> decltype(mutate_element(ffi::AnyView(), ffi::InplaceMode::kDisallow)) {
  const size_t size = self->seq.size();
  for (size_t i = 0; i < size; ++i) {
    auto pending = mutate_element(self->seq[i], ffi::InplaceMode::kDisallow);
    ffi::UnchangedOr<Stmt> mapped = ffi::Unchanged();
    if constexpr (std::is_same_v<decltype(pending), ffi::Expected<ffi::UnchangedOr<Stmt>>>) {
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, value, std::move(pending));
      mapped = std::move(value);
    } else {
      mapped = std::move(pending);
    }
    if (!mapped.UnchangedOrSameAs(self->seq[i])) {
      return MutateSeqStmtChanged(mutate_element, self, i, std::move(mapped).ValueUnchecked(),
                                  inplace_mode);
    }
    if (IsSeqStmtNoOp(self->seq[i]) || self->seq[i].as<SeqStmtNode>()) {
      return MutateSeqStmtChanged(mutate_element, self, i, self->seq[i], inplace_mode);
    }
  }
  if (size == 0) return Evaluate(0);
  if (size == 1) return self->seq[0];
  return ffi::Unchanged();
}

template <typename MutateElement>
auto MaybeInplaceMutateSeqStmtRaw(MutateElement& mutate_element, const SeqStmtNode* op)
    -> decltype(mutate_element(ffi::AnyView(), ffi::InplaceMode::kDisallow)) {
  SeqStmtNode* self = const_cast<SeqStmtNode*>(op);
  // The engine establishes ownership of the SeqStmt, but seq is a field and needs its own check.
  if (!self->seq.unique()) {
    return MutateSeqStmtRaw(mutate_element, self, ffi::InplaceMode::kAllow);
  }
  ffi::ArrayObj* seq = self->seq.GetArrayObj();
  ffi::Any* slots = const_cast<ffi::Any*>(seq->begin());
  const size_t size = seq->size();
  size_t delete_count = 0;
  size_t nested_extra = 0;
  bool has_nested = false;

  // Pass 1: rebuild each element in place and classify the final slot contents.
  for (size_t i = 0; i < size; ++i) {
    auto pending = mutate_element(slots[i], ffi::InplaceMode::kAllow);
    ffi::UnchangedOr<Stmt> mapped = ffi::Unchanged();
    if constexpr (std::is_same_v<decltype(pending), ffi::Expected<ffi::UnchangedOr<Stmt>>>) {
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, value, std::move(pending));
      mapped = std::move(value);
    } else {
      mapped = std::move(pending);
    }
    if (!mapped.UnchangedOrSameAs(slots[i].cast<Stmt>())) {
      slots[i] = ffi::Any(std::move(mapped).ValueUnchecked());
    }
    if (IsSeqStmtNoOp(slots[i])) {
      slots[i] = ffi::Any();  // A None slot is the delete marker consumed by compaction.
      ++delete_count;
      continue;
    }
    if (const auto* nested = slots[i].as<SeqStmtNode>()) {
      has_nested = true;
      if (nested->seq.empty()) {
        slots[i] = ffi::Any();
        ++delete_count;
      } else {
        nested_extra += nested->seq.size() - 1;
      }
    }
  }

  const size_t final_size = size - delete_count + nested_extra;
  if (delete_count == 0 && nested_extra == 0 && !has_nested) {
    if (size == 0) return Evaluate(0);
    if (size == 1) return slots[0].cast<Stmt>();
    return ffi::Unchanged();
  }

  // Splice in place when the result fits: compact out None slots, resize once, then expand.
  if (final_size <= seq->SeqBaseObj::capacity()) {
    size_t compacted = SeqStmtCompactNop(slots, size);
    seq->resize(final_size);
    SeqStmtExpandNested(slots, compacted, final_size);
  } else {
    // Splice beyond capacity: flatten into one exact-size array with pointer moves.
    std::vector<Stmt> results;
    results.reserve(final_size);
    for (size_t i = 0; i < size; ++i) {
      if (slots[i].type_index() == ffi::TypeIndex::kTVMFFINone) {
        continue;
      }
      if (const auto* nested = slots[i].as<SeqStmtNode>()) {
        for (const Stmt& child : nested->seq) {
          results.emplace_back(child);
        }
      } else {
        results.emplace_back(
            ffi::details::AnyUnsafe::MoveFromAnyAfterCheck<Stmt>(std::move(slots[i])));
      }
    }
    self->seq = ffi::Array<Stmt>(std::make_move_iterator(results.begin()),
                                 std::make_move_iterator(results.end()));
    seq = self->seq.GetArrayObj();
  }

  if (final_size == 0) {
    return Evaluate(0);
  }
  if (final_size == 1) {
    return seq->begin()[0].cast<Stmt>();
  }
  return ffi::Unchanged();
}

template <typename MutateElement>
auto MutateSeqStmt(const SeqStmtNode* self, ffi::InplaceMode inplace_mode,
                   MutateElement mutate_element)
    -> decltype(mutate_element(ffi::AnyView(), ffi::InplaceMode::kDisallow)) {
  if (inplace_mode == ffi::InplaceMode::kAllow) {
    return MaybeInplaceMutateSeqStmtRaw(mutate_element, self);
  }
  return MutateSeqStmtRaw(mutate_element, self, inplace_mode);
}

}  // namespace detail
}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_IR_SEQ_STMT_MUTATE_H_
