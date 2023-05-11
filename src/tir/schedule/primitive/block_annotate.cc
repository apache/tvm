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
#include <tvm/tir/expr.h>

#include "../../transforms/ir_utils.h"
#include "../utils.h"

namespace tvm {
namespace tir {

class StorageAlignAxisOutOfRangeError : public ScheduleError {
 public:
  explicit StorageAlignAxisOutOfRangeError(IRModule mod, Buffer buffer, int axis)
      : mod_(std::move(mod)), buffer_(std::move(buffer)), axis_(axis) {}

  String FastErrorString() const final {
    return "ScheduleError: The input `axis` is out of range. It is required to be in range "
           "[-ndim, ndim) where `ndim` is the number of dimensions of the buffer to set "
           "storage alignment.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    int ndim = static_cast<int>(buffer_->shape.size());
    os << "The buffer to set storage alignment of, " << buffer_->name << ", has " << ndim
       << " dimension(s), so `axis` is required to be in [" << -(ndim) << ", " << ndim
       << ") for storage_align. However, the input `axis` is " << axis_
       << ", which is out of the expected range.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {}; }

  static int CheckAndUpdate(const IRModule& mod, const Buffer& buffer, int axis) {
    int ndim = static_cast<int>(buffer->shape.size());
    if (axis < -ndim || axis >= ndim) {
      throw StorageAlignAxisOutOfRangeError(mod, buffer, axis);
    }
    // If axis is negative, convert it to a non-negative one.
    if (axis < 0) {
      axis += ndim;
    }
    return axis;
  }

 private:
  IRModule mod_;
  Buffer buffer_;
  int axis_;
};

class NonAllocatedBufferError : public ScheduleError {
 public:
  explicit NonAllocatedBufferError(IRModule mod, Buffer buffer) : mod_(mod), buffer_(buffer) {}

  String FastErrorString() const final {
    return "ScheduleError: The input buffer is not allocated by a block. This means the buffer is "
           " either a function parameter or defined in `match_buffer` of a block.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The input buffer " << buffer_->name
       << " is not allocated by a block. This means the buffer is either a function parameter or "
          "defined in `match_buffer` of a block.";
    return os.str();
  }

  static StmtSRef CheckAndGetBufferAllocationSite(const IRModule& mod, const StmtSRef& block_sref,
                                                  const Buffer& buffer) {
    auto [defining_site_sref, is_alloc] = GetBufferDefiningSite(block_sref, buffer);
    if (!defining_site_sref.defined() || !is_alloc) {
      throw NonAllocatedBufferError(mod, buffer);
    }

    return defining_site_sref.value();
  }

  Array<ObjectRef> LocationsOfInterest() const final { return {}; }
  IRModule mod() const final { return mod_; }

 private:
  IRModule mod_;
  Buffer buffer_;
};

class StorageAlignInvalidFactorError : public ScheduleError {
 public:
  explicit StorageAlignInvalidFactorError(IRModule mod, int factor)
      : mod_(std::move(mod)), factor_(factor) {}

  String FastErrorString() const final {
    return "ScheduleError: The input `factor` of storage_align is expected to be a positive "
           "number.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The input `factor` of storage_align is expected to be a positive number. However, the "
          "input `factor` is "
       << factor_ << ", which is out of the expected range.";
    return os.str();
  }

  static void Check(const IRModule& mod, int factor) {
    if (factor <= 0) {
      throw StorageAlignInvalidFactorError(mod, factor);
    }
  }

  Array<ObjectRef> LocationsOfInterest() const final { return {}; }
  IRModule mod() const final { return mod_; }

 private:
  IRModule mod_;
  int factor_;
};

class StorageAlignInvalidAnnotationError : public ScheduleError {
 public:
  explicit StorageAlignInvalidAnnotationError(IRModule mod, Block block)
      : mod_(std::move(mod)), block_(std::move(block)) {}

  String FastErrorString() const final {
    return "ScheduleError: The block annotation for storage align is expected to be an array of "
           "4-integer-tuples (buffer_index, axis, factor, offset).";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The block annotation for storage align is expected to be an array of 4-integer-tuples "
          "(buffer_index, axis, factor, offset). However, the block annotation with key "
       << attr::buffer_dim_align << " of the block {0} is "
       << block_->annotations.at(attr::buffer_dim_align) << ", which is unexpected.";
    return os.str();
  }

  static StorageAlignAnnotation CheckAndGetAnnotation(const IRModule& mod, const Block& block) {
    // Get existing annotation value.
    auto it = block->annotations.find(attr::buffer_dim_align);
    if (it != block->annotations.end()) {
      if (!IsValidAnnotation(block, (*it).second)) {
        throw StorageAlignInvalidAnnotationError(mod, block);
      }
      return Downcast<StorageAlignAnnotation>((*it).second);
    }

    // Create new annotation value
    StorageAlignAnnotation storage_align_annotation;
    return storage_align_annotation;
  }

  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
  IRModule mod() const final { return mod_; }

 private:
  static bool IsValidAnnotation(const Block& block, const ObjectRef& anno_value) {
    if (!anno_value->IsInstance<ArrayNode>()) {
      return false;
    }
    auto storage_align_annotations = Downcast<Array<ObjectRef>>(anno_value);
    for (const ObjectRef& storage_align_annotation : storage_align_annotations) {
      if (!storage_align_annotation->IsInstance<ArrayNode>()) {
        return false;
      }
      auto storage_align_tuple = Downcast<Array<ObjectRef>>(storage_align_annotation);
      // Check if the annotation is a 4-tuple.
      if (storage_align_tuple.size() != 4) {
        return false;
      }
      for (const ObjectRef& tuple_element : storage_align_tuple) {
        if (!tuple_element->IsInstance<IntImmNode>()) {
          return false;
        }
      }
    }
    return true;
  }

  IRModule mod_;
  Block block_;
};

/*!
 * \brief A helper mutator which recursively mutates the old buffer's storage scope and collects
 * the block sref reuse information for the following replacement.
 */
class StorageScopeMutator : private ReplaceBufferMutator {
 public:
  /*!
   * \param allocate_site The block where `old_buffer` was allocated.
   * \param old_buffer The old buffer
   * \param storage_scope The storage scope to be set
   * \param block_sref_reuse The block sref reuse map to be updated
   * \return The new block after the mutation
   */
  static Block Mutate(const Block& allocate_site, const Buffer& old_buffer,
                      const String& storage_scope, Map<Block, Block>* block_sref_reuse) {
    Buffer new_buffer = WithScope(old_buffer, storage_scope);
    StorageScopeMutator mutator(old_buffer, new_buffer, storage_scope, block_sref_reuse);
    Stmt new_block = mutator.VisitStmt(allocate_site);
    return Downcast<Block>(new_block);
  }

 private:
  StorageScopeMutator(const Buffer& old_buffer, Buffer new_buffer, String storage_scope,
                      Map<Block, Block>* block_sref_reuse)
      : ReplaceBufferMutator(old_buffer, std::move(new_buffer), block_sref_reuse) {}

  MatchBufferRegion VisitMatchBufferRegion(const MatchBufferRegion& match_buffer) final {
    auto it = buffer_var_map_.find(match_buffer->source->buffer->data.get());
    if (it != buffer_var_map_.end()) {
      Buffer new_target_buffer = WithScope(match_buffer->buffer, it->second.scope());
      buffer_var_map_[match_buffer->buffer->data.get()] = new_target_buffer;
      return MatchBufferRegion(new_target_buffer,
                               BufferRegion(it->second, match_buffer->source->region));
    } else {
      return match_buffer;
    }
  }
};

void StorageAlign(ScheduleState self, const StmtSRef& block_sref, int buffer_index, int axis,
                  int factor, int offset) {
  const BlockNode* block_ptr = TVM_SREF_TO_BLOCK(block_sref);
  Buffer buffer =
      GetNthAccessBuffer(self, GetRef<Block>(block_ptr), buffer_index, BufferIndexType::kWrite);
  StorageAlignInvalidFactorError::Check(self->mod, factor);
  axis = StorageAlignAxisOutOfRangeError::CheckAndUpdate(self->mod, buffer, axis);
  NonAllocatedBufferError::CheckAndGetBufferAllocationSite(self->mod, block_sref, buffer);

  // Step 1: Get existing or create new annotation value.
  StorageAlignAnnotation storage_align_annotation =
      StorageAlignInvalidAnnotationError::CheckAndGetAnnotation(self->mod,
                                                                GetRef<Block>(block_ptr));

  // Step 2: Update the annotation value
  bool found = false;
  StorageAlignTuple new_storage_align_tuple{Integer(buffer_index), Integer(axis), Integer(factor),
                                            Integer(offset)};
  for (size_t j = 0; j < storage_align_annotation.size(); ++j) {
    const auto& storage_align_tuple = storage_align_annotation[j];
    ICHECK(storage_align_tuple.size() == 4);
    if (storage_align_tuple[0] == buffer_index && storage_align_tuple[1] == axis) {
      storage_align_annotation.Set(j, std::move(new_storage_align_tuple));
      found = true;
      break;
    }
  }
  if (!found) {
    storage_align_annotation.push_back(std::move(new_storage_align_tuple));
  }

  // Step 3: Replace the block with the new annotation
  Block new_block = WithAnnotation(block_ptr, attr::buffer_dim_align, storage_align_annotation);
  self->Replace(block_sref, new_block, {{GetRef<Block>(block_ptr), new_block}});
}

void SetScope(ScheduleState self, const StmtSRef& block_sref, int buffer_index,
              const String& storage_scope) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  Buffer buffer =
      GetNthAccessBuffer(self, GetRef<Block>(block), buffer_index, BufferIndexType::kWrite);

  // Step 1. If `storage_scope` equals the original storage scope of the buffer, just return.
  if (buffer.scope() == storage_scope) {
    return;
  }

  // Step 2. Throw an error if the input storage scope is invalid.
  CheckStorageScope(self, storage_scope);

  // Step 3. Get the allocation site of the target buffer.
  StmtSRef alloc_site_sref =
      NonAllocatedBufferError::CheckAndGetBufferAllocationSite(self->mod, block_sref, buffer);
  const BlockNode* alloc_site = TVM_SREF_TO_BLOCK(alloc_site_sref);

  // Step 4. Recursively replace the old buffer to a new buffer, where the new buffer has the given
  // storage scope. In the meanwhile, collect the block sref reuse information.
  Map<Block, Block> block_reuse_map;
  Block new_block = StorageScopeMutator::Mutate(GetRef<Block>(alloc_site), buffer, storage_scope,
                                                &block_reuse_map);
  self->Replace(alloc_site_sref, new_block, block_reuse_map);
}

/*!
 * \brief A helper mutator which recursively mutates the old buffer's data type, inserts data type
 * conversions, and collecte the block sref reuse information for the following replacement.
 */
class DTypeMutator : private ReplaceBufferMutator {
 public:
  /*!
   * \param allocate_site The block where `old_buffer` was allocated.
   * \param old_buffer The old buffer
   * \param target_dtype The data type to be set
   * \param block_sref_reuse The block sref reuse map to be updated
   * \return The new block after the mutation
   */
  static Block Mutate(const Block& allocate_site, const Buffer& old_buffer, const DataType& dtype,
                      Map<Block, Block>* block_sref_reuse) {
    Buffer new_buffer = WithDType(old_buffer, dtype);
    DTypeMutator mutator(old_buffer, new_buffer, dtype, block_sref_reuse);
    Stmt new_block = mutator.VisitStmt(allocate_site);
    return Downcast<Block>(new_block);
  }

 private:
  DTypeMutator(const Buffer& old_buffer, Buffer new_buffer, const DataType& dtype,
               Map<Block, Block>* block_sref_reuse)
      : ReplaceBufferMutator(old_buffer, std::move(new_buffer), block_sref_reuse),
        src_dtype_(old_buffer->dtype),
        tgt_dtype_(dtype) {}

  MatchBufferRegion VisitMatchBufferRegion(const MatchBufferRegion& match_buffer) final {
    auto it = buffer_var_map_.find(match_buffer->source->buffer->data.get());
    if (it != buffer_var_map_.end()) {
      Buffer new_target_buffer = WithDType(match_buffer->buffer, it->second->dtype);
      buffer_var_map_[match_buffer->buffer->data.get()] = new_target_buffer;
      return MatchBufferRegion(new_target_buffer,
                               BufferRegion(it->second, match_buffer->source->region));
    } else {
      return match_buffer;
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore node = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto it = buffer_var_map_.find(node->buffer->data.get());
    if (it != buffer_var_map_.end()) {
      node.CopyOnWrite()->buffer = it->second;
      node.CopyOnWrite()->value = Cast(tgt_dtype_, node->value);
    }
    return node;
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    BufferLoad node = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto it = buffer_var_map_.find(node->buffer->data.get());
    if (it != buffer_var_map_.end()) {
      return Cast(src_dtype_, BufferLoad(it->second, node->indices));
    }
    return node;
  }

  DataType src_dtype_, tgt_dtype_;
};

void UnsafeSetDType(ScheduleState self, const StmtSRef& block_sref, int buffer_index,
                    const String& dtype) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  Buffer buffer =
      GetNthAccessBuffer(self, GetRef<Block>(block), buffer_index, BufferIndexType::kWrite);
  DataType target_dtype(runtime::String2DLDataType(dtype));

  // Step 1. If `dtype` equals the original data type, just return.
  if (buffer->dtype == target_dtype) {
    return;
  }

  // Step 2. Get the allocation site of the target buffer.
  StmtSRef alloc_site_sref =
      NonAllocatedBufferError::CheckAndGetBufferAllocationSite(self->mod, block_sref, buffer);
  const BlockNode* alloc_site = TVM_SREF_TO_BLOCK(alloc_site_sref);

  // Step 3. Recursively replace old buffer to a new buffer, where the new buffer has the given
  // dtype, and insert data type conversions.
  Map<Block, Block> block_reuse_map;
  Block new_block =
      DTypeMutator::Mutate(GetRef<Block>(alloc_site), buffer, target_dtype, &block_reuse_map);
  self->Replace(alloc_site_sref, new_block, block_reuse_map);
}

/******** InstructionKind Registration ********/

struct StorageAlignTraits : public UnpackedInstTraits<StorageAlignTraits> {
  static constexpr const char* kName = "StorageAlign";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 4;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, Integer buffer_index,
                                      Integer axis, Integer factor, Integer offset) {
    return sch->StorageAlign(block_rv, buffer_index->value, axis->value, factor->value,
                             offset->value);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, Integer buffer_index,
                                 Integer axis, Integer factor, Integer offset) {
    PythonAPICall py("storage_align");
    py.Input("block", block_rv);
    py.Input("buffer_index", buffer_index);
    py.Input("axis", axis);
    py.Input("factor", factor);
    py.Input("offset", offset);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct SetScopeTraits : public UnpackedInstTraits<SetScopeTraits> {
  static constexpr const char* kName = "SetScope";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, Integer buffer_index,
                                      String storage_scope) {
    return sch->SetScope(block_rv, buffer_index->value, storage_scope);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, Integer buffer_index,
                                 String storage_scope) {
    PythonAPICall py("set_scope");
    py.Input("block", block_rv);
    py.Input("buffer_index", buffer_index);
    py.Input("storage_scope", storage_scope);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct UnsafeSetDTypeTraits : public UnpackedInstTraits<UnsafeSetDTypeTraits> {
  static constexpr const char* kName = "UnsafeSetDType";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, Integer buffer_index,
                                      String dtype) {
    return sch->UnsafeSetDType(block_rv, buffer_index->value, dtype);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, Integer buffer_index,
                                 String dtype) {
    PythonAPICall py("unsafe_set_dtype");
    py.Input("block", block_rv);
    py.Input("buffer_index", buffer_index);
    py.Input("dtype", dtype);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(StorageAlignTraits);
TVM_REGISTER_INST_KIND_TRAITS(SetScopeTraits);
TVM_REGISTER_INST_KIND_TRAITS(UnsafeSetDTypeTraits);

}  // namespace tir
}  // namespace tvm
