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

/*!
 * \brief Find the defining site of the buffer in the given block and its ancestors
 * \param block_sref The block sref
 * \param buffer The buffer
 * \return The defining site of the buffer and whether the buffer is allocated (otherwise the
 *         buffer is from match_buffer).
 */
std::pair<StmtSRef, bool> GetBufferDefiningSite(const StmtSRef& block_sref, const Buffer& buffer) {
  // Climb up along the sref tree, and find the block where `buffer` is in alloc_buffers or
  // match_buffers.
  const StmtSRefNode* defining_site_sref = block_sref.get();
  while (defining_site_sref != nullptr) {
    const auto* block = defining_site_sref->StmtAs<BlockNode>();
    // If this sref is not a block sref, skip it.
    if (block == nullptr) {
      defining_site_sref = defining_site_sref->parent;
      continue;
    }
    // Try to find the buffer in `allloc_buffers`
    for (const Buffer& alloc_buffer : block->alloc_buffers) {
      if (buffer.same_as(alloc_buffer)) {
        return {GetRef<StmtSRef>(defining_site_sref), true};
      }
    }
    // We do not allow the buffer being defined in `match_buffer`.
    for (const MatchBufferRegion match_buffer : block->match_buffers) {
      if (buffer.same_as(match_buffer)) {
        return {GetRef<StmtSRef>(defining_site_sref), false};
      }
    }
    defining_site_sref = defining_site_sref->parent;
  }
  // If we cannot find the defining site block, it means that the buffer must be in the function's
  // buffer_map, which isn't an intermediate buffer.
  return {StmtSRef(), false};
}

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

  static void CheckBufferAllocated(const IRModule& mod, const StmtSRef& block_sref,
                                   const Buffer& buffer) {
    StmtSRef defining_site_sref;
    bool is_alloc;
    std::tie(defining_site_sref, is_alloc) = GetBufferDefiningSite(block_sref, buffer);
    if (!defining_site_sref.defined() || !is_alloc) {
      throw NonAllocatedBufferError(mod, buffer);
    }
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
           "3-integer-tuples (axis, factor, offset).";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The block annotation for storage align is expected to be an array of 3-integer-tuples "
          "(axis, factor, offset). However, the block annotation with key "
       << attr::buffer_dim_align << " of the block {0} is "
       << block_->annotations.at(attr::buffer_dim_align) << ", which is unexpected.";
    return os.str();
  }

  static Array<Array<Array<Integer>>> CheckAndGetAnnotation(const IRModule& mod,
                                                            const Block& block) {
    // Get existing annotation value.
    auto it = block->annotations.find(attr::buffer_dim_align);
    if (it != block->annotations.end()) {
      if (!IsValidAnnotation(block, (*it).second)) {
        throw StorageAlignInvalidAnnotationError(mod, block);
      }
      return Downcast<Array<Array<Array<Integer>>>>((*it).second);
    }

    // Create new annotation value
    Array<Array<Array<Integer>>> storage_align_annotation;
    storage_align_annotation.resize(block->writes.size());
    return storage_align_annotation;
  }

  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
  IRModule mod() const final { return mod_; }

 private:
  static bool IsValidAnnotation(const Block& block, const ObjectRef& anno_value) {
    if (!anno_value->IsInstance<ArrayNode>()) {
      return false;
    }
    const auto& buffer_annotations = Downcast<Array<ObjectRef>>(anno_value);
    if (buffer_annotations.size() != block->writes.size()) {
      return false;
    }
    for (const ObjectRef& buffer_annotation : buffer_annotations) {
      if (!buffer_annotation->IsInstance<ArrayNode>()) {
        return false;
      }
      const auto& dim_annotations = Downcast<Array<ObjectRef>>(buffer_annotation);
      for (const ObjectRef& dim_annotation : dim_annotations) {
        if (!dim_annotation->IsInstance<ArrayNode>()) {
          return false;
        }
        const auto& dim_anno = Downcast<Array<ObjectRef>>(dim_annotation);
        // Check if the annotations are consist of 3-tuples.
        if (dim_anno.size() != 3) {
          return false;
        }
        for (const ObjectRef& dim_anno_element : dim_anno) {
          if (!dim_anno_element->IsInstance<IntImmNode>()) {
            return false;
          }
        }
      }
    }
    return true;
  }

  IRModule mod_;
  Block block_;
};

void StorageAlign(ScheduleState self, const StmtSRef& block_sref, int buffer_index, int axis,
                  int factor, int offset) {
  const BlockNode* block_ptr = TVM_SREF_TO_BLOCK(block_ptr, block_sref);
  Buffer buffer = GetNthWriteBuffer(self, GetRef<Block>(block_ptr), buffer_index);
  StorageAlignInvalidFactorError::Check(self->mod, factor);
  axis = StorageAlignAxisOutOfRangeError::CheckAndUpdate(self->mod, buffer, axis);
  NonAllocatedBufferError::CheckBufferAllocated(self->mod, block_sref, buffer);

  // Step 1: Get existing or create new annotation value.
  auto storage_align_annotation = StorageAlignInvalidAnnotationError::CheckAndGetAnnotation(
      self->mod, GetRef<Block>(block_ptr));

  // Step 2: Update the annotation value
  Array<Array<Integer>> buffer_storage_align = storage_align_annotation[buffer_index];
  bool found = false;
  for (size_t j = 0; j < buffer_storage_align.size(); ++j) {
    ICHECK(buffer_storage_align[j].size() == 3);
    if (buffer_storage_align[j][0] == axis) {
      buffer_storage_align.Set(j, {Integer(axis), Integer(factor), Integer(offset)});
      found = true;
      break;
    }
  }
  if (!found) {
    buffer_storage_align.push_back({Integer(axis), Integer(factor), Integer(offset)});
  }
  storage_align_annotation.Set(buffer_index, std::move(buffer_storage_align));

  // Step 3: Replace the block with the new annotation
  Block new_block = WithAnnotation(block_ptr, attr::buffer_dim_align, storage_align_annotation);
  self->Replace(block_sref, new_block, {{GetRef<Block>(block_ptr), new_block}});
}

/******** Instruction Registration ********/

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

TVM_REGISTER_INST_KIND_TRAITS(StorageAlignTraits);

}  // namespace tir
}  // namespace tvm
