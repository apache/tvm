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
#include "../../transforms/ir_utils.h"
#include "../utils.h"

namespace tvm {
namespace tir {

/******** Error Classes ********/

namespace {
class BufTypeError : public ScheduleError {
 public:
  explicit BufTypeError(IRModule mod, const String& buf_type)
      : mod_(std::move(mod)), buf_type_(buf_type) {}

  String FastErrorString() const final {
    return "ScheduleError: Invalid buffer type for hide_buffer_access schedule.";
  }

  String DetailRenderTemplate() const final {
    return "The buffer type for hide_buffer_access schedule should either be 'read'"
           " or 'write', got " +
           buf_type_ + " instead.";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {}; }

 private:
  IRModule mod_;
  String buf_type_;
};

class InvalidIndexError : public ScheduleError {
 public:
  explicit InvalidIndexError(IRModule mod, int num_access_regions, int buf_idx)
      : mod_(std::move(mod)), num_access_regions_(num_access_regions), buf_idx_(buf_idx) {}

  String FastErrorString() const final {
    return "ScheduleError: Invalid buffer index array for hide_buffer_access schedule.";
  }

  String DetailRenderTemplate() const final {
    return "The buffer index array for hide_buffer_access schedule should be a list of integers"
           " between 0 and " +
           std::to_string(num_access_regions_ - 1) + ", got " + std::to_string(buf_idx_) +
           " instead.";
  }

  IRModule mod() const final { return mod_; }

  Array<ObjectRef> LocationsOfInterest() const final { return {}; }

 private:
  IRModule mod_;
  int num_access_regions_;
  int buf_idx_;
};

}  // namespace

/******** Implementation ********/

void UnsafeHideBufferAccess(ScheduleState self, const StmtSRef& block_sref, const String& buf_type,
                            const Array<IntImm>& buf_index_array) {
  /*!
   * Check:
   *   - validity of buf_index_array
   *   - validity of buf_type
   */
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  int num_access_regions = 0;
  if (buf_type == "read") {
    num_access_regions = block->reads.size();
  } else if (buf_type == "write") {
    num_access_regions = block->writes.size();
  } else {
    throw BufTypeError(self->mod, buf_type);
  }

  std::set<int> buf_indices;
  for (const IntImm& buf_idx : buf_index_array) {
    int buf_idx_val = buf_idx->value;
    if (buf_idx_val >= 0 && buf_idx_val < num_access_regions) {
      buf_indices.insert(buf_idx_val);
    } else {
      throw InvalidIndexError(self->mod, num_access_regions, buf_idx_val);
    }
  }

  /* Step 0: Collect new buffer access regions. */

  Array<BufferRegion> reads, writes;

  if (buf_type == "read") {
    for (size_t i = 0; i < block->reads.size(); ++i) {
      if (!buf_indices.count(i)) {
        reads.push_back(block->reads[i]);
      }
    }
    writes = block->writes;
  } else if (buf_type == "write") {
    for (size_t i = 0; i < block->writes.size(); ++i) {
      if (!buf_indices.count(i)) {
        writes.push_back(block->writes[i]);
      }
    }
    reads = block->reads;
  } else {
    CHECK(false) << "Unrecognized buffer type " << buf_type << ", only support read/write";
  }

  /* Step 1: Replace old block with the new block */

  auto n = make_object<BlockNode>(*block);
  n->reads = reads;
  n->writes = writes;
  Block new_block = Block(n);
  Map<Block, Block> blk_map;
  blk_map.Set(GetRef<Block>(block), new_block);
  self->Replace(block_sref, new_block, blk_map);
}

struct UnsafeHideBufferAccessTraits : public UnpackedInstTraits<UnsafeHideBufferAccessTraits> {
  static constexpr const char* kName = "UnsafeHideBufferAccess";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 3;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block, String buf_type,
                                      Array<IntImm> buf_index_array) {
    sch->UnsafeHideBufferAccess(block, buf_type, buf_index_array);
  }

  static String UnpackedAsPython(Array<String> outputs, String block, String buf_type,
                                 Array<IntImm> buf_index_array) {
    PythonAPICall py("unsafe_hide_buffer_access");
    py.Input("block", block);
    py.Input("buf_type", buf_type);
    py.Input("buf_index_array", buf_index_array);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(UnsafeHideBufferAccessTraits);

}  // namespace tir
}  // namespace tvm
