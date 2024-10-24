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

class AnnotateRegionRewriter : public StmtExprMutator {
 public:
  AnnotateRegionRewriter(Buffer buffer, int buffer_index, BufferRegion new_region,
                         BufferIndexType buffer_index_type)
      : buffer_(buffer),
        buffer_index_(buffer_index),
        new_region_(new_region),
        buffer_index_type_(buffer_index_type) {}

  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));

    Array<BufferRegion> regions =
        buffer_index_type_ == BufferIndexType::kWrite ? block->writes : block->reads;
    ICHECK_GE(buffer_index_, 0) << "Buffer index must be non-negative";
    ICHECK_LT(buffer_index_, static_cast<int>(regions.size())) << "Buffer index out of range";
    regions.Set(buffer_index_, new_region_);

    ObjectPtr<BlockNode> n = CopyOnWrite(block.get());
    if (buffer_index_type_ == BufferIndexType::kWrite) {
      n->writes = std::move(regions);
    } else {
      n->reads = std::move(regions);
    }

    // Annotate the block with explicit_read_region or explicit_write_region
    Map<String, ObjectRef> new_annotations = n->annotations;
    String annotation_key = buffer_index_type_ == BufferIndexType::kWrite
                                ? attr::explicit_write_region
                                : attr::explicit_read_region;
    if (new_annotations.count(annotation_key)) {
      Array<Integer> buffer_indices = Downcast<Array<Integer>>(new_annotations[annotation_key]);
      bool found = false;
      for (const Integer& index : buffer_indices) {
        if (index->value == buffer_index_) {
          found = true;
          break;
        }
      }
      if (!found) {
        buffer_indices.push_back(Integer(buffer_index_));
        new_annotations.Set(annotation_key, buffer_indices);
      }
    } else {
      new_annotations.Set(annotation_key, Array<Integer>{Integer(buffer_index_)});
    }
    n->annotations = std::move(new_annotations);

    return Block(n);
  }

 private:
  Buffer buffer_;
  int buffer_index_;
  BufferRegion new_region_;
  BufferIndexType buffer_index_type_;
};

void AnnotateBufferAccess(ScheduleState self, const StmtSRef& block_sref, int buffer_index,
                          BufferIndexType buffer_index_type, const IndexMap& index_map) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  Buffer buffer = GetNthAccessBuffer(self, GetRef<Block>(block), buffer_index, buffer_index_type);

  arith::Analyzer analyzer;
  Array<PrimExpr> block_iter_vars;
  for (const IterVar& iter_var : block->iter_vars) {
    block_iter_vars.push_back(iter_var->var);
  }
  Array<PrimExpr> new_indices = index_map->MapIndices(block_iter_vars, &analyzer);
  ICHECK_EQ(new_indices.size() % 2, 0) << "The size of new_indices should be even.";
  Array<Range> new_ranges;
  for (size_t i = 0; i < new_indices.size(); i += 2) {
    // (begin, end) represents a region
    new_ranges.push_back(Range::FromMinExtent(
        new_indices[i], analyzer.Simplify(new_indices[i + 1] - new_indices[i])));
  }

  BufferRegion new_region(buffer, new_ranges);

  AnnotateRegionRewriter mutator(buffer, buffer_index, new_region, buffer_index_type);
  Stmt new_stmt = mutator(GetRef<Stmt>(block_sref->stmt));

  self->Replace(block_sref, new_stmt, {{GetRef<Block>(block), Downcast<Block>(new_stmt)}});
}

struct AnnotateBufferAccessTraits : public UnpackedInstTraits<AnnotateBufferAccessTraits> {
  static constexpr const char* kName = "AnnotateBufferAccess";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 4;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block, Integer buffer_index,
                                      Integer buffer_index_type, IndexMap index_map) {
    return sch->AnnotateBufferAccess(block, buffer_index->value,
                                     static_cast<BufferIndexType>(buffer_index_type->value),
                                     index_map);
  }

  static String IndexMap2GenNewRangesLambda(const IndexMap& index_map) {
    std::ostringstream oss;
    oss << "lambda ";
    for (size_t i = 0; i < index_map->initial_indices.size(); ++i) {
      if (i != 0) oss << ", ";
      oss << index_map->initial_indices[i];
    }
    oss << ": [";
    for (size_t i = 0; i < index_map->final_indices.size(); i += 2) {
      if (i != 0) oss << ", ";
      if (index_map->final_indices[i].same_as(index_map->final_indices[i + 1])) {
        oss << index_map->final_indices[i];
      } else {
        oss << "(" << index_map->final_indices[i] << ", " << index_map->final_indices[i + 1] << ")";
      }
    }
    oss << "]";
    return String(oss.str());
  }

  static String UnpackedAsPython(Array<String> outputs, String block, Integer buffer_index,
                                 Integer buffer_index_type, IndexMap index_map) {
    PythonAPICall py("annotate_buffer_access");
    py.Input("block", block);
    py.Input("buffer_index", buffer_index->value);

    std::ostringstream os;
    os << "\"" << BufferIndexType2Str(static_cast<BufferIndexType>(buffer_index_type->value))
       << "\"";
    py.Input("buf_type", os.str());

    py.Input("gen_new_ranges", IndexMap2GenNewRangesLambda(index_map));
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(AnnotateBufferAccessTraits);

}  // namespace tir
}  // namespace tvm
