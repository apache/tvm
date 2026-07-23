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
#include <tvm/ffi/cast.h>
#include <tvm/s_tir/stmt.h>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

class AnnotateRegionRewriter : public StmtExprMutator {
 public:
  AnnotateRegionRewriter(Buffer buffer, int buffer_index, BufferRegion new_region,
                         BufferIndexType buffer_index_type)
      : buffer_(buffer),
        buffer_index_(buffer_index),
        new_region_(new_region),
        buffer_index_type_(buffer_index_type) {}

  Stmt VisitStmt_(const SBlockNode* op) final {
    SBlock block = StmtExprMutator::VisitStmt_(op).as_or_throw<SBlock>();

    ffi::Array<BufferRegion> regions =
        buffer_index_type_ == BufferIndexType::kWrite ? block->writes : block->reads;
    TVM_FFI_ICHECK_GE(buffer_index_, 0) << "Buffer index must be non-negative";
    TVM_FFI_ICHECK_LT(buffer_index_, static_cast<int>(regions.size()))
        << "Buffer index out of range";
    regions.Set(buffer_index_, new_region_);

    ffi::ObjectPtr<SBlockNode> n = CopyOnWrite(block.get());
    if (buffer_index_type_ == BufferIndexType::kWrite) {
      n->writes = std::move(regions);
    } else {
      n->reads = std::move(regions);
    }

    // Annotate the block with explicit_read_region or explicit_write_region
    ffi::Map<ffi::String, ffi::Any> new_annotations = n->annotations;
    ffi::String annotation_key = buffer_index_type_ == BufferIndexType::kWrite
                                     ? s_tir::attr::explicit_write_region
                                     : s_tir::attr::explicit_read_region;
    if (new_annotations.count(annotation_key)) {
      ffi::Array<int64_t> buffer_indices =
          new_annotations[annotation_key].as_or_throw<ffi::Array<int64_t>>();
      bool found = false;
      for (int64_t index : buffer_indices) {
        if (index == buffer_index_) {
          found = true;
          break;
        }
      }
      if (!found) {
        buffer_indices.push_back(static_cast<int64_t>(buffer_index_));
        new_annotations.Set(annotation_key, buffer_indices);
      }
    } else {
      new_annotations.Set(annotation_key, ffi::Array<int64_t>{static_cast<int64_t>(buffer_index_)});
    }
    n->annotations = std::move(new_annotations);

    return SBlock(n);
  }

 private:
  Buffer buffer_;
  int buffer_index_;
  BufferRegion new_region_;
  BufferIndexType buffer_index_type_;
};

void AnnotateBufferAccess(ScheduleState self, const StmtSRef& block_sref, int buffer_index,
                          BufferIndexType buffer_index_type, const IndexMap& index_map) {
  const SBlockNode* block = TVM_SREF_TO_SBLOCK(block_sref);
  Buffer buffer =
      GetNthAccessBuffer(self, ffi::GetRef<SBlock>(block), buffer_index, buffer_index_type);

  arith::Analyzer analyzer;
  ffi::Array<PrimExpr> block_iter_vars;
  for (const IterVar& iter_var : block->iter_vars) {
    block_iter_vars.push_back(iter_var->var);
  }
  ffi::Array<PrimExpr> new_indices = index_map->MapIndices(block_iter_vars, analyzer);
  TVM_FFI_ICHECK_EQ(new_indices.size() % 2, 0) << "The size of new_indices should be even.";
  ffi::Array<Range> new_ranges;
  for (size_t i = 0; i < new_indices.size(); i += 2) {
    // (begin, end) represents a region
    new_ranges.push_back(Range::FromMinExtent(
        new_indices[i], analyzer->Simplify(new_indices[i + 1] - new_indices[i])));
  }

  BufferRegion new_region(buffer, new_ranges);

  AnnotateRegionRewriter mutator(buffer, buffer_index, new_region, buffer_index_type);
  Stmt new_stmt = mutator(ffi::GetRef<Stmt>(block_sref->stmt));

  self->Replace(block_sref, new_stmt,
                {{ffi::GetRef<SBlock>(block), new_stmt.as_or_throw<SBlock>()}});
}

struct AnnotateBufferAccessTraits : public UnpackedInstTraits<AnnotateBufferAccessTraits> {
  static constexpr const char* kName = "AnnotateBufferAccess";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 4;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, SBlockRV block, IntImm buffer_index,
                                      IntImm buffer_index_type, IndexMap index_map) {
    return sch->AnnotateBufferAccess(block, buffer_index->value,
                                     static_cast<BufferIndexType>(buffer_index_type->value),
                                     index_map);
  }

  static ffi::String IndexMap2GenNewRangesLambda(const IndexMap& index_map) {
    std::ostringstream oss;
    auto print_expr = [&oss](const PrimExpr& expr) {
      if (auto var = expr.as<PrimVar>()) {
        oss << var.value()->name;
      } else {
        oss << expr;
      }
    };
    oss << "lambda ";
    for (size_t i = 0; i < index_map->initial_indices.size(); ++i) {
      if (i != 0) oss << ", ";
      oss << index_map->initial_indices[i]->name;
    }
    oss << ": [";
    for (size_t i = 0; i < index_map->final_indices.size(); i += 2) {
      if (i != 0) oss << ", ";
      if (index_map->final_indices[i].same_as(index_map->final_indices[i + 1])) {
        print_expr(index_map->final_indices[i]);
      } else {
        oss << "(";
        print_expr(index_map->final_indices[i]);
        oss << ", ";
        print_expr(index_map->final_indices[i + 1]);
        oss << ")";
      }
    }
    oss << "]";
    return ffi::String(oss.str());
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block,
                                      IntImm buffer_index, IntImm buffer_index_type,
                                      IndexMap index_map) {
    PythonAPICall py("annotate_buffer_access");
    py.Input("block", block);
    py.Input("buffer_index", buffer_index->value);

    std::ostringstream os;
    os << "\"" << BufferIndexType2Str(static_cast<BufferIndexType>(buffer_index_type->value))
       << "\"";
    py.Input("buf_type", ffi::String(os.str()));

    py.Input("gen_new_ranges", IndexMap2GenNewRangesLambda(index_map));
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(AnnotateBufferAccessTraits);

}  // namespace s_tir
}  // namespace tvm
