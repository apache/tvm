#include "../utils.h"

namespace tvm {
namespace tir {

class AnnotateReadRegionNode : public StmtExprMutator {
 public:
  AnnotateReadRegionNode(Buffer buffer, int buffer_index, BufferRegion new_region)
      : buffer_(buffer), buffer_index_(buffer_index), new_region_(new_region) {}

  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));

    ICHECK_GE(buffer_index_, 0) << "Buffer index must be non-negative";
    ICHECK_LT(buffer_index_, static_cast<int>(block->reads.size())) << "Buffer index out of range";
    Array<BufferRegion> new_reads = block->reads;
    new_reads.Set(buffer_index_, new_region_);

    ObjectPtr<BlockNode> n = CopyOnWrite(block.get());
    n->reads = std::move(new_reads);

    // Annotate the block with explicit_read_region
    Map<String, ObjectRef> new_annotations = n->annotations;
    new_annotations.Set(attr::explicit_read_region, Integer(buffer_index_));
    n->annotations = std::move(new_annotations);

    return Block(n);
  }

 private:
  Buffer buffer_;
  int buffer_index_;
  BufferRegion new_region_;
};

void AnnotateReadRegion(ScheduleState self, const StmtSRef& block_sref, int buffer_index,
                        const IndexMap& index_map) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  Buffer buffer =
      GetNthAccessBuffer(self, GetRef<Block>(block), buffer_index, BufferIndexType::kRead);

  arith::Analyzer analyzer;
  Array<PrimExpr> block_iter_vars;
  for (const IterVar& iter_var : block->iter_vars) {
    block_iter_vars.push_back(iter_var->var);
  }
  Array<PrimExpr> new_indices = index_map->MapIndices(block_iter_vars, &analyzer);
  ICHECK_EQ(new_indices.size() % 2, 0) << "The size of new_indices should be even.";
  Array<Range> new_ranges;
  for (size_t i = 0; i < new_indices.size(); i += 2) {
    if (analyzer.CanProveEqual(new_indices[i], new_indices[i + 1])) {
      new_ranges.push_back(Range::FromMinExtent(new_indices[i], 1));
    } else {
      new_ranges.push_back(Range::FromMinExtent(
          new_indices[i], analyzer.Simplify(new_indices[i + 1] - new_indices[i])));
    }
  }

  BufferRegion new_region(buffer, new_ranges);

  AnnotateReadRegionNode mutator(buffer, buffer_index, new_region);
  Stmt new_stmt = mutator(GetRef<Stmt>(block_sref->stmt));

  self->Replace(block_sref, new_stmt, {{GetRef<Block>(block), Downcast<Block>(new_stmt)}});
}

struct AnnotateReadRegionTraits : public UnpackedInstTraits<AnnotateReadRegionTraits> {
  static constexpr const char* kName = "AnnotateReadRegion";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 3;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block, Integer buffer_index,
                                      IndexMap index_map) {
    return sch->AnnotateReadRegion(block, buffer_index->value, index_map);
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
                                 IndexMap index_map) {
    PythonAPICall py("annotate_read_region");
    py.Input("block", block);
    py.Input("buffer_index", buffer_index->value);
    py.Input("gen_new_ranges", IndexMap2GenNewRangesLambda(index_map));
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(AnnotateReadRegionTraits);

}  // namespace tir
}  // namespace tvm