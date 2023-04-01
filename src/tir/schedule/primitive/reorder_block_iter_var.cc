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

/*!
 * \brief The reorder index is not a valid permutation of
 *   [0, 1, ..., n-1] where n is the number of block iter vars.
 */
class InvalidReorderIndex : public ScheduleError {
 public:
  explicit InvalidReorderIndex(IRModule mod, Block block, Array<Integer> new_order)
      : mod_(mod), block_(block), new_order_(new_order) {}
  IRModule mod() const final { return mod_; }
  String FastErrorString() const final {
    return "ScheduleError: The specified reorder indices are invalid.";
  }
  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The user provided block itervar index order " << new_order_
       << " is not a valid permutation of [0, 1, ..., num_block_iter_vars-1] in block {0}.";
    return String(os.str());
  }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

 private:
  IRModule mod_;
  Block block_;
  Array<Integer> new_order_;
};

class BlockIterVarRewriter : public StmtMutator {
 public:
  Map<Block, Block> block_map;
  explicit BlockIterVarRewriter(const BlockNode* block_n, std::vector<int> order)
      : order_(std::move(order)), block_to_rewrite(block_n) {}

 private:
  std::vector<int> order_;
  const BlockNode* block_to_rewrite;
  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    if (op->block.get() == block_to_rewrite) {
      auto block_n = CopyOnWrite(op->block.get());
      Block block = op->block;
      Array<IterVar> new_iter_vars;
      Array<PrimExpr> new_iter_values;
      for (int idx : order_) {
        new_iter_vars.push_back(block->iter_vars[idx]);
        new_iter_values.push_back(op->iter_values[idx]);
      }
      block_n->iter_vars = new_iter_vars;
      Block new_block(block_n);
      block_map.Set(block, new_block);
      auto block_realize_n = CopyOnWrite(op);
      block_realize_n->block = new_block;
      block_realize_n->iter_values = new_iter_values;
      return BlockRealize(block_realize_n);
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }
};

void ReorderBlockIterVar(ScheduleState self, const StmtSRef& block_sref,
                         const Array<Integer>& new_order) {
  const BlockNode* block_n = TVM_SREF_TO_BLOCK(block_sref);
  std::vector<int> new_order_vec;
  for (const Integer& x : new_order) {
    new_order_vec.push_back(x->value);
  }
  // check whether new_order is valid or not;
  size_t num_block_itervars = block_n->iter_vars.size();
  std::set<int> ind_set(new_order_vec.begin(), new_order_vec.end());
  bool is_full = new_order_vec.size() == num_block_itervars;
  bool is_unique = (ind_set.size() == new_order_vec.size());
  bool is_within_boundary = std::all_of(new_order_vec.begin(), new_order_vec.end(), [&](int x) {
    return x >= 0 && x < static_cast<int>(num_block_itervars);
  });
  if (!is_full || !is_unique || !is_within_boundary) {
    throw InvalidReorderIndex(self->mod, GetRef<Block>(block_n), new_order);
  }

  // find parent block
  const BlockNode* parent_block_n = nullptr;
  const StmtSRefNode* p = block_sref.get()->parent;
  while (p != nullptr) {
    if (p->stmt->IsInstance<BlockNode>()) {
      parent_block_n = TVM_SREF_TO_BLOCK(GetRef<StmtSRef>(p));
      break;
    }
    p = p->parent;
  }
  const StmtSRef parent_block_sref = GetRef<StmtSRef>(p);
  const Block& parent_block = GetRef<Block>(parent_block_n);

  // rewrite block and blockrealize
  BlockIterVarRewriter rewriter(block_n, std::move(new_order_vec));
  Block new_parent_block = Downcast<Block>(rewriter(parent_block));
  rewriter.block_map.Set(parent_block, new_parent_block);
  self->Replace(parent_block_sref, new_parent_block, rewriter.block_map);
}

struct ReorderBlockIterVarTraits : public UnpackedInstTraits<ReorderBlockIterVarTraits> {
  static constexpr const char* kName = "ReorderBlockIterVar";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block, Array<Integer> new_order) {
    sch->ReorderBlockIterVar(block, new_order);
  }

  static String UnpackedAsPython(Array<String> outputs, String block, Array<Integer> new_order) {
    PythonAPICall py("reorder_block_iter_var");
    py.Input("block", block);
    py.Input("new_order", new_order);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(ReorderBlockIterVarTraits);

}  // namespace tir
}  // namespace tvm
