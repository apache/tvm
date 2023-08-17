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

/*!
 * \file tvm/relax/analysis/dataflow_analysis.cc
 * \brief Implementation of functionality in dataflow_analysis.h
 */
#include <tvm/relax/dataflow_analysis.h>
#include <tvm/runtime/memory.h>

#include <queue>

namespace tvm {
namespace relax {

TVM_REGISTER_NODE_TYPE(BasicBlockNode);

BasicBlock BasicBlock::Create(const SeqExpr& seq, const Array<Var>& args, const Expr& ret,
                              size_t start_block_idx, size_t start_binding_idx,
                              size_t end_block_idx, size_t end_binding_idx) {
  ObjectPtr<BasicBlockNode> n = make_object<BasicBlockNode>();
  n->seq = seq;
  n->args = args;
  n->ret = ret;
  n->start_block_idx = start_block_idx;
  n->start_binding_idx = start_binding_idx;
  n->end_block_idx = end_block_idx;
  n->end_binding_idx = end_binding_idx;
  return BasicBlock(n);
}

TVM_REGISTER_NODE_TYPE(ControlFlowGraphNode);

ControlFlowGraph ControlFlowGraph::Create(const Array<BasicBlock>& blocks,
                                          const Array<Array<Integer>>& preds,
                                          const Array<Array<Integer>>& succs) {
  ObjectPtr<ControlFlowGraphNode> n = make_object<ControlFlowGraphNode>();
  n->blocks = blocks;
  n->preds = preds;
  n->succs = succs;
  return ControlFlowGraph(n);
}

// Extracts a basic block and updates the running lists blocks, preds, and succs.
// The return value is the index of the final basic block processed in the seq expression
// (useful for processing branches).
size_t ExtractCFGHelper(const SeqExpr& seq, const Array<Var>& args, size_t start_block_idx,
                        size_t start_binding_idx, std::vector<size_t> current_preds,
                        std::vector<BasicBlock>* blocks, std::vector<std::vector<size_t>>* preds,
                        std::vector<std::vector<size_t>>* succs) {
  size_t end_block_idx = 0;
  size_t end_binding_idx = 0;
  Expr ret;
  Optional<Var> branch_var;
  Optional<If> branch_expr;

  // go from the start index and continue until we hit the end of the block or a split point
  bool hit_branch = false;
  // note: if start_block_idx is past seq->blocks.size(), then the loop will not actually run
  // and we will not hit a branch, so we will produce a basic block comprised only of the
  // seq expr end expression
  for (size_t i = start_block_idx; i < seq->blocks.size(); i++) {
    for (size_t j = start_binding_idx; j < seq->blocks[i]->bindings.size(); j++) {
      Binding binding = seq->blocks[i]->bindings[j];
      if (auto* var_binding = binding.as<VarBindingNode>()) {
        if (var_binding->value.as<IfNode>()) {
          end_block_idx = i;
          end_binding_idx = j;
          branch_var = var_binding->var;
          branch_expr = Downcast<If>(var_binding->value);
          ret = branch_expr.value()->cond;
          hit_branch = true;
          break;
        }
      } else if (auto* match_binding = binding.as<MatchCastNode>()) {
        if (match_binding->value.as<IfNode>()) {
          end_block_idx = i;
          end_binding_idx = j;
          branch_var = var_binding->var;
          branch_expr = Downcast<If>(var_binding->value);
          ret = branch_expr.value()->cond;
          hit_branch = true;
          break;
        }
      } else {
        CHECK(false);  // will never happen
      }
    }
    if (hit_branch) {
      break;
    }
  }

  if (!hit_branch) {
    end_block_idx = seq->blocks.size();
    end_binding_idx = 0U;  // doesn't matter which we use
    ret = seq->body;
  }
  BasicBlock block = BasicBlock::Create(seq, args, ret, start_block_idx, start_binding_idx,
                                        end_block_idx, end_binding_idx);
  blocks->push_back(block);
  size_t block_idx = blocks->size() - 1U;
  succs->push_back({});
  preds->push_back(current_preds);
  for (size_t pred : current_preds) {
    succs->at(pred).push_back(block_idx);
  }
  // no branches: then we're done
  if (!hit_branch) {
    return block_idx;
  }
  // hit a branch: recurse down the branches and then set up the merge block
  SeqExpr true_branch = Downcast<SeqExpr>(branch_expr.value()->true_branch);
  SeqExpr false_branch = Downcast<SeqExpr>(branch_expr.value()->false_branch);
  // the branches could contain their own branches, which is why we return the final block index
  size_t end_true = ExtractCFGHelper(true_branch, {}, 0U, 0U, {block_idx}, blocks, preds, succs);
  size_t end_false = ExtractCFGHelper(false_branch, {}, 0U, 0U, {block_idx}, blocks, preds, succs);

  // work out the start indices for the merge point
  size_t next_start_block_idx = end_block_idx;
  size_t next_start_binding_idx = end_binding_idx;
  // figure out the next indices
  if (end_binding_idx == seq->blocks[end_block_idx]->bindings.size() - 1) {
    if (end_block_idx == seq->blocks.size() - 1) {
      next_start_block_idx = seq->blocks.size();
      next_start_binding_idx = 0U;
    } else {
      next_start_block_idx = end_block_idx + 1;
      next_start_binding_idx = 0U;
    }
  } else {
    next_start_binding_idx = end_binding_idx + 1;
  }
  return ExtractCFGHelper(seq, {branch_var.value()}, next_start_block_idx, next_start_binding_idx,
                          {end_true, end_false}, blocks, preds, succs);
}

ControlFlowGraph ExtractCFG(const Function& func) {
  std::vector<BasicBlock> blocks;
  std::vector<std::vector<size_t>> preds;
  std::vector<std::vector<size_t>> succs;
  ExtractCFGHelper(Downcast<SeqExpr>(func->body), func->params, 0U, 0U, {}, &blocks, &preds,
                   &succs);

  Array<Array<Integer>> pred_arr;
  for (auto pred_vec : preds) {
    Array<Integer> pred_ints;
    for (auto idx : pred_vec) {
      pred_ints.push_back(Integer(idx));
    }
    pred_arr.push_back(pred_ints);
  }
  Array<Array<Integer>> succ_arr;
  for (auto succ_vec : succs) {
    Array<Integer> succ_ints;
    for (auto idx : succ_vec) {
      succ_ints.push_back(Integer(idx));
    }
    succ_arr.push_back(succ_ints);
  }
  return ControlFlowGraph::Create(Array<BasicBlock>(blocks), pred_arr, succ_arr);
}

std::pair<Array<ObjectRef>, Array<ObjectRef>> DataflowAnalysis(
    const ControlFlowGraph& cfg, const ObjectRef& init,
    std::function<ObjectRef(const BasicBlock&, const ObjectRef&)> transfer_func,
    std::function<ObjectRef(const ObjectRef&, const ObjectRef&)> merge_func, bool forward) {
  std::vector<ObjectRef> in_map;
  std::vector<ObjectRef> out_map;
  for (size_t i = 0; i < cfg->blocks.size(); i++) {
    in_map.push_back(init);
    out_map.push_back(init);
  }

  // Modification from Adrian Sampson's version:
  // Since there are no loops in our AST, one traversal through the CFG suffices.
  // We will do BFS
  std::queue<size_t> worklist;
  worklist.push((forward) ? 0 : cfg->blocks.size() - 1);
  while (!worklist.empty()) {
    size_t idx = worklist.front();
    worklist.pop();
    Array<Integer> prev = (forward) ? cfg->preds[idx] : cfg->succs[idx];
    Array<Integer> next = (forward) ? cfg->succs[idx] : cfg->preds[idx];
    std::vector<ObjectRef>* results = (forward) ? &out_map : &in_map;
    std::vector<ObjectRef>* inputs = (forward) ? &in_map : &out_map;

    // Cases (for forward analysis):
    // 0 predecessors: The first block in the function
    // 1 predecessor: A branch in an If node (no merge needed)
    // 2 predecessors: The merge block after an If node (merge needed)
    // (Analogous for successors in backward analysis)
    inputs->operator[](idx) = (prev.size() == 0)   ? init
                              : (prev.size() == 1) ? results->at(prev[0].IntValue())
                                                   : merge_func(results->at(prev[0].IntValue()),
                                                                results->at(prev[1].IntValue()));
    results->operator[](idx) = transfer_func(cfg->blocks[idx], inputs->at(idx));

    for (Integer next_idx : next) {
      worklist.push(next_idx.IntValue());
    }
  }

  return {Array<ObjectRef>(in_map), Array<ObjectRef>(out_map)};
}

TVM_REGISTER_GLOBAL("relax.analysis.BasicBlock")
    .set_body_typed([](const SeqExpr& seq, const Array<Var>& args, const Expr& ret,
                       size_t start_block_idx, size_t start_binding_idx, size_t end_block_idx,
                       size_t end_binding_idx) {
      return BasicBlock::Create(seq, args, ret, start_block_idx, start_binding_idx, end_block_idx,
                                end_binding_idx);
    });

TVM_REGISTER_GLOBAL("relax.analysis.ControlFlowGraph")
    .set_body_typed([](const Array<BasicBlock>& blocks, const Array<Array<Integer>>& preds,
                       const Array<Array<Integer>>& succs) {
      return ControlFlowGraph::Create(blocks, preds, succs);
    });

TVM_REGISTER_GLOBAL("relax.analysis.ExtractCFG").set_body_typed(ExtractCFG);

TVM_REGISTER_GLOBAL("relax.analysis.DataflowAnalysis")
    .set_body_typed([](const ControlFlowGraph& cfg, const ObjectRef& init, PackedFunc transfer_func,
                       PackedFunc merge_func, bool forward) {
      auto ret = DataflowAnalysis(cfg, init, transfer_func, merge_func, forward);
      return Array<ObjectRef>({ret.first, ret.second});
    });

}  // namespace relax
}  // namespace tvm