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

TVM_REGISTER_NODE_TYPE(GraphBindingNode);

GraphBinding::GraphBinding(const SeqExpr& seq, const Array<Var>& args, size_t block_idx,
                           size_t binding_idx, BindingNodeKind kind) {
  ObjectPtr<GraphBindingNode> n = make_object<GraphBindingNode>();
  n->seq = seq;
  n->args = args;
  n->block_idx = block_idx;
  n->binding_idx = binding_idx;
  n->kind = kind;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ControlFlowGraphNode);

ControlFlowGraph::ControlFlowGraph(const Array<GraphBinding>& bindings,
                                   const Array<Array<Integer>>& preds,
                                   const Array<Array<Integer>>& succs) {
  ObjectPtr<ControlFlowGraphNode> n = make_object<ControlFlowGraphNode>();
  n->bindings = bindings;
  n->preds = preds;
  n->succs = succs;
  data_ = std::move(n);
}

// Extracts a basic block and updates the running lists bindings, preds, and succs.
// The return value is the index of the final binding processed in the seq expression
// (useful for processing branches).
size_t ExtractCFGHelper(const SeqExpr& seq, const Array<Var>& args, size_t block_idx,
                        size_t binding_idx, std::vector<size_t> current_preds,
                        std::vector<GraphBinding>* bindings,
                        std::vector<std::vector<size_t>>* preds,
                        std::vector<std::vector<size_t>>* succs) {
  // case 1: We're past the end -> this is the block body (base case)
  if (block_idx == seq->blocks.size()) {
    bindings->push_back(GraphBinding(seq, args, block_idx, 0U, BindingNodeKind::kSeqBody));
    preds->push_back(current_preds);
    // the final binding has no successors
    succs->push_back({});
    return bindings->size() - 1;
  }

  Binding binding = seq->blocks[block_idx]->bindings[binding_idx];
  Expr binding_value = GetBoundValue(binding);

  // case 2: Ordinary binding
  if (!binding_value.as<IfNode>()) {
    bindings->push_back(GraphBinding(seq, args, block_idx, binding_idx, BindingNodeKind::kBinding));
    size_t idx = bindings->size() - 1;
    preds->push_back(current_preds);
    // successor: the next binding (there will always be at least one binding after this,
    // even if it's the seq body)
    succs->push_back({idx + 1});
  } else {
    // case 3: dealing with a branch
    auto if_node = Downcast<If>(binding_value);
    // start with the cond node
    bindings->push_back(GraphBinding(seq, args, block_idx, binding_idx, BindingNodeKind::kIfCond));
    size_t idx = bindings->size() - 1;
    preds->push_back(current_preds);
    // there will be another successor, which we will add after recursing down the branches
    succs->push_back({idx + 1});
    size_t final_true_idx = ExtractCFGHelper(Downcast<SeqExpr>(if_node->true_branch), {}, 0U, 0U,
                                             {idx}, bindings, preds, succs);
    succs->at(idx).push_back(final_true_idx + 1);
    size_t final_false_idx = ExtractCFGHelper(Downcast<SeqExpr>(if_node->false_branch), {}, 0U, 0U,
                                              {idx}, bindings, preds, succs);
    // now create the merge
    bindings->push_back(GraphBinding(seq, {}, block_idx, binding_idx, BindingNodeKind::kIfMerge));
    size_t merge_idx = bindings->size() - 1;
    preds->push_back({final_true_idx, final_false_idx});
    succs->push_back({merge_idx + 1});
    // update the successors of the final true and false indices as well
    succs->at(final_true_idx).push_back(merge_idx);
    succs->at(final_false_idx).push_back(merge_idx);
  }
  // move on to next binding
  size_t next_block_idx = block_idx;
  size_t next_binding_idx = binding_idx + 1;
  if (next_binding_idx >= seq->blocks[block_idx]->bindings.size()) {
    next_block_idx = block_idx + 1;
    next_binding_idx = 0U;
  }
  return ExtractCFGHelper(seq, {}, next_block_idx, next_binding_idx, {bindings->size() - 1},
                          bindings, preds, succs);
}

ControlFlowGraph ExtractCFG(const Function& func) {
  std::vector<GraphBinding> bindings;
  std::vector<std::vector<size_t>> preds;
  std::vector<std::vector<size_t>> succs;
  ExtractCFGHelper(Downcast<SeqExpr>(func->body), func->params, 0U, 0U, {}, &bindings, &preds,
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
  return ControlFlowGraph(Array<GraphBinding>(bindings), pred_arr, succ_arr);
}

std::pair<Array<ObjectRef>, Array<ObjectRef>> DataflowAnalysis(
    const ControlFlowGraph& cfg, const ObjectRef& init,
    std::function<ObjectRef(const GraphBinding&, const ObjectRef&)> transfer_func,
    std::function<ObjectRef(const ObjectRef&, const ObjectRef&)> merge_func, bool forward) {
  std::vector<ObjectRef> in_map;
  std::vector<ObjectRef> out_map;
  for (size_t i = 0; i < cfg->bindings.size(); i++) {
    in_map.push_back(init);
    out_map.push_back(init);
  }

  // Modification from Adrian Sampson's version:
  // Since there are no loops in our AST, one traversal through the CFG suffices.
  // We will do BFS
  std::queue<size_t> worklist;
  worklist.push((forward) ? 0 : cfg->bindings.size() - 1);
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
    results->operator[](idx) = transfer_func(cfg->bindings[idx], inputs->at(idx));

    for (Integer next_idx : next) {
      worklist.push(next_idx.IntValue());
    }
  }

  return {Array<ObjectRef>(in_map), Array<ObjectRef>(out_map)};
}

size_t GetBindingIndex(const ControlFlowGraph& cfg, const SeqExpr& seq, size_t block_idx,
                       size_t binding_idx, bool match_cond) {
  bool is_body = (block_idx == seq->blocks.size());
  bool is_if =
      (!is_body && (GetBoundValue(seq->blocks[block_idx]->bindings[binding_idx]).as<IfNode>()));

  // This is an inefficient linear scan; it could be improved by keeping a map of
  // SeqExprs to indices in the CFG data structure.
  // That should be considered if this function poses performance issues (unlikely).
  for (size_t i = 0; i < cfg->bindings.size(); i++) {
    auto binding = cfg->bindings[i];
    if (binding->seq != seq) {
      continue;
    }
    if (is_body && binding->kind == BindingNodeKind::kSeqBody) {
      return i;
    }
    if (binding->block_idx == block_idx && binding->binding_idx == binding_idx) {
      if (!is_if || (match_cond && binding->kind == BindingNodeKind::kIfCond) ||
          (!match_cond && binding->kind == BindingNodeKind::kIfMerge)) {
        return i;
      }
    }
  }
  CHECK(false) << "Target binding does not appear in the given CFG";
  return cfg->bindings.size();
}

TVM_REGISTER_GLOBAL("relax.analysis.GraphBinding")
    .set_body_typed([](const SeqExpr& seq, const Array<Var>& args, size_t block_idx,
                       size_t binding_idx, int kind) {
      return GraphBinding(seq, args, block_idx, binding_idx, static_cast<BindingNodeKind>(kind));
    });

TVM_REGISTER_GLOBAL("relax.analysis.ControlFlowGraph")
    .set_body_typed([](const Array<GraphBinding>& blocks, const Array<Array<Integer>>& preds,
                       const Array<Array<Integer>>& succs) {
      return ControlFlowGraph(blocks, preds, succs);
    });

TVM_REGISTER_GLOBAL("relax.analysis.ExtractCFG").set_body_typed(ExtractCFG);

TVM_REGISTER_GLOBAL("relax.analysis.DataflowAnalysis")
    .set_body_typed([](const ControlFlowGraph& cfg, const ObjectRef& init, PackedFunc transfer_func,
                       PackedFunc merge_func, bool forward) {
      auto ret = DataflowAnalysis(cfg, init, transfer_func, merge_func, forward);
      return Array<ObjectRef>({ret.first, ret.second});
    });

// need to turn the size_t's into ints in order to cross the C++<->Python boundary
TVM_REGISTER_GLOBAL("relax.analysis.GetBindingIndex")
    .set_body_typed([](const ControlFlowGraph& cfg, const SeqExpr& seq, int block_idx,
                       int binding_idx, bool match_cond) -> int {
      return GetBindingIndex(cfg, seq, block_idx, binding_idx, match_cond);
    });

}  // namespace relax
}  // namespace tvm
