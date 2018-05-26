/*!
 *  Copyright (c) 2017 by Contributors
 * \file prune_graph.cc
 * \brief Prune the graph to do constant folding.
 *
 *  This pass breaks the graph into pre-compute graph
 *  and the execution graph.
 */
#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/pass.h>
#include <nnvm/compiler/op_attr_types.h>
#include <unordered_set>

namespace nnvm {
namespace compiler {

nnvm::Graph PruneGraph(nnvm::Graph src) {
  const auto& params = src.GetAttr<std::unordered_set<std::string> >("params");

  std::unordered_set<nnvm::Node*> pruned;
  nnvm::NodeEntryMap<nnvm::NodePtr> entry_var;
  DFSVisit(src.outputs, [&](const nnvm::NodePtr& n) {
    bool can_be_pruned = true;
    if (n->is_variable()) {
      if (params.count(n->attrs.name)) {
        pruned.emplace(n.get());
      }
      can_be_pruned = false;
    }

    for (const auto& e : n->inputs) {
      if (!pruned.count(e.node.get())) {
        can_be_pruned = false;
      }
    }
    if (can_be_pruned) {
      pruned.emplace(n.get());
    } else {
      // scan again to find edge nodes, skip variables
      for (auto& e : n->inputs) {
        if (!e.node->is_variable() && pruned.count(e.node.get())) {
          if (!entry_var.count(e)) {
            nnvm::NodePtr var = nnvm::Node::Create();
            var->attrs.name = e.node->attrs.name + "_output" + std::to_string(e.index);
            entry_var.emplace(e, var);
          }
          e = nnvm::NodeEntry{entry_var.at(e), 0, 0};
        }
      }
    }
  });

  nnvm::Graph pre_graph;
  pre_graph.outputs.reserve(entry_var.size());
  std::vector<std::string> output_names;
  output_names.reserve(entry_var.size());
  for (auto kv : entry_var) {
    if (kv.first.node->is_variable()) continue;
    pre_graph.outputs.emplace_back(kv.first);
    output_names.emplace_back(kv.second->attrs.name);
  }

  pre_graph.attrs["pruned_params"] =
    std::make_shared<dmlc::any>(std::move(output_names));
  src.attrs["pre_graph"] =
    std::make_shared<dmlc::any>(std::move(pre_graph));
  return src;
}

NNVM_REGISTER_PASS(PruneGraph)
.set_body(PruneGraph);

}  // namespace compiler
}  // namespace nnvm
