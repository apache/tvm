/*!
 *  Copyright (c) 2017 by Contributors
 * \file precompute_prune.cc
 * \brief Split the graph into a pre-compute graph and a execution graph.
 *
 *  The pre-compute graph outputs parameters that can be taken
 *  by execution graph during execution phase.
 */
#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/pass.h>
#include <nnvm/compiler/op_attr_types.h>
#include <unordered_set>

namespace nnvm {
namespace compiler {

nnvm::Graph PrecomputePrune(nnvm::Graph src) {
  const auto& plist
      = src.GetAttr<std::vector<std::string> >("param_name_list");
  std::unordered_set<std::string> params(plist.begin(), plist.end());

  std::unordered_set<nnvm::Node*> pruned;
  nnvm::NodeEntryMap<nnvm::NodePtr> entry_var;
  std::unordered_set<std::string> unique_name;
  // number of edges that are not variable
  int non_var_edge = 0;

  auto replace_pruned_entry = [&] (const NodeEntry& e) {
    if (!entry_var.count(e)) {
      if (!e.node->is_variable()) {
        ++non_var_edge;
      }
      nnvm::NodePtr var = nnvm::Node::Create();
      var->attrs.name = e.node->attrs.name;
      if (e.node->num_outputs() != 1) {
        var->attrs.name += "_output" + std::to_string(e.index);
      }
      entry_var.emplace(e, var);
      CHECK(!unique_name.count(var->attrs.name));
      unique_name.insert(var->attrs.name);
      return nnvm::NodeEntry{var, 0, 0};
    } else {
      return nnvm::NodeEntry{entry_var.at(e), 0, 0};
    }
  };

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
        if (pruned.count(e.node.get())) {
          e = replace_pruned_entry(e);
        }
      }
    }
  });

  // nothing being pruned.
  if (non_var_edge == 0) {
    return src;
  }

  for (auto& e : src.outputs) {
    if (pruned.count(e.node.get())) {
      e = replace_pruned_entry(e);
    }
  }

  nnvm::Graph pre_graph;
  pre_graph.outputs.reserve(entry_var.size());
  std::vector<std::string> output_names;
  output_names.reserve(entry_var.size());

  for (auto kv : entry_var) {
    pre_graph.outputs.emplace_back(kv.first);
    output_names.emplace_back(kv.second->attrs.name);
  }
  // new parameter list
  pre_graph.attrs["output_names"] =
      std::make_shared<dmlc::any>(std::move(output_names));
  src.attrs["precompute_graph"] =
      std::make_shared<dmlc::any>(std::move(pre_graph));
  return src;
}

NNVM_REGISTER_PASS(PrecomputePrune)
.set_body(PrecomputePrune);
}  // namespace compiler
}  // namespace nnvm
