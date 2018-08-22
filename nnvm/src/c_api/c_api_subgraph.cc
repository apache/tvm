/*!
 *  Copyright (c) 2018 by Contributors
 * \file c_api_subgraph.cc
 * \brief C API of nnvm for the ease of testing backend in Python
 */
#include <nnvm/c_api_subgraph.h>
#include <nnvm/pass.h>
#include "./c_api_common.h"
#include "../pass/subgraph/subgraph_property.h"

int NNPartitionGraph(GraphHandle graph_handle,
                     const char* prop_name,
                     const nn_uint num_ops,
                     const char** op_names,
                     GraphHandle* ret_graph_handle) {
  nnvm::Graph* g = new nnvm::Graph();
  API_BEGIN();
  std::unordered_set<std::string> op_name_set;
  for (size_t i = 0; i < num_ops; ++i) {
    op_name_set.emplace(op_names[i]);
  }
  nnvm::Graph* graph = static_cast<nnvm::Graph*>(graph_handle);
  nnvm::Symbol sym;
  sym.outputs = graph->outputs;
  sym = sym.Copy();
  nnvm::Graph tmp_graph;
  tmp_graph.outputs = sym.outputs;
  tmp_graph.attrs = graph->attrs;
  if (!op_name_set.empty()) {
    nnvm::pass::SubgraphPropertyPtr property
        = nnvm::pass::SubgraphPropertyRegistry::Get()->CreateSubgraphProperty(prop_name);
    property->SetAttr("op_names", op_name_set);
    tmp_graph.attrs["subgraph_property"] = std::make_shared<nnvm::any>(std::move(property));
  }
  tmp_graph = nnvm::ApplyPass(std::move(tmp_graph), "GraphPartition");
  g->outputs = tmp_graph.outputs;
  *ret_graph_handle = g;
  API_END_HANDLE_ERROR(delete g);
}

int NNSetSubgraphPropertyOpNames(const char* prop_name,
                                 const nn_uint num_ops,
                                 const char** op_names) {
  API_BEGIN();
  std::unordered_set<std::string> op_name_set;
  for (size_t i = 0; i < num_ops; ++i) {
    op_name_set.emplace(op_names[i]);
  }
  (*nnvm::pass::SubgraphPropertyOpNameSet::Get())[prop_name] = op_name_set;
  API_END();
}

int NNRemoveSubgraphPropertyOpNames(const char* prop_name) {
  API_BEGIN();
  nnvm::pass::SubgraphPropertyOpNameSet::Get()->erase(prop_name);
  API_END();
}
