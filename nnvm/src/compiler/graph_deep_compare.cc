/*!
 *  Copyright (c) 2017 by Contributors
 * \file graph_deep_compare.cc
 * \brief Deep compare two graph structure
 */
#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/packed_func_ext.h>
#include <tvm/runtime/packed_func.h>
#include "./node_attr.h"

namespace nnvm {
namespace compiler {

// deep compare the graph structure
// not considering the graph attributes
// return non-empty error message if the graph mismatch.
// the comparator won't match name of intermediate node.
// compare_var_attr
std::string DeepCompare(Graph a, Graph b,
                        bool compare_variable_attr) {
  const IndexedGraph& idxa = a.indexed_graph();
  const IndexedGraph& idxb = b.indexed_graph();
  std::ostringstream err;
  if (idxa.num_nodes() != idxb.num_nodes()) {
    err << "Number of nodes mismatch";
    return err.str();
  }
  if (idxa.num_node_entries() != idxb.num_node_entries()) {
    err << "Number of node entry mismatch";
    return err.str();
  }
  if (idxa.outputs().size() != idxb.outputs().size()) {
    err << "Number of outputs mismatch";
    return err.str();
  }
  for (size_t i = 0; i < idxa.outputs().size(); ++i) {
    if (idxa.outputs()[i].node_id != idxb.outputs()[i].node_id ||
        idxa.outputs()[i].index != idxb.outputs()[i].index) {
      err << "Output entry mismatch";
      return err.str();
    }
  }
  if (idxa.input_nodes().size() != idxb.input_nodes().size()) {
    err << "Number of inputs mismatch";
    return err.str();
  }

  for (uint32_t nid = 0; nid < idxa.num_nodes(); ++nid) {
    const IndexedGraph::Node& anode = idxa[nid];
    const IndexedGraph::Node& bnode = idxb[nid];
    if (anode.source->op() != bnode.source->op()) {
      err << "Node mismatch ";
      return err.str();
    }
    if (anode.source->is_variable()) {
      CHECK(bnode.source->is_variable());
      if (!compare_variable_attr) continue;
    }
    AttrDict adict = GetAttrDict(anode.source->attrs);
    AttrDict bdict = GetAttrDict(bnode.source->attrs);

    auto fmatch = [&err, &anode](const AttrDict& adict, const AttrDict& bdict) {
      for (const auto& kv : adict) {
        auto it = bdict.find(kv.first);
        if (it != bdict.end()) {
          if (it->second != kv.second) {
            err << "Node attr mismatch, op=" << anode.source->attrs.name
                << " attr_key=" << kv.first << " " << it->second
                << " v.s. " << kv.second;
            return false;
          }
        } else {
          err << "One attr_key=" << kv.first << " is missing in another "
               << "op=" << anode.source->attrs.name;
          return false;
        }
      }
      return true;
    };
    if (!fmatch(adict, bdict)) return err.str();
    if (adict.size() != bdict.size()) {
      CHECK(!fmatch(bdict, adict));
      return err.str();
    }
    if (anode.inputs.size() != bnode.inputs.size()) {
      err << "Node input mismatch, op=" << anode.source->attrs.name;
      return err.str();
    }
    if (anode.control_deps.size() != bnode.control_deps.size()) {
      err << "Node control_deps mistach, op=" << anode.source->attrs.name;
      return err.str();
    }
    for (size_t i = 0; i < anode.inputs.size(); ++i) {
      const IndexedGraph::NodeEntry& ae = anode.inputs[i];
      const IndexedGraph::NodeEntry& be = bnode.inputs[i];
      if (ae.node_id != be.node_id ||
          ae.index != be.index ||
          ae.version != be.version) {
        err << "Node input mismatch on, op=" << anode.source->attrs.name;
        return err.str();
      }
    }
    for (size_t i = 0; i < anode.control_deps.size(); ++i) {
      if (anode.control_deps[i] != bnode.control_deps[i]) {
        err << "Node control_dep mismatch on, op=" << anode.source->attrs.name;
        return err.str();
      }
    }
  }
  return "";
}

TVM_REGISTER_GLOBAL("nnvm.graph.DeepCompare")
.set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue *rv) {
    *rv = DeepCompare(args[0], args[1], args[2]);
  });
}  // namespace compiler
}  // namespace nnvm
