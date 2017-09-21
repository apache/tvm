/*!
 *  Copyright (c) 2017 by Contributors
 * \file print_graph_ir.cc
 * \brief Print the graph IR in LLVM style human readable format.
 */
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <iostream>

namespace nnvm {
namespace pass {

// print the graph ir in readable format
void PrintGraphIR_(Graph src, std::ostream& os) { // NOLINT(*)
  const IndexedGraph& idx = src.indexed_graph();
  os << "Graph(";
  if (idx.input_nodes().size() < 4) {
    for (size_t i = 0; i < idx.input_nodes().size(); ++i) {
      uint32_t nid = idx.input_nodes()[i];
      if (i != 0)  {
        os << ", ";
      }
      os << '%' << idx[nid].source->attrs.name;
    }
  } else {
    for (size_t i = 0; i < idx.input_nodes().size(); ++i) {
      uint32_t nid = idx.input_nodes()[i];
      if (i != 0)  {
        os << ",\n      ";
      }
      os << '%' << idx[nid].source->attrs.name;
    }
  }
  os << ") {\n";

  auto print_entry = [&](const IndexedGraph::NodeEntry& e) {
    if (idx[e.node_id].source->is_variable()) {
      os << '%' << idx[e.node_id].source->attrs.name;
    } else if (idx[e.node_id].source->num_outputs() == 1) {
      os << '%' << e.node_id;
    } else {
      os << '%' << e.node_id << "." << e.index;
    }
  };

  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    os << "  " << "%" << nid << " = "
       << inode.source->op()->name << "(";
    bool first = true;
    for (const IndexedGraph::NodeEntry& e : inode.inputs) {
      if (first) {
        first = false;
      } else {
        os << ", ";
      }
      print_entry(e);
    }
    for (const auto& kv : inode.source->attrs.dict) {
      if (first) {
        first = false;
      } else {
        os << ", ";
      }
      os << kv.first << "=\'" << kv.second << "\'";
    }
    os << ")";
    if (inode.control_deps.size() != 0) {
      os << ", control_deps=[";
      for (size_t i = 0; i < inode.control_deps.size(); ++i) {
        if (i != 0) os << ", ";
        uint32_t cid = inode.control_deps[i];
        if (idx[cid].source->is_variable()) {
          os << '%' << idx[cid].source->attrs.name;
        } else {
          os << '%' << cid;
        }
      }
      os << "]";
    }
    os << "\n";
  }
  os << "  ret ";
  {
    bool first = true;
    for (const IndexedGraph::NodeEntry& e : idx.outputs()) {
      if (first) {
        first = false;
      } else {
        os << ", ";
      }
      print_entry(e);
    }
  }
  os << "\n}";
  if (src.attrs.size() != 0) {
    os << "\ngraph_attr_keys = [";
    bool first = true;
    for (const auto& kv : src.attrs) {
      if (first) {
        first = false;
      } else {
        os << ", ";
      }
      os << kv.first;
    }
    os << "]\n";
  }
}

// save a graph to json
Graph PrintGraphIR(Graph src) {
  std::ostringstream os;
  PrintGraphIR_(src, os);
  Graph ret;
  ret.attrs["graphir"] = std::make_shared<any>(os.str());
  return ret;
}

// register pass
NNVM_REGISTER_PASS(PrintGraphIR)
.describe("Return a empty Graph, save ir to ret.attrs[\"graphir\"]")
.set_body(PrintGraphIR);

}  // namespace pass
}  // namespace nnvm
