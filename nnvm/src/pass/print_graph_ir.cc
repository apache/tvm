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
 * \file print_graph_ir.cc
 * \brief Print the graph IR in LLVM style human readable format.
 */
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <nnvm/tuple.h>

#include <iostream>

namespace nnvm {
namespace pass {

using AttrPrinter = std::function<void(uint32_t index, std::ostream& os)>;  // NOLINT(*)

template <typename T>
AttrPrinter GetVectorPrinter_(const T& vec) {
  return [&vec](uint32_t index, std::ostream& os) {  // NOLINT(*)
    os << vec[index];
  };
}

AttrPrinter GetVectorPrinter(const Graph& graph, const std::string& key) {
  auto it = graph.attrs.find(key);
  CHECK(it != graph.attrs.end()) << "Cannot find " << key << " in graph attr";
  const any& value = *(it->second);
  if (value.type() == typeid(std::vector<TShape>)) {
    return GetVectorPrinter_(nnvm::get<std::vector<TShape> >(value));
  } else if (value.type() == typeid(std::vector<int>)) {
    return GetVectorPrinter_(nnvm::get<std::vector<int> >(value));
  } else if (value.type() == typeid(std::vector<std::string>)) {
    return GetVectorPrinter_(nnvm::get<std::vector<std::string> >(value));
  } else {
    LOG(FATAL) << "Cannot handle type " << value.type().name();
    return nullptr;
  }
}

// print the graph ir in readable format
void PrintGraphIR_(Graph src, const std::vector<std::string>& join_entry_attrs,
                   const std::vector<std::string>& join_node_attrs,
                   std::ostream& os) {  // NOLINT(*)
  const IndexedGraph& idx = src.indexed_graph();
  std::vector<std::function<void(uint32_t, std::ostream&)> > trigger;  // NOLINT(*)

  for (const std::string& key : join_entry_attrs) {
    AttrPrinter fp = GetVectorPrinter(src, key);
    auto fprint = [&idx, key, fp](uint32_t nid, std::ostream& os) {  // NOLINT(*)
      const IndexedGraph::Node& inode = idx[nid];
      os << ", " << key << "=";
      if (inode.source->num_outputs() != 1) {
        os << '[';
        for (uint32_t i = 0; i < inode.source->num_outputs(); ++i) {
          if (i != 0) os << ", ";
          fp(idx.entry_id(nid, i), os);
        }
        os << ']';
      } else {
        fp(idx.entry_id(nid, 0), os);
      }
    };
    trigger.push_back(fprint);
  }
  for (const std::string& key : join_node_attrs) {
    AttrPrinter fp = GetVectorPrinter(src, key);
    auto fprint = [&idx, key, fp](uint32_t nid, std::ostream& os) {  // NOLINT(*)
      os << ", " << key << "=";
      fp(idx.entry_id(nid, 0), os);
    };
    trigger.push_back(fprint);
  }

  os << "Graph(";
  if (idx.input_nodes().size() < 4) {
    for (size_t i = 0; i < idx.input_nodes().size(); ++i) {
      uint32_t nid = idx.input_nodes()[i];
      if (i != 0) {
        os << ", ";
      }
      os << '%' << idx[nid].source->attrs.name;
    }
  } else {
    for (size_t i = 0; i < idx.input_nodes().size(); ++i) {
      uint32_t nid = idx.input_nodes()[i];
      if (i != 0) {
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

  if (trigger.size() != 0) {
    for (size_t i = 0; i < idx.input_nodes().size(); ++i) {
      uint32_t nid = idx.input_nodes()[i];
      os << "  %" << idx[nid].source->attrs.name;
      for (const auto& fp : trigger) {
        fp(nid, os);
      }
      os << '\n';
    }
  }

  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    os << "  "
       << "%" << nid << " = " << inode.source->op()->name << "(";
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
    // additional attribute trigger
    for (const auto& fp : trigger) {
      fp(nid, os);
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
Graph PrintGraphIRPass(Graph src) {
  std::ostringstream os;
  std::vector<std::string> join_entry_attrs, join_node_attrs;
  if (src.attrs.count("join_entry_attrs") != 0) {
    join_entry_attrs = src.MoveCopyAttr<std::vector<std::string> >("join_entry_attrs");
  }
  if (src.attrs.count("join_node_attrs") != 0) {
    join_node_attrs = src.MoveCopyAttr<std::vector<std::string> >("join_node_attrs");
  }
  PrintGraphIR_(src, join_entry_attrs, join_node_attrs, os);
  Graph ret;
  ret.attrs["graphir"] = std::make_shared<any>(os.str());
  return ret;
}

// register pass
NNVM_REGISTER_PASS(PrintGraphIR)
    .describe("Return a empty Graph, save ir to ret.attrs[\"graphir\"]")
    .set_body(PrintGraphIRPass);

}  // namespace pass
}  // namespace nnvm
