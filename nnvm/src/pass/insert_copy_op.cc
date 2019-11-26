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

#include <nnvm/graph.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/pass.h>

#include <algorithm>
#include <queue>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nnvm {

nnvm::Graph InsertDataCopy(nnvm::Graph g) {
  const nnvm::Op* copy_op = nnvm::Op::Get("device_copy_op");

  // Insert a copy node between two nodes if their device types are different.
  DFSVisit(g.outputs, [&copy_op](const nnvm::NodePtr& node) {
    auto device_type = node->attrs.device_type;
    for (size_t i = 0; i < node->inputs.size(); ++i) {
      const auto& entry = node->inputs[i];
      if (entry.node->attrs.device_type != device_type) {
        nnvm::NodePtr copy_node = nnvm::Node::Create();
        std::ostringstream os;
        os << "__copy_" << entry.node->attrs.name << "_to_" << node->attrs.name;
        copy_node->attrs.op = copy_op;
        copy_node->attrs.name = os.str();
        copy_node->attrs.device_type = node->attrs.device_type;
        copy_node->inputs.push_back(entry);
        if (copy_op->attr_parser != nullptr) {
          copy_node->attrs.op->attr_parser(&(copy_node->attrs));
        }
        node->inputs[i] = NodeEntry({copy_node, 0, 0});
      }
    }
  });

  const auto& idx = g.indexed_graph();
  DeviceVector device_vec(idx.num_node_entries(), 0);
  for (size_t i = 0; i < idx.num_nodes(); i++) {
    device_vec[idx.entry_id(i, 0)] = idx[i].source->attrs.device_type;
  }
  for (uint32_t nid = 0; nid < idx.num_nodes(); nid++) {
    const auto& inode = idx[nid];
    for (const auto& e : inode.inputs) {
      device_vec[idx.entry_id(e)] = idx[e.node_id].source->attrs.device_type;
    }
  }
  g.attrs["device_index"] = std::make_shared<any>(std::move(device_vec));
  return g;
}

NNVM_REGISTER_PASS(InsertDataCopy)
.describe("Insert cross device data copy nodes to transfer data between "
"opertors that are executed on different devices.")
.set_body(InsertDataCopy)
.set_change_graph(true);

}  // namespace nnvm
