/*!
 * Copyright (c) 2018 by Contributors
 * \file place_copy_op.cc
 * \brief Place corss device data copy nodes on entries where two nodes are
 * assigned to different devices.
 */
#include <dlpack/dlpack.h>
#include <nnvm/graph_annotate.h>
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <nnvm/graph_attr_types.h>

#include <algorithm>
#include <queue>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nnvm {
namespace pass {

nnvm::Graph PlaceDataCopy(nnvm::Graph g) {
  if (!g.HasAttr("annotated")) {
    LOG(ERROR) << "Nodes in the graph are not annotated with context info yet. "
                 "Run AnnotateGraph pass first.";
    return g;
  }
  const nnvm::Op* copy_op = nnvm::Op::Get("device_copy_op");

  // Insert a copy node between two nodes if their device types are different.
  DFSVisit(g.outputs, [&copy_op](const nnvm::NodePtr& node) {
    const auto& device_type = node->attrs.device;
    for (size_t i = 0; i < node->inputs.size(); ++i) {
      const auto& entry = node->inputs[i];
      if (entry.node->attrs.device != device_type) {
        nnvm::NodePtr copy_node = nnvm::Node::Create();
        std::ostringstream os;
        os << "__copy_" << entry.node->attrs.name << "_to_" << node->attrs.name;
        copy_node->attrs.op = copy_op;
        copy_node->attrs.name = os.str();
        copy_node->attrs.device = node->attrs.device;
        copy_node->inputs.push_back(entry);
        if (copy_op->attr_parser != nullptr) {
          copy_node->attrs.op->attr_parser(&(copy_node->attrs));
        }
        // node->inputs[i].node = copy_node;
        node->inputs[i] = NodeEntry({copy_node, 0, 0});
      }
    }
  });

  const auto& idx = g.indexed_graph();
  DeviceVector device_vec(idx.num_nodes(), -1);
  for (size_t i = 0; i < idx.num_nodes(); i++) {
    device_vec[i] = static_cast<int>(idx[i].source->attrs.device);
  }
  g.attrs["device"] = std::make_shared<any>(std::move(device_vec));

  return g;
}

NNVM_REGISTER_PASS(PlaceDataCopy)
    .describe("Insert cross device data copy nodes to transfer data between "
              "opertors that are executed on different devices.")
    .set_body(PlaceDataCopy)
    .set_change_graph(true)
    .depend_graph_attr("annotated");

}  // namespace pass
}  // namespace nnvm
