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

#include "graph_annotate.h"

#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>
#include <dlpack/dlpack.h>

#include <algorithm>
#include <memory>
#include <queue>
#include <stack>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nnvm {
using StringVector = std::vector<std::string>;
using IntVector = std::vector<int>;

enum class AnnotationType : int {
  kHomoTarget = 1,        // Only set target to the node attribute.
  kDeivceTarget = 2,  // Annotate both device type and target info to a node.
  kCopyInsertion = 3  // Annotate device type and target. Insert copy node.
};

// Only set the compliation target for graph nodes. For homogeneous execution,
// this is always how nodes should be annotated. The anntoated target could help
// layout altering and compute. For heterogeneous execution, all nodes in a
// graph should be scheduled to the same device (e.g. cpu) in the precompute
// pruning pass.
nnvm::Graph AnnotateTarget(nnvm::Graph&& g, const StringVector& targets,
                           const IntVector& device_types) {
  DFSVisit(g.outputs, [&](const nnvm::NodePtr& node) {
    if (device_types.size() == 1) {
        node->attrs.device_type = device_types[0];
        node->attrs.dict["target"] = targets[0];
    } else {
        const auto& it = std::find(device_types.begin(), device_types.end(),
                                   static_cast<int>(kDLCPU));
        CHECK(it != device_types.end()) << "No cpu target is found";
        node->attrs.device_type = static_cast<int>(kDLCPU);;
        node->attrs.dict["target"] =
            targets[std::distance(device_types.begin(), it)];
    }
  });
  return std::move(g);
}

// Annotate nodes with the device type.
// Also save the build target information. It will be used during lowering
// since FTVMCompute and FTVMSchedule will need target to get correct
// compute and schedule.
nnvm::Graph AnnotateDeviceTarget(nnvm::Graph&& g, const StringVector& targets,
                                 const IntVector& device_types,
                                 const ManualAnnotatorPtr& annotate) {
  DFSVisit(g.outputs, [&](const nnvm::NodePtr& node) {
    if (node->is_variable()) return;
    int device_type = annotate->AnnotateNode(node.get());
    node->attrs.device_type = device_type;
    for (const auto& e : node->inputs) {
      const nnvm::NodePtr& enode = e.node;
      if (enode->is_variable() && enode->attrs.device_type != device_type) {
        enode->attrs.device_type = device_type;
      }
    }
    const auto& it = std::find(device_types.begin(), device_types.end(),
                               node->attrs.device_type);
    CHECK(it != device_types.end())
        << "No compilation target is provided for device type: "
        << node->attrs.device_type;
    node->attrs.dict["target"] =
        targets[std::distance(device_types.begin(), it)];
  });

  return std::move(g);
}

nnvm::Graph AnnotateGraph(nnvm::Graph g) {
  CHECK(g.HasAttr("annotation_type"))
      << "annotate_type attribute is not specified. Please set it on the graph "
         "to indicate the annotation purpose.";
  AnnotationType annotation_type =
      static_cast<AnnotationType>(g.MoveCopyAttr<int>("annotation_type"));

  const auto& targets = g.GetAttr<StringVector>("target");
  const auto& device_types = g.GetAttr<IntVector>("device_type");

  if (annotation_type == AnnotationType::kHomoTarget) {
    g = AnnotateTarget(std::move(g), targets, device_types);
    return g;
  }

  const StringVector& op_names = g.HasAttr("op_name")
                                     ? g.MoveCopyAttr<StringVector>("op_name")
                                     : StringVector();
  const IntVector& op_devices = g.HasAttr("op_device")
                                    ? g.MoveCopyAttr<IntVector>("op_device")
                                    : IntVector();
  CHECK_EQ(op_names.size(), op_devices.size())
      << "The number of op names doesn't match the number of assigned device.";

  int fallback_device = 0;
  nnvm::ManualAnnotatorPtr annotate = nullptr;
  CHECK(g.HasAttr("fallback"))
      << "The fallback device is not attached to the graph.";
  fallback_device = g.MoveCopyAttr<int>("fallback");

  std::unordered_map<std::string, int> op_name_dev_map;
  for (size_t i = 0; i < op_names.size(); i++) {
    op_name_dev_map.emplace(std::make_pair(op_names[i], op_devices[i]));
  }
  annotate = std::make_shared<nnvm::ManualAnnotator>(op_name_dev_map,
                                                     fallback_device);

  if (annotation_type == AnnotationType::kDeivceTarget) {
    g = AnnotateDeviceTarget(std::move(g), targets, device_types, annotate);
  } else if (annotation_type == AnnotationType::kCopyInsertion) {
    g = AnnotateDeviceTarget(std::move(g), targets, device_types, annotate);
    g = nnvm::ApplyPass(g, "InsertDataCopy");
    // Fallback node to the default device if no device is set.
    DFSVisit(g.outputs, [fallback_device](const nnvm::NodePtr& node) {
      if (!node->is_variable() && node->attrs.device_type == 0) {
        node->attrs.device_type = fallback_device;
      }
    });
  } else {
    LOG(FATAL) << "The purpose of annotation has to be one of the types in "
                  "AnnotationType";
  }

  return g;
}

NNVM_REGISTER_PASS(AnnotateGraph)
    .describe(
        "Annotate the nodes in a graph to indicate where it should be "
        "executed.")
    .set_body(AnnotateGraph)
    .set_change_graph(true);

}  // namespace nnvm
