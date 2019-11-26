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

#ifndef NNVM_PASS_GRAPH_ANNOTATE_H_
#define NNVM_PASS_GRAPH_ANNOTATE_H_

#include <nnvm/graph.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace nnvm {

class ManualAnnotator;
/*
 * This class is an abstract class that can be derived by other classes to
 * implement how a node should be selected.
 */
class GraphAnnotator {
 public:
  explicit GraphAnnotator(int fallback_device)
      : fallback_device_(fallback_device) {}
  virtual ~GraphAnnotator() = default;
  // A virtual function that is implemented by different annotation methods.
  virtual int AnnotateNode(const nnvm::Node* n) const = 0;

  int GetFallbackDevice() const {
    return fallback_device_;
  }

 private:
  friend ManualAnnotator;
  /* The fallback device. */
  int fallback_device_;
};

/*
 * This class defines a manual way to annotate a graph node. In this method,
 * users are expected to provide the node name and also the device type that it
 * should be assigned to. However, if the operator contained in the graph node
 * is registered with a fallback property or the operator name has not been
 * saved, this node will be annotated with the fallback device.
 */
class ManualAnnotator : public GraphAnnotator {
  using OpNameDeviceMap = std::unordered_map<std::string, int>;
 public:
  explicit ManualAnnotator(const OpNameDeviceMap& op_name_dev_map,
                           int fallback_device)
      : GraphAnnotator(fallback_device),
        op_name_dev_map_(new OpNameDeviceMap(op_name_dev_map)) {}

  int AnnotateNode(const nnvm::Node* n) const final {
    if (n->is_variable()) return 0;
    if (n->op()->fallback) return fallback_device_;

    return op_name_dev_map_->count(n->op()->name)
               ? op_name_dev_map_->at(n->op()->name)
               : fallback_device_;
  }

 private:
  std::unique_ptr<const OpNameDeviceMap> op_name_dev_map_;
};

using ManualAnnotatorPtr = std::shared_ptr<ManualAnnotator>;

}  // namespace nnvm
#endif  // NNVM_PASS_GRAPH_ANNOTATE_H_
