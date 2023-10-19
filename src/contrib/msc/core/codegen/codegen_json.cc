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
 * \file src/contrib/msc/core/codegen/codegen_json.cc
 */

#include "codegen_json.h"

#include <memory>

namespace tvm {
namespace contrib {
namespace msc {

std::vector<JSONGraphNodeEntry> MSCJSONSerializer::VisitExpr_(const CallNode* call_node) {
  const auto& ref_node = graph_->FindNode(SpanUtils::GetAttr(call_node->span, "name"));
  std::vector<JSONGraphNodeEntry> inputs;
  for (const auto& arg : call_node->args) {
    auto res = VisitExpr(arg);
    inputs.insert(inputs.end(), res.begin(), res.end());
  }
  auto node =
      std::make_shared<JSONGraphNode>(ref_node->name, "kernel", inputs, ref_node->outputs.size());
  // add attributes
  AddNodeAttr(node, "optype", ref_node->optype);
  for (const auto& pair : ref_node->attrs) {
    AddNodeAttr(node, pair.first, pair.second);
  }
  if (!global_options_set_) {
    AddNodeAttr(node, "msc_global_options_num", std::to_string(options_.size()));
    for (const auto& pair : options_) {
      AddNodeAttr(node, "msc_global_" + pair.first, pair.second);
    }
    global_options_set_ = true;
  }
  return AddNode(node, GetRef<Expr>(call_node));
}

void MSCJSONSerializer::AddNodeAttr(JSONGraphObjectPtr node, const String& key,
                                    const String& value) {
  std::vector<std::string> array_value{std::string(value)};
  std::vector<dmlc::any> dmlc_value;
  dmlc_value.emplace_back(array_value);
  node->SetAttr(std::string(key), dmlc_value);
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
