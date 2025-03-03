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
 * \file src/contrib/msc/core/codegen/codegen_utils.cc
 */

#include "codegen_utils.h"

namespace tvm {
namespace contrib {
namespace msc {

const String CodeGenUtils::IdxNode(const MSCJoint& node, const String& prefix,
                                   const String& suffix) {
  return prefix + std::to_string(node->index) + suffix;
}

const String CodeGenUtils::IdxOutput(const MSCJoint& node, const String& prefix, int idx,
                                     const String& suffix) {
  const auto& idx_node = IdxNode(node, prefix, suffix);
  size_t output_size = node->outputs.size();
  if (output_size == 1 && node->optype != "tuple") {
    return idx_node;
  }
  size_t v_index = CommonUtils::GetIndex(idx, output_size);
  return idx_node + "[" + std::to_string(v_index) + "]";
}

const String CodeGenUtils::IdxInput(const MSCJoint& node, const String& prefix, int idx,
                                    const String& suffix) {
  const auto& pair = node->ProducerAndIdxOf(idx);
  return IdxOutput(pair.first, prefix, pair.second, suffix);
}

const String CodeGenUtils::IdxWeight(const MSCJoint& node, const String& wtype,
                                     const String& suffix) {
  return wtype + "_" + std::to_string(node->index) + suffix;
}

const Array<String> CodeGenUtils::GetPrims(const MSCTensor& tensor,
                                           const Map<String, String>& prims) {
  Array<String> dims;
  if (tensor->prims.size() == 0) {
    for (size_t i = 0; i < tensor->Ndim(); i++) {
      dims.push_back(StringUtils::ToString(tensor->DimAt(i)));
    }
    return dims;
  }
  for (size_t i = 0; i < tensor->Ndim(); i++) {
    const auto& prim = tensor->PrimAt(i);
    dims.push_back(prims.count(prim) ? prims[prim] : prim);
  }
  return dims;
}

const String CodeGenUtils::CommentNode(const MSCJoint& node, const String& prefix,
                                       const Map<String, String>& prims) {
  String comment = node->name + "(" + node->optype + "): <";
  for (size_t i = 0; i < node->inputs.size(); i++) {
    comment = comment + IdxInput(node, prefix, i) + (i == node->inputs.size() - 1 ? "> -> <" : ",");
  }
  for (size_t i = 0; i < node->outputs.size(); i++) {
    const auto& t_output = node->OutputAt(i);
    const auto& t_prims = GetPrims(t_output, prims);
    comment = comment + IdxOutput(node, prefix, i) + "|" + StringUtils::Join(t_prims, ":");
    comment = comment + "|" + t_output->DTypeName();
    if (t_output->layout.defined()) {
      comment = comment + "|" + t_output->layout->name;
    }
    comment = comment + (i == node->outputs.size() - 1 ? ">" : ", ");
  }
  return comment;
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
