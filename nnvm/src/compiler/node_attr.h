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
 * Copyright (c) 2017 by Contributors
 * \file node_attr.h
 * \brief utility to access node attributes
*/
#ifndef NNVM_COMPILER_NODE_ATTR_H_
#define NNVM_COMPILER_NODE_ATTR_H_

#include <nnvm/op.h>
#include <nnvm/compiler/op_attr_types.h>
#include <unordered_map>
#include <string>

namespace nnvm {
namespace compiler {

using AttrDict = std::unordered_map<std::string, std::string>;
/*!
 * \brief Get canonicalized attr dict from node
 * \param attrs The node attrs
 * \return The attribute dict
 */
inline AttrDict GetAttrDict(const NodeAttrs& attrs) {
  static auto& fgetdict = nnvm::Op::GetAttr<FGetAttrDict>("FGetAttrDict");
  if (fgetdict.count(attrs.op)) {
    return fgetdict[attrs.op](attrs);
  } else {
    return attrs.dict;
  }
}

}  // namespace compiler
}  // namespace nnvm
#endif  // NNVM_COMPILER_NODE_ATTR_H_
