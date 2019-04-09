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
 *  Copyright (c) 2019 by Contributors
 * \file param_dict.h
 * \brief Definitions for serializing and deserializing parameter dictionaries.
 */
#ifndef TVM_RELAY_BACKEND_PARAM_DICT_H_
#define TVM_RELAY_BACKEND_PARAM_DICT_H_

#include <tvm/node/node.h>
#include <tvm/packed_func_ext.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include <string>

namespace tvm {
namespace relay {

/*! \brief Magic number for NDArray list file  */
constexpr uint64_t kTVMNDArrayListMagic = 0xF7E58D4F05049CB7;

/*!
 * \brief Wrapper node for naming `NDArray`s.
 */
struct NamedNDArrayNode : public ::tvm::Node {
  std::string name;
  tvm::runtime::NDArray array;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("array", &array);
  }

  static constexpr const char* _type_key = "NamedNDArray";
  TVM_DECLARE_NODE_TYPE_INFO(NamedNDArrayNode, Node);
};

TVM_DEFINE_NODE_REF(NamedNDArray, NamedNDArrayNode);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_PARAM_DICT_H_
