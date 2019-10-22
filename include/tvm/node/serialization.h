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
 * Utility functions for serialization.
 * \file tvm/node/serialization.h
 */
#ifndef TVM_NODE_SERIALIZATION_H_
#define TVM_NODE_SERIALIZATION_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/object.h>

#include <string>

namespace tvm {
/*!
 * \brief save the node as well as all the node it depends on as json.
 *  This can be used to serialize any TVM object
 *
 * \return the string representation of the node.
 */
TVM_DLL std::string SaveJSON(const runtime::ObjectRef& node);

/*!
 * \brief Internal implementation of LoadJSON
 * Load tvm Node object from json and return a shared_ptr of Node.
 * \param json_str The json string to load from.
 *
 * \return The shared_ptr of the Node.
 */
TVM_DLL runtime::ObjectRef LoadJSON(std::string json_str);

}  // namespace tvm
#endif  // TVM_NODE_SERIALIZATION_H_
