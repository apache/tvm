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
 * \file tvm/ir/serialization.h
 * \brief Utility functions for serialization.
 *
 * This is a thin forwarding header to ffi/extra/serialization.h.
 * Prefer using ffi::ToJSONGraph / ffi::FromJSONGraph directly.
 */
#ifndef TVM_IR_SERIALIZATION_H_
#define TVM_IR_SERIALIZATION_H_

#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/extra/serialization.h>
#include <tvm/runtime/base.h>

#include <string>

namespace tvm {

/*!
 * \brief Save the node as well as all the node it depends on as json.
 *  This can be used to serialize any TVM object.
 *
 * \return the string representation of the node.
 */
TVM_DLL std::string SaveJSON(ffi::Any node);

/*!
 * \brief Load tvm Node object from json and return a shared_ptr of Node.
 * \param json_str The json string to load from.
 *
 * \return The shared_ptr of the Node.
 */
TVM_DLL ffi::Any LoadJSON(std::string json_str);

}  // namespace tvm
#endif  // TVM_IR_SERIALIZATION_H_
