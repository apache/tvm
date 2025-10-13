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
 * \file tvm/node/reflection.h
 * \brief Reflection utilities for IR/AST nodes.
 */
#ifndef TVM_NODE_REFLECTION_H_
#define TVM_NODE_REFLECTION_H_

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/string.h>

namespace tvm {

/*!
 * \brief Create an object from a type key and a map of fields.
 * \param type_key The type key of the object.
 * \param fields The fields of the object.
 * \return The created object.
 */
TVM_DLL ffi::Any CreateObject(const ffi::String& type_key,
                              const ffi::Map<ffi::String, ffi::Any>& fields);

}  // namespace tvm
#endif  // TVM_NODE_REFLECTION_H_
