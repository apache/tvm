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
 * \file src/relay/op/memory/memory.h
 * \brief Utilities related to memory allocation
 */

#ifndef TVM_RELAY_OP_MEMORY_UTILS_H_
#define TVM_RELAY_OP_MEMORY_UTILS_H_

#include <tvm/relay/type.h>

#include <vector>

namespace tvm {
namespace relay {

std::vector<TensorType> FlattenTupleType(const Type& type);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_OP_MEMORY_UTILS_H_
