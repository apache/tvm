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
 * \file tvm/ffi/extra/serialization.h
 * \brief Reflection-based serialization utilities
 */
#ifndef TVM_FFI_EXTRA_SERIALIZATION_H_
#define TVM_FFI_EXTRA_SERIALIZATION_H_

#include <tvm/ffi/extra/base.h>
#include <tvm/ffi/extra/json.h>

namespace tvm {
namespace ffi {

/**
 * \brief Serialize ffi::Any to a JSON that stores the object graph.
 *
 * The JSON graph structure is stored as follows:
 *
 * ```json
 * {
 *   "root_index": <int>,        // Index of root node in nodes array
 *   "nodes": [<node>, ...],     // Array of serialized nodes
 *   "metadata": <object>        // Optional metadata
 * }
 * ```
 *
 * Each node has the format: `{"type": "<type_key>", "data": <type_data>}`
 * For object types and strings, the data may contain indices to other nodes.
 * For object fields whose static type is known as a primitive type, it is stored directly,
 * otherwise, it is stored as a reference to the nodes array by an index.
 *
 * This function preserves the type and multiple references to the same object,
 * which is useful for debugging and serialization.
 *
 * \param value The ffi::Any value to serialize.
 * \param metadata Extra metadata attached to "metadata" field of the JSON object.
 * \return The serialized JSON value.
 */
TVM_FFI_EXTRA_CXX_API json::Value ToJSONGraph(const Any& value, const Any& metadata = Any(nullptr));

/**
 * \brief Deserialize a JSON that stores the object graph to an ffi::Any value.
 *
 * This function can be used to implement deserialization
 * and debugging.
 *
 * \param value The JSON value to deserialize.
 * \return The deserialized object graph.
 */
TVM_FFI_EXTRA_CXX_API Any FromJSONGraph(const json::Value& value);

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_EXTRA_SERIALIZATION_H_
