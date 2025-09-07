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
 * \file tvm/ffi/extra/json.h
 * \brief Minimal lightweight JSON parsing and serialization utilities
 */
#ifndef TVM_FFI_EXTRA_JSON_H_
#define TVM_FFI_EXTRA_JSON_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/base.h>

namespace tvm {
namespace ffi {
namespace json {

/*!
 * \brief alias Any as json Value.
 *
 * To keep things lightweight, we simply reuse the ffi::Any system.
 */
using Value = Any;

/*!
 * \brief alias Map<Any, Any> as json Object.
 * \note We use Map<Any, Any> instead of Map<String, Any> to avoid
 *      the overhead of key checking when doing as conversion,
 *      the check will be performed at runtime when we read each key
 */
using Object = ffi::Map<Any, Any>;

/*! \brief alias Array<Any> as json Array. */
using Array = ffi::Array<Any>;

/*!
 * \brief Parse a JSON string into an Any value.
 *
 * Besides the standard JSON syntax, this function also supports:
 * - Infinity/NaN as JavaScript syntax
 * - int64 integer value
 *
 * If error_msg is not nullptr, the error message will be written to it
 * and no exception will be thrown when parsing fails.
 *
 * \param json_str The JSON string to parse.
 * \param error_msg The output error message, can be nullptr.
 *
 * \return The parsed Any value.
 */
TVM_FFI_EXTRA_CXX_API json::Value Parse(const String& json_str, String* error_msg = nullptr);

/*!
 * \brief Serialize an Any value into a JSON string.
 *
 * \param value The Any value to serialize.
 * \param indent The number of spaces to indent the output.
 *               If not specified, the output will be compact.
 * \return The output JSON string.
 */
TVM_FFI_EXTRA_CXX_API String Stringify(const json::Value& value,
                                       Optional<int> indent = std::nullopt);

}  // namespace json
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_EXTRA_JSON_H_
