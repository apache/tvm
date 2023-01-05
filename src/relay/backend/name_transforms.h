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
 * \file relay/backend/name_transforms.h
 * \brief Transformations which are applied on names to generate appropriately named compiler
 * artifacts
 *
 * Example:
 * ToCFunctionStyle(PrefixName(CombineNames({"Device", "target", "Invoke"})))
 * // TVMDeviceTargetInvoke
 *
 * ToCFunctionStyle(PrefixGeneratedName(CombineNames({"model", "Run"})))
 * // TVMGenModelRun
 *
 * ToCVariableStyle(PrefixName(CombineNames({"Device", "target", "t"})))
 * // tvm_device_target_t
 *
 * ToCVariableStyle(PrefixGeneratedName(CombineNames({"model", "Devices"})))
 * // tvmgen_model_devices
 *
 * ToCConstantStyle(PrefixGeneratedName(CombineNames({"model", "Devices"})))
 * // TVMGEN_MODEL_DEVICES
 *
 */

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/logging.h>

#include <algorithm>
#include <iostream>
#include <string>

#ifndef TVM_RELAY_BACKEND_NAME_TRANSFORMS_H_
#define TVM_RELAY_BACKEND_NAME_TRANSFORMS_H_

namespace tvm {
namespace relay {
namespace backend {

/*!
 * \brief Transform a name to the C variable style assuming it is
 * appropriately constructed using the prefixing functions
 * \param original_name Original name
 * \return Transformed function in the C function style
 */
std::string ToCFunctionStyle(const std::string& original_name);

/*!
 * \brief Transform a name to the C variable style assuming it is
 * appropriately constructed using the prefixing functions
 * \param name Original name
 * \return Transformed function in the C variable style
 */
std::string ToCVariableStyle(const std::string& original_name);

/*!
 * \brief Transform a name to the C constant style assuming it is
 * appropriately constructed using the prefixing functions
 * \param name Original name
 * \return Transformed function in the C constant style
 */
std::string ToCConstantStyle(const std::string& original_name);

/*!
 * \brief Transform a name to the Rust struct style assuming it is
 * appropriately constructed using the combining functions
 * \param name Original name
 * \return Transformed function in the Rust struct style
 */
std::string ToRustStructStyle(const std::string& original_name);

/*!
 * \brief Transform a name to the Rust macro style assuming it is
 * appropriately constructed using the combining functions
 * \param name Original name
 * \return Transformed function in the Rust macro style
 */
std::string ToRustMacroStyle(const std::string& original_name);

/*!
 * \brief Transform a name to the Rust constant style assuming it is
 * appropriately constructed using the combining functions
 * \param name Original name
 * \return Transformed function in the Rust constant style
 */
std::string ToRustConstantStyle(const std::string& original_name);

/*!
 * \brief Combine names together for use as a generated name
 * \param names Vector of strings to combine
 * \return Combined together names
 */
std::string CombineNames(const Array<String>& names);

/*!
 * \brief Apply TVM-specific prefix to a name
 * \param names Vector of names to combine to form a combined name
 * \return Name with prefix applied or prefix-only if no name passed
 */
inline std::string PrefixName(const Array<String>& names) { return "TVM_" + CombineNames(names); }

/*!
 * \brief Apply generated TVM-specific prefix to a name
 * \param names Vector of names to combine to form a combined name
 * \return Name with prefix applied or prefix-only if no name passed
 */
inline std::string PrefixGeneratedName(const Array<String>& names) {
  return "TVMGen_" + CombineNames(names);
}

}  // namespace backend
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_NAME_TRANSFORMS_H_
