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
 * \file tvm/runtime/name_transforms.h
 * \brief Transformations which are applied on names to generate appropriately named.
 *  These functions are used in both Runtime and Backend.
 */
#ifndef TVM_RUNTIME_NAME_TRANSFORMS_H_
#define TVM_RUNTIME_NAME_TRANSFORMS_H_

#include <string>

namespace tvm {
namespace runtime {

/*!
 * \brief Sanitize name for output into compiler artifacts
 * \param name Original name
 * \return Sanitized name
 */
std::string SanitizeName(const std::string& name);

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_NAME_TRANSFORMS_H_
