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
 * \file source_utils.h
 * \brief Minimum source manipulation utils for runtime.
 */

#ifndef TVM_RUNTIME_SOURCE_UTILS_H_
#define TVM_RUNTIME_SOURCE_UTILS_H_

#include <string>
#include <unordered_map>

namespace tvm {
namespace runtime {
/*!
 * \brief Split the source file on separate kernels by specified delimiter.
 * \param source The source code of the kernels.
 * \param delimiter The delimiter which is using for splitting kernels.
 * \return Mapping from primitive name to kernel source
 */
std::unordered_map<std::string, std::string> SplitKernels(std::string source,
                                                          std::string delimiter = "// Function: ");
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_SOURCE_UTILS_H_
