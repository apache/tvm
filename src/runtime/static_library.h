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
 * \file runtime/static_library.h
 * \brief Represents a generic '.o' static library which can be linked into the final output
 * dynamic library by export_library.
 */

#ifndef TVM_RUNTIME_STATIC_LIBRARY_H_
#define TVM_RUNTIME_STATIC_LIBRARY_H_

#include <tvm/runtime/logging.h>
#include <tvm/runtime/module.h>

#include <array>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>

namespace tvm {
namespace runtime {

/*!
 * \brief Returns a static library with the contents loaded from filename which exports
 * func_names with the usual packed-func calling convention.
 */
Module LoadStaticLibrary(const std::string& filename, Array<String> func_names);

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_STATIC_LIBRARY_H_
