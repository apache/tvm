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
#ifndef TVM_SCRIPT_PRINTER_DIALECT_PREFIX_H_
#define TVM_SCRIPT_PRINTER_DIALECT_PREFIX_H_

#include <tvm/ffi/string.h>

namespace tvm {
namespace script {
namespace printer {

// Register during dialect static initialization so configuration can validate
// and reserve the prefix before any docsifier assigns variable names.
void RegisterDialectPrefix(const ffi::String& key, const ffi::String& default_prefix);

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_DIALECT_PREFIX_H_
