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
#ifndef SRC_SCRIPT_PRINTER_DOC_PRINTER_H_
#define SRC_SCRIPT_PRINTER_DOC_PRINTER_H_

#include <tvm/script/printer/config.h>
#include <tvm/script/printer/doc.h>

namespace tvm {
namespace script {
namespace printer {
namespace details {

// Render a Doc with this invocation's recovered paths and statement annotations.
ffi::String RenderPythonScript(Doc doc, const PrinterConfig& config,
                               const ffi::Array<ffi::Any>& header,
                               const ffi::Array<AccessPath>& underline_paths,
                               const ffi::Map<AccessPath, ffi::String>& annotations);

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // SRC_SCRIPT_PRINTER_DOC_PRINTER_H_
