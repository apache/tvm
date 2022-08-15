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
#ifndef TVM_SCRIPT_PRINTER_DOC_PRINTER_H_
#define TVM_SCRIPT_PRINTER_DOC_PRINTER_H_

#include <tvm/script/printer/doc.h>

namespace tvm {
namespace script {
namespace printer {

/*!
 * \brief Convert Doc into Python script.
 *
 * This function unpacks the DocPrinterOptions into function arguments
 * to be FFI friendly.
 *
 * \param doc Doc to be converted
 * \param indent_spaces Number of spaces used for indentation
 * \param print_line_numbers Whether to print line numbers
 * \param num_context_lines Number of context lines to print around the underlined text
 * \param path_to_underline Object path to be underlined
 */
String DocToPythonScript(Doc doc, int indent_spaces = 4, bool print_line_numbers = false,
                         int num_context_lines = -1,
                         Optional<ObjectPath> path_to_underline = NullOpt);

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_DOC_PRINTER_H_
