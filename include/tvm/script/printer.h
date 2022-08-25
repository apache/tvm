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
#ifndef TVM_SCRIPT_PRINTER_H_
#define TVM_SCRIPT_PRINTER_H_

#include <tvm/node/node.h>
#include <tvm/node/object_path.h>

namespace tvm {
namespace script {
namespace printer {

/*!
 * \brief Print IR graph as TVMScript code
 *
 * \param root_node The root node to print.
 * \param ir_name The dispatch token of the target IR, e.g., "tir", "relax".
 * \param ir_prefix The symbol name for TVMScript IR namespaces. For example, {"tir": "T"}.
 * \param indent_spaces Number of spaces used for indentation
 * \param print_line_numbers Whether to print line numbers
 * \param num_context_lines Number of context lines to print around the underlined text
 * \param path_to_underline Object path to be underlined
 *
 * \return the TVMScript code as string.
 */
String Script(                                        //
    const ObjectRef& root_node,                       //
    String ir_name,                                   //
    Map<String, String> ir_prefix,                    //
    int indent_spaces = 4,                            //
    bool print_line_numbers = false,                  //
    int num_context_lines = -1,                       //
    Optional<ObjectPath> path_to_underline = NullOpt  //
);

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_H_
