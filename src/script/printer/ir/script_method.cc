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
#include <tvm/tir/builtin.h>

#include "./utils.h"

namespace tvm {

std::string IRModuleNode::Script(int indent_spaces, bool print_line_numbers, int num_context_lines,
                                 Optional<ObjectPath> path_to_underline) const {
  using namespace tvm::script::printer;
  return DocToPythonScript(IRDocsifier()->AsDoc(GetRef<ObjectRef>(this), ObjectPath::Root()),
                           indent_spaces, print_line_numbers, num_context_lines, path_to_underline);
}

TVM_REGISTER_GLOBAL("ir.Module_Script").set_body_method<IRModule>(&IRModuleNode::Script);

}  // namespace tvm
