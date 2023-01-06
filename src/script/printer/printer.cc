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
#include <tvm/runtime/registry.h>
#include <tvm/script/printer/printer.h>

namespace tvm {
namespace script {
namespace printer {

String Script(ObjectRef obj, Map<String, String> ir_prefix, int indent_spaces,
              bool print_line_numbers, int num_context_lines,
              Optional<ObjectPath> path_to_underline) {
  IRDocsifier d(ir_prefix);
  Doc doc = d->AsDoc(obj, ObjectPath::Root());
  return DocToPythonScript(doc, indent_spaces, print_line_numbers, num_context_lines,
                           path_to_underline);
}

Default* Default::Instance() {
  static Default inst;
  return &inst;
}

TVM_REGISTER_GLOBAL("script.printer.Script").set_body_typed(Script);

}  // namespace printer
}  // namespace script
}  // namespace tvm
