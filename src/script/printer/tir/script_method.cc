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

std::string PrimExprNode::Script(int indent_spaces, bool print_line_numbers, int num_context_lines,
                                 Optional<ObjectPath> path_to_underline) const {
  using namespace tvm::script::printer;
  IRDocsifier d;
  ObjectRef obj = GetRef<ObjectRef>(this);
  With<TIRFrame> f(MakeDispatchFrame(d, obj, ObjectRef(nullptr)));
  return DocToPythonScript(Docsify(obj, d, *f), indent_spaces, print_line_numbers,
                           num_context_lines, path_to_underline);
}

namespace tir {

std::string StmtNode::Script(int indent_spaces, bool print_line_numbers, int num_context_lines,
                             Optional<ObjectPath> path_to_underline) const {
  using namespace tvm::script::printer;
  IRDocsifier d;
  ObjectRef obj = GetRef<ObjectRef>(this);
  With<TIRFrame> f(MakeDispatchFrame(d, obj, ObjectRef(nullptr)));
  return DocToPythonScript(Docsify(obj, d, *f), indent_spaces, print_line_numbers,
                           num_context_lines, path_to_underline);
}

std::string PrimFuncNode::Script(int indent_spaces, bool print_line_numbers, int num_context_lines,
                                 Optional<ObjectPath> path_to_underline) const {
  using namespace tvm::script::printer;
  return DocToPythonScript(IRDocsifier()->AsDoc(GetRef<ObjectRef>(this), ObjectPath::Root()),
                           indent_spaces, print_line_numbers, num_context_lines, path_to_underline);
}

TVM_REGISTER_GLOBAL("tir.PrimFuncScript").set_body_method<PrimFunc>(&PrimFuncNode::Script);
TVM_REGISTER_GLOBAL("tir.StmtScript").set_body_method<Stmt>(&StmtNode::Script);
TVM_REGISTER_GLOBAL("tir.PrimExprScript").set_body_method<PrimExpr>(&PrimExprNode::Script);

}  // namespace tir
}  // namespace tvm
