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

#include <tvm/script/printer/doc.h>

namespace tvm {
namespace script {
namespace printer {

TVM_REGISTER_NODE_TYPE(DocNode);
TVM_REGISTER_NODE_TYPE(ExprDocNode);

LiteralDoc::LiteralDoc(ObjectRef value) {
  ObjectPtr<LiteralDocNode> n = make_object<LiteralDocNode>();
  n->value = value;
  this->data_ = std::move(n);
}
TVM_REGISTER_NODE_TYPE(LiteralDocNode);
// Underscore is added to avoid syntax error in Python FFI binding
TVM_REGISTER_GLOBAL("script.printer.LiteralDoc.None_").set_body_typed(LiteralDoc::None);
TVM_REGISTER_GLOBAL("script.printer.LiteralDoc.Int").set_body_typed(LiteralDoc::Int);
TVM_REGISTER_GLOBAL("script.printer.LiteralDoc.Float").set_body_typed(LiteralDoc::Float);
TVM_REGISTER_GLOBAL("script.printer.LiteralDoc.Str").set_body_typed(LiteralDoc::Str);

}  // namespace printer
}  // namespace script
}  // namespace tvm
