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
#include <tvm/ir/function.h>

#include "../../../tirx/script/printer/utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

namespace {
bool PrintSTirFunction(DocTranslatorObj* d, const tirx::FunctionNode* func) {
  if (!func->attrs->dict.count(tvm::attr::kSTir)) return false;
  ExtraStateScope<bool> layout(d, "tirx.buffer_default_layout_none", true);
  PrintFunction(d, func, NamespaceDoc("s_tir")->Attr("function"), tvm::attr::kSTir);
  return true;
}

TVM_FFI_STATIC_INIT_BLOCK() { RegisterFunctionDocTranslate(PrintSTirFunction); }
}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
