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
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/index_map.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/stmt.h>

#include "../../../script/printer/utils.h"

namespace tvm {

TVM_FFI_STATIC_INIT_BLOCK() {
  using script::printer::details::RegisterScriptRepr;
  RegisterScriptRepr<tirx::TensorTypeNode>();
  RegisterScriptRepr<tirx::ComposeLayoutNode>();
  RegisterScriptRepr<tirx::ExecScopeNode>();
  RegisterScriptRepr<tirx::IndexMapNode>();
  RegisterScriptRepr<tirx::IterNode>();
  RegisterScriptRepr<tirx::FunctionNode>();
  RegisterScriptRepr<tirx::ScopeIdDefNode>();
  RegisterScriptRepr<tirx::ScopeIdDefStmtNode>();
  RegisterScriptRepr<tirx::TileLayoutNode>();
}

}  // namespace tvm
