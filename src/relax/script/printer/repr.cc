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
#include <tvm/relax/distributed/type.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/script/printer/printer.h>

#include "../../../script/printer/utils.h"

namespace tvm {

TVM_FFI_STATIC_INIT_BLOCK() {
  using script::printer::details::RegisterScriptRepr;
  ffi::reflection::GlobalDef().def("script.printer.ReprPrintRelax",
                                   [](const ffi::ObjectRef& obj, const PrinterConfig& config) {
                                     return tvm::Script(obj, config);
                                   });
  RegisterScriptRepr<relax::BindingBlockNode>();
  RegisterScriptRepr<relax::DataflowBlockNode>();
  RegisterScriptRepr<relax::DataflowVarNode>();
  RegisterScriptRepr<relax::ExternFuncNode>();
  RegisterScriptRepr<relax::FuncTypeNode>();
  RegisterScriptRepr<relax::FunctionNode>();
  RegisterScriptRepr<relax::IfExprNode>();
  RegisterScriptRepr<relax::MatchCastNode>();
  RegisterScriptRepr<relax::PackedFuncTypeNode>();
  RegisterScriptRepr<relax::SeqExprNode>();
  RegisterScriptRepr<relax::ShapeExprNode>();
  RegisterScriptRepr<relax::ShapeTypeNode>();
  RegisterScriptRepr<relax::TensorTypeNode>();
  RegisterScriptRepr<relax::VarBindingNode>();
  RegisterScriptRepr<relax::distributed::DTensorTypeNode>();
  RegisterScriptRepr<relax::distributed::DeviceMeshNode>();
  RegisterScriptRepr<relax::distributed::PlacementNode>();
}

}  // namespace tvm
