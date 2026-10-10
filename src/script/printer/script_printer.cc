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
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/module.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/vector_expr.h>
#include <tvm/ir/stmt.h>
#include <tvm/script/printer/printer.h>

#include <utility>

#include "utils.h"

namespace tvm {

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ffi::reflection;
  using script::printer::details::RegisterScriptRepr;
  refl::GlobalDef()
      .def("node.TVMScriptPrinterScript", tvm::Script)
      .def("script.printer.Script", tvm::Script);

  RegisterScriptRepr<DataTypeImmNode>();
  RegisterScriptRepr<GenericConstNode>();
  RegisterScriptRepr<AnyTypeNode>();
  RegisterScriptRepr<CallNode>();
  RegisterScriptRepr<DictAttrsNode>();
  RegisterScriptRepr<FloatImmNode>();
  RegisterScriptRepr<FuncTypeNode>();
  RegisterScriptRepr<GlobalVarNode>();
  RegisterScriptRepr<IRModuleNode>();
  RegisterScriptRepr<IntImmNode>();
  RegisterScriptRepr<MissingTypeNode>();
  RegisterScriptRepr<PointerTypeNode>();
  RegisterScriptRepr<PrimTypeNode>();
  RegisterScriptRepr<RangeNode>();
  RegisterScriptRepr<StringImmNode>();
  RegisterScriptRepr<TensorLoadNode>();
  RegisterScriptRepr<TensorRegionTypeNode>();
  RegisterScriptRepr<TupleNode>();
  RegisterScriptRepr<TupleGetItemNode>();
  RegisterScriptRepr<TupleTypeNode>();
  RegisterScriptRepr<VarNode>();
  RegisterScriptRepr<prim::AddNode>();
  RegisterScriptRepr<prim::AndNode>();
  RegisterScriptRepr<prim::BitwiseAndNode>();
  RegisterScriptRepr<prim::BitwiseNotNode>();
  RegisterScriptRepr<prim::BitwiseOrNode>();
  RegisterScriptRepr<prim::BitwiseXorNode>();
  RegisterScriptRepr<prim::BroadcastNode>();
  RegisterScriptRepr<prim::CastNode>();
  RegisterScriptRepr<prim::DivNode>();
  RegisterScriptRepr<prim::EQNode>();
  RegisterScriptRepr<prim::FloorDivNode>();
  RegisterScriptRepr<prim::FloorModNode>();
  RegisterScriptRepr<prim::GENode>();
  RegisterScriptRepr<prim::GTNode>();
  RegisterScriptRepr<prim::LENode>();
  RegisterScriptRepr<prim::LShiftNode>();
  RegisterScriptRepr<prim::LTNode>();
  RegisterScriptRepr<prim::LetNode>();
  RegisterScriptRepr<prim::MaxNode>();
  RegisterScriptRepr<prim::MinNode>();
  RegisterScriptRepr<prim::ModNode>();
  RegisterScriptRepr<prim::MulNode>();
  RegisterScriptRepr<prim::NENode>();
  RegisterScriptRepr<prim::NotNode>();
  RegisterScriptRepr<prim::OrNode>();
  RegisterScriptRepr<prim::RShiftNode>();
  RegisterScriptRepr<prim::RampNode>();
  RegisterScriptRepr<prim::SelectNode>();
  RegisterScriptRepr<prim::ShuffleNode>();
  RegisterScriptRepr<prim::SubNode>();
  RegisterScriptRepr<AssertStmtNode>();
  RegisterScriptRepr<RegionStmtNode>();
  RegisterScriptRepr<BindNode>();
  RegisterScriptRepr<BreakNode>();
  RegisterScriptRepr<TensorStoreNode>();
  RegisterScriptRepr<ContinueNode>();
  RegisterScriptRepr<EvaluateNode>();
  RegisterScriptRepr<ForNode>();
  RegisterScriptRepr<IfNode>();
  RegisterScriptRepr<LambdaExprNode>();
  RegisterScriptRepr<ReturnNode>();
  RegisterScriptRepr<SeqStmtNode>();
  RegisterScriptRepr<WhileNode>();
  RegisterScriptRepr<TensorRegionNode>();
}

}  // namespace tvm
