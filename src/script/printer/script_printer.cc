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
#include <tvm/relax/distributed/type.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/script/printer/printer.h>
#include <tvm/te/operation.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/index_map.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/stmt.h>

#include <utility>

#include "utils.h"

namespace tvm {

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ffi::reflection;
  using script::printer::details::RegisterScriptRepr;
  refl::GlobalDef()
      .def("node.TVMScriptPrinterScript", tvm::Script)
      .def("script.printer.Script", tvm::Script)
      .def("script.printer.ReprPrintRelax",
           [](const ffi::ObjectRef& obj, const PrinterConfig& config) {
             return tvm::Script(obj, config);
           });

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
  RegisterScriptRepr<relax::BindingBlockNode>();
  RegisterScriptRepr<relax::DataflowBlockNode>();
  RegisterScriptRepr<relax::DataflowVarNode>();
  RegisterScriptRepr<relax::ExternFuncNode>();
  RegisterScriptRepr<relax::FuncTypeNode>();
  RegisterScriptRepr<relax::FunctionNode>();
  RegisterScriptRepr<relax::IfNode>();
  RegisterScriptRepr<relax::MatchCastNode>();
  RegisterScriptRepr<relax::PackedFuncTypeNode>();
  RegisterScriptRepr<relax::SeqExprNode>();
  RegisterScriptRepr<relax::ShapeExprNode>();
  RegisterScriptRepr<relax::ShapeTypeNode>();
  RegisterScriptRepr<relax::TensorTypeNode>();
  RegisterScriptRepr<relax::TupleGetItemNode>();
  RegisterScriptRepr<relax::TupleNode>();
  RegisterScriptRepr<relax::VarBindingNode>();
  RegisterScriptRepr<relax::distributed::DTensorTypeNode>();
  RegisterScriptRepr<relax::distributed::DeviceMeshNode>();
  RegisterScriptRepr<relax::distributed::PlacementNode>();
  RegisterScriptRepr<te::CommReducerNode>();
  RegisterScriptRepr<te::ReduceNode>();
  RegisterScriptRepr<tirx::AssertStmtNode>();
  RegisterScriptRepr<tirx::RegionStmtNode>();
  RegisterScriptRepr<tirx::BindNode>();
  RegisterScriptRepr<tirx::BreakNode>();
  RegisterScriptRepr<tirx::TensorStoreNode>();
  RegisterScriptRepr<tirx::TensorTypeNode>();
  RegisterScriptRepr<tirx::ComposeLayoutNode>();
  RegisterScriptRepr<tirx::ContinueNode>();
  RegisterScriptRepr<tirx::EvaluateNode>();
  RegisterScriptRepr<tirx::ExecScopeNode>();
  RegisterScriptRepr<tirx::ForNode>();
  RegisterScriptRepr<tirx::IfThenElseNode>();
  RegisterScriptRepr<tirx::IndexMapNode>();
  RegisterScriptRepr<tirx::IterNode>();
  RegisterScriptRepr<s_tir::IterVarNode>();
  RegisterScriptRepr<LambdaExprNode>();
  RegisterScriptRepr<tirx::FunctionNode>();
  RegisterScriptRepr<tirx::ReturnNode>();
  RegisterScriptRepr<tirx::ScopeIdDefNode>();
  RegisterScriptRepr<tirx::ScopeIdDefStmtNode>();
  RegisterScriptRepr<tirx::SeqStmtNode>();
  RegisterScriptRepr<tirx::TileLayoutNode>();
  RegisterScriptRepr<tirx::TileOpCallNode>();
  RegisterScriptRepr<tirx::WhileNode>();
  RegisterScriptRepr<TensorRegionNode>();
}

}  // namespace tvm
