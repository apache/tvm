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
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/script/printer/doc_translator.h>
#include <tvm/script/printer/printer.h>
#include <tvm/te/operation.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/index_map.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/tile_primitive.h>

#include <sstream>
#include <utility>

namespace tvm {
namespace {

std::string RenderFallbackWithInvisiblePathInfo(const ffi::String& script,
                                                const PrinterConfig& config) {
  if (!config->render_invisible_path_info || config->path_to_underline.empty()) {
    return std::string(script);
  }

  std::ostringstream os;
  for (size_t i = 0; i < config->path_to_underline.size(); ++i) {
    if (i != 0) os << "\n";
    os << "Access path: " << config->path_to_underline[i]
       << "\nNote: No visible object for this path is rendered in TVMScript.";
  }
  os << "\n\n" << script;
  return os.str();
}

template <typename ObjectType>
void RegisterScriptRepr() {
  namespace refl = ffi::reflection;
  refl::TypeAttrDef<ObjectType>().def(refl::type_attr::kRepr,
                                      [](ffi::ObjectRef obj, ffi::Function) -> ffi::String {
                                        return RedirectedReprPrinterMethod(obj);
                                      });
}

}  // namespace

std::string Script(const ffi::ObjectRef& node, const ffi::Optional<PrinterConfig>& cfg) {
  PrinterConfig config = cfg.value_or(PrinterConfig());
  static ffi::reflection::TypeAttrColumn translate(script::printer::kDocTranslate);
  // Builtin runtime roots keep their native repr; hooks still translate them within IR.
  if (!node.defined() || node->type_index() < ffi::TypeIndex::kTVMFFIDynObjectBegin ||
      translate[node->type_index()].type_index() == ffi::TypeIndex::kTVMFFINone) {
    return RenderFallbackWithInvisiblePathInfo(ffi::ReprPrint(ffi::Any(node)), config);
  }
  return std::string(script::printer::Script(node, config));
}

std::string RedirectedReprPrinterMethod(const ffi::ObjectRef& obj) {
  try {
    PrinterConfig config;
    config->extra_config.Set("ir.comment_imports", true);
    // Call translation directly so an unsupported type cannot recurse through ffi repr.
    return std::string(script::printer::Script(obj, config));
  } catch (const tvm::ffi::Error& e) {
    LOG(WARNING) << "TVMScript printer falls back to the basic address printer with the error:\n"
                 << e.what();
    std::ostringstream os;
    os << obj->GetTypeKey() << '(' << obj.get() << ')';
    return os.str();
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ffi::reflection;
  refl::GlobalDef()
      .def("node.TVMScriptPrinterScript", tvm::Script)
      .def("script.printer.ReprPrintRelax",
           [](const ffi::ObjectRef& obj, const PrinterConfig& config) {
             return script::printer::Script(obj, config);
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
  RegisterScriptRepr<PointerTypeNode>();
  RegisterScriptRepr<PrimTypeNode>();
  RegisterScriptRepr<RangeNode>();
  RegisterScriptRepr<StringImmNode>();
  RegisterScriptRepr<TensorLoadNode>();
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
  RegisterScriptRepr<s_tir::MatchBufferRegionNode>();
  RegisterScriptRepr<s_tir::SBlockNode>();
  RegisterScriptRepr<s_tir::SBlockRealizeNode>();
  RegisterScriptRepr<te::CommReducerNode>();
  RegisterScriptRepr<te::ReduceNode>();
  RegisterScriptRepr<tirx::AssertStmtNode>();
  RegisterScriptRepr<tirx::AttrStmtNode>();
  RegisterScriptRepr<tirx::BindNode>();
  RegisterScriptRepr<tirx::BreakNode>();
  RegisterScriptRepr<tirx::BufferStoreNode>();
  RegisterScriptRepr<tirx::BufferTypeNode>();
  RegisterScriptRepr<tirx::ComposeLayoutNode>();
  RegisterScriptRepr<tirx::ContinueNode>();
  RegisterScriptRepr<tirx::EvaluateNode>();
  RegisterScriptRepr<tirx::ExecScopeNode>();
  RegisterScriptRepr<tirx::ForNode>();
  RegisterScriptRepr<tirx::IfThenElseNode>();
  RegisterScriptRepr<tirx::IndexMapNode>();
  RegisterScriptRepr<tirx::IterNode>();
  RegisterScriptRepr<tirx::IterVarNode>();
  RegisterScriptRepr<tirx::LambdaExprNode>();
  RegisterScriptRepr<tirx::PrimFuncNode>();
  RegisterScriptRepr<tirx::ReturnNode>();
  RegisterScriptRepr<tirx::ScopeIdDefNode>();
  RegisterScriptRepr<tirx::ScopeIdDefStmtNode>();
  RegisterScriptRepr<tirx::SeqStmtNode>();
  RegisterScriptRepr<tirx::TileLayoutNode>();
  RegisterScriptRepr<tirx::TilePrimitiveCallNode>();
  RegisterScriptRepr<tirx::WhileNode>();
  RegisterScriptRepr<TensorRegionNode>();
}

}  // namespace tvm
