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
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/global_info.h>
#include <tvm/ir/module.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/script/ir_builder/ir.h>
#include <tvm/sym/analyzer.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace ir_builder {
namespace ir {

using namespace tvm::prim;

using tvm::script::ir_builder::details::Namer;

TVM_FFI_STATIC_INIT_BLOCK() {
  Namer::vtable().SetDispatch<tvm::VarNode>(
      [](const ffi::ObjectRef& node, ffi::String name) -> void {
        VarNode* var = const_cast<VarNode*>(node.as<VarNode>());
        var->name = name;
      });
}

IRModuleFrame IRModule() {
  ffi::ObjectPtr<IRModuleFrameNode> n = ffi::make_object<IRModuleFrameNode>();
  n->global_var_map.clear();
  n->functions.clear();
  return IRModuleFrame(n);
}

// DeclFunction lives at the IR layer because an IRModule may host
// heterogeneous function kinds (e.g. relax::Function, tirx::Function).
// To derive the GlobalVar's ty without coupling the IR layer to
// any specific dialect, dispatch is keyed by the function's type-key:
// each dialect registers its own handler that maps a function of that
// type to the appropriate ty.
inline ffi::Optional<Type> GetGlobalVarType(const BaseFunc& func) {
  if (!func->ty.as<MissingType>().has_value()) {
    return func->ty;
  }
  // Registry: "script.ir_builder.decl_function.<type-key>" — per-function-kind
  // handler that derives the GlobalVar ty from the function signature.
  // Grep hint: grep -rn 'script.ir_builder.decl_function.' src/
  const std::string key = "script.ir_builder.decl_function." + func->GetTypeKey();
  if (auto fn = tvm::ffi::Function::GetGlobal(key)) {
    ffi::Optional<ffi::ObjectRef> result = (*fn)(func).cast<ffi::Optional<ffi::ObjectRef>>();
    if (result.has_value()) {
      return result.value().as_or_throw<Type>();
    }
  }
  return std::nullopt;
}

GlobalVar DeclFunction(const ffi::String& func_name, const BaseFunc& func_signature) {
  IRModuleFrame frame = FindModuleFrame();
  GlobalVar gv = frame->global_var_map.count(func_name) ? frame->global_var_map.at(func_name)
                                                        : GlobalVar(func_name);
  TVM_FFI_CHECK(!frame->functions.count(gv), ValueError)
      << "function " << func_name << " already exists";
  if (auto ty = GetGlobalVarType(func_signature)) {
    gv->ty = ty.value();
  } else {
    TVM_FFI_THROW(InternalError) << "Unsupported function type: " << func_signature->GetTypeKey();
  }
  TVM_FFI_CHECK(frame->functions.find(gv) == frame->functions.end(), ValueError)
      << "function " << func_name << " has already been defined.";
  frame->global_var_map.Set(func_name, gv);
  frame->functions.Set(gv, func_signature);
  return gv;
}

void DefFunction(const ffi::String& func_name, const BaseFunc& func) {
  IRModuleFrame frame = FindModuleFrame();
  auto it = frame->global_var_map.find(func_name);
  TVM_FFI_CHECK(it != frame->global_var_map.end(), ValueError)
      << "function " << func_name << " does not exist, please declare it first.";
  const GlobalVar& gv = (*it).second;
  frame->functions.Set(gv, func);
  if (auto ty = GetGlobalVarType(func)) {
    gv->ty = ty.value();
  } else {
    TVM_FFI_THROW(InternalError) << "Unsupported function type: " << func->GetTypeKey();
  }
}

void ModuleAttrs(ffi::Map<ffi::String, Any> attrs, bool allow_overwrite) {
  if (IRBuilder::IsInScope()) {
    // TODO(hongyi): add comments to explain why we need to check if the module frame is in scope
    IRModuleFrame frame = FindModuleFrame("I.ModuleAttr");
    if (!allow_overwrite && !frame->attrs.empty()) {
      TVM_FFI_THROW(ValueError) << "Duplicate module attrs, previous one is:\n" << frame->attrs;
    }
    frame->attrs = attrs;
  }
}

Any ModuleGetAttr(const ffi::String& key) {
  if (IRBuilder::IsInScope()) {
    IRModuleFrame frame = FindModuleFrame();
    if (frame->attrs.find(key) != frame->attrs.end()) {
      return frame->attrs[key];
    }
  }
  return Any();
}

void ModuleSetAttr(const ffi::String& key, const ffi::Optional<ffi::ObjectRef>& value,
                   bool allow_override) {
  if (IRBuilder::IsInScope()) {
    IRModuleFrame frame = FindModuleFrame();
    if (!allow_override && frame->attrs.find(key) != frame->attrs.end() && value.has_value()) {
      TVM_FFI_THROW(ValueError) << "Duplicate module attr " << key;
    }
    if (value.has_value()) {
      frame->attrs.Set(key, value.value());
    } else {
      frame->attrs.erase(key);
    }
  } else {
    TVM_FFI_THROW(ValueError) << "Currently in in the scope of a module.";
  }
}

void ModuleGlobalInfos(ffi::Map<ffi::String, ffi::Array<GlobalInfo>> global_infos) {
  if (IRBuilder::IsInScope()) {
    IRModuleFrame frame = FindModuleFrame("I.ModuleGlobalInfos");
    if (!frame->global_infos.empty()) {
      TVM_FFI_THROW(ValueError) << "Duplicate module global_infos, previous one is:\n"
                                << frame->global_infos;
    }
    frame->global_infos = global_infos;
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.ir.IRModule", IRModule)
      .def("script.ir_builder.ir.DeclFunction", DeclFunction)
      .def("script.ir_builder.ir.DefFunction", DefFunction)
      .def("script.ir_builder.ir.ModuleAttrs", ModuleAttrs)
      .def("script.ir_builder.ir.ModuleGetAttr", ModuleGetAttr)
      .def("script.ir_builder.ir.ModuleSetAttr", ModuleSetAttr)
      .def("script.ir_builder.ir.ModuleGlobalInfos", ModuleGlobalInfos);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  details::LocationAccessor::vtable().SetDispatch<tvm::StmtNode>(
      [](const ffi::ObjectRef& obj) -> Location* { return &obj.as<tvm::StmtNode>()->loc; });
}

/*!
 * \brief Determine the dtype of a loop var from its bounds, or validate an explicit one.
 *
 * Without an explicit dtype the bit width is the max of both bounds and the
 * signedness follows C-style promotion (unsigned wins), which is exactly what
 * `PromoteBinaryOpType` applies to the `stop - start` extent.
 */
PrimType InferLoopVarDtype(const PrimExpr& start, const PrimExpr& stop,
                           const ffi::Optional<PrimType>& dtype) {
  if (dtype.has_value()) {
    ir::CheckExplicitIndexDtype(dtype.value());
    return dtype.value();
  }
  PrimType start_ty = start.ty();
  PrimType stop_ty = stop.ty();
  bool is_unsigned =
      start_ty.MatchesCode(DLDataTypeCode::kDLUInt) || stop_ty.MatchesCode(DLDataTypeCode::kDLUInt);
  return PrimType(is_unsigned ? DLDataTypeCode::kDLUInt : DLDataTypeCode::kDLInt,
                  std::max(start_ty.bits(), stop_ty.bits()), 1);
}

/*!
 * \brief Coerce a loop bound to the loop var's dtype.
 *
 * Integer literals are re-created in the target dtype; any other mismatched
 * expression gets an explicit Cast, so the loop header stays the single place
 * where the index dtype has to be spelled out.
 */
PrimExpr ConvertLoopBound(const PrimExpr& e, const PrimType& var_ty) {
  if (e.ty() == var_ty) return e;
  if (const auto* imm = e.as<IntImmNode>()) {
    return tvm::IntImm(var_ty, imm->value);
  }
  return tvm::prim::Cast(var_ty, e);
}

#define TVM_IR_BUILDER_FOR_FRAME(Method, Kind)                                                     \
  ForFrame Method(PrimExpr start, PrimExpr stop,                                                   \
                  ffi::Optional<ffi::Map<ffi::String, Any>> annotations,                           \
                  ffi::Optional<PrimExpr> step, ffi::Optional<PrimType> dtype) {                   \
    PrimType var_ty = InferLoopVarDtype(start, stop, dtype);                                       \
    PrimExpr min = ConvertLoopBound(start, var_ty);                                                \
    PrimExpr extent = sym::Analyzer()->Simplify(ConvertLoopBound(stop, var_ty) - min);             \
    if (step.has_value()) {                                                                        \
      step = ConvertLoopBound(step.value(), var_ty);                                               \
    }                                                                                              \
    ffi::ObjectPtr<ForFrameNode> n = ffi::make_object<ForFrameNode>();                             \
    n->vars = {Var("v", var_ty)};                                                                  \
    n->doms = {Range::FromMinExtent(min, extent)};                                                 \
    n->steps = {step};                                                                             \
    n->f_make_for_loop = [annotations](ffi::Array<Var> vars, ffi::Array<Range> doms,               \
                                       ffi::Array<ffi::Optional<PrimExpr>> steps,                  \
                                       tvm::SeqStmt body, Location loc) {                          \
      TVM_FFI_ICHECK_EQ(vars.size(), 1);                                                           \
      TVM_FFI_ICHECK_EQ(doms.size(), 1);                                                           \
      TVM_FFI_ICHECK_EQ(steps.size(), 1);                                                          \
      auto loop =                                                                                  \
          tvm::For(vars[0].as_or_throw<tvm::PrimVar>(), doms[0]->min, doms[0]->extent, Kind, body, \
                   annotations.value_or(ffi::Map<ffi::String, Any>()), steps[0], loc);             \
      return loop;                                                                                 \
    };                                                                                             \
    return ForFrame(n);                                                                            \
  }

TVM_IR_BUILDER_FOR_FRAME(Serial, tvm::ForKind::kDefault);
TVM_IR_BUILDER_FOR_FRAME(Parallel, tvm::ForKind::kParallel);
TVM_IR_BUILDER_FOR_FRAME(Vectorized, tvm::ForKind::kVectorized);
TVM_IR_BUILDER_FOR_FRAME(Unroll, tvm::ForKind::kUnrolled);

#undef TVM_IR_BUILDER_FOR_FRAME

ForFrame Grid(ffi::Array<ffi::Variant<PrimExpr, ffi::Tuple<PrimExpr, PrimExpr>>> extents,
              ffi::Optional<PrimType> dtype) {
  if (dtype.has_value()) {
    ir::CheckExplicitIndexDtype(dtype.value());
  }
  ffi::ObjectPtr<ForFrameNode> n = ffi::make_object<ForFrameNode>();
  n->vars.reserve(extents.size());
  n->doms.reserve(extents.size());
  n->steps.resize(extents.size());
  for (const auto& extent : extents) {
    if (auto prim_expr = extent.as<PrimExpr>()) {
      // extent is a single PrimExpr
      PrimType var_ty = dtype.value_or(prim_expr.value().ty());
      n->vars.push_back(Var("v" + std::to_string(n->vars.size()), var_ty));
      n->doms.push_back(Range(tvm::IntImm(var_ty, 0), ConvertLoopBound(prim_expr.value(), var_ty)));
    } else if (auto tuple = extent.as<ffi::Tuple<PrimExpr, PrimExpr>>()) {
      // extent is a tuple of two PrimExpr (start, extent)
      PrimType var_ty = dtype.value_or(tuple.value().get<0>().ty());
      n->vars.push_back(Var("v" + std::to_string(n->vars.size()), var_ty));
      n->doms.push_back(Range::FromMinExtent(ConvertLoopBound(tuple.value().get<0>(), var_ty),
                                             ConvertLoopBound(tuple.value().get<1>(), var_ty)));
    } else {
      TVM_FFI_THROW(InternalError) << "TypeError: Invalid type for grid extent";
    }
  }
  n->f_make_for_loop = [](ffi::Array<Var> vars, ffi::Array<Range> doms,
                          ffi::Array<ffi::Optional<PrimExpr>> steps, SeqStmt body,
                          Location loc) -> Stmt {
    TVM_FFI_ICHECK_EQ(vars.size(), doms.size());
    TVM_FFI_ICHECK_EQ(vars.size(), steps.size());
    Stmt result = std::move(body);
    int n = vars.size();
    for (int i = n - 1; i >= 0; --i) {
      Range dom = doms[i];
      Var var = vars[i];
      result = For(var.as_or_throw<tvm::PrimVar>(), dom->min, dom->extent, ForKind::kDefault,
                   SeqStmt(std::move(result)),
                   /*annotations=*/{}, /*step=*/steps[i], loc);
    }
    return result;
  };
  return ForFrame(n);
}

AssertFrame Assert(PrimExpr condition, ffi::String error_kind,
                   ffi::Array<ffi::String> message_parts) {
  ffi::ObjectPtr<AssertFrameNode> n =
      ffi::make_object<AssertFrameNode>(condition, tvm::StringImm(error_kind));
  ffi::Array<tvm::StringImm> parts;
  for (const auto& p : message_parts) {
    parts.push_back(tvm::StringImm(p));
  }
  n->message_parts = parts;
  return AssertFrame(n);
}

Var Bind(Expr value, ffi::Optional<Type> type_annotation, ffi::Optional<Var> var) {
  Expr value_expr = value;
  Var bind_var = [&]() {
    if (var.has_value()) {
      return var.value();
    } else if (type_annotation.has_value()) {
      return Var("v", type_annotation.value());
    } else {
      return Var("v", value_expr->ty);
    }
  }();
  AddToParent(tvm::Bind(bind_var, value_expr));
  return bind_var;
}

RegionFrame Region(Op op, ffi::Array<Expr> args, ffi::Optional<ffi::Array<Var>> body_params,
                   DictAttrs attrs) {
  TVM_FFI_CHECK(tvm::IsRegionOp(op), ValueError)
      << op->name << " does not support region construction: FRegionGetBodyParams is required";
  auto params =
      body_params.has_value() ? body_params.value() : tvm::GetRegionBodyParams(op, args, attrs);
  auto n = ffi::make_object<RegionFrameNode>(std::move(op));
  n->args = std::move(args);
  n->body_params = std::move(params);
  n->attrs = std::move(attrs);
  return RegionFrame(n);
}

WhileFrame While(PrimExpr condition) {
  ffi::ObjectPtr<WhileFrameNode> n = ffi::make_object<WhileFrameNode>(condition);
  return WhileFrame(n);
}

tvm::Stmt Return(Expr value) {
  tvm::Stmt stmt = tvm::Return(std::move(value), Location());
  AddToParent(stmt);
  return stmt;
}

tvm::Stmt Break() {
  tvm::Stmt stmt = tvm::Break(Location());
  AddToParent(stmt);
  return stmt;
}

tvm::Stmt Continue() {
  tvm::Stmt stmt = tvm::Continue(Location());
  AddToParent(stmt);
  return stmt;
}

IfFrame If(PrimExpr condition) {
  ffi::ObjectPtr<IfFrameNode> n = ffi::make_object<IfFrameNode>(condition);
  n->then_stmts = std::nullopt;
  n->else_stmts = std::nullopt;
  return IfFrame(n);
}

ThenFrame Then() {
  ffi::ObjectPtr<ThenFrameNode> n = ffi::make_object<ThenFrameNode>();
  return ThenFrame(n);
}

ElseFrame Else() {
  ffi::ObjectPtr<ElseFrameNode> n = ffi::make_object<ElseFrameNode>();
  return ElseFrame(n);
}

tvm::Stmt Evaluate(Expr value) {
  tvm::Stmt stmt = tvm::Evaluate(value);
  AddToParent(stmt);
  return stmt;
}

// Preserve the historical FFI entry points with one shared implementation.
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("script.ir_builder.tirx.Grid", Grid)
      .def("script.ir_builder.tirx.Assert", Assert)
      .def("script.ir_builder.tirx.Bind", Bind)
      .def("script.ir_builder.tirx.While", While)
      .def("script.ir_builder.tirx.Return", Return)
      .def("script.ir_builder.tirx.Break", Break)
      .def("script.ir_builder.tirx.Continue", Continue)
      .def("script.ir_builder.tirx.If", If)
      .def("script.ir_builder.tirx.Then", Then)
      .def("script.ir_builder.tirx.Else", Else)
      .def("script.ir_builder.tirx.Region", Region)
      .def("script.ir_builder.tirx.Evaluate", Evaluate);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.ir_builder.tirx.AddToParent",
                        [](tvm::Stmt stmt) { AddToParent(std::move(stmt)); });
}

}  // namespace ir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
