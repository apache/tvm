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
#include <tvm/tirx/builtin.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

ExprDoc PrintVarCreation(const tirx::Var& var, const AccessPath& var_p, const IRDocsifier& d) {
  Type type = var->ty;
  AccessPath type_p = var_p->Attr("ty");
  ExprDoc rhs{ffi::UnsafeInit()};
  ffi::Array<ffi::String> kwargs_keys;
  ffi::Array<ExprDoc> kwargs_values;

  if (const auto* ptr_type = type.as<PointerTypeNode>()) {
    if (const auto* prim_type = ptr_type->element_type.as<PrimTypeNode>()) {
      rhs = TIR(d, "handle");
      rhs->source_paths.push_back(var_p->Attr("dtype"));
      if (ffi::GetRef<PrimType>(prim_type).IsVoid()) {
        if (ptr_type->storage_scope == "global") {
          rhs = rhs->Call({}, kwargs_keys, kwargs_values);
        } else {
          rhs =
              rhs->Call({}, {"storage_scope"},
                        {LiteralDoc::Str(ptr_type->storage_scope, type_p->Attr("storage_scope"))});
        }
      } else {
        ExprDoc element_type =
            LiteralDoc::DataType(prim_type->dtype, type_p->Attr("element_type")->Attr("dtype"));
        if (ptr_type->storage_scope == "") {
          rhs = rhs->Call({element_type}, kwargs_keys, kwargs_values);
        } else {
          rhs = rhs->Call({element_type,
                           LiteralDoc::Str(ptr_type->storage_scope,  //
                                           type_p->Attr("storage_scope"))},
                          kwargs_keys, kwargs_values);
        }
      }
    } else if (ptr_type->element_type->IsInstance<TensorMapTypeNode>()) {
      rhs = TIR(d, "TensorMap")->Call({}, {}, {});
    }
  } else {
    rhs = TIR(d, DType2Str(var->ty.as_or_throw<PrimType>()->dtype));
    rhs->source_paths.push_back(var_p->Attr("dtype"));
    rhs = rhs->Call({}, kwargs_keys, kwargs_values);
  }
  rhs->source_paths.push_back(type_p);
  return rhs;
}

Doc PrintVar(const tirx::Var& var, const AccessPath& var_p, const IRDocsifier& d) {
  if (!d->IsVarDefined(var)) {
    if (ffi::Optional<Frame> opt_f = FindLowestVarDef(var, d)) {
      ExprDoc lhs = DefineVar(var, opt_f.value(), d);
      ExprDoc rhs = PrintVarCreation(var, var_p, d);
      opt_f.value()->stmts.push_back(AssignDoc(lhs, rhs, std::nullopt));
    } else {
      LOG(WARNING) << "Didn't find variable definition for: " << var->name_hint;
    }
  }
  if (ffi::Optional<ExprDoc> doc = d->GetVarDoc(var)) {
    return doc.value();
  }
  TVM_FFI_THROW(InternalError) << "IndexError: Variable is not defined in the environment: "
                               << var->name_hint;
  TVM_FFI_UNREACHABLE();
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)  //
    .set_dispatch<tirx::Var>("", [](tirx::Var var, AccessPath p, IRDocsifier d) -> Doc {
      if (var->ty.as<PrimTypeNode>() || var->ty.as<PointerTypeNode>()) {
        return PrintVar(var, p, d);
      }
      if (!d->IsVarDefined(var)) {
        ExprDoc ann = d->AsDoc<ExprDoc>(var->ty, p->Attr("ty"));
        Frame f = d->frames.back();
        ExprDoc lhs = d->Define(var, f, var->name_hint.empty() ? "v" : var->name_hint);
        f->stmts.push_back(AssignDoc(lhs, std::nullopt, ann));
      }
      return d->GetVarDoc(var).value();
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::IterVar>("", [](tirx::IterVar var, AccessPath var_p, IRDocsifier d) -> Doc {
      return TIR(d, "iter_var")
          ->Call({
              d->AsDoc<ExprDoc>(var->var, var_p->Attr("var")),
              d->AsDoc<ExprDoc>(var->dom, var_p->Attr("dom")),
              LiteralDoc::Str(IterVarType2String(var->iter_type), var_p->Attr("iter_type")),
              LiteralDoc::Str(var->thread_tag, var_p->Attr("thread_tag")),
          });
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::Not>("", [](tirx::Not node, AccessPath p, IRDocsifier d) -> Doc {
      ExprDoc a = d->AsDoc<ExprDoc>(node->a, p->Attr("a"));
      if (a->IsInstance<LiteralDocNode>()) {
        return TIR(d, "Not")->Call({a});
      }
      return OperationDoc(OperationDocNode::Kind::kNot, {a});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::StringImm>("", [](tirx::StringImm s, AccessPath p, IRDocsifier d) -> Doc {
      if (HasMultipleLines(s->value)) {
        return d->AddMetadata(s);
      } else {
        return d->AsDoc<ExprDoc>(s->value, p->Attr("value"));
      }
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::Cast>("", [](tirx::Cast cast, AccessPath p, IRDocsifier d) -> Doc {
      ExprDoc dtype = LiteralDoc::DataType(cast.ty()->dtype, p->Attr("dtype"));
      ExprDoc value = d->AsDoc<ExprDoc>(cast->value, p->Attr("value"));
      return TIR(d, "Cast")->Call({dtype, value});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::Select>("", [](tirx::Select select, AccessPath p, IRDocsifier d) -> Doc {
      return TIR(d, "Select")
          ->Call({
              d->AsDoc<ExprDoc>(select->condition, p->Attr("condition")),
              d->AsDoc<ExprDoc>(select->true_value, p->Attr("true_value")),
              d->AsDoc<ExprDoc>(select->false_value, p->Attr("false_value")),
          });
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::Ramp>("", [](tirx::Ramp ramp, AccessPath ramp_p, IRDocsifier d) -> Doc {
      return TIR(d, "Ramp")->Call({
          d->AsDoc<ExprDoc>(ramp->base, ramp_p->Attr("base")),
          d->AsDoc<ExprDoc>(ramp->stride, ramp_p->Attr("stride")),
          d->AsDoc<ExprDoc>(ramp->lanes, ramp_p->Attr("lanes")),
      });
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::Broadcast>("",
                                   [](tirx::Broadcast bc, AccessPath bc_p, IRDocsifier d) -> Doc {
                                     return TIR(d, "Broadcast")
                                         ->Call({
                                             d->AsDoc<ExprDoc>(bc->value, bc_p->Attr("value")),
                                             d->AsDoc<ExprDoc>(bc->lanes, bc_p->Attr("lanes")),
                                         });
                                   });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::Shuffle>(  //
        "", [](tirx::Shuffle shuffle, AccessPath p, IRDocsifier d) -> Doc {
          return TIR(d, "Shuffle")
              ->Call({
                  d->AsDoc<ExprDoc>(shuffle->vectors, p->Attr("vectors")),
                  d->AsDoc<ExprDoc>(shuffle->indices, p->Attr("indices")),
              });
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::CommReducer>(  //
        "", [](tirx::CommReducer r, AccessPath p, IRDocsifier d) -> Doc {
          TVM_FFI_ICHECK_EQ(r->lhs.size(), r->rhs.size());
          ffi::Optional<LambdaDoc> lambda;
          {
            With<TIRFrame> f(d, r);
            int n_vars = r->lhs.size();
            ffi::Array<IdDoc> vars;
            vars.reserve(n_vars + n_vars);
            for (int i = 0; i < n_vars; ++i) {
              vars.push_back(DefineVar(r->lhs[i], *f, d).as_or_throw<IdDoc>());
            }
            for (int i = 0; i < n_vars; ++i) {
              vars.push_back(DefineVar(r->rhs[i], *f, d).as_or_throw<IdDoc>());
            }
            int n_results = r->result.size();
            ffi::Array<ExprDoc> results;
            results.reserve(n_results);
            for (int i = 0; i < n_results; ++i) {
              results.push_back(d->AsDoc<ExprDoc>(r->result[i], p->Attr("result")->ArrayItem(i)));
            }
            if (results.size() == 1) {
              lambda = LambdaDoc(vars, results[0]);
            } else {
              lambda = LambdaDoc(vars, TupleDoc(results));
            }
          }
          ExprDoc id = d->AsDoc<ExprDoc>(r->identity_element, p->Attr("identity_element"));
          return TIR(d, "comm_reducer")->Call({lambda.value(), id});
        });

LambdaDoc PrintIndexMap(const ffi::ObjectRef& map, const ffi::Array<tirx::PrimVar>& vs,
                        const AccessPath& vs_p, const ffi::Array<PrimExpr>& es,
                        const AccessPath& es_p, const IRDocsifier& d) {
  With<TIRFrame> f(d, map);
  ffi::Array<IdDoc> vars;
  for (int i = 0, l = vs.size(); i < l; ++i) {
    vars.push_back(DefineVar(static_cast<tirx::Var>(vs[i]), *f, d).as_or_throw<IdDoc>());
  }
  ffi::Array<ExprDoc> exprs;
  for (int i = 0, l = es.size(); i < l; ++i) {
    exprs.push_back(d->AsDoc<ExprDoc>(es[i], es_p->ArrayItem(i)));
  }
  return LambdaDoc(vars, TupleDoc(exprs));
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::IndexMap>(  //
        "", [](tirx::IndexMap m, AccessPath m_p, IRDocsifier d) -> Doc {
          LambdaDoc map = PrintIndexMap(m, m->initial_indices, m_p->Attr("initial_indices"),
                                        m->final_indices, m_p->Attr("final_indices"), d);
          if (m->inverse_index_map.has_value()) {
            tirx::IndexMap inverse = m->inverse_index_map.value().as_or_throw<tirx::IndexMap>();
            LambdaDoc inv = PrintIndexMap(inverse, inverse->initial_indices,
                                          m_p->Attr("inverse_index_map")->Attr("initial_indices"),
                                          inverse->final_indices,
                                          m_p->Attr("inverse_index_map")->Attr("final_indices"), d);
            return TIR(d, "index_map")->Call({map}, {"inverse_index_map"}, {inv});
          } else {
            return TIR(d, "index_map")->Call({map});
          }
        });

LambdaDoc PrintPredicate(const ffi::ObjectRef& pred, const ffi::Array<tirx::PrimVar>& vs,
                         const AccessPath& vs_p, const PrimExpr& p, const AccessPath& p_p,
                         const IRDocsifier& d) {
  With<TIRFrame> f(d, pred);
  ffi::Array<IdDoc> vars;
  for (int i = 0, l = vs.size(); i < l; ++i) {
    vars.push_back(DefineVar(static_cast<tirx::Var>(vs[i]), *f, d).as_or_throw<IdDoc>());
  }
  ExprDoc pred_doc = d->AsDoc<ExprDoc>(p, p_p);
  return LambdaDoc(vars, pred_doc);
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::Predicate>("",
                                   [](tirx::Predicate pred, AccessPath p, IRDocsifier d) -> Doc {
                                     return PrintPredicate(pred, pred->vars, p->Attr("vars"),
                                                           pred->pred, p->Attr("pred"), d);
                                   });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::Let>("", [](tirx::Let let, AccessPath p, IRDocsifier d) -> Doc {
      DictDoc where({d->AsDoc<ExprDoc>(let->var, p->Attr("var"))},
                    {d->AsDoc<ExprDoc>(let->value, p->Attr("value"))});
      return TIR(d, "Let")->Call({d->AsDoc<ExprDoc>(let->body, p->Attr("body"))},  //
                                 {"where"}, {where});
    });

Doc PrintTIRCall(Call call, AccessPath call_p, IRDocsifier d) {
  ffi::Optional<PrimType> call_prim_type = call->ty.as<PrimType>();
  auto get_call_type_doc = [&](AccessPath type_p) -> ExprDoc {
    if (call_prim_type.has_value()) {
      return LiteralDoc::DataType(call_prim_type.value()->dtype, type_p);
    }
    if (const auto* pointer_type = call->ty.as<PointerTypeNode>()) {
      ExprDoc pointer_type_doc = d->AsDoc<ExprDoc>(call->ty, type_p);
      if (const auto* element_type = pointer_type->element_type.as<PrimTypeNode>();
          element_type && ffi::GetRef<PrimType>(element_type).IsVoid() &&
          pointer_type->storage_scope == "global") {
        // The type annotation printer uses the concise bare `T.handle` for
        // function parameters.  A call's dtype position needs a value, so
        // materialize the corresponding type expression before selecting
        // `.ty`.
        pointer_type_doc = TIR(d, "handle")->Call({});
      }
      return pointer_type_doc->Attr("ty");
    }
    TVM_FFI_THROW(TypeError) << "Call dtype is only available for primitive or pointer return "
                                "types, but got "
                             << call->ty;
  };
  if (call->attrs.defined()) {
    ffi::Array<ExprDoc> call_args;
    int n_args = call->args.size();
    call_args.reserve(n_args);
    for (int i = 0; i < n_args; ++i) {
      call_args.push_back(d->AsDoc<ExprDoc>(call->args[i], call_p->Attr("args")->ArrayItem(i)));
    }
    ExprDoc op_doc = call->op.as<Op>()
                         ? LiteralDoc::Str(call->op.as<Op>().value()->name, call_p->Attr("op"))
                         : d->AsDoc<ExprDoc>(call->op, call_p->Attr("op"));
    ExprDoc ret_ty_doc = get_call_type_doc(call_p->Attr("ty"));
    return TIR(d, "Call")->Call(
        {op_doc, ListDoc(call_args)}, {"attrs", "ret_ty"},
        {d->AsDoc<ExprDoc>(call->attrs, call_p->Attr("attrs")), ret_ty_doc});
  }
  static const OpAttrMap<tirx::TScriptPrinterName>& op_names =
      Op::GetAttrMap<tirx::TScriptPrinterName>("TScriptPrinterName");
  static const OpAttrMap<tirx::TScriptDtypePrintLocation> dtype_locations =
      Op::GetAttrMap<tirx::TScriptDtypePrintLocation>("TScriptDtypePrintLocation");
  tirx::ScriptDtypePrintLocation dtype_print_location = tirx::ScriptDtypePrintLocation::kNone;
  ffi::Optional<ExprDoc> prefix;
  if (auto optional_op = call->op.as<Op>()) {
    auto op = optional_op.value();
    ffi::String name = op_names.get(op, op->name);
    if (op_names.count(op) == 0) {
      LOG(WARNING) << "No TScriptPrinterName attribute for " << op->name;
    }
    prefix = TIR(d, name);
    if (dtype_locations.count(op)) {
      dtype_print_location = static_cast<tirx::ScriptDtypePrintLocation>(dtype_locations[op]);
    }
    if (name == "call_llvm_pure_intrin" || name == "call_llvm_intrin") {
      int n_args = call->args.size();
      int64_t id = call->args[0].as<IntImmNode>()->value;
      auto f_llvm_lookup_intrinsic_name =
          tvm::ffi::Function::GetGlobal("target.llvm_get_intrinsic_name");

      ffi::Array<ExprDoc> args;
      args.reserve(n_args + 1);
      if (dtype_print_location == tirx::ScriptDtypePrintLocation::kFirst) {
        args.push_back(get_call_type_doc(call_p->Attr("dtype")));
      }

      for (int i = 0; i < n_args; ++i) {
        if ((i == 0) && (f_llvm_lookup_intrinsic_name)) {
          ffi::String name = (*f_llvm_lookup_intrinsic_name)(id).cast<ffi::String>();
          args.push_back(LiteralDoc::Str(name.c_str(), call_p->Attr("args")->ArrayItem(i)));
        } else {
          args.push_back(d->AsDoc<ExprDoc>(call->args[i], call_p->Attr("args")->ArrayItem(i)));
        }
      }
      if (dtype_print_location == tirx::ScriptDtypePrintLocation::kLast) {
        args.push_back(get_call_type_doc(call_p->Attr("dtype")));
      }
      return prefix.value()->Call(args);
    }
    // cuda_func_call: last arg is source_code (keyword-only in the Python API).
    // Print it as source_code=... to enable TVMScript round-trip.
    if (op->name == "tirx.cuda.func_call") {
      int n_args = call->args.size();
      ffi::Array<ExprDoc> args;
      // All args except the last (source_code) are positional.
      for (int i = 0; i < n_args - 1; ++i) {
        args.push_back(d->AsDoc<ExprDoc>(call->args[i], call_p->Attr("args")->ArrayItem(i)));
      }
      // source_code is the last arg, printed as keyword.
      // Extract the string value directly to avoid the StringImm printer
      // storing multiline source code in metadata (which can't be reparsed).
      ffi::Array<ffi::String> kw_keys;
      ffi::Array<ExprDoc> kw_vals;
      const auto* src_str = call->args[n_args - 1].as<tirx::StringImmNode>();
      TVM_FFI_ICHECK(src_str) << "cuda_func_call: last arg (source_code) must be StringImm";
      ExprDoc src = LiteralDoc::Str(src_str->value, call_p->Attr("args")->ArrayItem(n_args - 1));
      kw_keys.push_back("source_code");
      kw_vals.push_back(src);
      // If non-void return type, print return_type keyword.
      if (!call_prim_type || !call_prim_type.value().IsVoid()) {
        kw_keys.push_back("return_type");
        kw_vals.push_back(get_call_type_doc(call_p->Attr("dtype")));
      }
      return prefix.value()->Call(args, kw_keys, kw_vals);
    }
  } else if (call->op.as<GlobalVarNode>()) {
    prefix = d->AsDoc<ExprDoc>(call->op, call_p->Attr("op"));
  } else {
    TVM_FFI_THROW(InternalError) << "call: " << call;
  }
  ffi::Array<ExprDoc> args;
  int n_args = call->args.size();
  args.reserve(n_args + 1);
  if (dtype_print_location == tirx::ScriptDtypePrintLocation::kFirst) {
    args.push_back(get_call_type_doc(call_p->Attr("dtype")));
  }

  for (int i = 0; i < n_args; ++i) {
    args.push_back(d->AsDoc<ExprDoc>(call->args[i], call_p->Attr("args")->ArrayItem(i)));
  }
  if (dtype_print_location == tirx::ScriptDtypePrintLocation::kLast) {
    args.push_back(get_call_type_doc(call_p->Attr("dtype")));
  }
  return prefix.value()->Call(args);
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<Call>("tirx", PrintTIRCall);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::Reduce>("", [](tirx::Reduce r, AccessPath p, IRDocsifier d) -> Doc {
      ExprDoc combiner = d->AsDoc<ExprDoc>(r->combiner, p->Attr("combiner"));
      ExprDoc source = d->AsDoc<ExprDoc>(r->source, p->Attr("source"));
      ExprDoc init = d->AsDoc<ExprDoc>(r->init, p->Attr("init"));
      ExprDoc axis = d->AsDoc<ExprDoc>(r->axis, p->Attr("axis"));
      ExprDoc condition = d->AsDoc<ExprDoc>(r->condition, p->Attr("condition"));
      ExprDoc value_index = LiteralDoc::Int(r->value_index, p->Attr("value_index"));
      return TIR(d, "reduce")
          ->Call({combiner}, {"source", "init", "axis", "condition", "value_index"},
                 {source, init, axis, condition, value_index});
    });

#define TVM_SCRIPT_PRINTER_DEF_BINARY(NodeType, OpString)                                         \
  TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)                                                      \
      .set_dispatch<tirx::NodeType>("",                                                           \
                                    [](tirx::NodeType node, AccessPath p, IRDocsifier d) -> Doc { \
                                      ExprDoc a = d->AsDoc<ExprDoc>(node->a, p->Attr("a"));       \
                                      ExprDoc b = d->AsDoc<ExprDoc>(node->b, p->Attr("b"));       \
                                      return TIR(d, OpString)->Call({a, b});                      \
                                    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::Div>("", [](tirx::Div node, AccessPath p, IRDocsifier d) -> Doc {
      ExprDoc a = d->AsDoc<ExprDoc>(node->a, p->Attr("a"));
      ExprDoc b = d->AsDoc<ExprDoc>(node->b, p->Attr("b"));
      PrimExpr ret = tvm::div(node->a, node->b);
      if (!ret->IsInstance<tirx::DivNode>()) {
        return TIR(d, "Div")->Call({a, b});
      }
      PrimType a_ty = node->a.ty();
      PrimType b_ty = node->b.ty();
      if (a_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) &&
          b_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
        return TIR(d, "Div")->Call({a, b});
      }
      return OperationDoc(OperationDocNode::Kind::kDiv, {a, b});
    });

#define TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(NodeType, NodeObj, NodeFunc, OpString, OpKind) \
  TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)                                                    \
      .set_dispatch<tirx::NodeType>(                                                            \
          "", [](tirx::NodeType node, AccessPath p, IRDocsifier d) -> Doc {                     \
            ExprDoc a = d->AsDoc<ExprDoc>(node->a, p->Attr("a"));                               \
            ExprDoc b = d->AsDoc<ExprDoc>(node->b, p->Attr("b"));                               \
            PrimExpr ret = tvm::NodeFunc(node->a, node->b);                                     \
            if (const auto* ret_node = ret.as<tvm::tirx::NodeObj>()) {                          \
              if (ret_node->a.same_as(node->a) && ret_node->b.same_as(node->b)) {               \
                return OperationDoc(OperationDocNode::Kind::OpKind, {a, b});                    \
              }                                                                                 \
            }                                                                                   \
            return TIR(d, OpString)->Call({a, b});                                              \
          });

TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(Add, AddNode, add, "Add", kAdd);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(Sub, SubNode, sub, "Sub", kSub);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(Mul, MulNode, mul, "Mul", kMult);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(FloorDiv, FloorDivNode, floordiv, "FloorDiv", kFloorDiv);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(FloorMod, FloorModNode, floormod, "FloorMod", kMod);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(LT, LTNode, less, "LT", kLt);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(LE, LENode, less_equal, "LE", kLtE);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(EQ, EQNode, equal, "EQ", kEq);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(NE, NENode, not_equal, "NE", kNotEq);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(GT, GTNode, greater, "GT", kGt);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(GE, GENode, greater_equal, "GE", kGtE);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(And, AndNode, logical_and, "And", kAnd);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(Or, OrNode, logical_or, "Or", kOr);

TVM_SCRIPT_PRINTER_DEF_BINARY(Mod, "truncmod");
TVM_SCRIPT_PRINTER_DEF_BINARY(Min, "min");
TVM_SCRIPT_PRINTER_DEF_BINARY(Max, "max");

#undef TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR
#undef TVM_SCRIPT_PRINTER_DEF_BINARY

TVM_SCRIPT_REPR(tirx::IterVarNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::StringImmNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::CastNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::AddNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::SubNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::MulNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::DivNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::ModNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::FloorDivNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::FloorModNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::MinNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::MaxNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::LTNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::LENode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::EQNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::NENode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::GTNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::GENode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::AndNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::OrNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::NotNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::SelectNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::RampNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::BroadcastNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::LetNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::ShuffleNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::CommReducerNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::IndexMapNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::ReduceNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::PredicateNode, ReprPrintTIR);

}  // namespace printer
}  // namespace script
}  // namespace tvm
