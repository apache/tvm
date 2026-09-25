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
#include <tvm/ir/prim/builtin.h>
#include <tvm/te/operation.h>
#include <tvm/tirx/attrs.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/type.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<Tuple>(
      "tirx", [](Tuple tuple, AccessPath tuple_p, IRDocsifier d) -> Doc {
        return TupleDoc(d->AsDoc<ListDoc>(tuple->fields, tuple_p->Attr("fields"))->elements);
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<TupleGetItem>(
      "tirx", [](TupleGetItem get_item, AccessPath get_item_p, IRDocsifier d) -> Doc {
        ExprDoc index = LiteralDoc::Int(get_item->index, get_item_p->Attr("index"));
        return d->AsDoc<ExprDoc>(get_item->tuple, get_item_p->Attr("tuple"))[{index}];
      });
}

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
        if (ptr_type->storage_scope.empty()) {
          rhs = rhs->Call({element_type}, kwargs_keys, kwargs_values);
        } else {
          rhs = rhs->Call({element_type,
                           LiteralDoc::Str(ptr_type->storage_scope,  //
                                           type_p->Attr("storage_scope"))},
                          kwargs_keys, kwargs_values);
        }
      }
    } else if (ptr_type->element_type->IsInstance<tirx::TensorMapTypeNode>()) {
      rhs = TIR(d, "TensorMap")->Call({}, {}, {});
    }
  } else {
    rhs = IR(d, "dynamic")
              ->Call({LiteralDoc::Str(var->name, var_p->Attr("name"))}, {"dtype"},
                     {LiteralDoc::Str(DType2Str(var->ty.as_or_throw<PrimType>()->dtype), type_p)});
  }
  rhs->source_paths.push_back(type_p);
  return rhs;
}

Doc PrintVar(const tirx::Var& var, const AccessPath& var_p, const IRDocsifier& d) {
  if (!d->IsVarDefined(var)) {
    if (ffi::Optional<Frame> opt_f = FindLowestVarDef(var, d)) {
      Frame frame = var->ty.as<PrimTypeNode>() ? d->frames.front() : opt_f.value();
      ExprDoc lhs = DefineVar(var, frame, d);
      ExprDoc rhs = PrintVarCreation(var, var_p, d);
      frame->stmts.push_back(AssignDoc(lhs, rhs, std::nullopt));
    } else {
      LOG(WARNING) << "Didn't find variable definition for: " << var->name;
    }
  }
  if (ffi::Optional<ExprDoc> doc = d->GetVarDoc(var)) {
    return doc.value();
  }
  TVM_FFI_THROW(InternalError) << "IndexError: Variable is not defined in the environment: "
                               << var->name;
  TVM_FFI_UNREACHABLE();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable()  //
      .set_dispatch<tirx::Var>("", [](tirx::Var var, AccessPath p, IRDocsifier d) -> Doc {
        if (var->ty.as<tirx::BufferTypeNode>()) {
          tirx::BufferVar buffer(var);
          if (!d->IsVarDefined(buffer)) {
            if (ffi::Optional<Frame> opt_f = FindLowestVarDef(buffer, d)) {
              ExprDoc lhs = DefineBuffer(buffer, opt_f.value(), d);
              ExprDoc rhs = BufferDecl(buffer, "Buffer", {}, p, opt_f.value(), d,
                                       BufferVarDefinition::DataPointer);
              opt_f.value()->stmts.push_back(AssignDoc(lhs, rhs, std::nullopt));
            }
          }
          if (ffi::Optional<ExprDoc> doc = d->GetVarDoc(buffer)) {
            // special case for scalar buffer
            if (buffer.IsScalar()) {
              return doc.value()->Attr("source");
            }
            return doc.value();
          }
          TVM_FFI_THROW(IndexError) << "BufferVar is not defined in the environment: " << buffer;
        }
        if (var->ty.as<PrimTypeNode>() || var->ty.as<PointerTypeNode>()) {
          return PrintVar(var, p, d);
        }
        if (!d->IsVarDefined(var)) {
          ExprDoc ann = d->AsDoc<ExprDoc>(var->ty, p->Attr("ty"));
          Frame f = d->frames.back();
          ExprDoc lhs = d->Define(var, f, var->name.empty() ? "v" : var->name);
          f->stmts.push_back(AssignDoc(lhs, std::nullopt, ann));
        }
        return d->GetVarDoc(var).value();
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<tirx::IterVar>(
      "", [](tirx::IterVar var, AccessPath var_p, IRDocsifier d) -> Doc {
        return TIR(d, "iter_var")
            ->Call({
                d->AsDoc<ExprDoc>(var->var, var_p->Attr("var")),
                d->AsDoc<ExprDoc>(var->dom, var_p->Attr("dom")),
                LiteralDoc::Str(IterVarType2String(var->iter_type), var_p->Attr("iter_type")),
                LiteralDoc::Str(var->thread_tag, var_p->Attr("thread_tag")),
            });
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<te::CommReducer>(  //
      "", [](te::CommReducer r, AccessPath p, IRDocsifier d) -> Doc {
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
}

LambdaDoc PrintIndexMap(const ffi::ObjectRef& map, const ffi::Array<PrimVar>& vs,
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

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<tirx::IndexMap>(  //
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
}

LambdaDoc PrintLambda(const ffi::ObjectRef& pred, const ffi::Array<tirx::Var>& vs,
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

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<tirx::LambdaExpr>(
      "", [](tirx::LambdaExpr pred, AccessPath p, IRDocsifier d) -> Doc {
        return PrintLambda(pred, pred->vars, p->Attr("vars"), pred->pred, p->Attr("pred"), d);
      });
}

Doc PrintTIRCall(Call call, AccessPath call_p, IRDocsifier d) {
  if (call->op.same_as(tirx::builtin::buffer_data())) {
    TVM_FFI_ICHECK_EQ(call->args.size(), 1);
    return d->AsDoc<ExprDoc>(call->args[0], call_p->Attr("args")->ArrayItem(0))->Attr("data");
  }
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
  auto get_call_return_type_doc = [&]() -> ExprDoc {
    if (call->ty.IsMissing()) {
      return IdDoc("tvm")->Attr("ir")->Attr("Type")->Attr("missing")->Call({});
    }
    if (call_prim_type || call->ty.as<PointerTypeNode>()) {
      return get_call_type_doc(call_p->Attr("ty"));
    }
    // Annotation spellings such as None for an empty tuple are not type values.
    return d->AddMetadata(call->ty);
  };
  if (call->attrs.defined()) {
    ffi::Array<ExprDoc> call_args;
    int n_args = call->args.size();
    call_args.reserve(n_args);
    for (int i = 0; i < n_args; ++i) {
      call_args.push_back(d->AsDoc<ExprDoc>(call->args[i], call_p->Attr("args")->ArrayItem(i)));
    }
    if (call->op.same_as(tirx::builtin::tensormap_encode_tiled())) {
      const auto* attrs = call->attrs.as<tirx::TensorMapEncodeTiledAttr>();
      TVM_FFI_ICHECK(attrs);
      auto attr_p = call_p->Attr("attrs");
      return TIR(d, "tensormap_encode_tiled")
          ->Call(call_args,
                 {"descriptor_dtype", "rank", "interleave", "swizzle", "l2_promotion", "oob_fill",
                  "force_cu_dtype"},
                 {LiteralDoc::Str(ffi::DLDataTypeToString(attrs->descriptor_dtype),
                                  attr_p->Attr("descriptor_dtype")),
                  LiteralDoc::Int(attrs->rank, attr_p->Attr("rank")),
                  LiteralDoc::Int(attrs->interleave, attr_p->Attr("interleave")),
                  LiteralDoc::Int(attrs->swizzle, attr_p->Attr("swizzle")),
                  LiteralDoc::Int(attrs->l2_promotion, attr_p->Attr("l2_promotion")),
                  LiteralDoc::Int(attrs->oob_fill, attr_p->Attr("oob_fill")),
                  LiteralDoc::Int(attrs->force_cu_dtype, attr_p->Attr("force_cu_dtype"))});
    }
    if (call->op.same_as(tirx::builtin::call_ffi_kernel())) {
      const auto* attrs = call->attrs.as<tirx::CallFFIKernelAttr>();
      TVM_FFI_ICHECK(attrs);
      return TIR(d, "call_ffi_kernel")
          ->Call(call_args, {"launch_params", "ret_ty"},
                 {d->AsDoc<ExprDoc>(attrs->launch_params,
                                    call_p->Attr("attrs")->Attr("launch_params")),
                  get_call_return_type_doc()});
    }
    ExprDoc op_doc = call->op.as<Op>()
                         ? LiteralDoc::Str(call->op.as<Op>().value()->name, call_p->Attr("op"))
                         : d->AsDoc<ExprDoc>(call->op, call_p->Attr("op"));
    ExprDoc ret_ty_doc = get_call_return_type_doc();
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
      int64_t id = static_cast<int64_t>(call->args[0].as<IntImmNode>()->value);
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
      const auto* src_str = call->args[n_args - 1].as<StringImmNode>();
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
    TVM_FFI_THROW(TypeError) << "Cannot print a Call whose callee has type "
                             << call->op->GetTypeKey();
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

TVM_FFI_STATIC_INIT_BLOCK() { IRDocsifier::vtable().set_dispatch<Call>("tirx", PrintTIRCall); }

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<te::Reduce>(
      "", [](te::Reduce r, AccessPath p, IRDocsifier d) -> Doc {
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
}

TVM_REGISTER_SCRIPT_AS_REPR(tirx::IterVarNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(StringImmNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::CastNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::AddNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::SubNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::MulNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::DivNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::ModNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::FloorDivNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::FloorModNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::LShiftNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::RShiftNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::BitwiseAndNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::BitwiseOrNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::BitwiseXorNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::BitwiseNotNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::MinNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::MaxNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::LTNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::LENode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::EQNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::NENode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::GTNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::GENode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::AndNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::OrNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::NotNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::SelectNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::RampNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::BroadcastNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::LetNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(prim::ShuffleNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(te::CommReducerNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(tirx::IndexMapNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(te::ReduceNode, ReprPrintTIR);
TVM_REGISTER_SCRIPT_AS_REPR(tirx::LambdaExprNode, ReprPrintTIR);

}  // namespace printer
}  // namespace script
}  // namespace tvm
