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
#include <tvm/tir/builtin.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

ExprDoc PrintVarCreation(const tir::Var& var, const ObjectPath& var_p, const IRDocsifier& d) {
  Type type = var->type_annotation;
  ObjectPath type_p = var_p->Attr("type_annotation");
  ExprDoc rhs{nullptr};
  Array<String> kwargs_keys;
  Array<ExprDoc> kwargs_values;

  if (var->IsInstance<tir::SizeVarNode>()) {
    kwargs_keys.push_back("is_size_var");
    kwargs_values.push_back(LiteralDoc::Boolean(true, NullOpt));
  }

  if (const auto* ptr_type = type.as<PointerTypeNode>()) {
    const auto* prim_type = ptr_type->element_type.as<PrimTypeNode>();
    ICHECK(prim_type);
    ExprDoc element_type =
        LiteralDoc::DataType(prim_type->dtype, type_p->Attr("element_type")->Attr("dtype"));
    rhs = TIR(d, "handle");
    rhs->source_paths.push_back(var_p->Attr("dtype"));
    if (ptr_type->storage_scope == "") {
      rhs = rhs->Call({element_type}, kwargs_keys, kwargs_values);
    } else {
      rhs = rhs->Call({element_type,
                       LiteralDoc::Str(ptr_type->storage_scope,  //
                                       type_p->Attr("storage_scope"))},
                      kwargs_keys, kwargs_values);
    }
  } else {
    rhs = TIR(d, DType2Str(var->dtype));
    rhs->source_paths.push_back(var_p->Attr("dtype"));
    rhs = rhs->Call({}, kwargs_keys, kwargs_values);
  }
  rhs->source_paths.push_back(type_p);
  return rhs;
}

Doc PrintVar(const tir::Var& var, const ObjectPath& var_p, const IRDocsifier& d) {
  if (!d->IsVarDefined(var)) {
    if (Optional<Frame> opt_f = FindLowestVarDef(var, d)) {
      ExprDoc lhs = DefineVar(var, opt_f.value(), d);
      ExprDoc rhs = PrintVarCreation(var, var_p, d);
      opt_f.value()->stmts.push_back(AssignDoc(lhs, rhs, NullOpt));
    } else {
      LOG(WARNING) << "Didn't find variable definition for: " << var->name_hint;
    }
  }
  if (Optional<ExprDoc> doc = d->GetVarDoc(var)) {
    return doc.value();
  }
  LOG(FATAL) << "IndexError: Variable is not defined in the environment: " << var->name_hint;
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)  //
    .set_dispatch<tir::Var>("", [](tir::Var var, ObjectPath p, IRDocsifier d) -> Doc {
      return PrintVar(var, p, d);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)  //
    .set_dispatch<tir::SizeVar>("", [](tir::SizeVar var, ObjectPath p, IRDocsifier d) -> Doc {
      return PrintVar(var, p, d);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::IterVar>("", [](tir::IterVar var, ObjectPath var_p, IRDocsifier d) -> Doc {
      return TIR(d, "iter_var")
          ->Call({
              d->AsDoc<ExprDoc>(var->var, var_p->Attr("var")),
              d->AsDoc<ExprDoc>(var->dom, var_p->Attr("dom")),
              LiteralDoc::Str(IterVarType2String(var->iter_type), var_p->Attr("iter_type")),
              LiteralDoc::Str(var->thread_tag, var_p->Attr("thread_tag")),
          });
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Not>("", [](tir::Not node, ObjectPath p, IRDocsifier d) -> Doc {
      ExprDoc a = d->AsDoc<ExprDoc>(node->a, p->Attr("a"));
      if (a->IsInstance<LiteralDocNode>()) {
        return TIR(d, "Not")->Call({a});
      }
      return OperationDoc(OperationDocNode::Kind::kNot, {a});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::StringImm>("", [](tir::StringImm s, ObjectPath p, IRDocsifier d) -> Doc {
      if (HasMultipleLines(s->value)) {
        return d->AddMetadata(s);
      } else {
        return d->AsDoc<ExprDoc>(s->value, p->Attr("value"));
      }
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Cast>("", [](tir::Cast cast, ObjectPath p, IRDocsifier d) -> Doc {
      ExprDoc dtype = LiteralDoc::DataType(cast->dtype, p->Attr("dtype"));
      ExprDoc value = d->AsDoc<ExprDoc>(cast->value, p->Attr("value"));
      return TIR(d, "Cast")->Call({dtype, value});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Select>("", [](tir::Select select, ObjectPath p, IRDocsifier d) -> Doc {
      return TIR(d, "Select")
          ->Call({
              d->AsDoc<ExprDoc>(select->condition, p->Attr("condition")),
              d->AsDoc<ExprDoc>(select->true_value, p->Attr("true_value")),
              d->AsDoc<ExprDoc>(select->false_value, p->Attr("false_value")),
          });
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Ramp>("", [](tir::Ramp ramp, ObjectPath ramp_p, IRDocsifier d) -> Doc {
      return TIR(d, "Ramp")->Call({
          d->AsDoc<ExprDoc>(ramp->base, ramp_p->Attr("base")),
          d->AsDoc<ExprDoc>(ramp->stride, ramp_p->Attr("stride")),
          LiteralDoc::Int(ramp->lanes, ramp_p->Attr("lanes")),
      });
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Broadcast>("", [](tir::Broadcast bc, ObjectPath bc_p, IRDocsifier d) -> Doc {
      return TIR(d, "Broadcast")
          ->Call({
              d->AsDoc<ExprDoc>(bc->value, bc_p->Attr("value")),
              LiteralDoc::Int(bc->lanes, bc_p->Attr("lanes")),
          });
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Shuffle>(  //
        "", [](tir::Shuffle shuffle, ObjectPath p, IRDocsifier d) -> Doc {
          return TIR(d, "Shuffle")
              ->Call({
                  d->AsDoc<ExprDoc>(shuffle->vectors, p->Attr("vectors")),
                  d->AsDoc<ExprDoc>(shuffle->indices, p->Attr("indices")),
              });
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::CommReducer>(  //
        "", [](tir::CommReducer r, ObjectPath p, IRDocsifier d) -> Doc {
          ICHECK_EQ(r->lhs.size(), r->rhs.size());
          LambdaDoc lambda{nullptr};
          {
            With<TIRFrame> f(d, r);
            int n_vars = r->lhs.size();
            Array<IdDoc> vars;
            vars.reserve(n_vars + n_vars);
            for (int i = 0; i < n_vars; ++i) {
              vars.push_back(Downcast<IdDoc>(DefineVar(r->lhs[i], *f, d)));
            }
            for (int i = 0; i < n_vars; ++i) {
              vars.push_back(Downcast<IdDoc>(DefineVar(r->rhs[i], *f, d)));
            }
            int n_results = r->result.size();
            Array<ExprDoc> results;
            results.reserve(n_results);
            for (int i = 0; i < n_results; ++i) {
              results.push_back(d->AsDoc<ExprDoc>(r->result[i], p->Attr("result")->ArrayIndex(i)));
            }
            if (results.size() == 1) {
              lambda = LambdaDoc(vars, results[0]);
            } else {
              lambda = LambdaDoc(vars, TupleDoc(results));
            }
          }
          ExprDoc id = d->AsDoc<ExprDoc>(r->identity_element, p->Attr("identity_element"));
          return TIR(d, "comm_reducer")->Call({lambda, id});
        });

LambdaDoc PrintIndexMap(const ObjectRef& map, const Array<tir::Var>& vs, const ObjectPath& vs_p,
                        const Array<PrimExpr>& es, const ObjectPath& es_p, const IRDocsifier& d) {
  With<TIRFrame> f(d, map);
  Array<IdDoc> vars;
  for (int i = 0, l = vs.size(); i < l; ++i) {
    vars.push_back(Downcast<IdDoc>(DefineVar(vs[i], *f, d)));
  }
  Array<ExprDoc> exprs;
  for (int i = 0, l = es.size(); i < l; ++i) {
    exprs.push_back(d->AsDoc<ExprDoc>(es[i], es_p->ArrayIndex(i)));
  }
  return LambdaDoc(vars, TupleDoc(exprs));
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::IndexMap>(  //
        "", [](tir::IndexMap m, ObjectPath m_p, IRDocsifier d) -> Doc {
          LambdaDoc map = PrintIndexMap(m, m->initial_indices, m_p->Attr("initial_indices"),
                                        m->final_indices, m_p->Attr("final_indices"), d);
          if (m->inverse_index_map.defined()) {
            tir::IndexMap inverse = Downcast<tir::IndexMap>(m->inverse_index_map);
            LambdaDoc inv = PrintIndexMap(inverse, inverse->initial_indices,
                                          m_p->Attr("inverse_index_map")->Attr("initial_indices"),
                                          inverse->final_indices,
                                          m_p->Attr("inverse_index_map")->Attr("final_indices"), d);
            return TIR(d, "index_map")->Call({map}, {"inverse_index_map"}, {inv});
          } else {
            return TIR(d, "index_map")->Call({map});
          }
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Let>("", [](tir::Let let, ObjectPath p, IRDocsifier d) -> Doc {
      DictDoc where({d->AsDoc<ExprDoc>(let->var, p->Attr("var"))},
                    {d->AsDoc<ExprDoc>(let->value, p->Attr("value"))});
      return TIR(d, "Let")->Call({d->AsDoc<ExprDoc>(let->body, p->Attr("body"))},  //
                                 {"where"}, {where});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Call>("", [](tir::Call call, ObjectPath call_p, IRDocsifier d) -> Doc {
      static const OpAttrMap<tir::TScriptPrinterName>& op_names =
          Op::GetAttrMap<tir::TScriptPrinterName>("TScriptPrinterName");
      static const OpAttrMap<tir::TScriptDtypePrintLocation> dtype_locations =
          Op::GetAttrMap<tir::TScriptDtypePrintLocation>("TScriptDtypePrintLocation");
      tir::ScriptDtypePrintLocation dtype_print_location = tir::ScriptDtypePrintLocation::kNone;
      ExprDoc prefix{nullptr};
      if (auto optional_op = call->op.as<Op>()) {
        auto op = optional_op.value();
        String name = op_names.get(op, op->name);
        if (op_names.count(op) == 0) {
          LOG(WARNING) << "No TScriptPrinterName attribute for " << op->name;
        }
        prefix = TIR(d, name);
        if (dtype_locations.count(op)) {
          dtype_print_location =
              static_cast<tir::ScriptDtypePrintLocation>(dtype_locations[op].IntValue());
        }
        if (name == "call_llvm_pure_intrin" || name == "call_llvm_intrin") {
          int n_args = call->args.size();
          int64_t id = call->args[0].as<IntImmNode>()->value;
          auto f_llvm_lookup_intrinsic_name =
              tvm::runtime::Registry::Get("target.llvm_get_intrinsic_name");

          Array<ExprDoc> args;
          args.reserve(n_args + 1);
          if (dtype_print_location == tir::ScriptDtypePrintLocation::kFirst) {
            args.push_back(LiteralDoc::DataType(call->dtype, call_p->Attr("dtype")));
          }

          for (int i = 0; i < n_args; ++i) {
            if ((i == 0) && (f_llvm_lookup_intrinsic_name)) {
              String name = (*f_llvm_lookup_intrinsic_name)(id);
              args.push_back(LiteralDoc::Str(name.c_str(), call_p->Attr("args")->ArrayIndex(i)));
            } else {
              args.push_back(d->AsDoc<ExprDoc>(call->args[i], call_p->Attr("args")->ArrayIndex(i)));
            }
          }
          if (dtype_print_location == tir::ScriptDtypePrintLocation::kLast) {
            args.push_back(LiteralDoc::DataType(call->dtype, call_p->Attr("dtype")));
          }
          return prefix->Call(args);
        }
      } else if (call->op.as<GlobalVarNode>()) {
        prefix = d->AsDoc<ExprDoc>(call->op, call_p->Attr("op"));
      } else {
        LOG(FATAL) << "call: " << call;
      }
      Array<ExprDoc> args;
      int n_args = call->args.size();
      args.reserve(n_args + 1);
      if (dtype_print_location == tir::ScriptDtypePrintLocation::kFirst) {
        args.push_back(LiteralDoc::DataType(call->dtype, call_p->Attr("dtype")));
      }

      for (int i = 0; i < n_args; ++i) {
        args.push_back(d->AsDoc<ExprDoc>(call->args[i], call_p->Attr("args")->ArrayIndex(i)));
      }
      if (dtype_print_location == tir::ScriptDtypePrintLocation::kLast) {
        args.push_back(LiteralDoc::DataType(call->dtype, call_p->Attr("dtype")));
      }
      return prefix->Call(args);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Any>("", [](tir::Any any, ObjectPath p, IRDocsifier d) -> Doc {
      return TIR(d, "Any")->Call({});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Reduce>("", [](tir::Reduce r, ObjectPath p, IRDocsifier d) -> Doc {
      ExprDoc combiner = d->AsDoc<ExprDoc>(r->combiner, p->Attr("combiner"));
      ExprDoc source = d->AsDoc<ExprDoc>(r->source, p->Attr("source"));
      ExprDoc init = d->AsDoc<ExprDoc>(r->init, p->Attr("init"));
      ExprDoc axis = d->AsDoc<ExprDoc>(r->axis, p->Attr("axis"));
      ExprDoc condition = d->AsDoc<ExprDoc>(r->condition, p->Attr("condition"));
      ExprDoc value_index = LiteralDoc::Int(r->value_index, p->Attr("value_index"));
      return TIR(d, "reduce")
          ->Call({combiner}, {"source", "init", "axis", "condition", "value_index"},
                 {source, init, axis, condition, value_index});
      LOG(FATAL) << "ValueError: Reduce should never exist in TIR: " << r;
    });

#define TVM_SCRIPT_PRINTER_DEF_BINARY(NodeType, OpString)                                       \
  TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)                                                    \
      .set_dispatch<tir::NodeType>("",                                                          \
                                   [](tir::NodeType node, ObjectPath p, IRDocsifier d) -> Doc { \
                                     ExprDoc a = d->AsDoc<ExprDoc>(node->a, p->Attr("a"));      \
                                     ExprDoc b = d->AsDoc<ExprDoc>(node->b, p->Attr("b"));      \
                                     return TIR(d, OpString)->Call({a, b});                     \
                                   });

bool IsNumber(const ExprDoc& e) {
  if (const auto* n = e.as<LiteralDocNode>()) {
    if (n->value.defined()) {
      return n->value->IsInstance<IntImmNode>() || n->value->IsInstance<FloatImmNode>();
    }
  }
  return false;
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Div>("", [](tir::Div node, ObjectPath p, IRDocsifier d) -> Doc {
      ExprDoc a = d->AsDoc<ExprDoc>(node->a, p->Attr("a"));
      ExprDoc b = d->AsDoc<ExprDoc>(node->b, p->Attr("b"));
      PrimExpr ret = tvm::div(node->a, node->b);
      if (!ret->IsInstance<tir::DivNode>()) {
        return TIR(d, "Div")->Call({a, b});
      }
      if ((node->a->dtype.is_int() || node->a->dtype.is_uint()) &&
          (node->b->dtype.is_int() || node->b->dtype.is_uint())) {
        return TIR(d, "Div")->Call({a, b});
      }
      return OperationDoc(OperationDocNode::Kind::kDiv, {a, b});
    });

#define TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(NodeType, NodeObj, NodeFunc, OpString, OpKind) \
  TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)                                                    \
      .set_dispatch<tir::NodeType>(                                                             \
          "", [](tir::NodeType node, ObjectPath p, IRDocsifier d) -> Doc {                      \
            ExprDoc a = d->AsDoc<ExprDoc>(node->a, p->Attr("a"));                               \
            ExprDoc b = d->AsDoc<ExprDoc>(node->b, p->Attr("b"));                               \
            PrimExpr ret = tvm::NodeFunc(node->a, node->b);                                     \
            if (const auto* ret_node = ret.as<tvm::tir::NodeObj>()) {                           \
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

TVM_SCRIPT_REPR(tir::VarNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::SizeVarNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::IterVarNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::StringImmNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::CastNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::AddNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::SubNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::MulNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::DivNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::ModNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::FloorDivNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::FloorModNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::MinNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::MaxNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::LTNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::LENode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::EQNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::NENode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::GTNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::GENode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::AndNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::OrNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::NotNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::SelectNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::RampNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::BroadcastNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::LetNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::CallNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::ShuffleNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::CommReducerNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::IndexMapNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::AnyNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::ReduceNode, ReprPrintTIR);

}  // namespace printer
}  // namespace script
}  // namespace tvm
