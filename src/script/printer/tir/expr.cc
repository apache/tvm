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

Doc PrintVar(const tir::Var& var, const ObjectPath& p, const IRDocsifier& d) {
  if (!d->IsVarDefined(var)) {
    if (Optional<Frame> opt_f = FindLowestVarDef(var, d)) {
      ExprDoc lhs = DefineVar(var, opt_f.value(), d);
      Type type = var->type_annotation;
      if (const auto* ptr_type = type.as<PointerTypeNode>()) {
        ICHECK(ptr_type->element_type->IsInstance<PrimTypeNode>());
        ExprDoc rhs = d->AsDoc<ExprDoc>(type, p->Attr("type_annotation"));
        opt_f.value()->stmts.push_back(AssignDoc(lhs, rhs, NullOpt));
      } else {
        ExprDoc rhs = TIR(d)->Attr("var")->Call({LiteralDoc::DataType(var->dtype)});
        opt_f.value()->stmts.push_back(AssignDoc(lhs, rhs, NullOpt));
      }
    }
  }
  if (Optional<ExprDoc> doc = d->GetVarDoc(var)) {
    return doc.value();
  }
  LOG(FATAL) << "IndexError: Variable is not defined in the environment: " << var;
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
    .set_dispatch<tir::IterVar>("", [](tir::IterVar var, ObjectPath p, IRDocsifier d) -> Doc {
      return TIR(d)
          ->Attr("iter_var")
          ->Call({
              d->AsDoc<ExprDoc>(var->var, p->Attr("var")),
              d->AsDoc<ExprDoc>(var->dom, p->Attr("dom")),
              LiteralDoc::Str(IterVarType2String(var->iter_type)),
              LiteralDoc::Str(var->thread_tag),
          });
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)  //
    .set_dispatch<tir::Buffer>("", [](tir::Buffer buffer, ObjectPath p, IRDocsifier d) -> Doc {
      if (!d->IsVarDefined(buffer)) {
        if (Optional<Frame> opt_f = FindLowestVarDef(buffer, d)) {
          ExprDoc lhs = DefineBuffer(buffer, opt_f.value(), d);
          ExprDoc rhs = BufferDecl(buffer, "buffer_decl",  // TODO(@junrushao): name confusing
                                   {}, p, opt_f.value(), d);
          opt_f.value()->stmts.push_back(AssignDoc(lhs, rhs, NullOpt));
        }
      }
      if (Optional<ExprDoc> doc = d->GetVarDoc(buffer)) {
        return doc.value();
      }
      LOG(FATAL) << "IndexError: Buffer is not defined in the environment: " << buffer;
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Not>("", [](tir::Not node, ObjectPath p, IRDocsifier d) -> Doc {
      ExprDoc a = d->AsDoc<ExprDoc>(node->a, p->Attr("a"));
      if (a->IsInstance<LiteralDocNode>()) {
        return TIR(d)->Attr("Not")->Call({a});
      }
      return OperationDoc(OperationDocNode::Kind::kNot, {a});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::StringImm>("", [](tir::StringImm s, ObjectPath p, IRDocsifier d) -> Doc {
      return d->AsDoc<ExprDoc>(s->value, p->Attr("value"));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Cast>("", [](tir::Cast cast, ObjectPath p, IRDocsifier d) -> Doc {
      ExprDoc dtype = LiteralDoc::DataType(cast->dtype);
      ExprDoc value = d->AsDoc<ExprDoc>(cast->value, p->Attr("value"));
      return TIR(d)->Attr("Cast")->Call({dtype, value});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Select>("", [](tir::Select select, ObjectPath p, IRDocsifier d) -> Doc {
      return TIR(d)->Attr("Select")->Call({
          d->AsDoc<ExprDoc>(select->condition, p->Attr("condition")),
          d->AsDoc<ExprDoc>(select->true_value, p->Attr("true_value")),
          d->AsDoc<ExprDoc>(select->false_value, p->Attr("false_value")),
      });
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Ramp>("", [](tir::Ramp ramp, ObjectPath p, IRDocsifier d) -> Doc {
      return TIR(d)->Attr("Ramp")->Call({
          d->AsDoc<ExprDoc>(ramp->base, p->Attr("base")),
          d->AsDoc<ExprDoc>(ramp->stride, p->Attr("stride")),
          LiteralDoc::Int(ramp->lanes),
      });
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Broadcast>("", [](tir::Broadcast bc, ObjectPath p, IRDocsifier d) -> Doc {
      return TIR(d)
          ->Attr("Broadcast")
          ->Call({
              d->AsDoc<ExprDoc>(bc->value, p->Attr("value")),
              LiteralDoc::Int(bc->lanes),
          });
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Shuffle>(  //
        "", [](tir::Shuffle shuffle, ObjectPath p, IRDocsifier d) -> Doc {
          return TIR(d)->Attr("Shuffle")->Call({
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
              vars.push_back(DefineVar(r->lhs[i], *f, d));
            }
            for (int i = 0; i < n_vars; ++i) {
              vars.push_back(DefineVar(r->rhs[i], *f, d));
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
          return TIR(d)->Attr("comm_reducer")->Call({lambda, id});
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Let>("", [](tir::Let let, ObjectPath p, IRDocsifier d) -> Doc {
      return TIR(d)->Attr("let")->Call({
          d->AsDoc<ExprDoc>(let->var, p->Attr("var")),
          d->AsDoc<ExprDoc>(let->value, p->Attr("value")),
          d->AsDoc<ExprDoc>(let->body, p->Attr("body")),
      });
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Call>("", [](tir::Call call, ObjectPath p, IRDocsifier d) -> Doc {
      static const OpAttrMap<tir::TScriptPrinterName>& op_names =
          Op::GetAttrMap<tir::TScriptPrinterName>("TScriptPrinterName");
      static const std::unordered_set<const Object*> dtype_first_arg = {
          tir::builtin::reinterpret().get(),
          tir::builtin::call_extern().get(),
          tir::builtin::call_llvm_intrin().get(),       //
          tir::builtin::call_llvm_pure_intrin().get(),  //
          tir::builtin::call_pure_extern().get(),       //
          tir::builtin::ptx_mma().get(),
          tir::builtin::ptx_mma_sp().get(),
          tir::builtin::ptx_ldmatrix().get(),
          tir::builtin::ptx_cp_async().get(),
          tir::builtin::mma_store().get(),
          tir::builtin::mma_fill().get(),
          tir::builtin::vectorlow().get(),
          tir::builtin::vectorhigh().get(),
          tir::builtin::vectorcombine().get(),
          Op::Get("tir.type_annotation").get(),
      };
      static const std::unordered_set<const Object*> dtype_last_arg = {
          tir::builtin::tvm_struct_get().get(),
      };
      ExprDoc prefix{nullptr};
      if (const auto* op = call->op.as<OpNode>()) {
        String name = op_names[GetRef<Op>(op)];
        prefix = TIR(d)->Attr(name);
      } else if (const auto* gv = call->op.as<GlobalVarNode>()) {
        prefix = LiteralDoc::Str(gv->name_hint);
      } else {
        LOG(FATAL) << "call: " << call;
      }
      Array<ExprDoc> args;
      int n_args = call->args.size();
      args.reserve(n_args + 1);
      if (dtype_first_arg.count(call->op.get())) {
        args.push_back(LiteralDoc::DataType(call->dtype));
      }
      for (int i = 0; i < n_args; ++i) {
        args.push_back(d->AsDoc<ExprDoc>(call->args[i], p->Attr("args")->ArrayIndex(i)));
      }
      if (dtype_last_arg.count(call->op.get())) {
        args.push_back(LiteralDoc::DataType(call->dtype));
      }
      return prefix->Call(args);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Any>("", [](tir::Any any, ObjectPath p, IRDocsifier d) -> Doc {
      return TIR(d)->Attr("Any")->Call({});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Reduce>("", [](tir::Reduce r, ObjectPath p, IRDocsifier d) -> Doc {
      LOG(FATAL) << "ValueError: Reduce should never exist in TIR: " << r;
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::ProducerLoad>(
        "", [](tir::ProducerLoad load, ObjectPath p, IRDocsifier d) -> Doc {
          LOG(FATAL) << "ValueError: ProducerLoad should never exist in TIR: " << load;
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Load>("", [](tir::Load load, ObjectPath p, IRDocsifier d) -> Doc {
      LOG(FATAL) << "ValueError: Load has been deprecated for BufferLoad: " << load;
    });

#define TVM_SCRIPT_PRINTER_DEF_BINARY(NodeType, OpString)                                       \
  TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)                                                    \
      .set_dispatch<tir::NodeType>("",                                                          \
                                   [](tir::NodeType node, ObjectPath p, IRDocsifier d) -> Doc { \
                                     ExprDoc a = d->AsDoc<ExprDoc>(node->a, p->Attr("a"));      \
                                     ExprDoc b = d->AsDoc<ExprDoc>(node->b, p->Attr("b"));      \
                                     return TIR(d)->Attr(OpString)->Call({a, b});               \
                                   });

#define TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(NodeType, OpString, OpKind)          \
  TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)                                          \
      .set_dispatch<tir::NodeType>(                                                   \
          "", [](tir::NodeType node, ObjectPath p, IRDocsifier d) -> Doc {            \
            ExprDoc a = d->AsDoc<ExprDoc>(node->a, p->Attr("a"));                     \
            ExprDoc b = d->AsDoc<ExprDoc>(node->b, p->Attr("b"));                     \
            if (a->IsInstance<LiteralDocNode>() && b->IsInstance<LiteralDocNode>()) { \
              return TIR(d)->Attr(OpString)->Call({a, b});                            \
            }                                                                         \
            return OperationDoc(OperationDocNode::Kind::OpKind, {a, b});              \
          });

TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(Add, "Add", kAdd);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(Sub, "Sub", kSub);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(Mul, "Mul", kMult);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(Div, "Div", kDiv);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(FloorDiv, "FloorDiv", kFloorDiv);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(FloorMod, "FloorMod", kMod);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(LT, "LT", kLt);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(LE, "LE", kLtE);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(EQ, "EQ", kEq);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(NE, "NE", kNotEq);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(GT, "GT", kGt);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(GE, "GE", kGtE);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(And, "And", kAnd);
TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(Or, "Or", kOr);

TVM_SCRIPT_PRINTER_DEF_BINARY(Mod, "truncmod");
TVM_SCRIPT_PRINTER_DEF_BINARY(Min, "min");
TVM_SCRIPT_PRINTER_DEF_BINARY(Max, "max");

#undef TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR
#undef TVM_SCRIPT_PRINTER_DEF_BINARY

}  // namespace printer
}  // namespace script
}  // namespace tvm
