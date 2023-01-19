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
#include <tvm/runtime/device_api.h>
#include <tvm/tir/stmt_functor.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

String FindFunctionName(const IRDocsifier& d, const tir::PrimFunc& f) {
  if (!d->mod.defined()) {
    return "main";
  }
  for (const auto& kv : d->mod.value()->functions) {
    if (kv.second.same_as(f)) {
      return kv.first->name_hint;
    }
  }
  return "main";
}

bool IsSimpleBuffer(const tir::Buffer& buf) {
  if (!buf->strides.empty()) {
    return false;
  }
  for (const PrimExpr& shp_i : buf->shape) {
    if (!tir::UndefinedVars(shp_i).empty()) {
      return false;
    }
  }
  for (const PrimExpr& stride_i : buf->strides) {
    if (!tir::UndefinedVars(stride_i).empty()) {
      return false;
    }
  }
  if (!tir::UndefinedVars(buf->elem_offset).empty()) {
    return false;
  } else if (buf->elem_offset->IsInstance<IntImmNode>()) {
    IntImm elem_offset = Downcast<IntImm>(buf->elem_offset);
    if (elem_offset->value != 0) {
      return false;
    }
  }
  return buf.scope() == "global" && buf->data_alignment == runtime::kAllocAlignment &&
         buf->offset_factor == 1 && buf->buffer_type == tir::BufferType::kDefault &&
         !buf->axis_separators.size();
}

int CountVarOccurrence(const tir::PrimFunc& f, const tir::Var& v) {
  class OccurrenceCounter : public tir::StmtExprVisitor {
   public:
    int count = 0;
    const tir::VarNode* v = nullptr;

    void VisitExpr_(const tir::VarNode* op) final {
      if (op == v) {
        ++count;
      }
      tir::StmtExprVisitor::VisitExpr_(op);
    }

    void VisitStmt_(const tir::BufferStoreNode* op) final {
      VisitBuffer(op->buffer.get());
      tir::StmtExprVisitor::VisitStmt_(op);
    }

    void VisitExpr_(const tir::BufferLoadNode* op) final {
      VisitBuffer(op->buffer.get());
      tir::StmtExprVisitor::VisitExpr_(op);
    }

    void VisitStmt_(const tir::DeclBufferNode* op) final {
      VisitBuffer(op->buffer.get());
      tir::StmtExprVisitor::VisitStmt_(op);
    }

    void VisitBuffer(const tir::BufferNode* buffer) {
      VisitExpr(buffer->data);
      for (const PrimExpr& shape_i : buffer->shape) {
        VisitExpr(shape_i);
      }
      for (const PrimExpr& stride_i : buffer->strides) {
        VisitExpr(stride_i);
      }
      VisitExpr(buffer->elem_offset);
    }
  };

  OccurrenceCounter counter;
  counter.v = v.get();
  counter(f->body);
  for (const tir::Var& v : f->params) {
    counter(v);
  }
  for (const auto& pair : f->buffer_map) {
    counter(pair.first);
    counter.VisitBuffer(pair.second.get());
  }
  return counter.count;
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::PrimFunc>("", [](tir::PrimFunc func, ObjectPath p, IRDocsifier d) -> Doc {
      With<TIRFrame> frame(MakeDispatchFrame(d, func, func));
      int n_args = func->params.size();
      // Step 1. Handle `func->params`
      Array<AssignDoc> args;
      args.reserve(n_args);
      std::unordered_set<const tir::BufferNode*> buffer_inlined;
      for (int i = 0; i < n_args; ++i) {
        tir::Var var = func->params[i];
        ObjectPath var_p = p->Attr("params")->ArrayIndex(i);
        if (CountVarOccurrence(func, var) == 2 && func->buffer_map.count(var)) {
          tir::Buffer buffer = func->buffer_map[var];
          if (IsSimpleBuffer(buffer)) {
            ObjectPath buffer_p = p->Attr("buffer_map")->MapValue(var);
            args.push_back(AssignDoc(DefineBuffer(buffer, *frame, d), NullOpt,
                                     BufferAttn(buffer, buffer_p, *frame, d)));
            buffer_inlined.insert(buffer.get());
            continue;
          }
        }
        ExprDoc a = d->AsDoc<ExprDoc>(var->type_annotation, var_p->Attr("type_annotation"));
        args.push_back(AssignDoc(DefineVar(var, *frame, d), NullOpt, a));
      }
      // Step 2. Handle `func->attrs`
      if (func->attrs.defined() && !func->attrs->dict.empty()) {
        (*frame)->stmts.push_back(
            ExprStmtDoc(TIR("func_attr")  //
                            ->Call({d->AsDoc<ExprDoc>(func->attrs, p->Attr("attrs"))})));
      }
      // Step 3. Handle `func->buffer_map`
      for (int i = 0; i < n_args; ++i) {
        tir::Var param = func->params[i];
        if (func->buffer_map.count(param)) {
          tir::Buffer buffer = func->buffer_map[param];
          if (buffer_inlined.count(buffer.get())) {
            continue;
          }
          ExprDoc param = args[i]->lhs;
          ObjectPath buffer_p = p->Attr("buffer_map")->MapValue(param);
          ExprDoc lhs =
              DefineBuffer(buffer, *frame, d);  // TODO(@junrushao): switch `lhs` and `rhs`
          ExprDoc rhs = BufferDecl(buffer, "match_buffer", {param}, buffer_p, *frame, d);
          (*frame)->stmts.push_back(AssignDoc(lhs, rhs, NullOpt));
        }
      }
      // Step 4. Handle `func->body`
      AsDocBody(func->body, p->Attr("body"), frame->get(), d);
      Optional<ExprDoc> ret_type = NullOpt;
      if (func->ret_type.defined()) {
        const auto* as_tuple = func->ret_type.as<TupleTypeNode>();
        if (!as_tuple || as_tuple->fields.size()) {
          ret_type = d->AsDoc<ExprDoc>(func->ret_type, p->Attr("ret_type"));
        }
      }
      return FunctionDoc(
          /*name=*/IdDoc(FindFunctionName(d, func)),
          /*args=*/args,
          /*decorators=*/{TIR("prim_func")},
          /*return_type=*/ret_type,
          /*body=*/(*frame)->stmts);
    });

void ReprPrintPrimFunc(const ObjectRef& obj, ReprPrinter* p) {
  std::string res = DocToPythonScript(IRDocsifier()->AsDoc(obj, ObjectPath::Root()));
  p->stream << res;
}

TVM_SCRIPT_REPR(tir::PrimFuncNode, ReprPrintPrimFunc);

}  // namespace printer
}  // namespace script
}  // namespace tvm
