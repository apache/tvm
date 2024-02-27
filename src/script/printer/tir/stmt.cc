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
#include "../../../tir/transforms/ir_utils.h"  // For `GetPtrStorageScope`
#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

Doc DoConciseScoping(const Optional<ExprDoc>& lhs, const ExprDoc& rhs, Array<StmtDoc>* stmts,
                     bool concise_scoping) {
  if (concise_scoping) {
    if (lhs.defined()) {
      stmts->insert(stmts->begin(), AssignDoc(lhs.value(), rhs, NullOpt));
    } else {
      stmts->insert(stmts->begin(), ExprStmtDoc(rhs));
    }
    return StmtBlockDoc(*stmts);
  } else {
    return ScopeDoc(lhs, rhs, *stmts);
  }
}

bool AllowConciseScoping(const IRDocsifier& d, const ObjectRef& obj) {
  if (d->cfg.defined()) {
    if (d->cfg->obj_to_annotate.count(obj)) {
      // if the object requires annotation, do not fold this frame
      return false;
    }
  }
  ICHECK(!d->frames.empty());
  if (const auto* f = d->frames.back().as<TIRFrameNode>()) {
    return f->allow_concise_scoping;
  }
  LOG(FATAL) << "NotImplementedError: fragment printing";
}

bool IsAncestorOfAllVarUse(const tir::Stmt& node, const ObjectRef& var, const IRDocsifier& d) {
  if (!d->common_prefix.count(var.get())) {
    return false;
  }
  const std::vector<const Object*>& path = d->common_prefix.at(var.get());
  for (auto it = path.rbegin(); it != path.rend(); ++it) {
    if (*it == node.get()) {
      return true;
    }
  }
  return false;
}

Optional<PrimExpr> FindReturnValue(const tir::Stmt& node) {
  auto eval = node.as<tir::EvaluateNode>();
  if (!eval) return NullOpt;

  auto call = eval->value.as<tir::CallNode>();
  if (!call) return NullOpt;

  if (!call->op.same_as(tir::builtin::ret())) return NullOpt;

  if (call->args.size() != 1) return NullOpt;

  return call->args[0];
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Evaluate>("", [](tir::Evaluate eval, ObjectPath p, IRDocsifier d) -> Doc {
      if (d->cfg->syntax_sugar) {
        if (auto return_value = FindReturnValue(eval)) {
          ExprDoc value = d->AsDoc<ExprDoc>(return_value.value(),
                                            p->Attr("value")->Attr("args")->ArrayIndex(0));
          return ReturnDoc(value);
        }
      }

      ExprDoc value = d->AsDoc<ExprDoc>(eval->value, p->Attr("value"));
      if (eval->value->IsInstance<tir::CallNode>()) {
        return ExprStmtDoc(value);
      }
      return ExprStmtDoc(TIR(d, "evaluate")->Call({value}));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::LetStmt>("", [](tir::LetStmt stmt, ObjectPath p, IRDocsifier d) -> Doc {
      bool concise = AllowConciseScoping(d, stmt);
      // Step 1. Type annotation
      Optional<ExprDoc> type_doc = d->AsDoc<ExprDoc>(stmt->var->type_annotation,  //
                                                     p->Attr("var")->Attr("type_annotation"));
      if (const auto* tuple_type = stmt->var->type_annotation.as<TupleTypeNode>()) {
        if (tuple_type->fields.empty()) {
          type_doc = NullOpt;
        }
      }
      // Step 2. RHS
      ExprDoc rhs = d->AsDoc<ExprDoc>(stmt->value, p->Attr("value"));
      // Step 3. LHS and body
      With<TIRFrame> f(d, stmt);
      Array<StmtDoc>* stmts = &(*f)->stmts;
      bool var_defined = d->IsVarDefined(stmt->var);
      if (!var_defined) {
        DefineVar(stmt->var, *f, d);
      }
      ExprDoc lhs = d->AsDoc<ExprDoc>(stmt->var, p->Attr("var"));
      AsDocBody(stmt->body, p->Attr("body"), f->get(), d);
      // Step 4. Dispatch
      if (var_defined) {
        return ScopeDoc(NullOpt, TIR(d, "LetStmt")->Call({rhs}, {"var"}, {lhs}), *stmts);
      } else if (concise) {
        stmts->insert(stmts->begin(), AssignDoc(lhs, rhs, type_doc));
        return StmtBlockDoc(*stmts);
      } else if (type_doc.defined() && !stmt->var->type_annotation->IsInstance<PrimTypeNode>()) {
        return ScopeDoc(lhs, TIR(d, "LetStmt")->Call({rhs, type_doc.value()}), *stmts);
      } else {
        return ScopeDoc(lhs, TIR(d, "LetStmt")->Call({rhs}), *stmts);
      }
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::AssertStmt>(
        "", [](tir::AssertStmt stmt, ObjectPath p, IRDocsifier d) -> Doc {
          bool concise = AllowConciseScoping(d, stmt);
          ExprDoc cond = d->AsDoc<ExprDoc>(stmt->condition, p->Attr("condition"));
          ExprDoc msg = d->AsDoc<ExprDoc>(stmt->message, p->Attr("message"));
          With<TIRFrame> f(d, stmt);
          AsDocBody(stmt->body, p->Attr("body"), f->get(), d);
          if (concise) {
            Array<StmtDoc>* stmts = &(*f)->stmts;
            stmts->insert(stmts->begin(), AssertDoc(cond, msg));
            return StmtBlockDoc(*stmts);
          }
          return ScopeDoc(NullOpt, TIR(d, "Assert")->Call({cond, msg}), (*f)->stmts);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::While>("", [](tir::While stmt, ObjectPath p, IRDocsifier d) -> Doc {
      ExprDoc cond = d->AsDoc<ExprDoc>(stmt->condition, p->Attr("condition"));
      With<TIRFrame> f(d, stmt);
      AsDocBody(stmt->body, p->Attr("body"), f->get(), d);
      return WhileDoc(cond, (*f)->stmts);
    });

namespace {
Doc DeclBufferDoc(tir::DeclBuffer stmt, ObjectPath p, IRDocsifier d,
                  BufferVarDefinition var_definitions) {
  bool concise = AllowConciseScoping(d, stmt);
  ExprDoc rhs = BufferDecl(stmt->buffer, "decl_buffer", {}, p->Attr("buffer"), d->frames.back(), d,
                           var_definitions);
  With<TIRFrame> f(d, stmt);
  ExprDoc lhs = DefineBuffer(stmt->buffer, *f, d);
  AsDocBody(stmt->body, p->Attr("body"), f->get(), d);
  return DoConciseScoping(lhs, rhs, &(*f)->stmts, concise);
}
}  // namespace

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::DeclBuffer>(  //
        "", [](tir::DeclBuffer stmt, ObjectPath p, IRDocsifier d) -> Doc {
          return DeclBufferDoc(stmt, p, d, BufferVarDefinition::None);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::IfThenElse>(  //
        "", [](tir::IfThenElse stmt, ObjectPath p, IRDocsifier d) -> Doc {
          ExprDoc cond = d->AsDoc<ExprDoc>(stmt->condition, p->Attr("condition"));
          Array<StmtDoc> then_branch;
          Array<StmtDoc> else_branch;
          if (stmt->then_case.defined()) {
            With<TIRFrame> f(d, stmt->then_case);
            AsDocBody(stmt->then_case, p->Attr("then_case"), f->get(), d);
            then_branch = (*f)->stmts;
          }
          if (stmt->else_case.defined()) {
            With<TIRFrame> f(d, stmt->else_case);
            AsDocBody(stmt->else_case.value(), p->Attr("else_case"), f->get(), d);
            else_branch = (*f)->stmts;
          }
          return IfDoc(cond, then_branch, else_branch);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::SeqStmt>("", [](tir::SeqStmt stmt, ObjectPath p, IRDocsifier d) -> Doc {
      With<TIRFrame> f(d, stmt);
      AsDocBody(stmt, p, f->get(), d);
      return StmtBlockDoc((*f)->stmts);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Prefetch>(  //
        "", [](tir::Prefetch stmt, ObjectPath p, IRDocsifier d) -> Doc {
          return ExprStmtDoc(TIR(d, "prefetch")
                                 ->Call({
                                     d->AsDoc<ExprDoc>(stmt->buffer, p->Attr("buffer")),
                                     d->AsDoc<ExprDoc>(stmt->bounds, p->Attr("bounds")),
                                 }));
        });

bool IsAllocateDeclBufferPattern(const tir::AllocateNode* allocate) {
  const tir::Var& buffer_var = allocate->buffer_var;
  if (const tir::DeclBufferNode* decl_buffer = allocate->body.as<tir::DeclBufferNode>()) {
    const tir::Buffer& buffer = decl_buffer->buffer;
    if (buffer_var.same_as(buffer->data) && allocate->dtype == buffer->dtype &&
        tir::is_one(allocate->condition) && !allocate->annotations.size() &&
        allocate->extents.size() == buffer->shape.size()) {
      tir::ExprDeepEqual expr_equal;
      for (size_t i = 0, n = allocate->extents.size(); i < n; ++i) {
        if (!expr_equal(allocate->extents[i], buffer->shape[i])) {
          return false;
        }
      }
      return true;
    }
  }
  return false;
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Allocate>(  //
        "", [](tir::Allocate stmt, ObjectPath stmt_p, IRDocsifier d) -> Doc {
          bool concise = AllowConciseScoping(d, stmt_p);
          if (d->cfg->syntax_sugar && IsAllocateDeclBufferPattern(stmt.get())) {
            return DeclBufferDoc(Downcast<tir::DeclBuffer>(stmt->body), stmt_p->Attr("body"), d,
                                 BufferVarDefinition::DataPointer);
          }
          Array<ExprDoc> args;
          Array<String> kwargs_keys;
          Array<ExprDoc> kwargs_values;
          args.push_back(d->AsDoc<ExprDoc>(stmt->extents, stmt_p->Attr("extents")));
          args.push_back(LiteralDoc::DataType(stmt->dtype, stmt_p->Attr("dtype")));
          args.push_back(LiteralDoc::Str(tir::GetPtrStorageScope(stmt->buffer_var),
                                         stmt_p
                                             ->Attr("buffer_var")  //
                                             ->Attr("type_annotation")
                                             ->Attr("storage_scope")));
          if (!tir::is_one(stmt->condition)) {
            args.push_back(d->AsDoc<ExprDoc>(stmt->condition, stmt_p->Attr("condition")));
          }
          if (!stmt->annotations.empty()) {
            kwargs_keys.push_back("annotations");
            kwargs_values.push_back(
                d->AsDoc<ExprDoc>(stmt->annotations, stmt_p->Attr("annotations")));
          }
          ExprDoc lhs = DefineVar(stmt->buffer_var, d->frames.back(), d);
          With<TIRFrame> f(d, stmt);
          ExprDoc rhs = TIR(d, "allocate")->Call(args, kwargs_keys, kwargs_values);
          AsDocBody(stmt->body, stmt_p->Attr("body"), f->get(), d);
          return DoConciseScoping(lhs, rhs, &(*f)->stmts, concise);
        });

template <typename T>
ExprDoc PrintNDArray(::tvm::runtime::NDArray arr) {
  // FIXME(@junrushao): this is a hack and can be wrong in most of the cases
  constexpr int NUM_PRINT = 200;
  int ndim = arr->ndim;
  int tot_dim = 1;
  for (int i = 0; i < ndim; i++) {
    tot_dim *= arr->shape[i];
  }
  Array<ExprDoc> result;
  T* data_ptr = reinterpret_cast<T*>(arr->data);
  runtime::DataType dtype = arr.DataType();
  for (int i = 0; i < tot_dim; i++) {
    if (dtype.is_float()) {
      result.push_back(LiteralDoc::Float(data_ptr[i], NullOpt));
    } else {
      result.push_back(LiteralDoc::Int(data_ptr[i], NullOpt));
    }
    if (i == NUM_PRINT) {
      break;
    }
  }
  return ListDoc(result);
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::AllocateConst>(
        "", [](tir::AllocateConst stmt, ObjectPath stmt_p, IRDocsifier d) -> Doc {
          bool concise = AllowConciseScoping(d, stmt);
          String storage_scope = tir::GetPtrStorageScope(stmt->buffer_var);
          Array<ExprDoc> args;
          Array<String> kwargs_keys;
          Array<ExprDoc> kwargs_values;
          ExprDoc data_doc{nullptr};
          if (stmt->dtype.is_int()) {
            if (stmt->dtype.bits() == 8) {
              data_doc = PrintNDArray<int8_t>(stmt->data.value());
            } else if (stmt->dtype.bits() == 16) {
              data_doc = PrintNDArray<int16_t>(stmt->data.value());
            } else if (stmt->dtype.bits() == 32) {
              data_doc = PrintNDArray<int32_t>(stmt->data.value());
            } else if (stmt->dtype.bits() == 64) {
              data_doc = PrintNDArray<int64_t>(stmt->data.value());
            } else {
              LOG(FATAL) << "DataType not supported";
            }
          } else if (stmt->dtype.is_uint()) {
            if (stmt->dtype.bits() == 8) {
              data_doc = PrintNDArray<uint8_t>(stmt->data.value());
            } else if (stmt->dtype.bits() == 16) {
              data_doc = PrintNDArray<uint16_t>(stmt->data.value());
            } else if (stmt->dtype.bits() == 32) {
              data_doc = PrintNDArray<uint32_t>(stmt->data.value());
            } else if (stmt->dtype.bits() == 64) {
              data_doc = PrintNDArray<uint64_t>(stmt->data.value());
            } else {
              LOG(FATAL) << "DataType not supported";
            }
          } else if (stmt->dtype.is_float()) {
            if (stmt->dtype.bits() == 16) {
              data_doc = PrintNDArray<int16_t>(stmt->data.value());
            } else if (stmt->dtype.bits() == 32) {
              data_doc = PrintNDArray<float>(stmt->data.value());
            } else if (stmt->dtype.bits() == 64) {
              data_doc = PrintNDArray<double>(stmt->data.value());
            } else {
              LOG(FATAL) << "DataType not supported";
            }
          } else {
            LOG(FATAL) << "DataType not supported";
          }
          args.push_back(data_doc);
          args.push_back(LiteralDoc::DataType(stmt->dtype, stmt_p->Attr("dtype")));
          args.push_back(d->AsDoc<ExprDoc>(stmt->extents, stmt_p->Attr("extents")));
          ExprDoc rhs = TIR(d, "allocate_const")->Call(args, kwargs_keys, kwargs_values);
          With<TIRFrame> f(d, stmt);
          ExprDoc lhs = DefineVar(stmt->buffer_var, *f, d);
          AsDocBody(stmt->body, stmt_p->Attr("body"), f->get(), d);
          return DoConciseScoping(lhs, rhs, &(*f)->stmts, concise);
        });

ExprDoc DocsifyBufferRealize(const tir::BufferRealizeNode* stmt, Optional<ExprDoc> value,  //
                             ObjectPath p, IRDocsifier d) {
  ExprDoc buffer = d->AsDoc<ExprDoc>(stmt->buffer, p->Attr("buffer"));
  {
    Array<Doc> bounds;
    bounds.reserve(stmt->bounds.size());
    for (int i = 0, n = stmt->bounds.size(); i < n; ++i) {
      Range range = stmt->bounds[i];
      ObjectPath range_p = p->Attr("bounds")->ArrayIndex(i);
      bounds.push_back(
          SliceDoc(d->AsDoc<ExprDoc>(range->min, range_p->Attr("min")),
                   d->AsDoc<ExprDoc>(range->min + range->extent, range_p->Attr("extent")),  //
                   NullOpt));
    }
    buffer = buffer[bounds];
  }
  Array<ExprDoc> args{buffer};
  Array<String> kwargs_keys;
  Array<ExprDoc> kwargs_values;
  if (value.defined()) {
    args.push_back(value.value());
  }
  if (!tir::is_one(stmt->condition)) {
    kwargs_keys.push_back("condition");
    kwargs_values.push_back(d->AsDoc<ExprDoc>(stmt->condition, p->Attr("condition")));
  }
  return TIR(d, "realize")->Call(args, kwargs_keys, kwargs_values);
}

void InsertEnvThread(const tir::IterVar& iter_var, const ObjectPath& iter_var_p,
                     const IRDocsifier& d) {
  Frame f = FindLowestVarDef(iter_var->var, d).value();
  DefineVar(iter_var->var, f, d);
  ExprDoc rhs = TIR(d, "env_thread")
                    ->Call({LiteralDoc::Str(iter_var->thread_tag,  //
                                            iter_var_p->Attr("thread_tag"))});
  ExprDoc lhs = d->AsDoc<ExprDoc>(iter_var->var, iter_var_p->Attr("var"));
  f->stmts.push_back(AssignDoc(lhs, rhs, NullOpt));
}

ExprDoc DocsifyLaunchThread(const tir::AttrStmt& attr_stmt, const ObjectPath& attr_stmt_p,
                            Optional<tir::Var>* define_var, const IRDocsifier& d) {
  tir::IterVar iter_var = Downcast<tir::IterVar>(attr_stmt->node);
  ObjectPath iter_var_p = attr_stmt_p->Attr("node");

  ExprDoc var_doc{nullptr};
  if (d->IsVarDefined(iter_var->var)) {
    var_doc = d->AsDoc<ExprDoc>(iter_var->var, iter_var_p->Attr("var"));
  } else if (IsAncestorOfAllVarUse(attr_stmt, iter_var->var, d)) {
    var_doc = LiteralDoc::Str(iter_var->thread_tag, iter_var_p->Attr("thread_tag"));
    *define_var = iter_var->var;
  } else {
    InsertEnvThread(iter_var, iter_var_p, d);
    var_doc = d->AsDoc<ExprDoc>(iter_var->var, iter_var_p->Attr("var"));
  }
  return TIR(d, "launch_thread")
      ->Call({
          var_doc,
          d->AsDoc<ExprDoc>(attr_stmt->value, attr_stmt_p->Attr("value")),
      });
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BufferRealize>(  //
        "", [](tir::BufferRealize stmt, ObjectPath p, IRDocsifier d) -> Doc {
          bool concise = AllowConciseScoping(d, stmt);
          ExprDoc rhs = DocsifyBufferRealize(stmt.get(), NullOpt, p, d);
          With<TIRFrame> f(d, stmt);
          AsDocBody(stmt->body, p->Attr("body"), f->get(), d);
          return DoConciseScoping(NullOpt, rhs, &(*f)->stmts, concise);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::AttrStmt>(  //
        "", [](tir::AttrStmt stmt, ObjectPath stmt_p, IRDocsifier d) -> Doc {
          bool concise = AllowConciseScoping(d, stmt);
          Optional<ExprDoc> lhs = NullOpt;
          Optional<ExprDoc> rhs = NullOpt;
          Optional<tir::Var> define_var = NullOpt;
          tir::Stmt body = stmt->body;
          ObjectPath body_p = stmt_p->Attr("body");
          if (stmt->attr_key == "realize_scope") {
            if (const auto* realize = stmt->body.as<tir::BufferRealizeNode>()) {
              if (realize->buffer.same_as(stmt->node)) {
                rhs = DocsifyBufferRealize(
                    realize,
                    /*value=*/d->AsDoc<ExprDoc>(stmt->value, stmt_p->Attr("value")),
                    /*p=*/stmt_p->Attr("body"), d);
                body = realize->body;
                body_p = stmt_p->Attr("body")->Attr("body");
              }
            }
          }
          if (stmt->attr_key == "thread_extent" || stmt->attr_key == "virtual_thread") {
            if (stmt->node->IsInstance<tir::IterVarNode>()) {
              rhs = DocsifyLaunchThread(stmt, stmt_p, &define_var, d);
            }
          }
          if (!rhs.defined()) {
            rhs = TIR(d, "attr")->Call({
                d->AsDoc<ExprDoc>(stmt->node, stmt_p->Attr("node")),
                LiteralDoc::Str(stmt->attr_key, stmt_p->Attr("attr_key")),
                d->AsDoc<ExprDoc>(stmt->value, stmt_p->Attr("value")),
            });
          }
          With<TIRFrame> f(d, stmt);
          if (define_var.defined()) {
            lhs = DefineVar(define_var.value(), *f, d);
          }
          AsDocBody(body, body_p, f->get(), d);
          return DoConciseScoping(lhs, rhs.value(), &(*f)->stmts, concise);
        });

TVM_SCRIPT_REPR(tir::LetStmtNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::AttrStmtNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::AssertStmtNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::WhileNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::AllocateNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::AllocateConstNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::DeclBufferNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::PrefetchNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::SeqStmtNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::IfThenElseNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::EvaluateNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::BufferRealizeNode, ReprPrintTIR);

}  // namespace printer
}  // namespace script
}  // namespace tvm
