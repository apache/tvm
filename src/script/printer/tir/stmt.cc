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
#include "../../../tir/transform/ir_utils.h"  // For `GetPtrStorageScope`
#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

Doc DoConciseScoping(const ffi::Optional<ExprDoc>& lhs, const ExprDoc& rhs,
                     ffi::Array<StmtDoc>* stmts, bool concise_scoping) {
  if (concise_scoping) {
    if (lhs.defined()) {
      stmts->insert(stmts->begin(), AssignDoc(lhs.value(), rhs, std::nullopt));
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
  TVM_FFI_ICHECK(!d->frames.empty());
  if (const auto* f = d->frames.back().as<TIRFrameNode>()) {
    return f->allow_concise_scoping;
  }
  TVM_FFI_THROW(NotImplementedError) << "fragment printing";
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

ffi::Optional<PrimExpr> FindReturnValue(const tir::Stmt& node) {
  auto eval = node.as<tir::EvaluateNode>();
  if (!eval) return std::nullopt;

  auto call = eval->value.as<tir::CallNode>();
  if (!call) return std::nullopt;

  if (!call->op.same_as(tir::builtin::ret())) return std::nullopt;

  if (call->args.size() != 1) return std::nullopt;

  return call->args[0];
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Evaluate>("", [](tir::Evaluate eval, AccessPath p, IRDocsifier d) -> Doc {
      if (d->cfg->syntax_sugar) {
        if (auto return_value = FindReturnValue(eval)) {
          ExprDoc value =
              d->AsDoc<ExprDoc>(return_value.value(), p->Attr("value")->Attr("args")->ArrayItem(0));
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
    .set_dispatch<tir::LetStmt>("", [](tir::LetStmt stmt, AccessPath p, IRDocsifier d) -> Doc {
      bool concise = AllowConciseScoping(d, stmt);
      // Step 1. Type annotation
      ffi::Optional<ExprDoc> type_doc = d->AsDoc<ExprDoc>(stmt->var->type_annotation,  //
                                                          p->Attr("var")->Attr("type_annotation"));
      if (const auto* tuple_type = stmt->var->type_annotation.as<TupleTypeNode>()) {
        if (tuple_type->fields.empty()) {
          type_doc = std::nullopt;
        }
      }
      // Step 2. RHS
      ExprDoc rhs = d->AsDoc<ExprDoc>(stmt->value, p->Attr("value"));
      // Step 3. LHS and body
      With<TIRFrame> f(d, stmt);
      ffi::Array<StmtDoc>* stmts = &(*f)->stmts;
      bool var_defined = d->IsVarDefined(stmt->var);
      if (!var_defined) {
        DefineVar(stmt->var, *f, d);
      }
      ExprDoc lhs = d->AsDoc<ExprDoc>(stmt->var, p->Attr("var"));
      AsDocBody(stmt->body, p->Attr("body"), f->get(), d);
      // Step 4. Dispatch
      if (var_defined) {
        return ScopeDoc(std::nullopt, TIR(d, "LetStmt")->Call({rhs}, {"var"}, {lhs}), *stmts);
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
        "", [](tir::AssertStmt stmt, AccessPath p, IRDocsifier d) -> Doc {
          ExprDoc cond = d->AsDoc<ExprDoc>(stmt->condition, p->Attr("condition"));
          // Always emit the canonical tuple form: assert cond, ("Kind", ["part0", "part1", ...])
          ffi::Array<ExprDoc> parts;
          auto parts_path = p->Attr("message_parts");
          for (size_t i = 0; i < stmt->message_parts.size(); ++i) {
            parts.push_back(d->AsDoc<ExprDoc>(stmt->message_parts[i], parts_path->ArrayItem(i)));
          }
          ExprDoc kind_doc = d->AsDoc<ExprDoc>(stmt->error_kind, p->Attr("error_kind"));
          return AssertDoc(cond, TupleDoc({kind_doc, ListDoc(parts)}));
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::While>("", [](tir::While stmt, AccessPath p, IRDocsifier d) -> Doc {
      ExprDoc cond = d->AsDoc<ExprDoc>(stmt->condition, p->Attr("condition"));
      With<TIRFrame> f(d, stmt);
      AsDocBody(stmt->body, p->Attr("body"), f->get(), d);
      return WhileDoc(cond, (*f)->stmts);
    });

namespace {
Doc DeclBufferDoc(tir::DeclBuffer stmt, AccessPath p, IRDocsifier d,
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
        "", [](tir::DeclBuffer stmt, AccessPath p, IRDocsifier d) -> Doc {
          return DeclBufferDoc(stmt, p, d, BufferVarDefinition::None);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::IfThenElse>(  //
        "", [](tir::IfThenElse stmt, AccessPath p, IRDocsifier d) -> Doc {
          ExprDoc cond = d->AsDoc<ExprDoc>(stmt->condition, p->Attr("condition"));
          ffi::Array<StmtDoc> then_branch;
          ffi::Array<StmtDoc> else_branch;
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
    .set_dispatch<tir::SeqStmt>("", [](tir::SeqStmt stmt, AccessPath p, IRDocsifier d) -> Doc {
      With<TIRFrame> f(d, stmt);
      AsDocBody(stmt, p, f->get(), d);
      return StmtBlockDoc((*f)->stmts);
    });

void InsertEnvThread(const tir::IterVar& iter_var, const AccessPath& iter_var_p,
                     const IRDocsifier& d) {
  Frame f = FindLowestVarDef(iter_var->var, d).value();
  DefineVar(iter_var->var, f, d);
  ExprDoc rhs = TIR(d, "env_thread")
                    ->Call({LiteralDoc::Str(iter_var->thread_tag,  //
                                            iter_var_p->Attr("thread_tag"))});
  ExprDoc lhs = d->AsDoc<ExprDoc>(iter_var->var, iter_var_p->Attr("var"));
  f->stmts.push_back(AssignDoc(lhs, rhs, std::nullopt));
}

ExprDoc DocsifyLaunchThread(const tir::AttrStmt& attr_stmt, const AccessPath& attr_stmt_p,
                            ffi::Optional<tir::Var>* define_var, const IRDocsifier& d) {
  tir::IterVar iter_var = Downcast<tir::IterVar>(attr_stmt->node);
  AccessPath iter_var_p = attr_stmt_p->Attr("node");

  ExprDoc var_doc{ffi::UnsafeInit()};
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
    .set_dispatch<tir::AttrStmt>(  //
        "", [](tir::AttrStmt stmt, AccessPath stmt_p, IRDocsifier d) -> Doc {
          bool concise = AllowConciseScoping(d, stmt);
          ffi::Optional<ExprDoc> lhs = std::nullopt;
          ffi::Optional<ExprDoc> rhs = std::nullopt;
          ffi::Optional<tir::Var> define_var = std::nullopt;
          tir::Stmt body = stmt->body;
          AccessPath body_p = stmt_p->Attr("body");
          if (stmt->attr_key == "thread_extent" || stmt->attr_key == "virtual_thread") {
            if (stmt->node.as<tir::IterVarNode>()) {
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
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::AllocBuffer>(  //
        "", [](tir::AllocBuffer stmt, AccessPath p, IRDocsifier d) -> Doc {
          bool concise = AllowConciseScoping(d, stmt);
          // Always print AllocBuffer as T.decl_buffer (without data= parameter).
          // When the parser sees T.decl_buffer without data, DeclBufferFrame
          // creates AllocBuffer on exit, ensuring a clean roundtrip.
          // Annotations are passed as an extra kwarg when present.
          ffi::Array<ExprDoc> extra_args;
          ffi::Array<ffi::String> extra_kwargs_keys;
          ffi::Array<ExprDoc> extra_kwargs_values;
          if (!stmt->annotations.empty()) {
            extra_kwargs_keys.push_back("annotations");
            extra_kwargs_values.push_back(
                d->AsDoc<ExprDoc>(stmt->annotations, p->Attr("annotations")));
          }
          ExprDoc rhs = BufferDecl(stmt->buffer, "decl_buffer", {}, p->Attr("buffer"),
                                   d->frames.back(), d, BufferVarDefinition::DataPointer);
          // Append annotations kwarg if present
          if (!extra_kwargs_keys.empty()) {
            // BufferDecl returns a CallDoc; we need to extend it with annotations kwarg.
            // Build a new call with the extra kwargs.
            auto call = Downcast<CallDoc>(rhs);
            ffi::Array<ffi::String> all_keys;
            ffi::Array<ExprDoc> all_values;
            for (size_t i = 0; i < call->kwargs_keys.size(); ++i) {
              all_keys.push_back(call->kwargs_keys[i]);
              all_values.push_back(call->kwargs_values[i]);
            }
            for (size_t i = 0; i < extra_kwargs_keys.size(); ++i) {
              all_keys.push_back(extra_kwargs_keys[i]);
              all_values.push_back(extra_kwargs_values[i]);
            }
            rhs = CallDoc(call->callee, call->args, all_keys, all_values);
          }
          With<TIRFrame> f(d, stmt);
          ExprDoc lhs = DefineBuffer(stmt->buffer, *f, d);
          AsDocBody(stmt->body, p->Attr("body"), f->get(), d);
          return DoConciseScoping(lhs, rhs, &(*f)->stmts, concise);
        });

TVM_SCRIPT_REPR(tir::AllocBufferNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::DeclBufferNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::SeqStmtNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::IfThenElseNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::EvaluateNode, ReprPrintTIR);
}  // namespace printer
}  // namespace script
}  // namespace tvm
