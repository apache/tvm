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
#include <algorithm>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

Doc DoConciseScoping(const ffi::Optional<ExprDoc>& lhs, const ExprDoc& rhs,
                     ffi::Array<StmtDoc>* stmts, bool concise_scoping) {
  if (concise_scoping) {
    if (lhs.has_value()) {
      stmts->insert(stmts->begin(), AssignDoc(lhs.value(), rhs, std::nullopt));
    } else {
      stmts->insert(stmts->begin(), ExprStmtDoc(rhs));
    }
    return StmtBlockDoc(*stmts);
  } else {
    return ScopeDoc(lhs, rhs, *stmts);
  }
}

bool AllowConciseScoping(const IRDocsifier& d, const ffi::ObjectRef& obj) {
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
  TVM_FFI_UNREACHABLE();
}

bool IsAncestorOfAllVarUse(const tirx::Stmt& node, const ffi::ObjectRef& var,
                           const IRDocsifier& d) {
  if (!d->common_prefix.count(var.get())) {
    return false;
  }
  const std::vector<const ffi::Object*>& path = d->common_prefix.at(var.get());
  for (auto it = path.rbegin(); it != path.rend(); ++it) {
    if (*it == node.get()) {
      return true;
    }
  }
  return false;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<tirx::TilePrimitiveCall>(
      "", [](tirx::TilePrimitiveCall op_call, AccessPath p, IRDocsifier d) -> Doc {
        static const OpAttrMap<tirx::TScriptPrinterName>& op_names =
            Op::GetAttrMap<tirx::TScriptPrinterName>("TScriptPrinterName");
        auto op = op_call->op;
        if (op_names.count(op) == 0) {
          LOG(WARNING) << "No TScriptPrinterName attribute for " << op->name;
        }

        static const auto& category_map = Op::GetAttrMap<tirx::TIRxOpCategory>("TIRxOpCategory");
        bool is_tile_primitive = category_map.get(op, ffi::String("")) == "tile_primitive";
        TVM_FFI_ICHECK(is_tile_primitive)
            << "Only tile primitive ops can be used in tirx::TilePrimitiveCall";
        ffi::String name = op_names.get(op, op->name);
        // Per-call execution scope is printed as a namespace prefix on the op,
        // e.g. ``T.warp.copy(...)``. ``warpgroup`` prints as ``wg``. The
        // default ``thread`` scope prints through the explicit tile namespace,
        // e.g. ``T.tile.copy(...)``, so canonical script only needs the full
        // TIRx dialect import. ``Tx`` remains a handwritten shorthand for
        // ``T.tile`` and ``T.<scope>`` tile calls.
        auto scope_ns = [](tirx::ScopeKind k) -> ffi::Optional<ffi::String> {
          switch (k) {
            case tirx::ScopeKind::kWarp:
              return ffi::String("warp");
            case tirx::ScopeKind::kWarpgroup:
              return ffi::String("wg");
            case tirx::ScopeKind::kCta:
              return ffi::String("cta");
            case tirx::ScopeKind::kCluster:
              return ffi::String("cluster");
            default:  // kThread -> no prefix
              return std::nullopt;
          }
        };
        auto scoped_callee = [&](const ffi::String& op_name) -> ExprDoc {
          ffi::Optional<ffi::String> ns = scope_ns(op_call->scope->kind);
          if (ns.has_value()) {
            return TIR(d, ns.value())->Attr(op_name);
          }
          return TIR(d, "tile")->Attr(op_name);
        };
        // Trim trailing None args (e.g. optional bias=None, scale=None)
        size_t n_args = op_call->args.size();
        while (n_args > 0 &&
               op_call->args[n_args - 1].type_index() == ffi::TypeIndex::kTVMFFINone) {
          --n_args;
        }
        // Detect in-place unary ops: after trimming Nones, if exactly 2 args
        // and args[0]/args[1] refer to the same buffer region, collapse to 1 arg
        bool inplace_unary = false;
        if (n_args == 2) {
          auto dst_opt = op_call->args[0].as<tvm::TensorRegion>();
          auto src_opt = op_call->args[1].as<tvm::TensorRegion>();
          if (dst_opt.has_value() && src_opt.has_value() &&
              dst_opt.value()->source.same_as(src_opt.value()->source) &&
              StructuralEqual()(dst_opt.value()->region, src_opt.value()->region)) {
            inplace_unary = true;
          }
        }
        ffi::Array<Doc> args;
        for (size_t i = 0; i < n_args; ++i) {
          if (inplace_unary && i == 1) continue;  // skip duplicate src
          args.push_back(d->AsDoc<Doc>(op_call->args[i], p->Attr("args")->ArrayItem(i)));
        }
        ffi::Optional<ExprDoc> disp = std::nullopt;
        if (op_call->dispatch.has_value()) {
          disp = LiteralDoc::Str(op_call->dispatch.value(), p->Attr("dispatch"));
        }
        return OpCallDoc(scoped_callee(name), args,
                         d->AsDoc<DictDoc>(op_call->workspace, p->Attr("workspace")),
                         d->AsDoc<DictDoc>(op_call->config, p->Attr("config")), disp);
      });
}
TVM_FFI_STATIC_INIT_BLOCK() {
  TVMScriptPrinter::Register<tirx::TilePrimitiveCallNode>(ReprPrintTIR);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<tirx::Evaluate>(
      "", [](tirx::Evaluate eval, AccessPath p, IRDocsifier d) -> Doc {
        ExprDoc value = d->AsDoc<ExprDoc>(eval->value, p->Attr("value"));
        const auto* call = eval->value.as<CallNode>();
        if (call && !call->op.same_as(tirx::builtin::buffer_data())) {
          return ExprStmtDoc(value);
        }
        return ExprStmtDoc(TIR(d, "evaluate")->Call({value}));
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<tirx::Return>(
      "", [](tirx::Return stmt, AccessPath p, IRDocsifier d) -> Doc {
        ExprDoc value = d->AsDoc<ExprDoc>(stmt->value, p->Attr("value"));
        return ReturnDoc(value);
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<tirx::Bind>(
      "", [](tirx::Bind stmt, AccessPath p, IRDocsifier d) -> Doc {
        // Step 1. Type annotation
        TVM_FFI_ICHECK(!stmt->var->ty.IsMissing())
            << "Type annotation is required for variable: " << stmt->var->name;
        ffi::Optional<ExprDoc> type_doc;
        bool needs_annotation =
            stmt->var->ty.as<PrimTypeNode>() || stmt->var->ty.as<PointerTypeNode>() ||
            stmt->var->ty.as<TupleTypeNode>() || stmt->var->ty.as<StringTypeNode>() ||
            stmt->var->ty.as<FuncTypeNode>();
        if (needs_annotation) {
          type_doc = d->AsDoc<ExprDoc>(stmt->var->ty, p->Attr("var")->Attr("ty"));
          if (const auto* tuple_type = stmt->var->ty.as<TupleTypeNode>();
              tuple_type && tuple_type->fields.empty()) {
            type_doc = std::nullopt;
          }
        }
        // Step 2. RHS
        ExprDoc rhs = d->AsDoc<ExprDoc>(stmt->value, p->Attr("value"));
        // Step 3. LHS - Bind is flat, define var if new, otherwise just assign
        if (!d->IsVarDefined(stmt->var)) {
          TVM_FFI_ICHECK(!d->frames.empty());
          ExprDoc lhs = DefineVar(stmt->var, d->frames.back(), d);
          ffi::Optional<ExprDoc> let_ann;
          if (needs_annotation) {
            let_ann = type_doc.has_value() ? ExprDoc(IndexDoc(TIR(d, "let"), {type_doc.value()}))
                                           : TIR(d, "let");
          }
          return AssignDoc(lhs, rhs, let_ann);
        } else {
          ExprDoc lhs = d->GetVarDoc(stmt->var).value();
          lhs->source_paths.push_back(p->Attr("var"));
          return AssignDoc(lhs, rhs, std::nullopt);
        }
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<tirx::AssertStmt>(
      "", [](tirx::AssertStmt stmt, AccessPath p, IRDocsifier d) -> Doc {
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
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<tirx::While>(
      "", [](tirx::While stmt, AccessPath p, IRDocsifier d) -> Doc {
        ExprDoc cond = d->AsDoc<ExprDoc>(stmt->condition, p->Attr("condition"));
        With<TIRFrame> f(d, stmt);
        AsDocBody(stmt->body, p->Attr("body"), f->get(), d);
        return WhileDoc(cond, (*f)->stmts);
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<tirx::Break>(
      "", [](tirx::Break stmt, AccessPath p, IRDocsifier d) -> Doc { return BreakDoc(); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<tirx::Continue>(
      "", [](tirx::Continue stmt, AccessPath p, IRDocsifier d) -> Doc { return ContinueDoc(); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<tirx::IfThenElse>(  //
      "", [](tirx::IfThenElse stmt, AccessPath p, IRDocsifier d) -> Doc {
        ExprDoc cond = d->AsDoc<ExprDoc>(stmt->condition, p->Attr("condition"));
        ffi::Array<StmtDoc> then_branch;
        ffi::Array<StmtDoc> else_branch;
        if (stmt->then_case.defined()) {
          With<TIRFrame> f(d, stmt->then_case);
          AsDocBody(stmt->then_case, p->Attr("then_case"), f->get(), d);
          then_branch = (*f)->stmts;
        }
        if (stmt->else_case.has_value()) {
          With<TIRFrame> f(d, stmt->else_case.value());
          AsDocBody(stmt->else_case.value(), p->Attr("else_case"), f->get(), d);
          else_branch = (*f)->stmts;
        }
        return IfDoc(cond, then_branch, else_branch);
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<tirx::SeqStmt>(
      "", [](tirx::SeqStmt stmt, AccessPath p, IRDocsifier d) -> Doc {
        With<TIRFrame> f(d, stmt);
        AsDocBody(stmt, p, f->get(), d);
        return StmtBlockDoc((*f)->stmts);
      });
}

void InsertEnvThread(const tirx::IterVar& iter_var, const AccessPath& iter_var_p,
                     const IRDocsifier& d) {
  Frame f = FindLowestVarDef(iter_var->var, d).value();
  DefineVar(iter_var->var, f, d);
  ExprDoc rhs = TIR(d, "env_thread")
                    ->Call({LiteralDoc::Str(iter_var->thread_tag,  //
                                            iter_var_p->Attr("thread_tag"))});
  ExprDoc lhs = d->AsDoc<ExprDoc>(iter_var->var, iter_var_p->Attr("var"));
  f->stmts.push_back(AssignDoc(lhs, rhs, std::nullopt));
}

ExprDoc DocsifyLaunchThread(const tirx::AttrStmt& attr_stmt, const AccessPath& attr_stmt_p,
                            ffi::Optional<tirx::Var>* define_var, const IRDocsifier& d) {
  tirx::IterVar iter_var = attr_stmt->node.as_or_throw<tirx::IterVar>();
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

/*! \brief Check whether an AttrStmt has node=0 (the dict-attr pattern). */
static bool IsDictAttrPattern(const tirx::AttrStmt& stmt) {
  if (auto int_value = stmt->node.as<int64_t>()) {
    return int_value.value() == 0;
  }
  return false;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<tirx::AttrStmt>(  //
      "", [](tirx::AttrStmt stmt, AccessPath stmt_p, IRDocsifier d) -> Doc {
        bool concise = AllowConciseScoping(d, stmt);
        ffi::Optional<ExprDoc> lhs = std::nullopt;
        ffi::Optional<ExprDoc> rhs = std::nullopt;
        ffi::Optional<tirx::Var> define_var = std::nullopt;
        tirx::Stmt body = stmt->body;
        AccessPath body_p = stmt_p->Attr("body");
        if (stmt->attr_key == "thread_extent" ||
            stmt->attr_key == tvm::tirx::attr::virtual_thread) {
          if (stmt->node.as<tirx::IterVarNode>()) {
            rhs = DocsifyLaunchThread(stmt, stmt_p, &define_var, d);
          }
        }
        if (!rhs.has_value()) {
          // Try to collapse consecutive dict-attr-pattern AttrStmts into T.attr({...})
          if (IsDictAttrPattern(stmt)) {
            ffi::Array<ExprDoc> keys;
            ffi::Array<ExprDoc> values;
            tirx::AttrStmt cur = stmt;
            AccessPath cur_p = stmt_p;
            while (true) {
              keys.push_back(LiteralDoc::Str(cur->attr_key, cur_p->Attr("attr_key")));
              values.push_back(d->AsDoc<ExprDoc>(cur->value, cur_p->Attr("value")));
              if (auto next = cur->body.as<tirx::AttrStmt>()) {
                if (IsDictAttrPattern(next.value())) {
                  cur = next.value();
                  cur_p = cur_p->Attr("body");
                  continue;
                }
              }
              body = cur->body;
              body_p = cur_p->Attr("body");
              break;
            }
            rhs = TIR(d, "attr")->Call({DictDoc(keys, values)});
          } else {
            rhs = TIR(d, "attr")->Call({
                d->AsDoc<ExprDoc>(stmt->node, stmt_p->Attr("node")),
                LiteralDoc::Str(stmt->attr_key, stmt_p->Attr("attr_key")),
                d->AsDoc<ExprDoc>(stmt->value, stmt_p->Attr("value")),
            });
          }
        }
        With<TIRFrame> f(d, stmt);
        if (define_var.has_value()) {
          lhs = DefineVar(define_var.value(), *f, d);
        }
        AsDocBody(body, body_p, f->get(), d);
        return DoConciseScoping(lhs, rhs.value(), &(*f)->stmts, concise);
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  TVMScriptPrinter::Register<tirx::BindNode>(ReprPrintTIR);
  TVMScriptPrinter::Register<tirx::AttrStmtNode>(ReprPrintTIR);
  TVMScriptPrinter::Register<tirx::AssertStmtNode>(ReprPrintTIR);
  TVMScriptPrinter::Register<tirx::WhileNode>(ReprPrintTIR);
  TVMScriptPrinter::Register<tirx::ReturnNode>(ReprPrintTIR);
  TVMScriptPrinter::Register<tirx::BreakNode>(ReprPrintTIR);
  TVMScriptPrinter::Register<tirx::ContinueNode>(ReprPrintTIR);
  TVMScriptPrinter::Register<tirx::SeqStmtNode>(ReprPrintTIR);
  TVMScriptPrinter::Register<tirx::IfThenElseNode>(ReprPrintTIR);
  TVMScriptPrinter::Register<tirx::EvaluateNode>(ReprPrintTIR);
}
}  // namespace printer
}  // namespace script
}  // namespace tvm
