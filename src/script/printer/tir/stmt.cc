
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

#include <tvm/script/printer/doc.h>
#include <tvm/script/printer/ir_docsifier.h>
#include <tvm/script/printer/traced_object.h>
#include <tvm/script/printer/traced_object_functor.h>
#include <tvm/tir/stmt.h>

#include "../utils.h"
#include "./buffer.h"
#include "./tir.h"

namespace tvm {
namespace script {
namespace printer {

/*
 * \brief Helper to print stmt in the concise scoping form.
 *
 * For example, the allocate statment in TIR can be written as
 * \code
 * ...
 * with T.allocate([16], "float32", "global") as buf:
 *     buf[0] = 0.0  # inside the allocate
 * T.evaluate(T.call_extern(...))  # outside the allocate
 * \endcode
 * This representation is ambiguilty-free, but it adds one extra indent to
 * the code, which reduces readability if multiple statements are nested together.
 *
 * If the allocate statement is the last statement in its parent, it can be
 * written in the concise scoping form, avoiding adding extra level of indent.
 * \code
 * ...
 * buf = T.allocate([16], "float32", "global")
 * buf[0] = 0.0
 * ...
 * \endcode
 *
 * This builder class helps print stmt in the concise scoping form. The attributes
 * of this builder map to the output as,
 * \code
 * # Normal form
 * with <parent_expr> as <target>:
 *     <body>
 *
 * # Concise form
 * <target> = <parent_expr>
 * <body>
 *
 * # Concise form if the `concise_stmt_override` is defined
 * <concise_stmt_override>
 * <body>
 *
 * \endcode
 *
 */
class ConciseScopedStmtBuilder {
 public:
  Optional<ExprDoc> target{NullOpt};
  ExprDoc parent_expr{nullptr};
  Array<StmtDoc> body;
  Optional<StmtDoc> concise_stmt_override{NullOpt};

  ConciseScopedStmtBuilder() {}

  using TSelf = ConciseScopedStmtBuilder;

  TSelf& WithBody(Array<StmtDoc> body) {
    this->body = body;
    return *this;
  }

  TSelf& WithConciseFormStmt(StmtDoc stmt) {
    this->concise_stmt_override = stmt;
    return *this;
  }

  TSelf& WithTarget(ExprDoc target) {
    this->target = target;
    return *this;
  }

  TSelf& WithParentExpr(ExprDoc expr) {
    this->parent_expr = expr;
    return *this;
  }

  StmtBlockDoc ToDoc(const IRDocsifier& p) { return ToDoc(p->GetFrame<TIRFrame>().value()); }

  StmtBlockDoc ToDoc(const TIRFrame& frame) {
    ICHECK(parent_expr.defined());
    if (frame->allow_concise_scoping) {
      StmtDoc first_doc = ExprStmtDoc(parent_expr);
      if (concise_stmt_override) {
        first_doc = concise_stmt_override.value();
      } else if (target.defined()) {
        first_doc = AssignDoc(target.value(), parent_expr, NullOpt);
      }

      return StmtBlockDoc(runtime::Concat({first_doc}, body));
    } else {
      return StmtBlockDoc({ScopeDoc(target, parent_expr, body)});
    }
  }
};

std::vector<TracedObject<tir::Stmt>> FlattenSeqStmt(const TracedObject<tir::Stmt>& stmt) {
  std::vector<TracedObject<tir::Stmt>> result;

  if (stmt.IsInstance<tir::SeqStmt>()) {
    auto seq = stmt.Downcast<tir::SeqStmt>().GetAttr(&tir::SeqStmtNode::seq);
    for (const TracedObject<tir::Stmt>& child : seq) {
      std::vector<TracedObject<tir::Stmt>> flattened_child = FlattenSeqStmt(child);
      result.insert(result.end(), flattened_child.begin(), flattened_child.end());
    }
  } else {
    result.push_back(stmt);
  }

  return result;
}

Array<StmtDoc> FlattenStmtDoc(const Doc& doc) {
  if (const auto* stmt_block = doc.as<StmtBlockDocNode>()) {
    return stmt_block->stmts;
  } else if (const auto* stmt_doc = doc.as<StmtDocNode>()) {
    return {GetRef<StmtDoc>(stmt_doc)};
  } else {
    LOG(FATAL) << "Expect to get StmtBlockDoc or StmtDoc, got " << doc->GetTypeKey();
    throw;
  }
}

Array<StmtDoc> AsStmtDocArray(const TracedObject<tir::Stmt>& obj, IRDocsifier p) {
  Array<StmtDoc> result;
  std::vector<TracedObject<tir::Stmt>> flattened_stmts = FlattenSeqStmt(obj);

  const auto* frame_node = p->frames.back().as<TIRFrameNode>();
  ICHECK_NOTNULL(frame_node);

  size_t stmt_count = flattened_stmts.size();

  const bool original_concise_scoping_status = frame_node->allow_concise_scoping;
  frame_node->allow_concise_scoping = false;
  for (size_t i = 0; i < stmt_count; i++) {
    if (i == stmt_count - 1) {
      frame_node->allow_concise_scoping = true;
    }
    result = runtime::Concat(result, FlattenStmtDoc(p->AsDoc<Doc>(flattened_stmts[i])));
  }
  frame_node->allow_concise_scoping = original_concise_scoping_status;

  return result;
}

static TracedOptional<tir::Buffer> GetUsedBuffer(const TracedObject<ObjectRef>& stmt_or_expr) {
  if (auto load = stmt_or_expr.TryDowncast<tir::BufferLoad>()) {
    return load.value().GetAttr(&tir::BufferLoadNode::buffer);
  } else if (auto store = stmt_or_expr.TryDowncast<tir::BufferStore>()) {
    return store.value().GetAttr(&tir::BufferStoreNode::buffer);
  } else {
    return TracedOptional<tir::Buffer>(NullOpt, ObjectPath::Root());
  }
}

std::vector<TracedObject<tir::Buffer>> FindBufferVarUsage(tir::Var buffer_var,
                                                          TracedObject<tir::Stmt> body) {
  std::vector<TracedObject<tir::Buffer>> ret;
  PostOrderVisitStmtExprTraced(
      body, [&ret, buffer_var](const TracedObject<ObjectRef>& stmt_or_expr) {
        if (auto buffer_opt = GetUsedBuffer(stmt_or_expr)) {
          auto buffer = buffer_opt.value();
          if (buffer.Get()->data.same_as(buffer_var) &&
              std::find_if(ret.begin(), ret.end(),
                           [&](const auto& b) { return b.Get() == buffer.Get(); }) == ret.end()) {
            ret.push_back(buffer);
          }
        }
      });
  return ret;
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::SeqStmt>([](TracedObject<tir::SeqStmt> stmt, IRDocsifier p) -> Doc {
      if (!p->frames.back()->IsInstance<TIRTopLevelFrameNode>()) {
        // Throw error
        LOG(FATAL) << "tir::SeqStmt can only be printed when it's the top level statement. "
                      "Use AsStmtDocArray to print the body of statement";
        throw;
      }
      return StmtBlockDoc(AsStmtDocArray(stmt, p));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::AssertStmt>([](TracedObject<tir::AssertStmt> stmt, IRDocsifier p) {
      ExprDoc condition_expr = p->AsExprDoc(stmt.GetAttr(&tir::AssertStmtNode::condition));
      ExprDoc message_expr = p->AsExprDoc(stmt.GetAttr(&tir::AssertStmtNode::message));
      Array<StmtDoc> body = AsStmtDocArray(stmt.GetAttr(&tir::AssertStmtNode::body), p);

      return ConciseScopedStmtBuilder()
          .WithParentExpr(TIR(p)->Attr("Assert")->Call({condition_expr, message_expr}))
          .WithConciseFormStmt(AssertDoc(condition_expr, message_expr))
          .WithBody(body)
          .ToDoc(p);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BufferStore>([](TracedObject<tir::BufferStore> stmt, IRDocsifier p) {
      Array<ExprDoc> indices = AsExprDocArray(stmt.GetAttr(&tir::BufferStoreNode::indices), p);
      Array<Doc> index_docs(indices.begin(), indices.end());
      return AssignDoc(p->AsExprDoc(stmt.GetAttr(&tir::BufferStoreNode::buffer))[index_docs],
                       p->AsExprDoc(stmt.GetAttr(&tir::BufferStoreNode::value)), NullOpt);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::IfThenElse>([](TracedObject<tir::IfThenElse> stmt, IRDocsifier p) {
      ExprDoc predicate = p->AsExprDoc(stmt.GetAttr(&tir::IfThenElseNode::condition));
      Array<StmtDoc> then_branch = AsStmtDocArray(stmt.GetAttr(&tir::IfThenElseNode::then_case), p);
      Array<StmtDoc> else_branch = AsStmtDocArray(stmt.GetAttr(&tir::IfThenElseNode::else_case), p);
      return IfDoc(predicate, then_branch, else_branch);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::While>([](TracedObject<tir::While> stmt, IRDocsifier p) {
      return WhileDoc(p->AsExprDoc(stmt.GetAttr(&tir::WhileNode::condition)),
                      AsStmtDocArray(stmt.GetAttr(&tir::WhileNode::body), p));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Prefetch>([](TracedObject<tir::Prefetch> stmt, IRDocsifier p) {
      auto buffer = stmt.GetAttr(&tir::PrefetchNode::buffer);
      auto bounds = stmt.GetAttr(&tir::PrefetchNode::bounds);
      return ExprStmtDoc(
          TIR(p)->Attr("prefetch")->Call({p->AsExprDoc(buffer)[AsDocArray<Doc>(bounds, p)]}));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::LetStmt>([](TracedObject<tir::LetStmt> stmt, IRDocsifier p) {
      TIRFrame previous_frame = p->GetFrame<TIRFrame>().value();
      TIRGeneralFrame let_frame;
      WithCtx ctx = p->WithFrame(let_frame);

      auto var = stmt.GetAttr(&tir::LetStmtNode::var);
      bool is_var_defined_previously = p->vars->IsVarDefined(var.Get());
      ExprDoc var_doc{nullptr};
      if (is_var_defined_previously) {
        var_doc = p->vars->GetVarDoc(var).value();
      } else {
        var_doc = DefineTIRVar(var, let_frame, p);
      }

      auto value_doc = p->AsExprDoc(stmt.GetAttr(&tir::LetStmtNode::value));
      auto dtype = var.GetAttr(&tir::VarNode::dtype);
      auto type_annotation_doc = GetTypeAnnotationDocForVar(var, p);

      TracedObject<tir::Stmt> body_stmt = stmt.GetAttr(&tir::LetStmtNode::body);
      Array<StmtDoc> body_doc;

      // Print definition of buffers that aliases the variable of this Let stmt.
      std::vector<TracedObject<tir::Buffer>> aliasing_buffers =
          FindBufferVarUsage(var.Get(), body_stmt);
      std::vector<TracedObject<tir::Buffer>> buffers_to_define;
      for (const TracedObject<tir::Buffer>& buffer : aliasing_buffers) {
        if (!p->vars->IsVarDefined(buffer.Get())) {
          buffers_to_define.push_back(buffer);
        }
      }
      DefineBuffers(buffers_to_define, let_frame, p, TIR(p)->Attr("decl_buffer"),
                    [&body_doc](IdDoc buf_identifier, ExprDoc buf_definition) {
                      body_doc.push_back(AssignDoc(buf_identifier, buf_definition, NullOpt));
                    });

      body_doc = runtime::Concat(body_doc, AsStmtDocArray(body_stmt, p));

      if (previous_frame->allow_concise_scoping) {
        // dtype won't be linked to a doc object if it does concise scoping
        // here we manually link it to type annotation
        type_annotation_doc->source_paths.push_back(dtype.GetPath());
        AssignDoc var_assignment = AssignDoc(var_doc, value_doc, type_annotation_doc);
        return StmtBlockDoc(runtime::Concat({var_assignment}, body_doc));
      } else {
        Array<StmtDoc> result;
        if (!is_var_defined_previously) {
          result.push_back(AssignDoc(var_doc, TIR(p)->Attr("var")->Call({DType2Literal(dtype)}),
                                     type_annotation_doc));
        }
        ExprDoc let_call = TIR(p)->Attr("let")->Call({var_doc, value_doc});
        result.push_back(ScopeDoc(NullOpt, let_call, body_doc));
        return StmtBlockDoc(result);
      }
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Evaluate>([](TracedObject<tir::Evaluate> stmt, IRDocsifier p) {
      return ExprStmtDoc(p->AsExprDoc(stmt.GetAttr(&tir::EvaluateNode::value)));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Store>([](TracedObject<tir::Store> stmt, IRDocsifier p) -> Doc {
      LOG(FATAL) << "tir::Store cannot be printed. Store is replaced by BufferStore.";
      throw;
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BufferRealize>([](TracedObject<tir::BufferRealize> stmt,
                                         IRDocsifier p) -> Doc {
      LOG(FATAL)
          << "tir::BufferRealize cannot be printed. All the BufferRealize should be nested inside "
             "with AttrStmt.";
      throw;
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::ProducerStore>([](TracedObject<tir::ProducerStore> stmt,
                                         IRDocsifier p) -> Doc {
      LOG(FATAL) << "tir::ProducerStore cannot be printed";
      throw;
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::ProducerRealize>([](TracedObject<tir::ProducerRealize> stmt,
                                           IRDocsifier p) -> Doc {
      LOG(FATAL) << "tir::ProducerRealize cannot be printed";
      throw;
    });

}  // namespace printer
}  // namespace script
}  // namespace tvm
