
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
