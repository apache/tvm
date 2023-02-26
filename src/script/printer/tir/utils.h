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
#ifndef TVM_SCRIPT_PRINTER_TIR_UTILS_H_
#define TVM_SCRIPT_PRINTER_TIR_UTILS_H_

#include <tvm/script/printer/ir_docsifier.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../utils.h"

namespace tvm {
namespace script {
namespace printer {

/*! \brief A printer frame for TIR fragment */
class TIRFrameNode : public FrameNode {
 public:
  /*! \brief The TIR fragment the frame corresponds to */
  ObjectRef tir;
  /*! \brief Whether or not the frame allows concise scoping */
  bool allow_concise_scoping{false};

  void VisitAttrs(AttrVisitor* v) {
    FrameNode::VisitAttrs(v);
    v->Visit("tir", &tir);
    v->Visit("allow_concise_scoping", &allow_concise_scoping);
  }

  static constexpr const char* _type_key = "script.printer.TIRFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(TIRFrameNode, FrameNode);
};

/*! \brief Managed reference to TIRFrameNode */
class TIRFrame : public Frame {
 public:
  /*! \brief Constructor */
  explicit TIRFrame(const IRDocsifier& d, const ObjectRef& tir) {
    ObjectPtr<TIRFrameNode> n = make_object<TIRFrameNode>();
    n->stmts.clear();
    n->d = d.get();
    n->tir = tir;
    data_ = std::move(n);
  }

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TIRFrame, Frame, TIRFrameNode);
};

/*!
 * \brief Defines a variable in the IRDocsifier at the given frame,
 * and returns the corresponding IdDoc
 * \param var The variable to define
 * \param d The IRDocsifier
 * \param frame The frame to define the variable in
 * \return The IdDoc corresponding to the variable
 */
inline ExprDoc DefineVar(const tir::Var& var, const Frame& frame, const IRDocsifier& d) {
  if (Optional<ExprDoc> doc = d->GetVarDoc(var)) {
    return doc.value();
  }
  return d->Define(var, frame, var->name_hint.empty() ? "v" : var->name_hint);
}

/*!
 * \brief Defines a buffer in the IRDocsifier at the given frame,
 * and returns the corresponding IdDoc
 * \param buffer The buffer to define
 * \param frame The frame to define the buffer in
 * \param d The IRDocsifier
 * \return The IdDoc corresponding to the buffer
 */
inline IdDoc DefineBuffer(const tir::Buffer& buffer, const Frame& frame, const IRDocsifier& d) {
  return d->Define(buffer, frame, buffer->name.empty() ? "buffer" : buffer->name);
}

/*!
 * \brief Recursively process the body statements of a TIR fragment represented by a frame
 * \param stmt The body statement to process
 * \param p The object path
 * \param f The frame
 * \param d The IRDocsifier
 */
inline void AsDocBody(const tir::Stmt& stmt, ObjectPath p, TIRFrameNode* f, const IRDocsifier& d) {
  if (const auto* seq_stmt = stmt.as<tir::SeqStmtNode>()) {
    Array<tir::Stmt> body = seq_stmt->seq;
    for (int i = 0, n = body.size(); i < n; ++i) {
      f->allow_concise_scoping = (i == n - 1);
      Doc doc = d->AsDoc(body[i], p->Attr("seq")->ArrayIndex(i));
      doc->source_paths.push_back(p);
      if (const auto* block = doc.as<StmtBlockDocNode>()) {
        f->stmts.insert(f->stmts.end(), block->stmts.begin(), block->stmts.end());
      } else {
        f->stmts.push_back(Downcast<StmtDoc>(doc));
      }
    }
  } else {
    f->allow_concise_scoping = true;
    Doc doc = d->AsDoc(stmt, p);
    if (const auto* block = doc.as<StmtBlockDocNode>()) {
      f->stmts.insert(f->stmts.end(), block->stmts.begin(), block->stmts.end());
    } else {
      f->stmts.push_back(Downcast<StmtDoc>(doc));
    }
  }
}

/*!
 * \brief Find the top frame in the stack that could place a var definition
 * \param var The var to be defined
 * \param d The IRDocsifier
 * \return The frame that could place the var definition
 */
inline Optional<Frame> FindLowestVarDef(const ObjectRef& var, const IRDocsifier& d) {
  if (!d->common_prefix.count(var.get())) {
    return NullOpt;
  }
  int n_frames = d->frames.size();
  std::unordered_map<const Object*, const FrameNode*> tir_to_frame;
  const FrameNode* fallback_frame = nullptr;
  tir_to_frame.reserve(n_frames);
  for (int i = n_frames - 1; i >= 0; --i) {
    if (const auto* f = d->frames[i].as<TIRFrameNode>()) {
      if (f->tir.defined()) {
        tir_to_frame[f->tir.get()] = f;
      } else if (fallback_frame == nullptr) {
        fallback_frame = f;
      }
    }
  }
  const std::vector<const Object*>& path = d->common_prefix.at(var.get());
  for (auto it = path.rbegin(); it != path.rend(); ++it) {
    if (tir_to_frame.count(*it)) {
      return GetRef<Frame>(tir_to_frame.at(*it));
    }
  }
  if (fallback_frame != nullptr) {
    return GetRef<Frame>(fallback_frame);
  }
  return NullOpt;
}

/*! \brief Redirected method for the ReprPrinter */
inline std::string ReprPrintTIR(const ObjectRef& obj, const PrinterConfig& cfg) {
  IRDocsifier d(cfg);
  d->SetCommonPrefix(obj, [](const ObjectRef& obj) {
    return obj->IsInstance<tir::VarNode>() || obj->IsInstance<tir::BufferNode>();
  });
  With<TIRFrame> f(d, ObjectRef{nullptr});
  (*f)->AddDispatchToken(d, "tir");
  return Docsify(obj, d, *f, cfg);
}

/*!
 * \brief Declare and define a buffer
 * \param buffer The buffer to be defined
 * \param method The method used to declare the buffer
 * \param args The extra arguments used to declare the buffer
 * \param p The object path
 * \param f The frame
 * \param d The IRDocsifier
 * \return The ExprDoc corresponding to the buffer declaration
 */
ExprDoc BufferDecl(const tir::Buffer& buffer, const String& method, const Array<ExprDoc>& args,
                   const ObjectPath& p, const Frame& frame, const IRDocsifier& d);

/*!
 * \brief Declare and define a buffer as annotation
 * \param buffer The buffer to be defined
 * \param p The object path
 * \param f The frame
 * \param d The IRDocsifier
 * \return The ExprDoc corresponding to the buffer declaration
 */
ExprDoc BufferAttn(const tir::Buffer& buffer, const ObjectPath& p, const Frame& frame,
                   const IRDocsifier& d);

/*! \brief A Var occurrence counter visitor */
class OccurrenceCounter : public tir::StmtExprVisitor {
 public:
  /*! \brief The occurrence counter */
  int count = 0;
  /*! \brief The Var to count occurrence */
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

  explicit OccurrenceCounter(const tir::VarNode* var) { v = var; }
};

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_TIR_UTILS_H_
