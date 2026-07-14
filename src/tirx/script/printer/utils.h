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

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/script/printer/ir_docsifier.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/index_map.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/predicate.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tirx_op.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../../script/printer/utils.h"

namespace tvm {
namespace script {
namespace printer {

using tvm::ffi::StructuralEqual;

/*! \brief A printer frame for TIR fragment */
class TIRFrameNode : public FrameNode {
 public:
  /*! \brief The TIR fragment the frame corresponds to */
  ffi::ObjectRef tirx;
  /*! \brief Whether or not the frame allows concise scoping */
  bool allow_concise_scoping{false};

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TIRFrameNode>()
        .def_ro("tirx", &TIRFrameNode::tirx)
        .def_ro("allow_concise_scoping", &TIRFrameNode::allow_concise_scoping);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.TIRFrame", TIRFrameNode, FrameNode);
};

/*! \brief Managed reference to TIRFrameNode */
class TIRFrame : public Frame {
 public:
  /*! \brief Constructor */
  explicit TIRFrame(const IRDocsifier& d, const ffi::ObjectRef& tirx) {
    ffi::ObjectPtr<TIRFrameNode> n = ffi::make_object<TIRFrameNode>();
    n->stmts.clear();
    n->d = d.get();
    n->tirx = tirx;
    data_ = std::move(n);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TIRFrame, Frame, TIRFrameNode);
};

/*!
 * \brief Defines a variable in the IRDocsifier at the given frame,
 * and returns the corresponding IdDoc
 * \param var The variable to define
 * \param d The IRDocsifier
 * \param frame The frame to define the variable in
 * \return The IdDoc corresponding to the variable
 */
inline ExprDoc DefineVar(const tirx::Var& var, const Frame& frame, const IRDocsifier& d) {
  if (ffi::Optional<ExprDoc> doc = d->GetVarDoc(var)) {
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
inline IdDoc DefineBuffer(const tirx::Buffer& buffer, const Frame& frame, const IRDocsifier& d) {
  return d->Define(buffer, frame, buffer->name.empty() ? "buffer" : buffer->name);
}

/*!
 * \brief Recursively process the body statements of a TIR fragment represented by a frame
 * \param stmt The body statement to process
 * \param p The object path
 * \param f The frame
 * \param d The IRDocsifier
 */
inline void AsDocBody(const tirx::Stmt& stmt, AccessPath p, TIRFrameNode* f, const IRDocsifier& d) {
  if (const auto* seq_stmt = stmt.as<tirx::SeqStmtNode>()) {
    ffi::Array<tirx::Stmt> body = seq_stmt->seq;
    auto value_refs_buffer = [](const PrimExpr& value, const tirx::Buffer& buffer) {
      bool found = false;
      tirx::PostOrderVisit(value, [&](const ffi::ObjectRef& node) {
        if (const auto* load = node.as<tirx::BufferLoadNode>()) {
          if (load->buffer.same_as(buffer)) {
            found = true;
          }
        }
      });
      return found;
    };

    for (int i = 0, n = body.size(); i < n;) {
      int consumed = 1;
      AccessPath item_p = p->Attr("seq")->ArrayItem(i);
      Doc doc{ffi::UnsafeInit()};

      const auto* alloc = body[i].as<tirx::AllocBufferNode>();
      if (d->cfg->syntax_sugar && alloc != nullptr && alloc->buffer.IsScalar(true) && i + 1 < n) {
        const auto* store = body[i + 1].as<tirx::BufferStoreNode>();
        bool can_merge_init = store != nullptr && store->buffer.same_as(alloc->buffer) &&
                              !store->predicate.has_value() && store->indices.size() == 1 &&
                              tirx::is_zero(store->indices[0]) &&
                              !value_refs_buffer(store->value, alloc->buffer);
        if (can_merge_init) {
          Doc alloc_doc = d->AsDoc(body[i], item_p);
          if (const auto* assign = alloc_doc.as<AssignDocNode>()) {
            if (assign->annotation.has_value() && !assign->rhs.has_value()) {
              ExprDoc init_rhs =
                  d->AsDoc<ExprDoc>(store->value, p->Attr("seq")->ArrayItem(i + 1)->Attr("value"));
              auto fused = AssignDoc(assign->lhs, init_rhs, assign->annotation);
              // Preserve comments that obj_to_annotate attached to either the
              // AllocBuffer (alloc_doc) or the BufferStore source, since the
              // user only sees the single fused line.
              ffi::Optional<ffi::String> merged_comment = assign->comment;
              if (d->cfg->obj_to_annotate.count(body[i + 1])) {
                ffi::String store_comment = d->cfg->obj_to_annotate.at(body[i + 1]);
                merged_comment = merged_comment.has_value()
                                     ? merged_comment.value() + "\n" + store_comment
                                     : store_comment;
              }
              fused->comment = merged_comment;
              doc = fused;
              consumed = 2;
            } else {
              doc = alloc_doc;
            }
          } else {
            doc = alloc_doc;
          }
        } else {
          doc = d->AsDoc(body[i], item_p);
        }
      } else {
        doc = d->AsDoc(body[i], item_p);
      }

      f->allow_concise_scoping = (i + consumed >= n);
      doc->source_paths.push_back(p);
      if (const auto* block = doc.as<StmtBlockDocNode>()) {
        f->stmts.insert(f->stmts.end(), block->stmts.begin(), block->stmts.end());
      } else {
        f->stmts.push_back(doc.as_or_throw<StmtDoc>());
      }
      i += consumed;
    }
  } else {
    f->allow_concise_scoping = true;
    Doc doc = d->AsDoc(stmt, p);
    if (const auto* block = doc.as<StmtBlockDocNode>()) {
      f->stmts.insert(f->stmts.end(), block->stmts.begin(), block->stmts.end());
    } else {
      f->stmts.push_back(doc.as_or_throw<StmtDoc>());
    }
  }
}

inline ffi::String ScopeIdApiName(const tirx::ScopeBinding& binding) {
  auto [parent, cur] = tirx::ScopeBindingToStringPair(binding);
  if (parent == "kernel" && cur == "cluster") {
    return "cluster_id";
  } else if (parent == "kernel" && cur == "cta") {
    return "cta_id";
  } else if (parent == "cluster" && cur == "cta") {
    return "cta_id_in_cluster";
  } else if (parent == "cluster" && cur == "cta_pair") {
    return "cta_id_in_pair";
  } else if (parent == "cta" && cur == "warpgroup") {
    return "warpgroup_id";
  } else if (parent == "cta" && cur == "warp") {
    return "warp_id";
  } else if (parent == "warpgroup" && cur == "warp") {
    return "warp_id_in_wg";
  } else if (parent == "warp" && cur == "thread") {
    return "lane_id";
  } else if (parent == "cta" && cur == "thread") {
    return "thread_id";
  } else if (parent == "warpgroup" && cur == "thread") {
    return "thread_id_in_wg";
  }
  LOG(FATAL) << "Unknown scope id binding: parent=" << parent << " cur=" << cur;
  return "";
}

/*!
 * \brief Find the top frame in the stack that could place a var definition
 * \param var The var to be defined
 * \param d The IRDocsifier
 * \return The frame that could place the var definition
 */
inline ffi::Optional<Frame> FindLowestVarDef(const ffi::ObjectRef& var, const IRDocsifier& d) {
  if (!d->common_prefix.count(var.get())) {
    return std::nullopt;
  }
  int n_frames = d->frames.size();
  std::unordered_map<const ffi::Object*, const FrameNode*> tir_to_frame;
  const FrameNode* fallback_frame = nullptr;
  tir_to_frame.reserve(n_frames);
  for (int i = n_frames - 1; i >= 0; --i) {
    if (const auto* f = d->frames[i].as<TIRFrameNode>()) {
      if (f->tirx.defined()) {
        tir_to_frame[f->tirx.get()] = f;
      } else if (fallback_frame == nullptr) {
        fallback_frame = f;
      }
    }
  }
  const std::vector<const ffi::Object*>& path = d->common_prefix.at(var.get());
  for (auto it = path.rbegin(); it != path.rend(); ++it) {
    if (tir_to_frame.count(*it)) {
      return ffi::GetRef<Frame>(tir_to_frame.at(*it));
    }
  }
  if (fallback_frame != nullptr) {
    return ffi::GetRef<Frame>(fallback_frame);
  }
  return std::nullopt;
}

/*! \brief Redirected method for the ffi repr hook */
inline std::string ReprPrintTIR(const ffi::ObjectRef& obj, const PrinterConfig& cfg) {
  IRDocsifier d(cfg);
  d->SetCommonPrefix(obj, [](const ffi::ObjectRef& obj) {
    return obj->IsInstance<tirx::VarNode>() || obj->IsInstance<tirx::BufferNode>();
  });
  With<TIRFrame> f(d, ffi::ObjectRef{nullptr});
  (*f)->AddDispatchToken(d, "tirx");
  return Docsify(obj, d, *f, cfg);
}

Doc PrintTIRCall(Call call, AccessPath call_p, IRDocsifier d);

/* \brief Specify which variables are defined along with the buffer
 *
 * Depending on the context, defining a buffer may define additional
 * variables associated with the buffer.
 */
enum class BufferVarDefinition {
  // All parameters in the buffer must be defined prior to this call.
  // For example, DeclBuffer.
  None,

  // The data pointer is defined along with the buffer, but buffer
  // parameters (shape/stride/elem_offset) must be defined prior to
  // use.  For example, `BlockNode::alloc_buffers`, or the
  // syntax-sugar representation of an `AllocBuffer`.
  DataPointer,

  // The data pointer is defined along with the buffer, along with any
  // buffer parameters (shape/stride/elem_offset) that have not
  // previously been defined.  For example,
  // `BlockNode::match_buffers`, or the `PrimFuncNode::buffer_map`.
  MatchBuffer,
};

/*!
 * \brief Declare and define a buffer
 * \param buffer The buffer to be defined
 * \param method The method used to declare the buffer
 * \param args The extra arguments used to declare the buffer
 * \param p The object path
 * \param f The frame
 * \param d The IRDocsifier
 * \param var_definitions Which variables are implicitly defined with
 *     the buffer.
 * \return The ExprDoc corresponding to the buffer declaration
 */
ExprDoc BufferDecl(const tirx::Buffer& buffer, const ffi::String& method,
                   const ffi::Array<ExprDoc>& args, const AccessPath& p, const Frame& frame,
                   const IRDocsifier& d, BufferVarDefinition var_definitions);

/*!
 * \brief Declare and define a buffer as annotation
 * \param buffer The buffer to be defined
 * \param p The object path
 * \param f The frame
 * \param d The IRDocsifier
 * \return The ExprDoc corresponding to the buffer declaration
 */
ExprDoc BufferAttn(const tirx::Buffer& buffer, const AccessPath& p, const Frame& frame,
                   const IRDocsifier& d);

/*!
 * \brief Print the creation of a Var
 * \param var The Var to be printed
 * \param var_p The object path of the Var
 * \param d The IRDocsifier
 * \return The ExprDoc corresponding to the Var creation
 */
ExprDoc PrintVarCreation(const tirx::Var& var, const AccessPath& var_p, const IRDocsifier& d);

/*! \brief A Var occurrence counter visitor */
class OccurrenceCounter : public tirx::StmtExprVisitor {
 public:
  /*! \brief The occurrence counter */
  int count = 0;
  /*! \brief The Var to count occurrence */
  const tirx::VarNode* v = nullptr;

  void VisitVar(const tirx::Var& var) { VisitExpr(static_cast<const Expr&>(var)); }

  void VisitExpr_(const tirx::VarNode* op) final {
    if (op == v) {
      ++count;
    }
    tirx::StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const tirx::BufferStoreNode* op) final {
    VisitBuffer(op->buffer.get());
    tirx::StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const tirx::BufferLoadNode* op) final {
    VisitBuffer(op->buffer.get());
    tirx::StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const tirx::AllocBufferNode* op) final {
    VisitBuffer(op->buffer.get());
    tirx::StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const tirx::DeclBufferNode* op) final {
    VisitBuffer(op->buffer.get());
    tirx::StmtExprVisitor::VisitStmt_(op);
  }

  void VisitBuffer(const tirx::BufferNode* buffer) {
    VisitExpr(buffer->data);
    for (const PrimExpr& shape_i : buffer->shape) {
      VisitExpr(shape_i);
    }
    for (const PrimExpr& stride_i : buffer->strides) {
      VisitExpr(stride_i);
    }
    VisitExpr(buffer->elem_offset);
  }

  explicit OccurrenceCounter(const tirx::VarNode* var) { v = var; }
};

#ifndef TVM_SCRIPT_REPR
#define TVM_SCRIPT_REPR(ObjectType, Method) TVM_REGISTER_SCRIPT_AS_REPR(ObjectType, Method)
#endif

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_TIRX_SCRIPT_PRINTER_UTILS_H_
