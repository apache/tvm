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

/*!
 * \file tirx/ir/tir_visitor_with_path.cc
 * \brief Provide a TIR visitor that tracks the current location
 */
#include "tir_visitor_with_path.h"

#include <tvm/ffi/reflection/access_path.h>

#include <algorithm>
#include <optional>
#include <unordered_set>
#include <variant>
#include <vector>

namespace tvm {
namespace tirx {
using AccessPath = ffi::reflection::AccessPath;

void TIRVisitorWithPath::Visit(const IRModule& mod, AccessPath path) {
  // To ensure deterministic order of visits, sort the GlobalVar first
  // by visibility (public then private), then alphabetically by name.
  std::vector<GlobalVar> gvars;
  std::unordered_set<GlobalVar> externally_exposed;
  for (const auto& [gvar, func] : mod->functions) {
    gvars.push_back(gvar);
    if (func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol).has_value()) {
      externally_exposed.insert(gvar);
    }
  }

  std::sort(gvars.begin(), gvars.end(),
            [&externally_exposed](const GlobalVar& a, const GlobalVar& b) {
              bool a_exposed = externally_exposed.count(a);
              bool b_exposed = externally_exposed.count(b);
              if (a_exposed != b_exposed) {
                return a_exposed > b_exposed;
              } else {
                return a->name_hint < b->name_hint;
              }
            });

  std::vector<DefContext<GlobalVar>> context;

  for (const auto& gvar : gvars) {
    context.push_back(WithDef(gvar, path->Attr("global_var_map_")->MapItem(gvar->name_hint)));
  }

  for (const auto& gvar : gvars) {
    auto base_func = mod->functions[gvar];
    if (auto prim_func = base_func.as<PrimFunc>()) {
      Visit(prim_func.value(), path->Attr("functions")->MapItem(gvar));
    }
  }

  while (context.size()) context.pop_back();
}

void TIRVisitorWithPath::Visit(const PrimFunc& func, AccessPath path) {
  // BufferType metadata may introduce symbolic dimensions.  Define those
  // symbols before entering the buffer parameter itself.
  std::vector<std::variant<DefContext<Var>, DefContext<BufferVar>>> context;

  auto ppath = path->Attr("params");
  for (size_t i = 0; i < func->params.size(); i++) {
    const Var& param = func->params[i];
    if (!param->ty.as<BufferTypeNode>()) {
      context.push_back(WithDef(param, ppath->ArrayItem(i)));
    }
  }

  for (size_t i = 0; i < func->params.size(); i++) {
    if (auto opt = func->params[i].as<BufferVar>()) {
      auto buf = opt.value();
      auto buf_path = ppath->ArrayItem(i)->Attr("ty");

      for (auto& def : WithMatchBufferDefs(buf, buf_path)) {
        context.push_back(std::move(def));
      }
    }
  }

  // Only after all the implicit definitions have been visited can we
  // visit the buffer definition itself.
  for (size_t i = 0; i < func->params.size(); i++) {
    if (auto opt = func->params[i].as<BufferVar>()) {
      context.push_back(WithDef(opt.value(), ppath->ArrayItem(i)));
    }
  }

  bind_scope_.WithNewScope([&]() { Visit(func->body, path->Attr("body")); });

  while (context.size()) context.pop_back();
}

void TIRVisitorWithPath::EnterDef(const IterVar& iter_var, AccessPath path) {
  if (iter_var->dom.defined()) {
    Visit(iter_var->dom, path->Attr("dom"));
  }
  EnterDef(iter_var->var, path->Attr("var"));
}

void TIRVisitorWithPath::ExitDef(const IterVar& iter_var, AccessPath path) {
  ExitDef(iter_var->var, path->Attr("var"));
}

void TIRVisitorWithPath::EnterDef(const BufferVar& buffer, AccessPath path) {
  // BufferVar is a checked view over an ordinary Var.  Its definition
  // therefore introduces both the variable identity and the buffer metadata.
  EnterDef(buffer.var(), path);
  // Defining a buffer counts as using all parameters in the buffer
  // (e.g. shape/strides).
  VisitBufferDef(buffer, path);
}
void TIRVisitorWithPath::ExitDef(const BufferVar& buffer, AccessPath path) {
  ExitDef(buffer.var(), path);
}

void TIRVisitorWithPath::VisitBufferDef(const BufferVar& buffer, AccessPath path) {
  Visit(buffer->shape, path->Attr("shape"));
  Visit(buffer->strides, path->Attr("strides"));
  Visit(buffer->elem_offset, path->Attr("elem_offset"));
  Visit(buffer->allocated_addr, path->Attr("allocated_addr"));
}

// Default: buffer use sites do not re-visit buffer fields. BufferVar fields
// (shape, strides, elem_offset) are visited at the definition site via
// VisitBufferDef/EnterDef. Re-visiting at use sites would require those
// variables to be in scope at every use, which may not hold when buffers
// are allocated in a different scope than where they are used.
void TIRVisitorWithPath::VisitBufferUse(const BufferVar& buffer, AccessPath path) {}

void TIRVisitorWithPath::Visit(const TensorRegion& region, AccessPath path) {
  if (auto buffer = region->source.as<BufferVar>()) {
    VisitBufferUse(buffer.value(), path->Attr("source"));
  } else {
    Visit(region->source, path->Attr("source"));
  }
  Visit(region->region, path->Attr("region"));
}

void TIRVisitorWithPath::Visit(const IterVar& iter_var, AccessPath path) {
  if (iter_var->dom.defined()) {
    Visit(iter_var->dom, path->Attr("dom"));
  }
  Visit(iter_var->var, path->Attr("var"));
}

void TIRVisitorWithPath::Visit(const Range& range, AccessPath path) {
  Visit(range->min, path->Attr("min"));
  Visit(range->extent, path->Attr("extent"));
}

void TIRVisitorWithPath::Dispatch_(const BindNode* op, AccessPath path) {
  Visit(op->value, path->Attr("value"));
  // Push the Bind's var definition into the current scope.
  // The def lives until the enclosing scope (body-carrying stmt) exits.
  bind_scope_.Current().push_back(WithDef(op->var, path->Attr("var")));
}

void TIRVisitorWithPath::Dispatch_(const AttrStmtNode* op, AccessPath path) {
  Visit(op->value, path->Attr("value"));

  std::vector<std::variant<DefContext<IterVar>, DefContext<Var>, DefContext<BufferVar>>> context;
  if (auto iter_var = op->node.as<IterVar>();
      iter_var && (op->attr_key == attr::thread_extent || op->attr_key == "virtual_thread")) {
    // Some attributes serve as a source of definition for the
    // tirx::Var they annotate.
    context.push_back(WithDef(iter_var.value(), path->Attr("node")));

  } else if (auto expr = op->node.as<PrimExpr>()) {
    Visit(expr.value(), path->Attr("node"));
  }
  bind_scope_.WithNewScope([&]() { Visit(op->body, path->Attr("body")); });

  while (context.size()) {
    context.pop_back();
  }
}

void TIRVisitorWithPath::Dispatch_(const ForNode* op, AccessPath path) {
  Visit(op->min, path->Attr("min"));
  Visit(op->extent, path->Attr("extent"));
  auto context = WithDef(op->loop_var, path->Attr("loop_var"));
  bind_scope_.WithNewScope([&]() { Visit(op->body, path->Attr("body")); });
}

void TIRVisitorWithPath::Dispatch_(const WhileNode* op, AccessPath path) {
  Visit(op->condition, path->Attr("condition"));
  bind_scope_.WithNewScope([&]() { Visit(op->body, path->Attr("body")); });
}

void TIRVisitorWithPath::Dispatch_(const ReturnNode* op, AccessPath path) {
  Visit(op->value, path->Attr("value"));
}

void TIRVisitorWithPath::Dispatch_(const BreakNode* op, AccessPath path) {}

void TIRVisitorWithPath::Dispatch_(const ContinueNode* op, AccessPath path) {}

void TIRVisitorWithPath::Dispatch_(const AllocBufferNode* op, AccessPath path) {
  // Push definitions into the current scope so they are visible to subsequent siblings.
  auto buf_path = path->Attr("buffer");
  bind_scope_.Current().push_back(WithDef(op->buffer, buf_path));
}

void TIRVisitorWithPath::Dispatch_(const DeclBufferNode* op, AccessPath path) {
  Visit(op->data, path->Attr("data"));
  // Push buffer definition into the current scope so it is visible to subsequent siblings.
  bind_scope_.Current().push_back(WithDef(op->buffer, path->Attr("buffer")));
}

void TIRVisitorWithPath::Dispatch_(const BufferStoreNode* op, AccessPath path) {
  Visit(op->value, path->Attr("value"));
  VisitBufferUse(op->buffer, path->Attr("buffer"));
  Visit(op->indices, path->Attr("indices"));
}

void TIRVisitorWithPath::Dispatch_(const IfThenElseNode* op, AccessPath path) {
  Visit(op->condition, path->Attr("condition"));
  bind_scope_.WithNewScope([&]() { Visit(op->then_case, path->Attr("then_case")); });
  bind_scope_.WithNewScope([&]() { Visit(op->else_case, path->Attr("else_case")); });
}

void TIRVisitorWithPath::Dispatch_(const AssertStmtNode* op, AccessPath path) {
  Visit(op->condition, path->Attr("condition"));
  Visit(op->error_kind, path->Attr("error_kind"));
  Visit(op->message_parts, path->Attr("message_parts"));
}

void TIRVisitorWithPath::Dispatch_(const SeqStmtNode* op, AccessPath path) {
  auto seq_path = path->Attr("seq");
  for (size_t i = 0; i < op->seq.size(); i++) {
    Visit(op->seq[i], seq_path->ArrayItem(i));
  }
}

void TIRVisitorWithPath::Dispatch_(const EvaluateNode* op, AccessPath path) {
  Visit(op->value, path->Attr("value"));
}

void TIRVisitorWithPath::VisitStmt_(const tirx::TilePrimitiveCallNode* op, AccessPath path) {
  for (size_t i = 0; i < op->args.size(); i++) {
    if (op->args[i] == nullptr) {
      continue;
    }
    if (auto buf_region = op->args[i].as<TensorRegion>()) {
      Visit(buf_region.value(), path->Attr("args")->ArrayItem(i));
    } else if (auto expr = op->args[i].as<PrimExpr>()) {
      Visit(expr.value(), path->Attr("args")->ArrayItem(i));
    } else if (auto stmt = op->args[i].as<Stmt>()) {
      Visit(stmt.value(), path->Attr("args")->ArrayItem(i));
    } else if (auto buf = op->args[i].as<BufferVar>()) {
      VisitBufferUse(buf.value(), path->Attr("args")->ArrayItem(i));
    }
  }
}

void TIRVisitorWithPath::Dispatch_(const ScopeIdDefStmtNode* op, AccessPath path) {
  // Flat stmt -- no body. Visit extents and preferred_extents (if present),
  // then push the bound Var(s) into the current scope so subsequent siblings
  // see them as defined.
  auto def_path = path->Attr("def");
  if (op->def->extents.has_value()) {
    Visit(op->def->extents.value(), def_path->Attr("extents"));
  }
  if (op->def->preferred_extents.has_value()) {
    Visit(op->def->preferred_extents.value(), def_path->Attr("preferred_extents"));
  }
  auto def_ids_path = def_path->Attr("def_ids");
  for (size_t i = 0; i < op->def->def_ids.size(); ++i) {
    bind_scope_.Current().push_back(
        WithDef(static_cast<Var>(op->def->def_ids[i]), def_ids_path->ArrayItem(i)));
  }
}

void TIRVisitorWithPath::Dispatch_(const VarNode* op, AccessPath path) {}

void TIRVisitorWithPath::Dispatch_(const TensorLoadNode* op, AccessPath path) {
  VisitBufferUse(op->source.as_or_throw<tvm::tirx::BufferVar>(), path->Attr("source"));
  Visit(op->indices, path->Attr("indices"));
}

void TIRVisitorWithPath::Dispatch_(const TensorRegionNode* op, AccessPath path) {
  Visit(ffi::GetRef<TensorRegion>(op), path);
}

void TIRVisitorWithPath::Dispatch_(const OpaqueExprNode* op, AccessPath path) {}

void TIRVisitorWithPath::Dispatch_(const TupleNode* op, AccessPath path) {
  Visit(op->fields, path->Attr("fields"));
}

void TIRVisitorWithPath::Dispatch_(const TupleGetItemNode* op, AccessPath path) {
  Visit(op->tuple, path->Attr("tuple"));
}

void TIRVisitorWithPath::Dispatch_(const prim::LetNode* op, AccessPath path) {
  Visit(op->value, path->Attr("value"));
  auto context = WithDef(op->var, path->Attr("var"));
  Visit(op->body, path->Attr("body"));
}

void TIRVisitorWithPath::Dispatch_(const CallNode* op, AccessPath path) {
  if (auto gvar = op->op.as<GlobalVar>()) {
    Visit(gvar.value(), path->Attr("op"));
  } else if (op->op.as<OpaqueExprNode>()) {
    Visit(op->op, path->Attr("op"));
  }
  Visit(op->args, path->Attr("args"));
}

#define DEFINE_BINOP_VISIT_(OP)                                       \
  void TIRVisitorWithPath::Dispatch_(const OP* op, AccessPath path) { \
    Visit(op->a, path->Attr("a"));                                    \
    Visit(op->b, path->Attr("b"));                                    \
  }

DEFINE_BINOP_VISIT_(prim::AddNode);
DEFINE_BINOP_VISIT_(prim::SubNode);
DEFINE_BINOP_VISIT_(prim::MulNode);
DEFINE_BINOP_VISIT_(prim::DivNode);
DEFINE_BINOP_VISIT_(prim::ModNode);
DEFINE_BINOP_VISIT_(prim::FloorDivNode);
DEFINE_BINOP_VISIT_(prim::FloorModNode);
DEFINE_BINOP_VISIT_(prim::MinNode);
DEFINE_BINOP_VISIT_(prim::MaxNode);
DEFINE_BINOP_VISIT_(prim::EQNode);
DEFINE_BINOP_VISIT_(prim::NENode);
DEFINE_BINOP_VISIT_(prim::LTNode);
DEFINE_BINOP_VISIT_(prim::LENode);
DEFINE_BINOP_VISIT_(prim::GTNode);
DEFINE_BINOP_VISIT_(prim::GENode);
DEFINE_BINOP_VISIT_(prim::AndNode);
DEFINE_BINOP_VISIT_(prim::OrNode);

#undef DEFINE_BINOP_VISIT_

void TIRVisitorWithPath::Dispatch_(const IntImmNode* op, AccessPath path) {}
void TIRVisitorWithPath::Dispatch_(const FloatImmNode* op, AccessPath path) {}
void TIRVisitorWithPath::Dispatch_(const prim::StringImmNode* op, AccessPath path) {}

void TIRVisitorWithPath::Dispatch_(const prim::CastNode* op, AccessPath path) {
  Visit(op->value, path->Attr("value"));
}

void TIRVisitorWithPath::Dispatch_(const prim::NotNode* op, AccessPath path) {
  Visit(op->a, path->Attr("a"));
}

void TIRVisitorWithPath::Dispatch_(const prim::SelectNode* op, AccessPath path) {
  Visit(op->condition, path->Attr("condition"));
  Visit(op->true_value, path->Attr("true_value"));
  Visit(op->false_value, path->Attr("false_value"));
}

void TIRVisitorWithPath::Dispatch_(const prim::RampNode* op, AccessPath path) {
  Visit(op->base, path->Attr("base"));
  Visit(op->stride, path->Attr("stride"));
  Visit(op->lanes, path->Attr("lanes"));
}

void TIRVisitorWithPath::Dispatch_(const prim::ShuffleNode* op, AccessPath path) {
  Visit(op->indices, path->Attr("indices"));
  Visit(op->vectors, path->Attr("vectors"));
}

void TIRVisitorWithPath::Dispatch_(const prim::BroadcastNode* op, AccessPath path) {
  Visit(op->value, path->Attr("value"));
  Visit(op->lanes, path->Attr("lanes"));
}

}  // namespace tirx
}  // namespace tvm
