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
 * \file tir/ir/tir_visitor_with_path.cc
 * \brief Provide a TIR visitor that tracks the current location
 */

#include "tir_visitor_with_path.h"

#include <algorithm>
#include <optional>
#include <unordered_set>
#include <variant>
#include <vector>

namespace tvm {
namespace tir {

void TIRVisitorWithPath::Visit(const IRModule& mod, ObjectPath path) {
  // To ensure deterministic order of visits, sort the GlobalVar first
  // by visibility (public then private), then alphabetically by name.
  std::vector<GlobalVar> gvars;
  std::unordered_set<GlobalVar, ObjectPtrHash, ObjectPtrEqual> externally_exposed;
  for (const auto& [gvar, func] : mod->functions) {
    gvars.push_back(gvar);
    if (func->GetAttr<String>(tvm::attr::kGlobalSymbol).defined()) {
      externally_exposed.insert(gvar);
    }
  }

  std::sort(gvars.begin(), gvars.end(),
            [&externally_exposed](const GlobalVar& a, const GlobalVar& b) {
              bool a_exposed = externally_exposed.count(a);
              bool b_exposed = externally_exposed.count(b);
              if (a_exposed != b_exposed) {
                return a < b;
              } else {
                return a->name_hint < b->name_hint;
              }
            });

  std::vector<DefContext<GlobalVar>> context;

  for (const auto& gvar : gvars) {
    context.push_back(WithDef(gvar, path->Attr("global_var_map_")->MapValue(gvar->name_hint)));
  }

  for (const auto& gvar : gvars) {
    auto base_func = mod->functions[gvar];
    if (auto prim_func = base_func.as<PrimFunc>()) {
      Visit(prim_func.value(), path->Attr("functions")->MapValue(gvar));
    }
  }

  while (context.size()) context.pop_back();
}

void TIRVisitorWithPath::Visit(const PrimFunc& func, ObjectPath path) {
  // The implicit definitions from a PrimFunc::buffer_map are pretty
  // weird.  They only apply if no previous definition of that
  // variable has occurred.  Therefore, to ensure that we only avoid
  // duplicate calls to VisitVarDef, these semantics need to be
  // checked.
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> defined_params;
  std::vector<std::variant<DefContext<Var>, DefContext<Buffer>>> context;

  auto ppath = path->Attr("params");
  for (size_t i = 0; i < func->params.size(); i++) {
    context.push_back(WithDef(func->params[i], ppath->ArrayIndex(i)));
    defined_params.insert(func->params[i]);
  }

  auto try_visit_implicit_var_def = [this, &defined_params, &context](const PrimExpr& expr,
                                                                      ObjectPath path) {
    if (auto opt = expr.as<Var>()) {
      auto var = opt.value();
      if (!defined_params.count(var)) {
        context.push_back(WithDef(var, path));
        defined_params.insert(var);
      }
    }
  };
  auto try_visit_implicit_var_def_array = [&try_visit_implicit_var_def](const Array<PrimExpr>& arr,
                                                                        ObjectPath path) {
    for (size_t i = 0; i < arr.size(); i++) {
      try_visit_implicit_var_def(arr[i], path->ArrayIndex(i));
    }
  };

  auto buffer_map_path = path->Attr("buffer_map");
  for (size_t i = 0; i < func->params.size(); i++) {
    if (auto opt = func->buffer_map.Get(func->params[i])) {
      auto buf = opt.value();
      auto buf_path = buffer_map_path->MapValue(ppath->ArrayIndex(i));

      // A buffer in the buffer_map always defines its data pointer
      context.push_back(WithDef(buf->data, buf_path->Attr("data")));

      // But other implicit definitions only apply if they weren't
      // provided as explicit parameters, and they weren't defined
      // implicitly by any previous buffer.
      try_visit_implicit_var_def_array(buf->shape, buf_path->Attr("shape"));
      try_visit_implicit_var_def_array(buf->strides, buf_path->Attr("strides"));
      try_visit_implicit_var_def(buf->elem_offset, buf_path->Attr("elem_offset"));
    }
  }

  // Only after all the implicit definitions have been visited can we
  // visit the buffer definition itself.
  for (size_t i = 0; i < func->params.size(); i++) {
    if (auto opt = func->buffer_map.Get(func->params[i])) {
      auto buf_path = buffer_map_path->MapValue(ppath->ArrayIndex(i));
      EnterDef(opt.value(), buf_path);
    }
  }

  Visit(func->body, path->Attr("body"));

  while (context.size()) context.pop_back();
}

void TIRVisitorWithPath::EnterDef(const IterVar& iter_var, ObjectPath path) {
  if (iter_var->dom.defined()) {
    Visit(iter_var->dom, path->Attr("dom"));
  }
  EnterDef(iter_var->var, path->Attr("var"));
}

void TIRVisitorWithPath::ExitDef(const IterVar& iter_var, ObjectPath path) {
  ExitDef(iter_var->var, path->Attr("var"));
}

void TIRVisitorWithPath::EnterDef(const Buffer& buffer, ObjectPath path) {
  // Defining a buffer counts as using all parameters in the buffer
  // (e.g. shape/strides).
  Visit(buffer->data, path->Attr("data"));
  Visit(buffer->shape, path->Attr("shape"));
  Visit(buffer->strides, path->Attr("strides"));
  Visit(buffer->elem_offset, path->Attr("elem_offset"));
}
void TIRVisitorWithPath::ExitDef(const Buffer& buffer, ObjectPath path) {}

void TIRVisitorWithPath::Visit(const Buffer& buffer, ObjectPath path) {
  // Using a buffer *also* counts as using all parameters in the buffer.
  Visit(buffer->data, path->Attr("data"));
  Visit(buffer->shape, path->Attr("shape"));
  Visit(buffer->strides, path->Attr("strides"));
  Visit(buffer->elem_offset, path->Attr("elem_offset"));
}

void TIRVisitorWithPath::Visit(const BufferRegion& region, ObjectPath path) {
  Visit(region->buffer, path->Attr("buffer"));
  Visit(region->region, path->Attr("region"));
}

void TIRVisitorWithPath::Visit(const MatchBufferRegion& match, ObjectPath path) {
  Visit(match->source, path->Attr("source"));

  // MatchBufferRegion define the match->buffer, but do not own the
  // body in which the match->buffer is defined.  Therefore, the
  // definitions are handled in the BlockNode visitor.
}

void TIRVisitorWithPath::Visit(const IterVar& iter_var, ObjectPath path) {
  if (iter_var->dom.defined()) {
    Visit(iter_var->dom, path->Attr("dom"));
  }
  Visit(iter_var->var, path->Attr("var"));
}

void TIRVisitorWithPath::Visit(const Range& range, ObjectPath path) {
  Visit(range->min, path->Attr("min"));
  Visit(range->extent, path->Attr("extent"));
}

void TIRVisitorWithPath::VisitStmt_(const LetStmtNode* op, ObjectPath path) {
  Visit(op->value, path->Attr("value"));
  auto context = WithDef(op->var, path->Attr("var"));
  Visit(op->body, path->Attr("body"));
}

void TIRVisitorWithPath::VisitStmt_(const AttrStmtNode* op, ObjectPath path) {
  Visit(op->value, path->Attr("value"));

  std::optional<DefContext<IterVar>> context = std::nullopt;
  if (auto iter_var = op->node.as<IterVar>();
      iter_var && (op->attr_key == attr::thread_extent || op->attr_key == attr::virtual_thread)) {
    // Some attributes serve as a source of definition for the
    // tir::Var they annotate.
    context = WithDef(iter_var.value(), path->Attr("node"));
  } else if (auto expr = op->node.as<PrimExpr>()) {
    Visit(expr.value(), path->Attr("node"));
  }
  Visit(op->body, path->Attr("body"));
}

void TIRVisitorWithPath::VisitStmt_(const ForNode* op, ObjectPath path) {
  Visit(op->min, path->Attr("min"));
  Visit(op->extent, path->Attr("extent"));
  auto context = WithDef(op->loop_var, path->Attr("loop_var"));
  Visit(op->body, path->Attr("body"));
}

void TIRVisitorWithPath::VisitStmt_(const WhileNode* op, ObjectPath path) {
  Visit(op->condition, path->Attr("condition"));
  Visit(op->body, path->Attr("body"));
}

void TIRVisitorWithPath::VisitStmt_(const AllocateNode* op, ObjectPath path) {
  Visit(op->condition, path->Attr("condition"));
  Visit(op->extents, path->Attr("extents"));
  auto context = WithDef(op->buffer_var, path->Attr("buffer_var"));
  Visit(op->body, path->Attr("body"));
}

void TIRVisitorWithPath::VisitStmt_(const AllocateConstNode* op, ObjectPath path) {
  Visit(op->extents, path->Attr("extents"));
  auto context = WithDef(op->buffer_var, path->Attr("buffer_var"));
  Visit(op->body, path->Attr("body"));
}

void TIRVisitorWithPath::VisitStmt_(const DeclBufferNode* op, ObjectPath path) {
  auto context = WithDef(op->buffer, path->Attr("buffer"));
  Visit(op->body, path->Attr("body"));
}

void TIRVisitorWithPath::VisitStmt_(const BufferStoreNode* op, ObjectPath path) {
  Visit(op->value, path->Attr("value"));
  Visit(op->buffer, path->Attr("buffer"));
  Visit(op->indices, path->Attr("indices"));
}

void TIRVisitorWithPath::VisitStmt_(const BufferRealizeNode* op, ObjectPath path) {
  Visit(op->condition, path->Attr("condition"));
  Visit(op->bounds, path->Attr("bounds"));
  auto context = WithDef(op->buffer, path->Attr("buffer"));
  Visit(op->body, path->Attr("body"));
}

void TIRVisitorWithPath::VisitStmt_(const IfThenElseNode* op, ObjectPath path) {
  Visit(op->condition, path->Attr("condition"));
  Visit(op->then_case, path->Attr("then_case"));
  Visit(op->else_case, path->Attr("else_case"));
}

void TIRVisitorWithPath::VisitStmt_(const AssertStmtNode* op, ObjectPath path) {
  Visit(op->condition, path->Attr("condition"));
  Visit(op->message, path->Attr("message"));
  Visit(op->body, path->Attr("body"));
}

void TIRVisitorWithPath::VisitStmt_(const ProducerStoreNode* op, ObjectPath path) {
  Visit(op->indices, path->Attr("indices"));
  Visit(op->value, path->Attr("value"));
}

void TIRVisitorWithPath::VisitStmt_(const ProducerRealizeNode* op, ObjectPath path) {
  Visit(op->bounds, path->Attr("bounds"));
  Visit(op->body, path->Attr("body"));
  Visit(op->condition, path->Attr("condition"));
}

void TIRVisitorWithPath::VisitStmt_(const PrefetchNode* op, ObjectPath path) {
  Visit(op->bounds, path->Attr("bounds"));
}

void TIRVisitorWithPath::VisitStmt_(const SeqStmtNode* op, ObjectPath path) {
  Visit(op->seq, path->Attr("seq"));
}

void TIRVisitorWithPath::VisitStmt_(const EvaluateNode* op, ObjectPath path) {
  Visit(op->value, path->Attr("value"));
}

void TIRVisitorWithPath::VisitStmt_(const BlockNode* op, ObjectPath path) {
  std::vector<std::variant<DefContext<Var>, DefContext<IterVar>, DefContext<Buffer>>> context;

  {
    auto iter_path = path->Attr("iter_vars");
    for (size_t i = 0; i < op->iter_vars.size(); i++) {
      context.push_back(WithDef(op->iter_vars[i], iter_path->ArrayIndex(i)));
    }
  }
  Visit(op->reads, path->Attr("reads"));
  Visit(op->writes, path->Attr("writes"));

  {
    auto alloc_path = path->Attr("alloc_buffers");
    for (size_t i = 0; i < op->alloc_buffers.size(); i++) {
      auto buffer_path = alloc_path->ArrayIndex(i);
      auto buf = op->alloc_buffers[i];
      context.push_back(WithDef(buf->data, buffer_path->Attr("data")));
      context.push_back(WithDef(buf, buffer_path));
    }
  }

  {
    auto match_path = path->Attr("match_buffers");
    Visit(op->match_buffers, match_path);

    for (size_t i = 0; i < op->match_buffers.size(); i++) {
      auto buf = op->match_buffers[i]->buffer;
      auto buffer_path = match_path->ArrayIndex(i)->Attr("buffer");
      context.push_back(WithDef(buf->data, buffer_path->Attr("data")));
      context.push_back(WithDef(buf, buffer_path));
    }
  }

  Visit(op->init, path->Attr("init"));
  Visit(op->body, path->Attr("body"));

  while (context.size()) context.pop_back();
}

void TIRVisitorWithPath::VisitStmt_(const BlockRealizeNode* op, ObjectPath path) {
  Visit(op->iter_values, path->Attr("iter_values"));
  Visit(op->predicate, path->Attr("predicate"));
  Visit(op->block, path->Attr("block"));
}

void TIRVisitorWithPath::VisitExpr_(const VarNode* op, ObjectPath path) {}

void TIRVisitorWithPath::VisitExpr_(const SizeVarNode* op, ObjectPath path) {
  VisitExpr_(static_cast<const VarNode*>(op), path);
}

void TIRVisitorWithPath::VisitExpr_(const AnyNode* op, ObjectPath path) {}

void TIRVisitorWithPath::VisitExpr_(const BufferLoadNode* op, ObjectPath path) {
  Visit(op->buffer, path->Attr("buffer"));
  Visit(op->indices, path->Attr("indices"));
}

void TIRVisitorWithPath::VisitExpr_(const ProducerLoadNode* op, ObjectPath path) {
  Visit(op->indices, path->Attr("indices"));
}

void TIRVisitorWithPath::VisitExpr_(const LetNode* op, ObjectPath path) {
  Visit(op->value, path->Attr("value"));
  auto context = WithDef(op->var, path->Attr("var"));
  Visit(op->body, path->Attr("body"));
}

void TIRVisitorWithPath::VisitExpr_(const CallNode* op, ObjectPath path) {
  if (auto gvar = op->op.as<GlobalVar>()) {
    Visit(gvar.value(), path->Attr("op"));
  }
  Visit(op->args, path->Attr("args"));
}

#define DEFINE_BINOP_VISIT_(OP)                                        \
  void TIRVisitorWithPath::VisitExpr_(const OP* op, ObjectPath path) { \
    Visit(op->a, path->Attr("a"));                                     \
    Visit(op->b, path->Attr("b"));                                     \
  }

DEFINE_BINOP_VISIT_(AddNode);
DEFINE_BINOP_VISIT_(SubNode);
DEFINE_BINOP_VISIT_(MulNode);
DEFINE_BINOP_VISIT_(DivNode);
DEFINE_BINOP_VISIT_(ModNode);
DEFINE_BINOP_VISIT_(FloorDivNode);
DEFINE_BINOP_VISIT_(FloorModNode);
DEFINE_BINOP_VISIT_(MinNode);
DEFINE_BINOP_VISIT_(MaxNode);
DEFINE_BINOP_VISIT_(EQNode);
DEFINE_BINOP_VISIT_(NENode);
DEFINE_BINOP_VISIT_(LTNode);
DEFINE_BINOP_VISIT_(LENode);
DEFINE_BINOP_VISIT_(GTNode);
DEFINE_BINOP_VISIT_(GENode);
DEFINE_BINOP_VISIT_(AndNode);
DEFINE_BINOP_VISIT_(OrNode);

#undef DEFINE_BINOP_VISIT_

void TIRVisitorWithPath::VisitExpr_(const IntImmNode* op, ObjectPath path) {}
void TIRVisitorWithPath::VisitExpr_(const FloatImmNode* op, ObjectPath path) {}
void TIRVisitorWithPath::VisitExpr_(const StringImmNode* op, ObjectPath path) {}

void TIRVisitorWithPath::VisitExpr_(const ReduceNode* op, ObjectPath path) {
  Visit(op->axis, path->Attr("axis"));
  Visit(op->source, path->Attr("source"));
  Visit(op->init, path->Attr("init"));
  Visit(op->condition, path->Attr("condition"));
}

void TIRVisitorWithPath::VisitExpr_(const CastNode* op, ObjectPath path) {
  Visit(op->value, path->Attr("value"));
}

void TIRVisitorWithPath::VisitExpr_(const NotNode* op, ObjectPath path) {
  Visit(op->a, path->Attr("a"));
}

void TIRVisitorWithPath::VisitExpr_(const SelectNode* op, ObjectPath path) {
  Visit(op->condition, path->Attr("condition"));
  Visit(op->true_value, path->Attr("true_value"));
  Visit(op->false_value, path->Attr("false_value"));
}

void TIRVisitorWithPath::VisitExpr_(const RampNode* op, ObjectPath path) {
  Visit(op->base, path->Attr("base"));
  Visit(op->stride, path->Attr("stride"));
}

void TIRVisitorWithPath::VisitExpr_(const ShuffleNode* op, ObjectPath path) {
  Visit(op->indices, path->Attr("indices"));
  Visit(op->vectors, path->Attr("vectors"));
}

void TIRVisitorWithPath::VisitExpr_(const BroadcastNode* op, ObjectPath path) {
  Visit(op->value, path->Attr("value"));
}

}  // namespace tir
}  // namespace tvm
