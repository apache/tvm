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
 * \file tvm/tir/stmt.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace tir {

// LetStmt
LetStmt::LetStmt(Var var, PrimExpr value, Stmt body, Span span) {
  ICHECK(value.defined());
  ICHECK(body.defined());
  ICHECK_EQ(value.dtype(), var.dtype());

  ObjectPtr<LetStmtNode> node = make_object<LetStmtNode>();
  node->var = std::move(var);
  node->value = std::move(value);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.LetStmt")
    .set_body_typed([](Var var, PrimExpr value, Stmt body, Span span) {
      return LetStmt(var, value, body, span);
    });

TVM_REGISTER_NODE_TYPE(LetStmtNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<LetStmtNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const LetStmtNode*>(node.get());
      p->PrintIndent();
      p->stream << "let " << op->var << " = ";
      p->Print(op->value);
      p->stream << '\n';
      p->Print(op->body);
    });

// AttrStmt
AttrStmt::AttrStmt(ObjectRef node, String attr_key, PrimExpr value, Stmt body, Span span) {
  auto n = make_object<AttrStmtNode>();
  n->node = node;
  n->attr_key = std::move(attr_key);
  n->value = std::move(value);
  n->body = std::move(body);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("tir.AttrStmt")
    .set_body_typed([](ObjectRef node, String attr_key, PrimExpr value, Stmt body, Span span) {
      return AttrStmt(node, attr_key, value, body, span);
    });

TVM_REGISTER_NODE_TYPE(AttrStmtNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AttrStmtNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const AttrStmtNode*>(node.get());
      p->PrintIndent();
      p->stream << "// attr [";
      p->Print(op->node);
      p->stream << "] " << op->attr_key << " = ";
      p->Print(op->value);
      p->stream << '\n';
      p->Print(op->body);
    });

// AssertStmt
AssertStmt::AssertStmt(PrimExpr condition, PrimExpr message, Stmt body, Span span) {
  ICHECK(condition.defined());
  ICHECK(message.dtype() == DataType::Int(32) || message.as<StringImmNode>())
      << "TypeError: AssertStmt message must be an int or string:" << message << "\n";

  ObjectPtr<AssertStmtNode> node = make_object<AssertStmtNode>();
  node->condition = std::move(condition);
  node->message = std::move(message);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(AssertStmtNode);

TVM_REGISTER_GLOBAL("tir.AssertStmt")
    .set_body_typed([](PrimExpr condition, ObjectRef message, Stmt body, Span span) {
      if (const auto* str = message.as<StringObj>()) {
        auto msg = StringImm(str->data);
        return AssertStmt(condition, msg, body, span);
      } else {
        return AssertStmt(condition, Downcast<PrimExpr>(message), body, span);
      }
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AssertStmtNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const AssertStmtNode*>(node.get());
      p->PrintIndent();
      p->stream << "assert(";
      p->Print(op->condition);
      p->stream << ", ";
      p->Print(op->message);
      p->stream << ")\n";
      p->Print(op->body);
    });

// For
For::For(Var loop_var, PrimExpr min, PrimExpr extent, ForKind kind, Stmt body,
         Optional<IterVar> thread_binding, Map<String, ObjectRef> annotations, Span span) {
  ICHECK(min.defined());
  ICHECK(extent.defined());
  ICHECK(min.dtype().is_scalar());
  ICHECK(extent.dtype().is_scalar());
  ICHECK(loop_var.dtype().is_scalar());
  ICHECK(body.defined());

  ObjectPtr<ForNode> node = make_object<ForNode>();
  node->loop_var = std::move(loop_var);
  node->min = std::move(min);
  node->extent = std::move(extent);
  node->kind = kind;
  node->body = std::move(body);
  node->thread_binding = std::move(thread_binding);
  node->annotations = std::move(annotations);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.For").set_body_typed(
    [](Var loop_var, PrimExpr min, PrimExpr extent, int kind, Stmt body,
       Optional<IterVar> thread_binding, Optional<Map<String, ObjectRef>> annotations, Span span) {
      return For(loop_var, min, extent, static_cast<ForKind>(kind), body, thread_binding,
                 annotations.value_or(Map<String, ObjectRef>()), span);
    });

TVM_REGISTER_NODE_TYPE(ForNode);

std::ostream& operator<<(std::ostream& out, ForKind type) {  // NOLINT(*)
  switch (type) {
    case ForKind::kSerial:
      out << "for";
      break;
    case ForKind::kParallel:
      out << "parallel";
      break;
    case ForKind::kUnrolled:
      out << "unrolled";
      break;
    case ForKind::kVectorized:
      out << "vectorized";
      break;
    case ForKind::kThreadBinding:
      out << "launch_thread";
      break;
  }
  return out;
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ForNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ForNode*>(node.get());
      p->PrintIndent();
      p->stream << op->kind << " (" << op->loop_var << ", ";
      p->Print(op->min);
      p->stream << ", ";
      p->Print(op->extent);
      p->stream << ") {\n";

      p->indent += 2;
      p->Print(op->body);
      p->indent -= 2;

      p->PrintIndent();
      p->stream << "}\n";
    });

// While
While::While(PrimExpr condition, Stmt body, Span span) {
  ICHECK(condition.defined());
  ICHECK(condition.dtype().is_scalar());
  ICHECK(condition.as<tir::IntImmNode>() == nullptr) << "The condition should not be trivial.";
  ICHECK(body.defined());

  ObjectPtr<WhileNode> node = make_object<WhileNode>();
  node->condition = std::move(condition);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.While").set_body_typed([](PrimExpr condition, Stmt body, Span span) {
  return While(condition, body, span);
});

TVM_REGISTER_NODE_TYPE(WhileNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<WhileNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const WhileNode*>(node.get());
      p->PrintIndent();
      p->stream << "while(" << op->condition << ") {\n";
      p->indent += 2;
      p->Print(op->body);
      p->indent -= 2;
      p->PrintIndent();
      p->stream << "}\n";
    });

// Store
Store::Store(Var buffer_var, PrimExpr value, PrimExpr index, PrimExpr predicate, Span span) {
  ICHECK(value.defined());
  ICHECK(index.defined());
  ICHECK(predicate.defined());
  ICHECK_EQ(value.dtype().lanes(), index.dtype().lanes());
  ICHECK_EQ(value.dtype().lanes(), predicate.dtype().lanes());

  ObjectPtr<StoreNode> node = make_object<StoreNode>();
  node->buffer_var = std::move(buffer_var);
  node->value = std::move(value);
  node->index = std::move(index);
  node->predicate = std::move(predicate);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.Store").set_body([](TVMArgs args, TVMRetValue* ret) {
  PrimExpr value = args[1];
  if (args.size() == 3) {
    *ret = Store(args[0], value, args[2], const_true(value.dtype().lanes()), Span());
  } else if (args.size() == 4) {
    *ret = Store(args[0], value, args[2], args[3], Span());
  } else {
    *ret = Store(args[0], value, args[2], args[3], args[4]);
  }
});

TVM_REGISTER_NODE_TYPE(StoreNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<StoreNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const StoreNode*>(node.get());
      p->PrintIndent();
      p->stream << op->buffer_var << "[";
      p->Print(op->index);
      p->stream << "] = ";
      p->Print(op->value);
      if (!is_one(op->predicate)) {
        p->stream << " if ";
        p->Print(op->predicate);
      }
      p->stream << '\n';
    });

// ProducerStore
ProducerStore::ProducerStore(DataProducer producer, PrimExpr value, Array<PrimExpr> indices,
                             Span span) {
  ObjectPtr<ProducerStoreNode> node = make_object<ProducerStoreNode>();
  node->producer = std::move(producer);
  node->value = std::move(value);
  node->indices = std::move(indices);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.ProducerStore")
    .set_body_typed([](DataProducer producer, PrimExpr value, Array<PrimExpr> indices, Span span) {
      return ProducerStore(producer, value, indices, span);
    });

TVM_REGISTER_NODE_TYPE(ProducerStoreNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ProducerStoreNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ProducerStoreNode*>(node.get());
      p->PrintIndent();
      p->stream << op->producer->GetNameHint() << "[";
      for (size_t i = 0; i < op->indices.size(); ++i) {
        p->Print(op->indices[i]);
        if (i < op->indices.size() - 1) p->stream << ", ";
      }
      p->stream << "]";
      p->stream << " =";
      p->Print(op->value);
      p->stream << '\n';
    });

// Allocate
Allocate::Allocate(Var buffer_var, DataType dtype, Array<PrimExpr> extents, PrimExpr condition,
                   Stmt body, Span span) {
  CHECK(IsPointerType(buffer_var->type_annotation, dtype))
      << "The allocated data type (" << dtype
      << ") does not match the type annotation of the buffer " << buffer_var << " ("
      << buffer_var->type_annotation
      << "). The data type should be an element of the pointer type.";

  for (size_t i = 0; i < extents.size(); ++i) {
    ICHECK(extents[i].defined());
    ICHECK(extents[i].dtype().is_scalar());
  }
  ICHECK(body.defined());
  ICHECK(condition.defined());
  ICHECK(condition.dtype().is_bool());

  ObjectPtr<AllocateNode> node = make_object<AllocateNode>();
  node->buffer_var = std::move(buffer_var);
  node->dtype = dtype;
  node->extents = std::move(extents);
  node->condition = std::move(condition);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

int32_t AllocateNode::constant_allocation_size(const Array<PrimExpr>& extents) {
  int64_t result = 1;
  for (size_t i = 0; i < extents.size(); ++i) {
    if (const IntImmNode* int_size = extents[i].as<IntImmNode>()) {
      result *= int_size->value;
      if (result > std::numeric_limits<int32_t>::max()) {
        return 0;
      }
    } else {
      return 0;
    }
  }
  return static_cast<int32_t>(result);
}

TVM_REGISTER_GLOBAL("tir.Allocate")
    .set_body_typed([](Var buffer_var, DataType type, Array<PrimExpr> extents, PrimExpr condition,
                       Stmt body, Span span) {
      return Allocate(buffer_var, type, extents, condition, body, span);
    });

TVM_REGISTER_NODE_TYPE(AllocateNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AllocateNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const AllocateNode*>(node.get());
      p->PrintIndent();
      p->stream << "allocate " << op->buffer_var << "[" << op->dtype;
      for (size_t i = 0; i < op->extents.size(); ++i) {
        p->stream << " * ";
        p->Print(op->extents[i]);
      }
      p->stream << "]";
      if (!is_one(op->condition)) {
        p->stream << " if ";
        p->Print(op->condition);
      }
      p->stream << "\n";
      p->Print(op->body);
    });

// ProducerRealize
ProducerRealize::ProducerRealize(DataProducer producer, Region bounds, PrimExpr condition,
                                 Stmt body, Span span) {
  for (size_t i = 0; i < bounds.size(); ++i) {
    ICHECK(bounds[i]->min.defined());
    ICHECK(bounds[i]->extent.defined());
    ICHECK(bounds[i]->min.dtype().is_scalar());
    ICHECK(bounds[i]->extent.dtype().is_scalar());
  }
  ICHECK(body.defined());
  ICHECK(condition.defined());
  ICHECK(condition.dtype().is_bool());

  ObjectPtr<ProducerRealizeNode> node = make_object<ProducerRealizeNode>();
  node->producer = std::move(producer);
  node->bounds = std::move(bounds);
  node->condition = std::move(condition);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.ProducerRealize")
    .set_body_typed([](DataProducer producer, Region bounds, PrimExpr condition, Stmt body,
                       Span span) {
      return ProducerRealize(producer, bounds, condition, body, span);
    });

TVM_REGISTER_NODE_TYPE(ProducerRealizeNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ProducerRealizeNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ProducerRealizeNode*>(node.get());
      p->PrintIndent();
      p->stream << "producer_realize " << op->producer->GetNameHint() << "(";
      for (size_t i = 0; i < op->bounds.size(); ++i) {
        p->stream << "[";
        p->Print(op->bounds[i]->min);
        p->stream << ", ";
        p->Print(op->bounds[i]->extent);
        p->stream << "]";
        if (i < op->bounds.size() - 1) p->stream << ", ";
      }
      p->stream << ")";
      if (!is_one(op->condition)) {
        p->stream << " if ";
        p->Print(op->condition);
      }
      p->stream << " {\n";

      p->indent += 2;
      p->Print(op->body);
      p->indent -= 2;

      p->PrintIndent();
      p->stream << "}\n";
    });

// Prefetch
Prefetch::Prefetch(Buffer buffer, Array<Range> bounds, Span span) {
  data_ = make_object<PrefetchNode>(buffer, bounds, span);
}

TVM_REGISTER_GLOBAL("tir.Prefetch")
    .set_body_typed([](Buffer buffer, Array<Range> bounds, Span span) {
      return Prefetch(buffer, bounds, span);
    });

TVM_REGISTER_NODE_TYPE(PrefetchNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrefetchNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrefetchNode*>(node.get());
      p->PrintIndent();
      p->stream << "prefetch " << op->buffer << "(";
      for (size_t i = 0; i < op->bounds.size(); ++i) {
        p->stream << "[";
        p->Print(op->bounds[i]->min);
        p->stream << ", ";
        p->Print(op->bounds[i]->extent);
        p->stream << "]";
        if (i < op->bounds.size() - 1) p->stream << ", ";
      }
      p->stream << ")";
    });

// SeqStmt
SeqStmt::SeqStmt(Array<Stmt> seq, Span span) {
  auto node = make_object<SeqStmtNode>();
  node->seq = std::move(seq);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.SeqStmt").set_body_typed([](Array<Stmt> seq, Span span) {
  return SeqStmt(std::move(seq), span);
});

TVM_REGISTER_NODE_TYPE(SeqStmtNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SeqStmtNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SeqStmtNode*>(node.get());
      for (Stmt stmt : op->seq) {
        p->Print(stmt);
      }
    });

// IfThenElse
IfThenElse::IfThenElse(PrimExpr condition, Stmt then_case, Stmt else_case, Span span) {
  ICHECK(condition.defined());
  ICHECK(then_case.defined());
  // else_case may be null.
  ObjectPtr<IfThenElseNode> node = make_object<IfThenElseNode>();
  node->condition = std::move(condition);
  node->then_case = std::move(then_case);
  node->else_case = std::move(else_case);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(IfThenElseNode);

TVM_REGISTER_GLOBAL("tir.IfThenElse")
    .set_body_typed([](PrimExpr condition, Stmt then_case, Stmt else_case, Span span) {
      return IfThenElse(condition, then_case, else_case, span);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IfThenElseNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const IfThenElseNode*>(node.get());
      p->PrintIndent();
      while (true) {
        p->stream << "if (" << op->condition << ") {\n";
        p->indent += 2;
        p->Print(op->then_case);
        p->indent -= 2;

        if (!op->else_case.defined()) {
          break;
        }

        if (const IfThenElseNode* nested_if = op->else_case.as<IfThenElseNode>()) {
          p->PrintIndent();
          p->stream << "} else ";
          op = nested_if;
        } else {
          p->PrintIndent();
          p->stream << "} else {\n";
          p->indent += 2;
          p->Print(op->else_case);
          p->indent -= 2;
          break;
        }
      }
      p->PrintIndent();
      p->stream << "}\n";
    });

// Evaluate
Evaluate::Evaluate(PrimExpr value, Span span) {
  ICHECK(value.defined());

  ObjectPtr<EvaluateNode> node = make_object<EvaluateNode>();
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.Evaluate").set_body_typed([](PrimExpr value, Span span) {
  return Evaluate(value, span);
});

TVM_REGISTER_NODE_TYPE(EvaluateNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<EvaluateNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const EvaluateNode*>(node.get());
      p->PrintIndent();
      p->Print(op->value);
      p->stream << "\n";
    });

// BufferStore
BufferStore::BufferStore(Buffer buffer, PrimExpr value, Array<PrimExpr> indices, Span span) {
  ObjectPtr<BufferStoreNode> node = make_object<BufferStoreNode>();
  node->buffer = std::move(buffer);
  node->value = std::move(value);
  node->indices = std::move(indices);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.BufferStore")
    .set_body_typed([](Buffer buffer, PrimExpr value, Array<PrimExpr> indices, Span span) {
      return BufferStore(buffer, value, indices, span);
    });

TVM_REGISTER_NODE_TYPE(BufferStoreNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BufferStoreNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const BufferStoreNode*>(node.get());
      p->PrintIndent();
      p->stream << op->buffer->name << "[";
      for (size_t i = 0; i < op->indices.size(); ++i) {
        p->Print(op->indices[i]);
        if (i < op->indices.size() - 1) p->stream << ", ";
      }
      p->stream << "]";
      p->stream << " = ";
      p->Print(op->value);
      p->stream << '\n';
    });

// BufferRealize
BufferRealize::BufferRealize(Buffer buffer, Array<Range> bounds, PrimExpr condition, Stmt body,
                             Span span) {
  data_ = make_object<BufferRealizeNode>(buffer, bounds, condition, body, span);
}

TVM_REGISTER_GLOBAL("tir.BufferRealize")
    .set_body_typed([](Buffer buffer, Array<Range> bounds, PrimExpr condition, Stmt body,
                       Span span) { return BufferRealize(buffer, bounds, condition, body, span); });

TVM_REGISTER_NODE_TYPE(BufferRealizeNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BufferRealizeNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const BufferRealizeNode*>(node.get());
      p->PrintIndent();
      p->stream << "buffer_realize " << op->buffer->name << "(";
      for (size_t i = 0; i < op->bounds.size(); ++i) {
        p->stream << "[";
        p->Print(op->bounds[i]->min);
        p->stream << ", ";
        p->Print(op->bounds[i]->extent);
        p->stream << "]";
        if (i < op->bounds.size() - 1) p->stream << ", ";
      }
      p->stream << ")";
      if (!is_one(op->condition)) {
        p->stream << " if ";
        p->Print(op->condition);
      }
      p->stream << " {\n";

      p->indent += 2;
      p->Print(op->body);
      p->indent -= 2;

      p->PrintIndent();
      p->stream << "}\n";
    });

// BufferRegion
BufferRegion::BufferRegion(Buffer buffer, Array<Range> region) {
  ObjectPtr<BufferRegionNode> node = make_object<BufferRegionNode>();
  node->buffer = std::move(buffer);
  node->region = std::move(region);
  data_ = std::move(node);
}

BufferRegion BufferRegion::FullRegion(Buffer buffer) {
  Array<Range> region;
  for (PrimExpr extent : buffer->shape) {
    region.push_back(Range::FromMinExtent(0, extent));
  }
  return BufferRegion(buffer, region);
}

TVM_REGISTER_GLOBAL("tir.BufferRegion").set_body_typed([](Buffer buffer, Array<Range> region) {
  return BufferRegion(buffer, region);
});

TVM_REGISTER_NODE_TYPE(BufferRegionNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BufferRegionNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const BufferRegionNode*>(node.get());
      p->stream << op->buffer->name;
      p->stream << "[";
      for (size_t i = 0; i < op->region.size(); ++i) {
        const auto& range = op->region[i];
        p->Print(range->min);
        if (!is_one(range->extent)) {
          p->stream << ":";
          p->Print(range->min + range->extent);
        }
        if (i != op->region.size() - 1) p->stream << ", ";
      }
      p->stream << "]";
    });

// MatchBufferRegion
MatchBufferRegion::MatchBufferRegion(Buffer buffer, BufferRegion source) {
  ObjectPtr<MatchBufferRegionNode> node = make_object<MatchBufferRegionNode>();
  node->buffer = std::move(buffer);
  node->source = std::move(source);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.MatchBufferRegion").set_body_typed([](Buffer buffer, BufferRegion source) {
  return MatchBufferRegion(buffer, source);
});

TVM_REGISTER_NODE_TYPE(MatchBufferRegionNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MatchBufferRegionNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const MatchBufferRegionNode*>(node.get());
      p->PrintIndent();
      p->stream << op->buffer->name << " = match_buffer_region(";
      p->Print(op->source);
      p->stream << ")\n";
    });

// Block
Block::Block(Array<IterVar> iter_vars, Array<BufferRegion> reads, Array<BufferRegion> writes,
             String name_hint, Stmt body, Optional<Stmt> init, Array<Buffer> alloc_buffers,
             Array<MatchBufferRegion> match_buffers, Map<String, ObjectRef> annotations,
             Span span) {
  ObjectPtr<BlockNode> node = make_object<BlockNode>();
  node->iter_vars = std::move(iter_vars);
  node->reads = std::move(reads);
  node->writes = std::move(writes);
  node->name_hint = std::move(name_hint);
  node->body = std::move(body);
  node->init = std::move(init);
  node->alloc_buffers = std::move(alloc_buffers);
  node->match_buffers = std::move(match_buffers);
  node->annotations = std::move(annotations);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.Block")
    .set_body_typed([](Array<IterVar> iter_vars, Array<BufferRegion> reads,
                       Array<BufferRegion> writes, String name_hint, Stmt body, Optional<Stmt> init,
                       Array<Buffer> alloc_buffers, Array<MatchBufferRegion> match_buffers,
                       Map<String, ObjectRef> annotations, Span span) {
      return Block(iter_vars, reads, writes, name_hint, body, init, alloc_buffers, match_buffers,
                   annotations, span);
    });

TVM_REGISTER_NODE_TYPE(BlockNode);

void PrintBlockTitle(const BlockNode* op, ReprPrinter* p) {
  p->stream << "block " << op->name_hint << "(";
  for (size_t i = 0; i < op->iter_vars.size(); i++) {
    p->Print(op->iter_vars[i]);
    if (i < op->iter_vars.size() - 1) p->stream << ", ";
  }
  p->stream << ")";
}

void PrintBlockSignature(const BlockNode* op, ReprPrinter* p) {
  // print read/write regions
  p->PrintIndent();
  p->stream << "reads(";
  p->Print(op->reads);
  p->stream << ")\n";
  p->PrintIndent();
  p->stream << "writes(";
  p->Print(op->writes);
  p->stream << ")\n";
  // Print alloc_buffers
  for (const auto& alloc_buf : op->alloc_buffers) {
    p->PrintIndent();
    p->stream << alloc_buf->name << " = alloc_buffer(" << alloc_buf->dtype << "[";
    for (size_t i = 0; i < alloc_buf->shape.size(); ++i) {
      if (i > 0) p->stream << ", ";
      p->Print(alloc_buf->shape[i]);
    }
    p->stream << "])\n";
  }
  // Print match_buffer_regions
  for (const auto& match_buf : op->match_buffers) {
    p->Print(match_buf);
  }
  if (!op->annotations.empty()) {
    p->PrintIndent();
    p->stream << "annotations(" << op->annotations << ")\n";
  }
}

void PrintBlockBody(const BlockNode* op, ReprPrinter* p) {
  // Print init
  if (op->init.defined()) {
    p->PrintIndent();
    p->stream << "with init() {\n";
    p->indent += 2;
    p->Print(op->init.value());
    p->indent -= 2;
    p->PrintIndent();
    p->stream << "}\n";
  }
  // Print body
  p->Print(op->body);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BlockNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const BlockNode*>(node.get());
      p->PrintIndent();
      PrintBlockTitle(op, p);
      p->stream << " {\n";
      p->indent += 2;

      // Print block elements (e.g. reads/writes, etc)
      PrintBlockSignature(op, p);
      // Print block init and body
      PrintBlockBody(op, p);

      p->indent -= 2;
      p->PrintIndent();
      p->stream << "}\n";
    });

// BlockRealize
BlockRealize::BlockRealize(Array<PrimExpr> values, PrimExpr predicate, Block block, Span span) {
  CHECK_EQ(block->iter_vars.size(), values.size())
      << "ValueError: BlockRealize needs to have the same number of iter_vars and binding values";
  CHECK(predicate.dtype().is_bool()) << "TypeError: Expect Block.predicate to be a bool expression";
  ObjectPtr<BlockRealizeNode> node = make_object<BlockRealizeNode>();
  node->iter_values = std::move(values);
  node->predicate = std::move(predicate);
  node->block = std::move(block);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.BlockRealize")
    .set_body_typed([](Array<PrimExpr> iter_values, PrimExpr predicate, Block block, Span span) {
      return BlockRealize(iter_values, predicate, block, span);
    });

TVM_REGISTER_NODE_TYPE(BlockRealizeNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BlockRealizeNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const BlockRealizeNode*>(node.get());
      auto* block_op = op->block.get();
      p->PrintIndent();
      PrintBlockTitle(block_op, p);
      p->stream << " {\n";
      p->indent += 2;

      // Print binding iter_values
      for (size_t i = 0; i < block_op->iter_vars.size(); ++i) {
        p->PrintIndent();
        p->stream << "bind(";
        p->Print(block_op->iter_vars[i]->var);
        p->stream << ", ";
        p->Print(op->iter_values[i]);
        p->stream << ")\n";
      }
      // Print predicate
      if (!is_one(op->predicate)) {
        p->PrintIndent();
        p->stream << "where(";
        p->Print(op->predicate);
        p->stream << ")\n";
      }
      // Print block elements (e.g. reads/writes, etc)
      PrintBlockSignature(block_op, p);
      // Print block init and body
      PrintBlockBody(block_op, p);

      p->indent -= 2;
      p->PrintIndent();
      p->stream << "}\n";
    });

PrimExpr TypeAnnotation(DataType dtype, Span span) {
  static auto op = Op::Get("tir.type_annotation");
  return tir::Call(dtype, op, {}, span);
}

TVM_REGISTER_OP("tir.type_annotation")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

}  // namespace tir
}  // namespace tvm
