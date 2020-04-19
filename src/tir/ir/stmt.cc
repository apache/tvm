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
#include <tvm/tir/stmt.h>
#include <tvm/tir/ir_pass.h>
#include "../pass/ir_util.h"

namespace tvm {
namespace tir {

Stmt LetStmtNode::make(Var var, PrimExpr value, Stmt body) {
  CHECK(value.defined());
  CHECK(body.defined());
  CHECK_EQ(value.dtype(), var.dtype());

  ObjectPtr<LetStmtNode> node = make_object<LetStmtNode>();
  node->var = std::move(var);
  node->value = std::move(value);
  node->body = std::move(body);
  return Stmt(node);
}

TVM_REGISTER_GLOBAL("tir.LetStmt")
.set_body_typed(LetStmtNode::make);

Stmt AttrStmtNode::make(ObjectRef node,
                    std::string attr_key,
                    PrimExpr value,
                    Stmt body) {
  auto n = make_object<AttrStmtNode>();
  n->node = node;
  n->attr_key = std::move(attr_key);
  n->value = std::move(value);
  n->body = std::move(body);
  return Stmt(n);
}

TVM_REGISTER_GLOBAL("tir.AttrStmt")
.set_body_typed(AttrStmtNode::make);

Stmt AssertStmtNode::make(PrimExpr condition, PrimExpr message, Stmt body) {
  CHECK(condition.defined());
  CHECK(message.dtype() == DataType::Int(32) ||
        message.as<StringImmNode>())
      << "TypeError: AssertStmt message must be an int or string:"
      << message << "\n";

  ObjectPtr<AssertStmtNode> node = make_object<AssertStmtNode>();
  node->condition = std::move(condition);
  node->message = std::move(message);
  node->body = std::move(body);
  return Stmt(node);
}

TVM_REGISTER_GLOBAL("tir.AssertStmt")
.set_body_typed([](PrimExpr condition, ObjectRef message, Stmt body) {
  if (const auto* str = message.as<StringObj>()) {
    auto msg = StringImmNode::make(str->data);
    return AssertStmtNode::make(condition, msg, body);
  } else {
    return AssertStmtNode::make(condition, Downcast<PrimExpr>(message), body);
  }
});

Stmt ForNode::make(Var loop_var,
                   PrimExpr min,
                   PrimExpr extent,
                   ForType for_type,
                   DeviceAPI device_api,
                   Stmt body) {
  CHECK(min.defined());
  CHECK(extent.defined());
  CHECK(min.dtype().is_scalar());
  CHECK(extent.dtype().is_scalar());
  CHECK(loop_var.dtype().is_scalar());
  CHECK(body.defined());

  ObjectPtr<ForNode> node = make_object<ForNode>();
  node->loop_var = std::move(loop_var);
  node->min = std::move(min);
  node->extent = std::move(extent);
  node->for_type = for_type;
  node->device_api = device_api;
  node->body = std::move(body);
  return Stmt(node);
}

TVM_REGISTER_GLOBAL("tir.For")
.set_body_typed([](
  Var loop_var, PrimExpr min, PrimExpr extent,
  int for_type, int device_api, Stmt body) {
  return ForNode::make(loop_var,
                       min,
                       extent,
                       static_cast<ForType>(for_type),
                       static_cast<DeviceAPI>(device_api),
                       body);
});


Stmt StoreNode::make(Var buffer_var, PrimExpr value, PrimExpr index, PrimExpr predicate) {
  CHECK(value.defined());
  CHECK(index.defined());
  CHECK(predicate.defined());
  CHECK_EQ(value.dtype().lanes(), index.dtype().lanes());
  CHECK_EQ(value.dtype().lanes(), predicate.dtype().lanes());

  ObjectPtr<StoreNode> node = make_object<StoreNode>();
  node->buffer_var = std::move(buffer_var);
  node->value = std::move(value);
  node->index = std::move(index);
  node->predicate = std::move(predicate);
  return Stmt(node);
}


TVM_REGISTER_GLOBAL("tir.Store")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    PrimExpr value = args[1];
    if (args.size() == 3) {
      *ret = StoreNode::make(args[0], value, args[2], const_true(value.dtype().lanes()));
    } else {
      *ret = StoreNode::make(args[0], value, args[2], args[3]);
    }
  });


Stmt ProvideNode::make(FunctionRef func, int value_index, PrimExpr value, Array<PrimExpr> args) {
  CHECK(value_index >=0 && value_index < func->num_outputs())
      << "value index output function return value bound";
  CHECK(value.defined()) << "Provide of undefined value\n";

  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i].defined()) << "Provide to undefined location\n";
  }

  ObjectPtr<ProvideNode> node = make_object<ProvideNode>();
  node->func = std::move(func);
  node->value_index = value_index;
  node->value = std::move(value);
  node->args = std::move(args);
  return Stmt(node);
}

TVM_REGISTER_GLOBAL("tir.Provide")
.set_body_typed(ProvideNode::make);


Stmt AllocateNode::make(Var buffer_var,
                        DataType dtype,
                        Array<PrimExpr> extents,
                        PrimExpr condition,
                        Stmt body) {
    for (size_t i = 0; i < extents.size(); ++i) {
      CHECK(extents[i].defined());
      CHECK(extents[i].dtype().is_scalar());
    }
    CHECK(body.defined());
    CHECK(condition.defined());
    CHECK(condition.dtype().is_bool());

    ObjectPtr<AllocateNode> node = make_object<AllocateNode>();
    node->buffer_var = std::move(buffer_var);
    node->dtype = dtype;
    node->extents = std::move(extents);
    node->condition = std::move(condition);
    node->body = std::move(body);
    return Stmt(node);
}

// overloaded, needs special handling
// has default args
TVM_REGISTER_GLOBAL("tir.Allocate")
.set_body_typed([](
    Var buffer_var, DataType type, Array<PrimExpr> extents, PrimExpr condition, Stmt body
                   ){
  return AllocateNode::make(buffer_var, type, extents, condition, body);
});

int32_t AllocateNode::constant_allocation_size(const Array<PrimExpr>& extents) {
  int64_t result = 1;
  for (size_t i = 0; i < extents.size(); ++i) {
    if (const IntImmNode *int_size = extents[i].as<IntImmNode>()) {
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

Stmt FreeNode::make(Var buffer_var) {
  ObjectPtr<FreeNode> node = make_object<FreeNode>();
  node->buffer_var = buffer_var;
  return Stmt(node);
}

TVM_REGISTER_GLOBAL("tir.Free")
.set_body_typed(FreeNode::make);


Stmt RealizeNode::make(FunctionRef func,
                       int value_index,
                       DataType dtype,
                       Region bounds,
                       PrimExpr condition,
                       Stmt body) {
  for (size_t i = 0; i < bounds.size(); ++i) {
    CHECK(bounds[i]->min.defined());
    CHECK(bounds[i]->extent.defined());
    CHECK(bounds[i]->min.dtype().is_scalar());
    CHECK(bounds[i]->extent.dtype().is_scalar());
  }
  CHECK(body.defined());
  CHECK(condition.defined());
  CHECK(condition.dtype().is_bool());

  ObjectPtr<RealizeNode> node = make_object<RealizeNode>();
  node->func = std::move(func);
  node->value_index = value_index;
  node->dtype = dtype;
  node->bounds = std::move(bounds);
  node->condition = std::move(condition);
  node->body = std::move(body);
  return Stmt(node);
}


TVM_REGISTER_GLOBAL("tir.Realize")
.set_body_typed(RealizeNode::make);


Prefetch::Prefetch(Buffer buffer, Array<Range> bounds) {
  data_ = make_object<PrefetchNode>(buffer, bounds);
}

TVM_REGISTER_GLOBAL("tir.Prefetch")
.set_body_typed([](Buffer buffer, Array<Range> bounds) {
  return Prefetch(buffer, bounds);
});


SeqStmt::SeqStmt(Array<Stmt> seq) {
  auto node = make_object<SeqStmtNode>();
  node->seq = std::move(seq);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.SeqStmt")
.set_body_typed([](Array<Stmt> seq) {
  return SeqStmt(std::move(seq));
});

Stmt IfThenElseNode::make(PrimExpr condition, Stmt then_case, Stmt else_case) {
  CHECK(condition.defined());
  CHECK(then_case.defined());
  // else_case may be null.

  ObjectPtr<IfThenElseNode> node = make_object<IfThenElseNode>();
  node->condition = std::move(condition);
  node->then_case = std::move(then_case);
  node->else_case = std::move(else_case);
  return Stmt(node);
}

TVM_REGISTER_GLOBAL("tir.IfThenElse")
.set_body_typed(IfThenElseNode::make);


Stmt EvaluateNode::make(PrimExpr value) {
  CHECK(value.defined());

  ObjectPtr<EvaluateNode> node = make_object<EvaluateNode>();
  node->value = std::move(value);
  return Stmt(node);
}

TVM_REGISTER_GLOBAL("tir.Evaluate")
.set_body_typed(EvaluateNode::make);

BufferStore::BufferStore(Buffer buffer, PrimExpr value, Array<PrimExpr> indices) {
  ObjectPtr<BufferStoreNode> node = make_object<BufferStoreNode>();
  node->buffer = std::move(buffer);
  node->value = std::move(value);
  node->indices = std::move(indices);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.BufferStore")
.set_body_typed([](Buffer buffer, PrimExpr value, Array<PrimExpr> indices) {
  return BufferStore(buffer, value, indices);
});

TVM_REGISTER_NODE_TYPE(BufferStoreNode);


BufferRealize::BufferRealize(Buffer buffer,
                             Array<Range> bounds,
                             PrimExpr condition,
                             Stmt body) {
  data_ = make_object<BufferRealizeNode>(
      buffer, bounds, condition, body);
}

TVM_REGISTER_GLOBAL("tir.BufferRealize")
.set_body_typed([](Buffer buffer,
                   Array<Range> bounds,
                   PrimExpr condition,
                   Stmt body) {
  return BufferRealize(buffer, bounds, condition, body);
});

TVM_REGISTER_NODE_TYPE(BufferRealizeNode);

// Printers

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<LetStmtNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const LetStmtNode*>(node.get());
    p->PrintIndent();
    p->stream << "let " << op->var << " = ";
    p->Print(op->value);
    p->stream << '\n';
    p->Print(op->body);
  });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<AttrStmtNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const AttrStmtNode*>(node.get());
    p->PrintIndent();
    p->stream << "// attr [";
    p->Print(op->node);
    p->stream << "] "
              << op->attr_key << " = ";
    p->Print(op->value);
    p->stream << '\n';
    p->Print(op->body);
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

std::ostream &operator<<(std::ostream& out, ForType type) { // NOLINT(*)
  switch (type) {
    case ForType::Serial:
      out << "for";
      break;
    case ForType::Parallel:
      out << "parallel";
      break;
    case ForType::Unrolled:
      out << "unrolled";
      break;
    case ForType::Vectorized:
      out << "vectorized";
      break;
  }
  return out;
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<ForNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const ForNode*>(node.get());
    p->PrintIndent();
    p->stream << op->for_type << " (" << op->loop_var << ", ";
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

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<ProvideNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const ProvideNode*>(node.get());
    p->PrintIndent();
    p->stream << op->func->func_name() << "(";
    for (size_t i = 0; i < op->args.size(); ++i) {
      p->Print(op->args[i]);
      if (i < op->args.size() - 1) p->stream << ", ";
    }
    p->stream << ")";
    if (op->func->num_outputs() != 1) {
      p->stream << ".value[" << op->value_index << "]";
    }
    p->stream << " =";
    p->Print(op->value);
    p->stream << '\n';
  });

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

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<FreeNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const FreeNode*>(node.get());
    p->PrintIndent();
    p->stream << "free " << op->buffer_var;
    p->stream << '\n';
  });

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

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<RealizeNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const RealizeNode*>(node.get());
    p->PrintIndent();
    p->stream << "realize " << op->func->func_name() << "(";
    for (size_t i = 0; i < op->bounds.size(); ++i) {
      p->stream << "[";
      p->Print(op->bounds[i]->min);
      p->stream << ", ";
      p->Print(op->bounds[i]->extent);
      p->stream << "]";
      if (i < op->bounds.size() - 1) p->stream << ", ";
    }
    p->stream << ")";
    if (op->func->num_outputs() != 1) {
      p->stream << ".value[" << op->value_index << "]";
    }
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

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<SeqStmtNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const SeqStmtNode*>(node.get());
    for (Stmt stmt : op->seq) {
      p->Print(stmt);
    }
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

      if (const IfThenElseNode *nested_if = op->else_case.as<IfThenElseNode>()) {
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

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<EvaluateNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const EvaluateNode*>(node.get());
    p->PrintIndent();
    p->Print(op->value);
    p->stream << "\n";
  });

template<typename T>
void PrintList(const Array<T> &exprs, ReprPrinter* p) {
  for (size_t i = 0; i < exprs.size(); ++i) {
    p->Print(exprs[i]);
    if (i < exprs.size() - 1) {
      p->stream << ", ";
    }
  }
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<ShuffleNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const ShuffleNode*>(node.get());
    p->stream << "shuffle(";
    PrintList(op->vectors, p);
    p->stream << ", ";
    PrintList(op->indices, p);
    p->stream << ")";
  });

TVM_REGISTER_NODE_TYPE(AttrStmtNode);
TVM_REGISTER_NODE_TYPE(PrefetchNode);
TVM_REGISTER_NODE_TYPE(CallNode);
TVM_REGISTER_NODE_TYPE(LetNode);
TVM_REGISTER_NODE_TYPE(LetStmtNode);
TVM_REGISTER_NODE_TYPE(AssertStmtNode);
TVM_REGISTER_NODE_TYPE(ForNode);
TVM_REGISTER_NODE_TYPE(StoreNode);
TVM_REGISTER_NODE_TYPE(ProvideNode);
TVM_REGISTER_NODE_TYPE(AllocateNode);
TVM_REGISTER_NODE_TYPE(FreeNode);
TVM_REGISTER_NODE_TYPE(RealizeNode);
TVM_REGISTER_NODE_TYPE(SeqStmtNode);
TVM_REGISTER_NODE_TYPE(IfThenElseNode);
TVM_REGISTER_NODE_TYPE(EvaluateNode);

}  // namespace tir
}  // namespace tvm
