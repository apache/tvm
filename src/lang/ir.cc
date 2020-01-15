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
 * \file ir.cc
 */
#include <tvm/base.h>
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <memory>
#include "../pass/ir_util.h"

namespace tvm {
namespace ir {

// constructors

PrimExpr FloatImm(DataType t, double value) {
  CHECK_EQ(t.lanes(), 1)
      << "ValueError: FloatImm can only take scalar";
  ObjectPtr<FloatImmNode> node = make_object<FloatImmNode>();
  node->dtype = t;
  node->value = value;
  return PrimExpr(node);
}

PrimExpr StringImmNode::make(std::string value) {
  ObjectPtr<StringImmNode> node = make_object<StringImmNode>();
  node->dtype = DataType::Handle();
  node->value = std::move(value);
  return PrimExpr(node);
}

PrimExpr CastNode::make(DataType t, PrimExpr value) {
  CHECK(value.defined());
  CHECK_EQ(t.lanes(), value.dtype().lanes());
  ObjectPtr<CastNode> node = make_object<CastNode>();
  node->dtype = t;
  node->value = std::move(value);
  return PrimExpr(node);
}

PrimExpr AndNode::make(PrimExpr a, PrimExpr b) {
  CHECK(a.defined()) << "ValueError: a is undefined";
  CHECK(b.defined()) << "ValueError: b is undefined";
  CHECK(a.dtype().is_bool());
  CHECK(b.dtype().is_bool());
  CHECK(a.dtype() == b.dtype()) << "TypeError: mismatched types";

  ObjectPtr<AndNode> node = make_object<AndNode>();
  node->dtype = DataType::Bool(a.dtype().lanes());
  node->a = std::move(a);
  node->b = std::move(b);
  return PrimExpr(node);
}

PrimExpr OrNode::make(PrimExpr a, PrimExpr b) {
  CHECK(a.defined()) << "ValueError: a is undefined";
  CHECK(b.defined()) << "ValueError: b is undefined";
  CHECK(a.dtype().is_bool());
  CHECK(b.dtype().is_bool());
  CHECK(a.dtype() == b.dtype()) << "TypeError: mismatched types";

  ObjectPtr<OrNode> node = make_object<OrNode>();
  node->dtype = DataType::Bool(a.dtype().lanes());
  node->a = std::move(a);
  node->b = std::move(b);
  return PrimExpr(node);
}

PrimExpr NotNode::make(PrimExpr a) {
  CHECK(a.defined()) << "ValueError: a is undefined";
  CHECK(a.dtype().is_bool());

  ObjectPtr<NotNode> node = make_object<NotNode>();
  node->dtype = DataType::Bool(a.dtype().lanes());
  node->a = std::move(a);
  return PrimExpr(node);
}

PrimExpr SelectNode::make(PrimExpr condition, PrimExpr true_value, PrimExpr false_value) {
  CHECK(condition.defined()) << "ValueError: condition is undefined";
  CHECK(true_value.defined()) << "ValueError: true_value is undefined";
  CHECK(false_value.defined()) << "ValueError: true_value is undefined";
  CHECK(condition.dtype().is_bool());
  CHECK_EQ(condition.dtype().lanes(), true_value.dtype().lanes());
  CHECK(false_value.dtype() == true_value.dtype()) << "TypeError: mismatched types";

  ObjectPtr<SelectNode> node = make_object<SelectNode>();
  node->dtype = true_value.dtype();
  node->condition = std::move(condition);
  node->true_value = std::move(true_value);
  node->false_value = std::move(false_value);
  return PrimExpr(node);
}

PrimExpr LoadNode::make(DataType dtype, Var buffer_var, PrimExpr index, PrimExpr predicate) {
  CHECK(buffer_var.defined());
  CHECK(predicate.defined());
  CHECK(index.defined());
  CHECK_EQ(dtype.lanes(), index.dtype().lanes());
  CHECK_EQ(dtype.lanes(), predicate.dtype().lanes());

  ObjectPtr<LoadNode> node = make_object<LoadNode>();
  node->dtype = dtype;
  node->buffer_var = std::move(buffer_var);
  node->index = std::move(index);
  node->predicate = std::move(predicate);

  return PrimExpr(node);
}

PrimExpr RampNode::make(PrimExpr base, PrimExpr stride, int lanes) {
  CHECK(base.defined());
  CHECK(stride.defined());
  CHECK(base.dtype().is_scalar());
  CHECK(stride.dtype().is_scalar());
  CHECK_GT(lanes, 1);
  CHECK_EQ(stride.dtype(), base.dtype());

  ObjectPtr<RampNode> node = make_object<RampNode>();
  node->dtype = base.dtype().with_lanes(lanes);
  node->base = base;
  node->stride = stride;
  node->lanes = lanes;
  return PrimExpr(node);
}

PrimExpr BroadcastNode::make(PrimExpr value, int lanes) {
  CHECK(value.defined());
  CHECK(value.dtype().is_scalar());
  CHECK_GT(lanes, 1);

  ObjectPtr<BroadcastNode> node = make_object<BroadcastNode>();
  node->dtype = value.dtype().with_lanes(lanes);
  node->value = std::move(value);
  node->lanes = lanes;
  return PrimExpr(node);
}

PrimExpr LetNode::make(Var var, PrimExpr value, PrimExpr body) {
  CHECK(value.defined());
  CHECK(body.defined());
  CHECK_EQ(value.dtype(), var.dtype());

  ObjectPtr<LetNode> node = make_object<LetNode>();
  node->dtype = body.dtype();
  node->var = std::move(var);
  node->value = std::move(value);
  node->body = std::move(body);
  return PrimExpr(node);
}

const char* CallNode::vectorizable_intrinsics[] = {
    "floor", "ceil", "sign", "trunc", "fabs", "round", "exp", "tanh", "sqrt",
    "log", "sin", "cos", "pow", ir::CallNode::shift_left, ir::CallNode::shift_right,
    ir::CallNode::likely, ir::CallNode::popcount
};

bool CallNode::is_vectorizable() const {
  size_t cnt = sizeof(CallNode::vectorizable_intrinsics) / sizeof(char*);
  for (size_t i = 0; i < cnt; ++i) {
    if (name == CallNode::vectorizable_intrinsics[i]) {
      return true;
    }
  }
  return false;
}

PrimExpr CallNode::make(DataType dtype,
                std::string name,
                Array<PrimExpr> args,
                CallType call_type,
                FunctionRef func,
                int value_index) {
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i].defined());
  }

  if (call_type == Halide) {
    for (size_t i = 0; i < args.size(); ++i) {
      CHECK(args[i].dtype().is_int());
    }
  }

  ObjectPtr<CallNode> node = make_object<CallNode>();
  node->dtype = dtype;
  node->name = std::move(name);
  node->args = std::move(args);
  node->call_type = call_type;
  node->func = std::move(func);
  node->value_index = value_index;
  return PrimExpr(node);
}

PrimExpr ShuffleNode::make(Array<PrimExpr> vectors,
                   Array<PrimExpr> indices) {
  CHECK_NE(vectors.size(), 0U);
  CHECK_NE(indices.size(), 0U);

  DataType base_type = vectors[0].dtype().element_of();
  int total_lanes = 0;

  for (PrimExpr val : vectors) {
    CHECK(val.dtype().element_of() == base_type);
    total_lanes += val.dtype().lanes();
  }
  CHECK_LE(indices.size(), static_cast<size_t>(total_lanes));

  ObjectPtr<ShuffleNode> node = make_object<ShuffleNode>();
  node->dtype = base_type.with_lanes(static_cast<int>(indices.size()));
  node->vectors = std::move(vectors);
  node->indices = std::move(indices);
  return PrimExpr(node);
}

PrimExpr ShuffleNode::make_concat(Array<PrimExpr> vectors) {
  CHECK_NE(vectors.size(), 0);
  if (vectors.size() == 1) {
    return vectors[0];
  }
  Array<PrimExpr> indices;
  int index = 0;
  for (const PrimExpr& e : vectors) {
    for (int i = 0; i < e.dtype().lanes(); ++i) {
      indices.push_back(IntImm(DataType::Int(32), index++));
    }
  }
  return make(vectors, indices);
}

PrimExpr ShuffleNode::make_extract_element(PrimExpr vector, int index) {
  return make({vector}, {Integer(index)});
}

CommReducer CommReducerNode::make(Array<Var> lhs,
                                  Array<Var> rhs,
                                  Array<PrimExpr> result,
                                  Array<PrimExpr> identity_element) {
  auto node = make_object<CommReducerNode>();
  node->lhs = lhs;
  node->rhs = rhs;
  node->result = result;
  node->identity_element = identity_element;
  return CommReducer(node);
}

Array<PrimExpr> CommReducerNode::operator()(Array<PrimExpr> a, Array<PrimExpr> b) const {
  CHECK_EQ(a.size(), b.size());
  CHECK_EQ(lhs.size(), a.size());
  CHECK_EQ(rhs.size(), b.size());
  Map<Var, PrimExpr> value_map;
  for (size_t i = 0; i < a.size(); ++i) {
    value_map.Set(lhs[i], a[i]);
    value_map.Set(rhs[i], b[i]);
  }
  return UpdateArray(result, [&value_map] (const PrimExpr& e) {
      return Substitute(e, value_map);
    });
}

PrimExpr ReduceNode::make(CommReducer combiner, Array<PrimExpr> source,
                  Array<IterVar> axis, PrimExpr condition, int value_index) {
  for (size_t i = 0; i < axis.size(); ++i) {
    CHECK_EQ(axis[i]->iter_type, kCommReduce)
        << "Can only take axis created by reduce_axis";
  }
  if (!condition.defined()) {
    condition = const_true();
  }
  auto n = make_object<ReduceNode>();
  CHECK(source.defined());
  for (size_t i = 0; i < axis.size(); ++i) {
    CHECK(axis[i].defined());
  }
  n->dtype = source[value_index].dtype();
  n->combiner = std::move(combiner);
  n->source = std::move(source);
  n->axis = std::move(axis);
  n->condition = condition;
  n->value_index = value_index;
  return PrimExpr(n);
}

PrimExpr AnyNode::make() {
  auto n = make_object<AnyNode>();
  return PrimExpr(n);
}

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

Stmt ProducerConsumerNode::make(FunctionRef func, bool is_producer, Stmt body) {
  CHECK(body.defined());

  ObjectPtr<ProducerConsumerNode> node = make_object<ProducerConsumerNode>();
  node->func = std::move(func);
  node->is_producer = is_producer;
  node->body = std::move(body);
  return Stmt(node);
}

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

Stmt AllocateNode::make(Var buffer_var,
                    DataType dtype,
                    Array<PrimExpr> extents,
                    PrimExpr condition,
                    Stmt body,
                    PrimExpr new_expr,
                    std::string free_function) {
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
    node->new_expr = std::move(new_expr);
    node->free_function = std::move(free_function);
    return Stmt(node);
}

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

Stmt PrefetchNode::make(FunctionRef func, int value_index, DataType dtype, Region bounds) {
  for (size_t i = 0; i < bounds.size(); ++i) {
    CHECK(bounds[i]->min.defined());
    CHECK(bounds[i]->extent.defined());
    CHECK(bounds[i]->min.dtype().is_scalar());
    CHECK(bounds[i]->extent.dtype().is_scalar());
  }

  ObjectPtr<PrefetchNode> node = make_object<PrefetchNode>();
  node->func = std::move(func);
  node->value_index = value_index;
  node->dtype = dtype;
  node->bounds = std::move(bounds);
  return Stmt(node);
}

SeqStmt::SeqStmt(Array<Stmt> seq) {
  auto node = make_object<SeqStmtNode>();
  node->seq = std::move(seq);
  data_ = std::move(node);
}

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

Stmt EvaluateNode::make(PrimExpr value) {
  CHECK(value.defined());

  ObjectPtr<EvaluateNode> node = make_object<EvaluateNode>();
  node->value = std::move(value);
  return Stmt(node);
}

// Printers

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<FloatImmNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const FloatImmNode*>(node.get());
    auto& stream = p->stream;
    switch (op->dtype.bits()) {
      case 64:
        stream << op->value;
        break;
      case 32:
        stream << op->value << 'f';
        break;
      case 16:
        stream << op->value << 'h';
        break;
      default:
        LOG(FATAL) << "Unknown float type bits=" << op->dtype.bits();
    }
  });

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<StringImmNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const StringImmNode*>(node.get());
    auto& stream = p->stream;
    stream << '"';
    for (size_t i = 0; i < op->value.size(); ++i) {
      unsigned char c = op->value[i];
      if (c >= ' ' && c <= '~' && c != '\\' && c != '"') {
        stream << c;
      } else {
        stream << '\\';
        switch (c) {
          case '"':
            stream << '"';
            break;
          case '\\':
            stream << '\\';
            break;
          case '\t':
            stream << 't';
            break;
          case '\r':
            stream << 'r';
            break;
          case '\n':
            stream << 'n';
            break;
          default:
            const char* hex_digits = "0123456789ABCDEF";
            stream << 'x' << hex_digits[c >> 4] << hex_digits[c & 0xf];
        }
      }
    }
    stream << '"';
  });

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<CastNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const CastNode*>(node.get());
    p->stream << op->dtype << '(';
    p->Print(op->value);
    p->stream << ')';
  })
.set_dispatch<VarNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const VarNode*>(node.get());
    // omit the type
    // stream << op->name << "." << op->type;
    p->stream << op->name_hint;
  })
.set_dispatch<AddNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const AddNode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " + ";
    p->Print(op->b);
    p->stream << ')';
  })
.set_dispatch<SubNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const SubNode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " - ";
    p->Print(op->b);
    p->stream << ')';
  })
.set_dispatch<MulNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const MulNode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << "*";
    p->Print(op->b);
    p->stream << ')';
  })
.set_dispatch<DivNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const DivNode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << "/";
    p->Print(op->b);
    p->stream << ')';
  })
.set_dispatch<ModNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const ModNode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " % ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<MinNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const MinNode*>(node.get());
    p->stream << "min(";
    p->Print(op->a);
    p->stream << ", ";
    p->Print(op->b);
    p->stream << ")";
})
.set_dispatch<MaxNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const MaxNode*>(node.get());
    p->stream << "max(";
    p->Print(op->a);
    p->stream << ", ";
    p->Print(op->b);
    p->stream << ")";
})
.set_dispatch<EQNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const EQNode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " == ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<NENode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const NENode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " != ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<LTNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const LTNode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " < ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<LENode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const LENode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " <= ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<GTNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const GTNode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " > ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<GENode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const GENode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " >= ";
    p->Print(op->b);
    p->stream << ')';
});

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<FloorDivNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const FloorDivNode*>(node.get());
  p->stream << "floordiv(" << op->a << ", " << op->b << ")";
});

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<FloorModNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const FloorModNode*>(node.get());
  p->stream << "floormod(" << op->a << ", " << op->b << ")";
});

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<AndNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const AndNode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " && ";
    p->Print(op->b);
    p->stream << ')';
});

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<OrNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const OrNode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " || ";
    p->Print(op->b);
    p->stream << ')';
});

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<NotNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const NotNode*>(node.get());
    p->stream << '!';
    p->Print(op->a);
});

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<SelectNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const SelectNode*>(node.get());
    p->stream << "select(";
    p->Print(op->condition);
    p->stream << ", ";
    p->Print(op->true_value);
    p->stream << ", ";
    p->Print(op->false_value);
    p->stream << ")";
});

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<LoadNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const LoadNode*>(node.get());
    p->stream << op->buffer_var << "[";
    p->Print(op->index);
    p->stream << "]";
    if (!is_one(op->predicate)) {
        p->stream << " if ";
        p->Print(op->predicate);
    }
});

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<RampNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const RampNode*>(node.get());
    p->stream << "ramp(";
    p->Print(op->base);
    p->stream << ", ";
    p->Print(op->stride);
    p->stream << ", " << op->lanes << ")";
});

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<BroadcastNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const BroadcastNode*>(node.get());
    p->stream << "x" << op->lanes << "(";
    p->Print(op->value);
    p->stream << ")";
});

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<CallNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const CallNode*>(node.get());
    p->stream << op->name << "(";
    for (size_t i = 0; i < op->args.size(); ++i) {
      p->Print(op->args[i]);
      if (i < op->args.size() - 1) {
        p->stream << ", ";
      }
    }
    p->stream << ")";
  });

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<LetNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const LetNode*>(node.get());
    p->stream << "(let " << op->var << " = ";
    p->Print(op->value);
    p->stream << " in ";
    p->Print(op->body);
    p->stream << ")";
});

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<LetStmtNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const LetStmtNode*>(node.get());
    p->PrintIndent();
    p->stream << "let " << op->var << " = ";
    p->Print(op->value);
    p->stream << '\n';
    p->Print(op->body);
  });

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<AttrStmtNode>([](const ObjectRef& node, NodePrinter* p) {
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

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<AssertStmtNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const AssertStmtNode*>(node.get());
    p->PrintIndent();
    p->stream << "assert(";
    p->Print(op->condition);
    p->stream << ", ";
    p->Print(op->message);
    p->stream << ")\n";
    p->Print(op->body);
  });

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<ProducerConsumerNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const ProducerConsumerNode*>(node.get());
    if (op->is_producer) {
      p->PrintIndent();
      p->stream << "produce " << op->func->func_name() << " {\n";
      p->indent += 2;
      p->Print(op->body);
      p->indent -= 2;
      p->PrintIndent();
      p->stream << "}\n";
    } else {
      p->Print(op->body);
    }
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

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<ForNode>([](const ObjectRef& node, NodePrinter* p) {
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

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<StoreNode>([](const ObjectRef& node, NodePrinter* p) {
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

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<ProvideNode>([](const ObjectRef& node, NodePrinter* p) {
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

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<AllocateNode>([](const ObjectRef& node, NodePrinter* p) {
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

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<FreeNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const FreeNode*>(node.get());
    p->PrintIndent();
    p->stream << "free " << op->buffer_var;
    p->stream << '\n';
  });

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<RealizeNode>([](const ObjectRef& node, NodePrinter* p) {
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

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<PrefetchNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const PrefetchNode*>(node.get());
    p->PrintIndent();
    p->stream << "prefetch " << op->func->func_name() << "(";
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
  });

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<SeqStmtNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const SeqStmtNode*>(node.get());
    for (Stmt stmt : op->seq) {
      p->Print(stmt);
    }
  });

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<IfThenElseNode>([](const ObjectRef& node, NodePrinter* p) {
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

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<EvaluateNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const EvaluateNode*>(node.get());
    p->PrintIndent();
    p->Print(op->value);
    p->stream << "\n";
  });

template<typename T>
void PrintList(const Array<T> &exprs, NodePrinter* p) {
  for (size_t i = 0; i < exprs.size(); ++i) {
    p->Print(exprs[i]);
    if (i < exprs.size() - 1) {
      p->stream << ", ";
    }
  }
}

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<ShuffleNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const ShuffleNode*>(node.get());
    p->stream << "shuffle(";
    PrintList(op->vectors, p);
    p->stream << ", ";
    PrintList(op->indices, p);
    p->stream << ")";
  });

// Container printer
TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<ArrayNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const ArrayNode*>(node.get());
    p->stream << '[';
    for (size_t i = 0 ; i < op->data.size(); ++i) {
      if (i != 0) {
        p->stream << ", ";
      }
      p->Print(op->data[i]);
    }
    p->stream << ']';
});

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<MapNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const MapNode*>(node.get());
    p->stream << '{';
    for (auto it = op->data.begin(); it != op->data.end(); ++it) {
      if (it != op->data.begin()) {
        p->stream << ", ";
      }
      p->Print(it->first);
      p->stream << ": ";
      p->Print(it->second);
    }
    p->stream << '}';
  });

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<StrMapNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const StrMapNode*>(node.get());
    p->stream << '{';
    for (auto it = op->data.begin(); it != op->data.end(); ++it) {
      if (it != op->data.begin()) {
        p->stream << ", ";
      }
      p->stream << '\"' << it->first << "\": ";
      p->Print(it->second);
    }
    p->stream << '}';
  });

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<ReduceNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const ReduceNode*>(node.get());
    p->stream << "reduce(combiner="
              << op->combiner;
    p->stream << ", source=" << op->source;
    p->stream << ", axis=" << op->axis;
    p->stream << ", where=" << op->condition;
    p->stream << ", value_index=" << op->value_index;
    p->stream << ")";
  });

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<CommReducerNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const CommReducerNode*>(node.get());
    p->stream << "comm_reducer(result=" << op->result
              << ", lhs=" << op->lhs
              << ", rhs=" << op->rhs
              << ", identity_element=" << op->identity_element
              << ")";
  });

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<AnyNode>([](const ObjectRef& node, NodePrinter* p) {
    p->stream << "?";
});

TVM_REGISTER_NODE_TYPE(CommReducerNode);
TVM_REGISTER_NODE_TYPE(ReduceNode);
TVM_REGISTER_NODE_TYPE(AnyNode);
TVM_REGISTER_NODE_TYPE(AttrStmtNode);
TVM_REGISTER_NODE_TYPE(FloatImmNode);
TVM_REGISTER_NODE_TYPE(IntImmNode);
TVM_REGISTER_NODE_TYPE(StringImmNode);
TVM_REGISTER_NODE_TYPE(CastNode);
TVM_REGISTER_NODE_TYPE(VarNode);
TVM_REGISTER_NODE_TYPE(AddNode);
TVM_REGISTER_NODE_TYPE(SubNode);
TVM_REGISTER_NODE_TYPE(MulNode);
TVM_REGISTER_NODE_TYPE(DivNode);
TVM_REGISTER_NODE_TYPE(ModNode);
TVM_REGISTER_NODE_TYPE(FloorDivNode);
TVM_REGISTER_NODE_TYPE(FloorModNode);
TVM_REGISTER_NODE_TYPE(MinNode);
TVM_REGISTER_NODE_TYPE(MaxNode);
TVM_REGISTER_NODE_TYPE(EQNode);
TVM_REGISTER_NODE_TYPE(NENode);
TVM_REGISTER_NODE_TYPE(LTNode);
TVM_REGISTER_NODE_TYPE(LENode);
TVM_REGISTER_NODE_TYPE(GTNode);
TVM_REGISTER_NODE_TYPE(GENode);
TVM_REGISTER_NODE_TYPE(AndNode);
TVM_REGISTER_NODE_TYPE(OrNode);
TVM_REGISTER_NODE_TYPE(NotNode);
TVM_REGISTER_NODE_TYPE(SelectNode);
TVM_REGISTER_NODE_TYPE(LoadNode);
TVM_REGISTER_NODE_TYPE(RampNode);
TVM_REGISTER_NODE_TYPE(BroadcastNode);
TVM_REGISTER_NODE_TYPE(ShuffleNode);
TVM_REGISTER_NODE_TYPE(PrefetchNode);
TVM_REGISTER_NODE_TYPE(CallNode);
TVM_REGISTER_NODE_TYPE(LetNode);
TVM_REGISTER_NODE_TYPE(LetStmtNode);
TVM_REGISTER_NODE_TYPE(AssertStmtNode);
TVM_REGISTER_NODE_TYPE(ProducerConsumerNode);
TVM_REGISTER_NODE_TYPE(ForNode);
TVM_REGISTER_NODE_TYPE(StoreNode);
TVM_REGISTER_NODE_TYPE(ProvideNode);
TVM_REGISTER_NODE_TYPE(AllocateNode);
TVM_REGISTER_NODE_TYPE(FreeNode);
TVM_REGISTER_NODE_TYPE(RealizeNode);
TVM_REGISTER_NODE_TYPE(SeqStmtNode);
TVM_REGISTER_NODE_TYPE(IfThenElseNode);
TVM_REGISTER_NODE_TYPE(EvaluateNode);

}  // namespace ir
}  // namespace tvm
