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
Expr UIntImm::make(DataType t, uint64_t value) {
  CHECK(t.is_uint() && t.lanes() == 1)
      << "ValueError: UIntImm can only take scalar";
  NodePtr<UIntImm> node = make_node<UIntImm>();
  node->type = t;
  node->value = value;
  return Expr(node);
}

Expr FloatImm::make(DataType t, double value) {
  CHECK_EQ(t.lanes(), 1)
      << "ValueError: FloatImm can only take scalar";
  NodePtr<FloatImm> node = make_node<FloatImm>();
  node->type = t;
  node->value = value;
  return Expr(node);
}

Expr StringImm::make(std::string value) {
  NodePtr<StringImm> node = make_node<StringImm>();
  node->type = Handle();
  node->value = std::move(value);
  return Expr(node);
}

Expr Cast::make(DataType t, Expr value) {
  CHECK(value.defined());
  CHECK_EQ(t.lanes(), value.type().lanes());
  NodePtr<Cast> node = make_node<Cast>();
  node->type = t;
  node->value = std::move(value);
  return Expr(node);
}

Expr And::make(Expr a, Expr b) {
  CHECK(a.defined()) << "ValueError: a is undefined";
  CHECK(b.defined()) << "ValueError: b is undefined";
  CHECK(a.type().is_bool());
  CHECK(b.type().is_bool());
  CHECK(a.type() == b.type()) << "TypeError: mismatched types";

  NodePtr<And> node = make_node<And>();
  node->type = Bool(a.type().lanes());
  node->a = std::move(a);
  node->b = std::move(b);
  return Expr(node);
}

Expr Or::make(Expr a, Expr b) {
  CHECK(a.defined()) << "ValueError: a is undefined";
  CHECK(b.defined()) << "ValueError: b is undefined";
  CHECK(a.type().is_bool());
  CHECK(b.type().is_bool());
  CHECK(a.type() == b.type()) << "TypeError: mismatched types";

  NodePtr<Or> node = make_node<Or>();
  node->type = Bool(a.type().lanes());
  node->a = std::move(a);
  node->b = std::move(b);
  return Expr(node);
}

Expr Not::make(Expr a) {
  CHECK(a.defined()) << "ValueError: a is undefined";
  CHECK(a.type().is_bool());

  NodePtr<Not> node = make_node<Not>();
  node->type = Bool(a.type().lanes());
  node->a = std::move(a);
  return Expr(node);
}

Expr Select::make(Expr condition, Expr true_value, Expr false_value) {
  CHECK(condition.defined()) << "ValueError: condition is undefined";
  CHECK(true_value.defined()) << "ValueError: true_value is undefined";
  CHECK(false_value.defined()) << "ValueError: true_value is undefined";
  CHECK(condition.type().is_bool());
  CHECK_EQ(condition.type().lanes(), true_value.type().lanes());
  CHECK(false_value.type() == true_value.type()) << "TypeError: mismatched types";

  NodePtr<Select> node = make_node<Select>();
  node->type = true_value.type();
  node->condition = std::move(condition);
  node->true_value = std::move(true_value);
  node->false_value = std::move(false_value);
  return Expr(node);
}

Expr Load::make(DataType type, Var buffer_var, Expr index, Expr predicate) {
  CHECK(buffer_var.defined());
  CHECK(predicate.defined());
  CHECK(index.defined());
  CHECK_EQ(type.lanes(), index.type().lanes());
  CHECK_EQ(type.lanes(), predicate.type().lanes());

  NodePtr<Load> node = make_node<Load>();
  node->type = type;
  node->buffer_var = std::move(buffer_var);
  node->index = std::move(index);
  node->predicate = std::move(predicate);

  return Expr(node);
}

Expr Ramp::make(Expr base, Expr stride, int lanes) {
  CHECK(base.defined());
  CHECK(stride.defined());
  CHECK(base.type().is_scalar());
  CHECK(stride.type().is_scalar());
  CHECK_GT(lanes, 1);
  CHECK_EQ(stride.type(), base.type());

  NodePtr<Ramp> node = make_node<Ramp>();
  node->type = base.type().with_lanes(lanes);
  node->base = base;
  node->stride = stride;
  node->lanes = lanes;
  return Expr(node);
}

Expr Broadcast::make(Expr value, int lanes) {
  CHECK(value.defined());
  CHECK(value.type().is_scalar());
  CHECK_GT(lanes, 1);

  NodePtr<Broadcast> node = make_node<Broadcast>();
  node->type = value.type().with_lanes(lanes);
  node->value = std::move(value);
  node->lanes = lanes;
  return Expr(node);
}

Expr Let::make(Var var, Expr value, Expr body) {
  CHECK(value.defined());
  CHECK(body.defined());
  CHECK_EQ(value.type(), var.type());

  NodePtr<Let> node = make_node<Let>();
  node->type = body.type();
  node->var = std::move(var);
  node->value = std::move(value);
  node->body = std::move(body);
  return Expr(node);
}

Expr Call::make(DataType type,
                std::string name,
                Array<Expr> args,
                CallType call_type,
                FunctionRef func,
                int value_index) {
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i].defined());
  }

  if (call_type == Halide) {
    for (size_t i = 0; i < args.size(); ++i) {
      CHECK(args[i].type().is_int());
    }
  }

  NodePtr<Call> node = make_node<Call>();
  node->type = type;
  node->name = std::move(name);
  node->args = std::move(args);
  node->call_type = call_type;
  node->func = std::move(func);
  node->value_index = value_index;
  return Expr(node);
}

Expr Shuffle::make(Array<Expr> vectors,
                   Array<Expr> indices) {
  CHECK_NE(vectors.size(), 0U);
  CHECK_NE(indices.size(), 0U);

  Type base_type = vectors[0].type().element_of();
  int total_lanes = 0;

  for (Expr val : vectors) {
    CHECK(val.type().element_of() == base_type);
    total_lanes += val.type().lanes();
  }
  CHECK_LE(indices.size(), static_cast<size_t>(total_lanes));

  NodePtr<Shuffle> node = make_node<Shuffle>();
  node->type = base_type.with_lanes(static_cast<int>(indices.size()));
  node->vectors = std::move(vectors);
  node->indices = std::move(indices);
  return Expr(node);
}

Expr Shuffle::make_concat(Array<Expr> vectors) {
  CHECK_NE(vectors.size(), 0);
  if (vectors.size() == 1) {
    return vectors[0];
  }
  Array<Expr> indices;
  int index = 0;
  for (const Expr& e : vectors) {
    for (int i = 0; i < e.type().lanes(); ++i) {
      indices.push_back(IntImm::make(Int(32), index++));
    }
  }
  return make(vectors, indices);
}

Expr Shuffle::make_extract_element(Expr vector, int index) {
  return make({vector}, {Integer(index)});
}

CommReducer CommReducerNode::make(Array<Var> lhs,
                                  Array<Var> rhs,
                                  Array<Expr> result,
                                  Array<Expr> identity_element) {
  auto node = make_node<CommReducerNode>();
  node->lhs = lhs;
  node->rhs = rhs;
  node->result = result;
  node->identity_element = identity_element;
  return CommReducer(node);
}

Array<Expr> CommReducerNode::operator()(Array<Expr> a, Array<Expr> b) const {
  CHECK_EQ(a.size(), b.size());
  CHECK_EQ(lhs.size(), a.size());
  CHECK_EQ(rhs.size(), b.size());
  Map<Var, Expr> value_map;
  for (size_t i = 0; i < a.size(); ++i) {
    value_map.Set(lhs[i], a[i]);
    value_map.Set(rhs[i], b[i]);
  }
  return UpdateArray(result, [&value_map] (const Expr& e) {
      return Substitute(e, value_map);
    });
}

Expr Reduce::make(CommReducer combiner, Array<Expr> source,
                  Array<IterVar> axis, Expr condition, int value_index) {
  for (size_t i = 0; i < axis.size(); ++i) {
    CHECK_EQ(axis[i]->iter_type, kCommReduce)
        << "Can only take axis created by reduce_axis";
  }
  if (!condition.defined()) {
    condition = const_true();
  }
  auto n = make_node<Reduce>();
  CHECK(source.defined());
  for (size_t i = 0; i < axis.size(); ++i) {
    CHECK(axis[i].defined());
  }
  n->type = source[value_index].type();
  n->combiner = std::move(combiner);
  n->source = std::move(source);
  n->axis = std::move(axis);
  n->condition = condition;
  n->value_index = value_index;
  return Expr(n);
}

Expr Any::make() {
  auto n = make_node<Any>();
  return Expr(n);
}

Stmt LetStmt::make(Var var, Expr value, Stmt body) {
  CHECK(value.defined());
  CHECK(body.defined());
  CHECK_EQ(value.type(), var.type());

  NodePtr<LetStmt> node = make_node<LetStmt>();
  node->var = std::move(var);
  node->value = std::move(value);
  node->body = std::move(body);
  return Stmt(node);
}

Stmt AttrStmt::make(NodeRef node,
                    std::string attr_key,
                    Expr value,
                    Stmt body) {
  auto n = make_node<AttrStmt>();
  n->node = node;
  n->attr_key = std::move(attr_key);
  n->value = std::move(value);
  n->body = std::move(body);
  return Stmt(n);
}

Stmt AssertStmt::make(Expr condition, Expr message, Stmt body) {
  CHECK(condition.defined());
  CHECK(message.type() == Int(32) ||
        message.as<StringImm>())
      << "TypeError: AssertStmt message must be an int or string:"
      << message << "\n";

  NodePtr<AssertStmt> node = make_node<AssertStmt>();
  node->condition = std::move(condition);
  node->message = std::move(message);
  node->body = std::move(body);
  return Stmt(node);
}

Stmt ProducerConsumer::make(FunctionRef func, bool is_producer, Stmt body) {
  CHECK(body.defined());

  NodePtr<ProducerConsumer> node = make_node<ProducerConsumer>();
  node->func = std::move(func);
  node->is_producer = is_producer;
  node->body = std::move(body);
  return Stmt(node);
}

Stmt For::make(Var loop_var,
               Expr min,
               Expr extent,
               ForType for_type,
               DeviceAPI device_api,
               Stmt body) {
  CHECK(min.defined());
  CHECK(extent.defined());
  CHECK(min.type().is_scalar());
  CHECK(extent.type().is_scalar());
  CHECK(loop_var.type().is_scalar());
  CHECK(body.defined());

  NodePtr<For> node = make_node<For>();
  node->loop_var = std::move(loop_var);
  node->min = std::move(min);
  node->extent = std::move(extent);
  node->for_type = for_type;
  node->device_api = device_api;
  node->body = std::move(body);
  return Stmt(node);
}

Stmt Store::make(Var buffer_var, Expr value, Expr index, Expr predicate) {
  CHECK(value.defined());
  CHECK(index.defined());
  CHECK(predicate.defined());
  CHECK_EQ(value.type().lanes(), index.type().lanes());
  CHECK_EQ(value.type().lanes(), predicate.type().lanes());

  NodePtr<Store> node = make_node<Store>();
  node->buffer_var = std::move(buffer_var);
  node->value = std::move(value);
  node->index = std::move(index);
  node->predicate = std::move(predicate);
  return Stmt(node);
}

Stmt Provide::make(FunctionRef func, int value_index, Expr value, Array<Expr> args) {
  CHECK(value_index >=0 && value_index < func->num_outputs())
      << "value index output function return value bound";
  CHECK(value.defined()) << "Provide of undefined value\n";

  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i].defined()) << "Provide to undefined location\n";
  }

  NodePtr<Provide> node = make_node<Provide>();
  node->func = std::move(func);
  node->value_index = value_index;
  node->value = std::move(value);
  node->args = std::move(args);
  return Stmt(node);
}

Stmt Allocate::make(Var buffer_var,
                    DataType type,
                    Array<Expr> extents,
                    Expr condition,
                    Stmt body,
                    Expr new_expr,
                    std::string free_function) {
    for (size_t i = 0; i < extents.size(); ++i) {
      CHECK(extents[i].defined());
      CHECK(extents[i].type().is_scalar());
    }
    CHECK(body.defined());
    CHECK(condition.defined());
    CHECK(condition.type().is_bool());

    NodePtr<Allocate> node = make_node<Allocate>();
    node->buffer_var = std::move(buffer_var);
    node->type = type;
    node->extents = std::move(extents);
    node->condition = std::move(condition);
    node->body = std::move(body);
    node->new_expr = std::move(new_expr);
    node->free_function = std::move(free_function);
    return Stmt(node);
}

int32_t Allocate::constant_allocation_size(const Array<Expr>& extents) {
  int64_t result = 1;
  for (size_t i = 0; i < extents.size(); ++i) {
    if (const IntImm *int_size = extents[i].as<IntImm>()) {
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

Stmt Free::make(Var buffer_var) {
  NodePtr<Free> node = make_node<Free>();
  node->buffer_var = buffer_var;
  return Stmt(node);
}

Stmt Realize::make(FunctionRef func,
                   int value_index,
                   DataType type,
                   Region bounds,
                   Expr condition,
                   Stmt body) {
  for (size_t i = 0; i < bounds.size(); ++i) {
    CHECK(bounds[i]->min.defined());
    CHECK(bounds[i]->extent.defined());
    CHECK(bounds[i]->min.type().is_scalar());
    CHECK(bounds[i]->extent.type().is_scalar());
  }
  CHECK(body.defined());
  CHECK(condition.defined());
  CHECK(condition.type().is_bool());

  NodePtr<Realize> node = make_node<Realize>();
  node->func = std::move(func);
  node->value_index = value_index;
  node->type = type;
  node->bounds = std::move(bounds);
  node->condition = std::move(condition);
  node->body = std::move(body);
  return Stmt(node);
}

Stmt Prefetch::make(FunctionRef func, int value_index, DataType type, Region bounds) {
  for (size_t i = 0; i < bounds.size(); ++i) {
    CHECK(bounds[i]->min.defined());
    CHECK(bounds[i]->extent.defined());
    CHECK(bounds[i]->min.type().is_scalar());
    CHECK(bounds[i]->extent.type().is_scalar());
  }

  NodePtr<Prefetch> node = make_node<Prefetch>();
  node->func = std::move(func);
  node->value_index = value_index;
  node->type = type;
  node->bounds = std::move(bounds);
  return Stmt(node);
}

Stmt Block::make(Stmt first, Stmt rest) {
  CHECK(first.defined());
  CHECK(rest.defined());
  NodePtr<Block> node = make_node<Block>();

  // canonicalize.
  if (const Block* b = first.as<Block>()) {
    node->first = b->first;
    node->rest  = Block::make(b->rest, rest);
  } else {
    node->first = std::move(first);
    node->rest = std::move(rest);
  }
  return Stmt(node);
}

Stmt Block::make(const std::vector<Stmt>& stmts) {
  if (stmts.empty()) {
    return Stmt();
  }
  Stmt result = stmts.back();
  for (size_t i = stmts.size() - 1; i != 0; --i) {
    result = Block::make(stmts[i - 1], result);
  }
  return result;
}

Stmt IfThenElse::make(Expr condition, Stmt then_case, Stmt else_case) {
  CHECK(condition.defined());
  CHECK(then_case.defined());
  // else_case may be null.

  NodePtr<IfThenElse> node = make_node<IfThenElse>();
  node->condition = std::move(condition);
  node->then_case = std::move(then_case);
  node->else_case = std::move(else_case);
  return Stmt(node);
}

Stmt Evaluate::make(Expr value) {
  CHECK(value.defined());

  NodePtr<Evaluate> node = make_node<Evaluate>();
  node->value = std::move(value);
  return Stmt(node);
}

// Printers
TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<UIntImm>([](const UIntImm* op, IRPrinter* p) {
    p->stream << "(" << op->type << ")" << op->value;
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<FloatImm>([](const FloatImm* op, IRPrinter* p) {
    auto& stream = p->stream;
    switch (op->type.bits()) {
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
        LOG(FATAL) << "Unknown float type bits=" << op->type.bits();
    }
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<StringImm>([](const StringImm* op, IRPrinter* p) {
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

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Cast>([](const Cast* op, IRPrinter* p) {
    p->stream << op->type << '(';
    p->Print(op->value);
    p->stream << ')';
  })
.set_dispatch<Variable>([](const Variable* op, IRPrinter* p) {
    // omit the type
    // stream << op->name << "." << op->type;
    p->stream << op->name_hint;
  })
.set_dispatch<Add>([](const Add* op, IRPrinter* p) {
    p->stream << '(';
    p->Print(op->a);
    p->stream << " + ";
    p->Print(op->b);
    p->stream << ')';
  })
.set_dispatch<Sub>([](const Sub* op, IRPrinter* p) {
    p->stream << '(';
    p->Print(op->a);
    p->stream << " - ";
    p->Print(op->b);
    p->stream << ')';
  })
.set_dispatch<Mul>([](const Mul* op, IRPrinter* p) {
    p->stream << '(';
    p->Print(op->a);
    p->stream << "*";
    p->Print(op->b);
    p->stream << ')';
  })
.set_dispatch<Div>([](const Div* op, IRPrinter* p) {
    p->stream << '(';
    p->Print(op->a);
    p->stream << "/";
    p->Print(op->b);
    p->stream << ')';
  })
.set_dispatch<Mod>([](const Mod* op, IRPrinter* p) {
    p->stream << '(';
    p->Print(op->a);
    p->stream << " % ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<Min>([](const Min* op, IRPrinter* p) {
    p->stream << "min(";
    p->Print(op->a);
    p->stream << ", ";
    p->Print(op->b);
    p->stream << ")";
})
.set_dispatch<Max>([](const Max* op, IRPrinter* p) {
    p->stream << "max(";
    p->Print(op->a);
    p->stream << ", ";
    p->Print(op->b);
    p->stream << ")";
})
.set_dispatch<EQ>([](const EQ* op, IRPrinter* p) {
    p->stream << '(';
    p->Print(op->a);
    p->stream << " == ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<NE>([](const NE* op, IRPrinter* p) {
    p->stream << '(';
    p->Print(op->a);
    p->stream << " != ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<LT>([](const LT* op, IRPrinter* p) {
    p->stream << '(';
    p->Print(op->a);
    p->stream << " < ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<LE>([](const LE* op, IRPrinter* p) {
    p->stream << '(';
    p->Print(op->a);
    p->stream << " <= ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<GT>([](const GT* op, IRPrinter* p) {
    p->stream << '(';
    p->Print(op->a);
    p->stream << " > ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<GE>([](const GE* op, IRPrinter* p) {
    p->stream << '(';
    p->Print(op->a);
    p->stream << " >= ";
    p->Print(op->b);
    p->stream << ')';
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<FloorDiv>([](const FloorDiv* op, IRPrinter *p) {
  p->stream << "floordiv(" << op->a << ", " << op->b << ")";
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<FloorMod>([](const FloorMod* op, IRPrinter *p) {
  p->stream << "floormod(" << op->a << ", " << op->b << ")";
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<And>([](const And* op, IRPrinter* p) {
    p->stream << '(';
    p->Print(op->a);
    p->stream << " && ";
    p->Print(op->b);
    p->stream << ')';
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Or>([](const Or* op, IRPrinter* p) {
    p->stream << '(';
    p->Print(op->a);
    p->stream << " || ";
    p->Print(op->b);
    p->stream << ')';
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Not>([](const Not* op, IRPrinter* p) {
    p->stream << '!';
    p->Print(op->a);
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Select>([](const Select* op, IRPrinter* p) {
    p->stream << "select(";
    p->Print(op->condition);
    p->stream << ", ";
    p->Print(op->true_value);
    p->stream << ", ";
    p->Print(op->false_value);
    p->stream << ")";
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Load>([](const Load* op, IRPrinter* p) {
    p->stream << op->buffer_var << "[";
    p->Print(op->index);
    p->stream << "]";
    if (!is_one(op->predicate)) {
        p->stream << " if ";
        p->Print(op->predicate);
    }
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Ramp>([](const Ramp* op, IRPrinter* p) {
    p->stream << "ramp(";
    p->Print(op->base);
    p->stream << ", ";
    p->Print(op->stride);
    p->stream << ", " << op->lanes << ")";
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Broadcast>([](const Broadcast* op, IRPrinter* p) {
    p->stream << "x" << op->lanes << "(";
    p->Print(op->value);
    p->stream << ")";
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Call>([](const Call* op, IRPrinter* p) {
    p->stream << op->name << "(";
    for (size_t i = 0; i < op->args.size(); ++i) {
      p->Print(op->args[i]);
      if (i < op->args.size() - 1) {
        p->stream << ", ";
      }
    }
    p->stream << ")";
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Let>([](const Let* op, IRPrinter* p) {
    p->stream << "(let " << op->var << " = ";
    p->Print(op->value);
    p->stream << " in ";
    p->Print(op->body);
    p->stream << ")";
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<LetStmt>([](const LetStmt* op, IRPrinter* p) {
    p->PrintIndent();
    p->stream << "let " << op->var << " = ";
    p->Print(op->value);
    p->stream << '\n';
    p->Print(op->body);
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<AttrStmt>([](const AttrStmt* op, IRPrinter* p) {
    p->PrintIndent();
    p->stream << "// attr [";
    p->Print(op->node);
    p->stream << "] "
              << op->attr_key << " = ";
    p->Print(op->value);
    p->stream << '\n';
    p->Print(op->body);
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<AssertStmt>([](const AssertStmt* op, IRPrinter* p) {
    p->PrintIndent();
    p->stream << "assert(";
    p->Print(op->condition);
    p->stream << ", ";
    p->Print(op->message);
    p->stream << ")\n";
    p->Print(op->body);
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<ProducerConsumer>([](const ProducerConsumer* op, IRPrinter* p) {
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

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<For>([](const For* op, IRPrinter* p) {
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

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Store>([](const Store* op, IRPrinter* p) {
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

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Provide>([](const Provide* op, IRPrinter* p) {
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

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Allocate>([](const Allocate* op, IRPrinter* p) {
    p->PrintIndent();
    p->stream << "allocate " << op->buffer_var << "[" << op->type;
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

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Free>([](const Free* op, IRPrinter* p) {
    p->PrintIndent();
    p->stream << "free " << op->buffer_var;
    p->stream << '\n';
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Realize>([](const Realize* op, IRPrinter* p) {
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

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Prefetch>([](const Prefetch* op, IRPrinter* p) {
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

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Block>([](const Block* op, IRPrinter* p) {
    p->Print(op->first);
    if (op->rest.defined()) p->Print(op->rest);
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<IfThenElse>([](const IfThenElse* op, IRPrinter* p) {
    p->PrintIndent();
    while (true) {
      p->stream << "if (" << op->condition << ") {\n";
      p->indent += 2;
      p->Print(op->then_case);
      p->indent -= 2;

      if (!op->else_case.defined()) {
        break;
      }

      if (const IfThenElse *nested_if = op->else_case.as<IfThenElse>()) {
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

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Evaluate>([](const Evaluate* op, IRPrinter* p) {
    p->PrintIndent();
    p->Print(op->value);
    p->stream << "\n";
  });

template<typename T>
void PrintList(const Array<T> &exprs, IRPrinter* p) {
  for (size_t i = 0; i < exprs.size(); ++i) {
    p->Print(exprs[i]);
    if (i < exprs.size() - 1) {
      p->stream << ", ";
    }
  }
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Shuffle>([](const Shuffle* op, IRPrinter* p) {
    p->stream << "shuffle(";
    PrintList(op->vectors, p);
    p->stream << ", ";
    PrintList(op->indices, p);
    p->stream << ")";
  });

// Container printer
TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<ArrayNode>([](const ArrayNode* op, IRPrinter* p) {
    p->stream << '[';
    for (size_t i = 0 ; i < op->data.size(); ++i) {
      if (i != 0) {
        p->stream << ", ";
      }
      p->Print(NodeRef(op->data[i]));
    }
    p->stream << ']';
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<MapNode>([](const MapNode* op, IRPrinter* p) {
    p->stream << '{';
    for (auto it = op->data.begin(); it != op->data.end(); ++it) {
      if (it != op->data.begin()) {
        p->stream << ", ";
      }
      p->Print(NodeRef(it->first));
      p->stream << ": ";
      p->Print(NodeRef(it->second));
    }
    p->stream << '}';
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<StrMapNode>([](const StrMapNode* op, IRPrinter* p) {
    p->stream << '{';
    for (auto it = op->data.begin(); it != op->data.end(); ++it) {
      if (it != op->data.begin()) {
        p->stream << ", ";
      }
      p->stream << '\"' << it->first << "\": ";
      p->Print(NodeRef(it->second));
    }
    p->stream << '}';
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Reduce>([](const Reduce* op, IRPrinter* p) {
    p->stream << "reduce(combiner="
              << op->combiner;
    p->stream << ", source=" << op->source;
    p->stream << ", axis=" << op->axis;
    p->stream << ", where=" << op->condition;
    p->stream << ", value_index=" << op->value_index;
    p->stream << ")";
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<CommReducerNode>([](const CommReducerNode* op, IRPrinter* p) {
    p->stream << "comm_reducer(result=" << op->result
              << ", lhs=" << op->lhs
              << ", rhs=" << op->rhs
              << ", identity_element=" << op->identity_element
              << ")";
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Any>([](const Any *op, IRPrinter *p) {
  p->stream << "?";
});

TVM_REGISTER_NODE_TYPE(CommReducerNode);
TVM_REGISTER_NODE_TYPE(Reduce);
TVM_REGISTER_NODE_TYPE(Any);
TVM_REGISTER_NODE_TYPE(AttrStmt);
TVM_REGISTER_NODE_TYPE(FloatImm);
TVM_REGISTER_NODE_TYPE(IntImm);
TVM_REGISTER_NODE_TYPE(UIntImm);
TVM_REGISTER_NODE_TYPE(StringImm);
TVM_REGISTER_NODE_TYPE(Cast);
TVM_REGISTER_NODE_TYPE(Variable);
TVM_REGISTER_NODE_TYPE(Add);
TVM_REGISTER_NODE_TYPE(Sub);
TVM_REGISTER_NODE_TYPE(Mul);
TVM_REGISTER_NODE_TYPE(Div);
TVM_REGISTER_NODE_TYPE(Mod);
TVM_REGISTER_NODE_TYPE(FloorDiv);
TVM_REGISTER_NODE_TYPE(FloorMod);
TVM_REGISTER_NODE_TYPE(Min);
TVM_REGISTER_NODE_TYPE(Max);
TVM_REGISTER_NODE_TYPE(EQ);
TVM_REGISTER_NODE_TYPE(NE);
TVM_REGISTER_NODE_TYPE(LT);
TVM_REGISTER_NODE_TYPE(LE);
TVM_REGISTER_NODE_TYPE(GT);
TVM_REGISTER_NODE_TYPE(GE);
TVM_REGISTER_NODE_TYPE(And);
TVM_REGISTER_NODE_TYPE(Or);
TVM_REGISTER_NODE_TYPE(Not);
TVM_REGISTER_NODE_TYPE(Select);
TVM_REGISTER_NODE_TYPE(Load);
TVM_REGISTER_NODE_TYPE(Ramp);
TVM_REGISTER_NODE_TYPE(Broadcast);
TVM_REGISTER_NODE_TYPE(Call);
TVM_REGISTER_NODE_TYPE(Let);
TVM_REGISTER_NODE_TYPE(LetStmt);
TVM_REGISTER_NODE_TYPE(AssertStmt);
TVM_REGISTER_NODE_TYPE(ProducerConsumer);
TVM_REGISTER_NODE_TYPE(For);
TVM_REGISTER_NODE_TYPE(Store);
TVM_REGISTER_NODE_TYPE(Provide);
TVM_REGISTER_NODE_TYPE(Allocate);
TVM_REGISTER_NODE_TYPE(Free);
TVM_REGISTER_NODE_TYPE(Realize);
TVM_REGISTER_NODE_TYPE(Block);
TVM_REGISTER_NODE_TYPE(IfThenElse);
TVM_REGISTER_NODE_TYPE(Evaluate);

}  // namespace ir
}  // namespace tvm
