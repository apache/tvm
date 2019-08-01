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
 * \file expr.cc
 */
#include <tvm/base.h>
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/expr_operator.h>
#include <memory>
#include <limits>

namespace tvm {

// maximum and min values
Expr DataType::max() const {
  using namespace ir;
  CHECK_EQ(lanes(), 1);
  if (is_int()) {
    if (bits() == 64) {
      return IntImm::make(*this, std::numeric_limits<int64_t>::max());
    } else if (bits() < 64) {
      int64_t val = 1;
      val = (val << (bits() - 1)) - 1;
      return IntImm::make(*this, val);
    }
  } else if (is_uint()) {
    if (bits() == 64) {
      return UIntImm::make(*this, std::numeric_limits<uint64_t>::max());
    } else if (bits() < 64) {
      uint64_t val = 1;
      val = (val << static_cast<uint64_t>(bits())) - 1;
      return UIntImm::make(*this, val);
    }
  } else if (is_float()) {
    if (bits() == 64) {
      return FloatImm::make(*this, std::numeric_limits<double>::max());
    } else if (bits() == 32) {
      return FloatImm::make(*this, std::numeric_limits<float>::max());
    } else if (bits() == 16) {
      return FloatImm::make(*this, 65504.0);
    }
  }
  LOG(FATAL) << "Cannot decide max_value for type" << *this;
  return Expr();
}

Expr DataType::min() const {
  using namespace ir;
  CHECK_EQ(lanes(), 1);
  if (is_int()) {
    if (bits() == 64) {
      return IntImm::make(*this, std::numeric_limits<int64_t>::lowest());
    } else if (bits() < 64) {
      int64_t val = 1;
      val = -(val << (bits() - 1));
      return IntImm::make(*this, val);
    }
  } else if (is_uint()) {
    return UIntImm::make(*this, 0);
  } else if (is_float()) {
    if (bits() == 64) {
      return FloatImm::make(*this, std::numeric_limits<double>::lowest());
    } else if (bits() == 32) {
      return FloatImm::make(*this, std::numeric_limits<float>::lowest());
    } else if (bits() == 16) {
      return FloatImm::make(*this, -65504.0);
    }
  }
  LOG(FATAL) << "Cannot decide min_value for type" << *this;
  return Expr();
}

Expr::Expr(int32_t value)
    : Expr(IntImm::make(Int(32), value)) {}

Expr::Expr(float value)
    : Expr(ir::FloatImm::make(Float(32), value)) {}

Expr::Expr(std::string str)
    : Expr(ir::StringImm::make(str)) {}

Var::Var(std::string name_hint, DataType t)
    : Var(Variable::make(t, name_hint)) {}

Var Variable::make(DataType t, std::string name_hint) {
  NodePtr<Variable> node = make_node<Variable>();
  node->type = t;
  node->name_hint = std::move(name_hint);
  return Var(node);
}

Range::Range(Expr begin, Expr end)
    : Range(make_node<RangeNode>(
          begin,
          is_zero(begin) ? end : (end - begin))) {
}

Integer IntImm::make(Type t, int64_t value) {
  CHECK(t.is_int() && t.is_scalar())
      << "ValueError: IntImm can only take scalar.";
  NodePtr<IntImm> node = make_node<IntImm>();
  node->type = t;
  node->value = value;
  return Integer(node);
}

Range Range::make_by_min_extent(Expr min, Expr extent) {
  return Range(make_node<RangeNode>(min, extent));
}

IterVar IterVarNode::make(Range dom,
                          Var var,
                          IterVarType t,
                          std::string thread_tag) {
  NodePtr<IterVarNode> n = make_node<IterVarNode>();
  n->dom = dom;
  n->var = var;
  n->iter_type = t;
  n->thread_tag = thread_tag;
  return IterVar(n);
}

IterVar thread_axis(Range dom, std::string tag) {
  return IterVarNode::make(
      dom, Var(tag), kThreadIndex, tag);
}

IterVar reduce_axis(Range dom, std::string name) {
  return IterVarNode::make(
      dom, Var(name), kCommReduce);
}

void Dump(const NodeRef& n) {
  std::cerr << n << "\n";
}

Var var(std::string name_hint, Type t) {
  return Var(name_hint, t);
}

void IRPrinter::Print(const NodeRef& ir) {
  static const FType& f = vtable();
  if (!ir.defined()) {
    stream << "(nullptr)";
  } else {
    if (f.can_dispatch(ir)) {
      f(ir, this);
    } else {
      // default value, output type key and addr.
      stream << ir->type_key() << "(" << ir.get() << ")";
    }
  }
}

void IRPrinter::PrintIndent() {
  for (int i = 0; i < indent; ++i) {
    stream << ' ';
  }
}

IRPrinter::FType& IRPrinter::vtable() {
  static FType inst;
  return inst;
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<IntImm>([](const IntImm *op, IRPrinter *p) {
    if (op->type == Int(32)) {
      p->stream << op->value;
    } else {
      p->stream << "(" << op->type << ")" << op->value;
    }
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<IterVarNode>([](const IterVarNode *op, IRPrinter *p) {
    p->stream << "iter_var(";
    if (op->var->name_hint.length() != 0) {
      p->stream  << op->var->name_hint << ", ";
    }
    if (op->dom.defined()) {
      p->stream << op->dom;
    }
    if (op->thread_tag.length() != 0) {
      p->stream << ", " << op->thread_tag;
    }
    p->stream << ")";
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<RangeNode>([](const RangeNode* op, IRPrinter* p) {
    p->stream << "range(min=" << op->min << ", ext=" << op->extent << ')';
  });

TVM_REGISTER_NODE_TYPE(ArrayNode);
TVM_REGISTER_NODE_TYPE(MapNode);
TVM_REGISTER_NODE_TYPE(StrMapNode);
TVM_REGISTER_NODE_TYPE(RangeNode);
TVM_REGISTER_NODE_TYPE(IterVarNode);

}  // namespace tvm
