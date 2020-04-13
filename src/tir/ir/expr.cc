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
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/op.h>
#include <tvm/tir/ir_pass.h>
#include <memory>
#include <limits>
#include "../pass/ir_util.h"
#include "../../support/str_escape.h"

namespace tvm {
namespace tir {

Var::Var(std::string name_hint, DataType dtype) {
  auto n = make_object<VarNode>();
  n->name_hint = std::move(name_hint);
  n->dtype = std::move(dtype);
  data_ = std::move(n);
}

Var::Var(std::string name_hint, Type type_annotation) {
  auto n = make_object<VarNode>();
  n->name_hint = std::move(name_hint);
  n->dtype = GetRuntimeDataType(type_annotation);
  n->type_annotation = std::move(type_annotation);
  data_ = std::move(n);
}

Var Var::copy_with_suffix(const std::string& suffix) const {
  const VarNode* node = get();
  ObjectPtr<VarNode> new_ptr;
  if (auto* ptr = this->as<SizeVarNode>()) {
    new_ptr = make_object<SizeVarNode>(*ptr);
  } else {
    new_ptr = make_object<VarNode>(*node);
  }
  new_ptr->name_hint += suffix;

  return Var(new_ptr);
}

SizeVar::SizeVar(std::string name_hint, DataType dtype) {
  auto n = make_object<SizeVarNode>();
  n->name_hint = std::move(name_hint);
  n->dtype = std::move(dtype);
  data_ = std::move(n);
}


TVM_REGISTER_GLOBAL("tir.Var")
.set_body_typed([](std::string name_hint, runtime::TVMArgValue type) {
  if (type.IsObjectRef<Type>()) {
    return Var(name_hint, type.operator Type());
  } else {
    return Var(name_hint, type.operator DataType());
  }
});

TVM_REGISTER_GLOBAL("tir.SizeVar")
.set_body_typed([](std::string s, DataType t) {
    return SizeVar(s, t);
});


IterVar IterVarNode::make(Range dom,
                          Var var,
                          IterVarType t,
                          std::string thread_tag) {
  ObjectPtr<IterVarNode> n = make_object<IterVarNode>();
  n->dom = dom;
  n->var = var;
  n->iter_type = t;
  n->thread_tag = thread_tag;
  return IterVar(n);
}

TVM_REGISTER_GLOBAL("tir.IterVar")
.set_body_typed([](Range dom, Var var, int iter_type, std::string thread_tag) {
  return IterVarNode::make(
      dom, var,
      static_cast<IterVarType>(iter_type),
      thread_tag);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<IterVarNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const IterVarNode*>(node.get());
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


TVM_REGISTER_NODE_TYPE(IterVarNode);

PrimExpr StringImmNode::make(std::string value) {
  ObjectPtr<StringImmNode> node = make_object<StringImmNode>();
  node->dtype = DataType::Handle();
  node->value = std::move(value);
  return PrimExpr(node);
}

TVM_REGISTER_GLOBAL("tir.StringImm")
.set_body_typed(StringImmNode::make);


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
  CHECK(condition.dtype().lanes() == true_value.dtype().lanes() ||
        condition.dtype().lanes() == 1);
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
    "log", "sin", "cos", "pow", "tan", tir::CallNode::shift_left, tir::CallNode::shift_right,
    tir::CallNode::likely, tir::CallNode::popcount
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

TVM_REGISTER_GLOBAL("tir.CommReducer")
.set_body_typed(CommReducerNode::make);

TVM_REGISTER_GLOBAL("tir.CommReducerCombine")
.set_body_method<tir::CommReducer>(&tir::CommReducerNode::operator());


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


TVM_REGISTER_GLOBAL("tir.Reduce")
.set_body_typed(ReduceNode::make);


PrimExpr AnyNode::make() {
  auto n = make_object<AnyNode>();
  return PrimExpr(n);
}

BufferLoad::BufferLoad(Buffer buffer, Array<PrimExpr> indices) {
  ObjectPtr<BufferLoadNode> node = make_object<BufferLoadNode>();
  node->dtype = buffer->dtype;
  node->buffer = std::move(buffer);
  node->indices = std::move(indices);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.BufferLoad")
.set_body_typed([](Buffer buffer, Array<PrimExpr> indices) {
  return BufferLoad(buffer, indices);
});

TVM_REGISTER_NODE_TYPE(BufferLoadNode);


TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<StringImmNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const StringImmNode*>(node.get());
    p->stream << '\"' << support::StrEscape(op->value) << '\"';
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<CastNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const CastNode*>(node.get());
    p->stream << op->dtype << '(';
    p->Print(op->value);
    p->stream << ')';
  })
.set_dispatch<VarNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const VarNode*>(node.get());
    // omit the type
    // stream << op->name << "." << op->type;
    p->stream << op->name_hint;
  })
.set_dispatch<SizeVarNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const SizeVarNode*>(node.get());
    p->stream << "{" << op->name_hint << "|" << op->name_hint << ">=0}";
  })
.set_dispatch<AddNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const AddNode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " + ";
    p->Print(op->b);
    p->stream << ')';
  })
.set_dispatch<SubNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const SubNode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " - ";
    p->Print(op->b);
    p->stream << ')';
  })
.set_dispatch<MulNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const MulNode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << "*";
    p->Print(op->b);
    p->stream << ')';
  })
.set_dispatch<DivNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const DivNode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << "/";
    p->Print(op->b);
    p->stream << ')';
  })
.set_dispatch<ModNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const ModNode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " % ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<MinNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const MinNode*>(node.get());
    p->stream << "min(";
    p->Print(op->a);
    p->stream << ", ";
    p->Print(op->b);
    p->stream << ")";
})
.set_dispatch<MaxNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const MaxNode*>(node.get());
    p->stream << "max(";
    p->Print(op->a);
    p->stream << ", ";
    p->Print(op->b);
    p->stream << ")";
})
.set_dispatch<EQNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const EQNode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " == ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<NENode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const NENode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " != ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<LTNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const LTNode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " < ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<LENode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const LENode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " <= ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<GTNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const GTNode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " > ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<GENode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const GENode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " >= ";
    p->Print(op->b);
    p->stream << ')';
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<FloorDivNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const FloorDivNode*>(node.get());
  p->stream << "floordiv(" << op->a << ", " << op->b << ")";
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<FloorModNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const FloorModNode*>(node.get());
  p->stream << "floormod(" << op->a << ", " << op->b << ")";
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<AndNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const AndNode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " && ";
    p->Print(op->b);
    p->stream << ')';
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<OrNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const OrNode*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " || ";
    p->Print(op->b);
    p->stream << ')';
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<NotNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const NotNode*>(node.get());
    p->stream << '!';
    p->Print(op->a);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<SelectNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const SelectNode*>(node.get());
    p->stream << "select(";
    p->Print(op->condition);
    p->stream << ", ";
    p->Print(op->true_value);
    p->stream << ", ";
    p->Print(op->false_value);
    p->stream << ")";
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<LoadNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const LoadNode*>(node.get());
    p->stream << op->buffer_var << "[";
    p->Print(op->index);
    p->stream << "]";
    if (!is_one(op->predicate)) {
        p->stream << " if ";
        p->Print(op->predicate);
    }
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<RampNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const RampNode*>(node.get());
    p->stream << "ramp(";
    p->Print(op->base);
    p->stream << ", ";
    p->Print(op->stride);
    p->stream << ", " << op->lanes << ")";
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<BroadcastNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const BroadcastNode*>(node.get());
    p->stream << "x" << op->lanes << "(";
    p->Print(op->value);
    p->stream << ")";
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<CallNode>([](const ObjectRef& node, ReprPrinter* p) {
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

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<LetNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const LetNode*>(node.get());
    p->stream << "(let " << op->var << " = ";
    p->Print(op->value);
    p->stream << " in ";
    p->Print(op->body);
    p->stream << ")";
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<AnyNode>([](const ObjectRef& node, ReprPrinter* p) {
    p->stream << "?";
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<ReduceNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const ReduceNode*>(node.get());
    p->stream << "reduce(combiner="
              << op->combiner;
    p->stream << ", source=" << op->source;
    p->stream << ", axis=" << op->axis;
    p->stream << ", where=" << op->condition;
    p->stream << ", value_index=" << op->value_index;
    p->stream << ")";
  });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<CommReducerNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const CommReducerNode*>(node.get());
    p->stream << "comm_reducer(result=" << op->result
              << ", lhs=" << op->lhs
              << ", rhs=" << op->rhs
              << ", identity_element=" << op->identity_element
              << ")";
  });

TVM_REGISTER_NODE_TYPE(StringImmNode);
TVM_REGISTER_NODE_TYPE(CastNode);
TVM_REGISTER_NODE_TYPE(VarNode);
TVM_REGISTER_NODE_TYPE(SizeVarNode);
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
TVM_REGISTER_NODE_TYPE(CommReducerNode);
TVM_REGISTER_NODE_TYPE(ReduceNode);
TVM_REGISTER_NODE_TYPE(AnyNode);


TVM_REGISTER_GLOBAL("tir.Add")
.set_body_typed(AddNode::make);

TVM_REGISTER_GLOBAL("tir.Sub")
.set_body_typed(SubNode::make);

TVM_REGISTER_GLOBAL("tir.Mul")
.set_body_typed(MulNode::make);

TVM_REGISTER_GLOBAL("tir.Div")
.set_body_typed(DivNode::make);

TVM_REGISTER_GLOBAL("tir.Mod")
.set_body_typed(ModNode::make);

TVM_REGISTER_GLOBAL("tir.FloorDiv")
.set_body_typed(FloorDivNode::make);

TVM_REGISTER_GLOBAL("tir.FloorMod")
.set_body_typed(FloorModNode::make);

TVM_REGISTER_GLOBAL("tir.Min")
.set_body_typed(MinNode::make);

TVM_REGISTER_GLOBAL("tir.Max")
.set_body_typed(MaxNode::make);

TVM_REGISTER_GLOBAL("tir.EQ")
.set_body_typed(EQNode::make);

TVM_REGISTER_GLOBAL("tir.NE")
.set_body_typed(NENode::make);

TVM_REGISTER_GLOBAL("tir.LT")
.set_body_typed(LTNode::make);

TVM_REGISTER_GLOBAL("tir.LE")
.set_body_typed(LENode::make);

TVM_REGISTER_GLOBAL("tir.GT")
.set_body_typed(GTNode::make);

TVM_REGISTER_GLOBAL("tir.GE")
.set_body_typed(GENode::make);

TVM_REGISTER_GLOBAL("tir.And")
.set_body_typed(AndNode::make);

TVM_REGISTER_GLOBAL("tir.Or")
.set_body_typed(OrNode::make);

TVM_REGISTER_GLOBAL("tir.Not")
.set_body_typed(NotNode::make);

TVM_REGISTER_GLOBAL("tir.Select")
.set_body_typed(SelectNode::make);

TVM_REGISTER_GLOBAL("tir.Ramp")
.set_body_typed(RampNode::make);

TVM_REGISTER_GLOBAL("tir.Cast")
.set_body_typed(CastNode::make);

TVM_REGISTER_GLOBAL("tir.Broadcast")
.set_body_typed(BroadcastNode::make);

TVM_REGISTER_GLOBAL("tir.Shuffle")
.set_body_typed(ShuffleNode::make);

TVM_REGISTER_GLOBAL("tir.Let")
.set_body_typed(LetNode::make);

TVM_REGISTER_GLOBAL("tir.Load")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    DataType t = args[0];
    if (args.size() == 3) {
      *ret = LoadNode::make(t, args[1], args[2], const_true(t.lanes()));
    } else {
      *ret = LoadNode::make(t, args[1], args[2], args[3]);
    }
  });

TVM_REGISTER_GLOBAL("tir.Call")
.set_body_typed([](
  DataType type, std::string name,
  Array<ObjectRef> args, int call_type,
  FunctionRef func, int value_index
) {
  Array<PrimExpr> prim_expr_args;
  for (const auto& it : args) {
    CHECK(it->IsInstance<runtime::StringObj>() ||
          it->IsInstance<PrimExprNode>());
    if (const auto* str = it.as<runtime::StringObj>()) {
      prim_expr_args.push_back(StringImmNode::make(str->data));
    } else {
      prim_expr_args.push_back(Downcast<PrimExpr>(it));
    }
  }
  return CallNode::make(type,
                        name,
                        prim_expr_args,
                        static_cast<CallNode::CallType>(call_type),
                        func,
                        value_index);
});

}  // namespace tir
}  // namespace tvm
