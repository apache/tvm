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
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../../support/str_escape.h"
#include "buffer_common.h"

namespace tvm {
namespace tir {

#define TVM_DEFINE_BINOP_CONSTRUCTOR(Name)                                               \
  Name::Name(PrimExpr a, PrimExpr b, Span span) {                                        \
    using T = Name::ContainerType;                                                       \
    ICHECK(a.defined()) << "ValueError: a is undefined\n";                               \
    ICHECK(b.defined()) << "ValueError: b is undefined\n";                               \
    ICHECK(a.dtype() == b.dtype())                                                       \
        << "TypeError: mismatched types. " << a.dtype() << " vs. " << b.dtype() << "\n"; \
    ObjectPtr<T> node = make_object<T>();                                                \
    node->dtype = a.dtype();                                                             \
    node->a = std::move(a);                                                              \
    node->b = std::move(b);                                                              \
    node->span = std::move(span);                                                        \
    data_ = std::move(node);                                                             \
  }

#define TVM_DEFINE_CMPOP_CONSTRUCTOR(Name)                             \
  Name::Name(PrimExpr a, PrimExpr b, Span span) {                      \
    using T = Name::ContainerType;                                     \
    ICHECK(a.defined()) << "ValueError: a is undefined\n";             \
    ICHECK(b.defined()) << "ValueError: b is undefined\n";             \
    ICHECK(a.dtype() == b.dtype()) << "TypeError: mismatched types\n"; \
    ObjectPtr<T> node = make_object<T>();                              \
    node->dtype = DataType::Bool(a.dtype().lanes());                   \
    node->a = std::move(a);                                            \
    node->b = std::move(b);                                            \
    node->span = std::move(span);                                      \
    data_ = std::move(node);                                           \
  }

// Var
Var::Var(String name_hint, DataType dtype, Span span) {
  auto n = make_object<VarNode>();
  n->name_hint = std::move(name_hint);
  n->dtype = std::move(dtype);
  n->span = std::move(span);
  data_ = std::move(n);
}

Var::Var(String name_hint, Type type_annotation, Span span) {
  auto n = make_object<VarNode>();
  n->name_hint = std::move(name_hint);
  n->dtype = GetRuntimeDataType(type_annotation);
  n->type_annotation = std::move(type_annotation);
  n->span = std::move(span);
  data_ = std::move(n);
}

Var Var::copy_with_suffix(const String& suffix) const {
  const VarNode* node = get();
  ObjectPtr<VarNode> new_ptr;
  if (auto* ptr = this->as<SizeVarNode>()) {
    new_ptr = make_object<SizeVarNode>(*ptr);
  } else {
    new_ptr = make_object<VarNode>(*node);
  }
  new_ptr->name_hint = new_ptr->name_hint + suffix;
  return Var(new_ptr);
}

TVM_REGISTER_GLOBAL("tir.Var").set_body_typed([](String name_hint, runtime::TVMArgValue type,
                                                 Span span) {
  if (type.IsObjectRef<Type>()) {
    return Var(name_hint, type.operator Type(), span);
  } else {
    return Var(name_hint, type.operator DataType(), span);
  }
});

TVM_REGISTER_NODE_TYPE(VarNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<VarNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const VarNode*>(node.get());
      // omit the type
      // stream << op->name << "." << op->type;
      p->stream << op->name_hint;
    });

// SizeVar
SizeVar::SizeVar(String name_hint, DataType dtype, Span span) {
  auto n = make_object<SizeVarNode>();
  n->name_hint = std::move(name_hint);
  n->dtype = std::move(dtype);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("tir.SizeVar").set_body_typed([](String s, DataType t, Span span) {
  return SizeVar(s, t, span);
});

TVM_REGISTER_NODE_TYPE(SizeVarNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SizeVarNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SizeVarNode*>(node.get());
      p->stream << "{" << op->name_hint << "|" << op->name_hint << ">=0}";
    });

// IterVar
IterVar::IterVar(Range dom, Var var, IterVarType t, String thread_tag, Span span) {
  ObjectPtr<IterVarNode> n = make_object<IterVarNode>();
  n->dom = dom;
  n->var = var;
  n->iter_type = t;
  n->thread_tag = thread_tag;
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("tir.IterVar")
    .set_body_typed([](Range dom, Var var, int iter_type, String thread_tag, Span span) {
      return IterVar(dom, var, static_cast<IterVarType>(iter_type), thread_tag, span);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IterVarNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const IterVarNode*>(node.get());
      p->stream << "iter_var(";
      if (op->var->name_hint.length() != 0) {
        p->stream << op->var->name_hint << ", ";
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

// StringImm
StringImm::StringImm(String value, Span span) {
  ObjectPtr<StringImmNode> node = make_object<StringImmNode>();
  node->dtype = DataType::Handle();
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.StringImm").set_body_typed([](String value, Span span) {
  return StringImm(value, span);
});

TVM_REGISTER_NODE_TYPE(StringImmNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<StringImmNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const StringImmNode*>(node.get());
      p->stream << '\"' << support::StrEscape(op->value) << '\"';
    });

// Cast
Cast::Cast(DataType t, PrimExpr value, Span span) {
  ICHECK(value.defined());
  ICHECK_EQ(t.lanes(), value.dtype().lanes());
  ObjectPtr<CastNode> node = make_object<CastNode>();
  node->dtype = t;
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.Cast").set_body_typed([](DataType dtype, PrimExpr value, Span span) {
  return Cast(dtype, value, span);
});

TVM_REGISTER_NODE_TYPE(CastNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<CastNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const CastNode*>(node.get());
      p->stream << op->dtype << '(';
      p->Print(op->value);
      p->stream << ')';
    });

// Add
TVM_DEFINE_BINOP_CONSTRUCTOR(Add);

TVM_REGISTER_GLOBAL("tir.Add").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return Add(a, b, span);
});

TVM_REGISTER_NODE_TYPE(AddNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AddNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const AddNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " + ";
      p->Print(op->b);
      p->stream << ')';
    });

// Sub
TVM_DEFINE_BINOP_CONSTRUCTOR(Sub);

TVM_REGISTER_GLOBAL("tir.Sub").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return Sub(a, b, span);
});

TVM_REGISTER_NODE_TYPE(SubNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SubNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SubNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " - ";
      p->Print(op->b);
      p->stream << ')';
    });

// Mul
TVM_DEFINE_BINOP_CONSTRUCTOR(Mul);

TVM_REGISTER_GLOBAL("tir.Mul").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return Mul(a, b, span);
});

TVM_REGISTER_NODE_TYPE(MulNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MulNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const MulNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << "*";
      p->Print(op->b);
      p->stream << ')';
    });

// Div
TVM_DEFINE_BINOP_CONSTRUCTOR(Div);

TVM_REGISTER_GLOBAL("tir.Div").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return Div(a, b, span);
});

TVM_REGISTER_NODE_TYPE(DivNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<DivNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const DivNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << "/";
      p->Print(op->b);
      p->stream << ')';
    });

// Mod
TVM_DEFINE_BINOP_CONSTRUCTOR(Mod);

TVM_REGISTER_GLOBAL("tir.Mod").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return Mod(a, b, span);
});

TVM_REGISTER_NODE_TYPE(ModNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ModNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ModNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " % ";
      p->Print(op->b);
      p->stream << ')';
    });

// FloorDiv
TVM_DEFINE_BINOP_CONSTRUCTOR(FloorDiv);

TVM_REGISTER_GLOBAL("tir.FloorDiv").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return FloorDiv(a, b, span);
});

TVM_REGISTER_NODE_TYPE(FloorDivNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<FloorDivNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const FloorDivNode*>(node.get());
      p->stream << "floordiv(" << op->a << ", " << op->b << ")";
    });

// FloorMod
TVM_DEFINE_BINOP_CONSTRUCTOR(FloorMod);

TVM_REGISTER_GLOBAL("tir.FloorMod").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return FloorMod(a, b, span);
});

TVM_REGISTER_NODE_TYPE(FloorModNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<FloorModNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const FloorModNode*>(node.get());
      p->stream << "floormod(" << op->a << ", " << op->b << ")";
    });

// Min
TVM_DEFINE_BINOP_CONSTRUCTOR(Min);

TVM_REGISTER_GLOBAL("tir.Min").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return Min(a, b, span);
});

TVM_REGISTER_NODE_TYPE(MinNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MinNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const MinNode*>(node.get());
      p->stream << "min(";
      p->Print(op->a);
      p->stream << ", ";
      p->Print(op->b);
      p->stream << ")";
    });

// Max
TVM_DEFINE_BINOP_CONSTRUCTOR(Max);

TVM_REGISTER_GLOBAL("tir.Max").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return Max(a, b, span);
});

TVM_REGISTER_NODE_TYPE(MaxNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MaxNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const MaxNode*>(node.get());
      p->stream << "max(";
      p->Print(op->a);
      p->stream << ", ";
      p->Print(op->b);
      p->stream << ")";
    });

// EQ
TVM_DEFINE_CMPOP_CONSTRUCTOR(EQ);

TVM_REGISTER_GLOBAL("tir.EQ").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return EQ(a, b, span);
});

TVM_REGISTER_NODE_TYPE(EQNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<EQNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const EQNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " == ";
      p->Print(op->b);
      p->stream << ')';
    });

// NE
TVM_DEFINE_CMPOP_CONSTRUCTOR(NE);

TVM_REGISTER_GLOBAL("tir.NE").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return NE(a, b, span);
});

TVM_REGISTER_NODE_TYPE(NENode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<NENode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const NENode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " != ";
      p->Print(op->b);
      p->stream << ')';
    });

// LT
TVM_DEFINE_CMPOP_CONSTRUCTOR(LT);

TVM_REGISTER_GLOBAL("tir.LT").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return LT(a, b, span);
});

TVM_REGISTER_NODE_TYPE(LTNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<LTNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const LTNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " < ";
      p->Print(op->b);
      p->stream << ')';
    });

// LE
TVM_DEFINE_CMPOP_CONSTRUCTOR(LE);

TVM_REGISTER_GLOBAL("tir.LE").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return LE(a, b, span);
});

TVM_REGISTER_NODE_TYPE(LENode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<LENode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const LENode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " <= ";
      p->Print(op->b);
      p->stream << ')';
    });

// GT
TVM_DEFINE_CMPOP_CONSTRUCTOR(GT);

TVM_REGISTER_GLOBAL("tir.GT").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return GT(a, b, span);
});

TVM_REGISTER_NODE_TYPE(GTNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<GTNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const GTNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " > ";
      p->Print(op->b);
      p->stream << ')';
    });

// GE
TVM_DEFINE_CMPOP_CONSTRUCTOR(GE);

TVM_REGISTER_GLOBAL("tir.GE").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return GE(a, b, span);
});

TVM_REGISTER_NODE_TYPE(GENode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<GENode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const GENode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " >= ";
      p->Print(op->b);
      p->stream << ')';
    });

// And
And::And(PrimExpr a, PrimExpr b, Span span) {
  ICHECK(a.defined()) << "ValueError: a is undefined";
  ICHECK(b.defined()) << "ValueError: b is undefined";
  ICHECK(a.dtype().is_bool());
  ICHECK(b.dtype().is_bool());
  ICHECK(a.dtype() == b.dtype()) << "TypeError: mismatched types";

  ObjectPtr<AndNode> node = make_object<AndNode>();
  node->dtype = DataType::Bool(a.dtype().lanes());
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.And").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return And(a, b, span);
});

TVM_REGISTER_NODE_TYPE(AndNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AndNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const AndNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " && ";
      p->Print(op->b);
      p->stream << ')';
    });

// Or
Or::Or(PrimExpr a, PrimExpr b, Span span) {
  ICHECK(a.defined()) << "ValueError: a is undefined";
  ICHECK(b.defined()) << "ValueError: b is undefined";
  ICHECK(a.dtype().is_bool());
  ICHECK(b.dtype().is_bool());
  ICHECK(a.dtype() == b.dtype()) << "TypeError: mismatched types";

  ObjectPtr<OrNode> node = make_object<OrNode>();
  node->dtype = DataType::Bool(a.dtype().lanes());
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.Or").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return Or(a, b, span);
});

TVM_REGISTER_NODE_TYPE(OrNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<OrNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const OrNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << " || ";
      p->Print(op->b);
      p->stream << ')';
    });

// Not
Not::Not(PrimExpr a, Span span) {
  ICHECK(a.defined()) << "ValueError: a is undefined";
  ICHECK(a.dtype().is_bool());

  ObjectPtr<NotNode> node = make_object<NotNode>();
  node->dtype = DataType::Bool(a.dtype().lanes());
  node->a = std::move(a);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.Not").set_body_typed([](PrimExpr a, Span span) { return Not(a, span); });

TVM_REGISTER_NODE_TYPE(NotNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<NotNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const NotNode*>(node.get());
      p->stream << '!';
      p->Print(op->a);
    });

// Select
Select::Select(PrimExpr condition, PrimExpr true_value, PrimExpr false_value, Span span) {
  ICHECK(condition.defined()) << "ValueError: condition is undefined";
  ICHECK(true_value.defined()) << "ValueError: true_value is undefined";
  ICHECK(false_value.defined()) << "ValueError: true_value is undefined";
  ICHECK(condition.dtype().is_bool());
  ICHECK(condition.dtype().lanes() == true_value.dtype().lanes() || condition.dtype().lanes() == 1);
  ICHECK(false_value.dtype() == true_value.dtype()) << "TypeError: mismatched types";

  ObjectPtr<SelectNode> node = make_object<SelectNode>();
  node->dtype = true_value.dtype();
  node->condition = std::move(condition);
  node->true_value = std::move(true_value);
  node->false_value = std::move(false_value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.Select")
    .set_body_typed([](PrimExpr condition, PrimExpr true_value, PrimExpr false_value, Span span) {
      return Select(condition, true_value, false_value, span);
    });

TVM_REGISTER_NODE_TYPE(SelectNode);

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

// Load
Load::Load(DataType dtype, Var buffer_var, PrimExpr index, PrimExpr predicate, Span span) {
  ICHECK(buffer_var.defined());
  ICHECK(predicate.defined());
  ICHECK(index.defined());

  // Assume that the array elements have 1 lane, unless a type
  // annotation tells us otherwise.
  int element_lanes = 1;
  auto pointer_type = tir::GetPointerType(buffer_var->type_annotation);
  if (pointer_type.first) {
    // Cannot check element type of array, as it may be different than
    // the loaded type in some cases.
    //
    // 1. Booleans use DataType::Int(8) while stored, and the codegens
    // handle cast to boolean.
    //
    // 2. The StorageRewrite pass can merge multiple allocations at
    // the same scope, regardless of element type.  The codegen is
    // then responsible for casting to the output type.

    // TODO(Lunderberg): Uncomment this check once it can be applied.
    // See https://discuss.tvm.apache.org/t/pre-rfc-vectorized-tir-buffers/10615
    // for discussion.

    // ICHECK(dtype.element_of() == pointer_type.second.element_of())
    //     << "Type mismatch, cannot load type " << dtype << " from buffer " <<
    //     buffer_var->name_hint
    //     << " of type " << pointer_type.second;
    element_lanes = pointer_type.second.lanes();
  }

  // The C-based codegens assume that all loads occur on a array with
  // non-vectorized elements, and cast between
  // vectorized/non-vectorized arrays as needed.  Ideally, these
  // should be changed to explicit casts in the TIR graph, rather than
  // being handled at the code-gen level.
  ICHECK((dtype.lanes() == element_lanes * index.dtype().lanes()) ||
         (dtype.lanes() == index.dtype().lanes()));
  ICHECK((dtype.lanes() == element_lanes * predicate.dtype().lanes()) ||
         (dtype.lanes() == index.dtype().lanes()));

  ObjectPtr<LoadNode> node = make_object<LoadNode>();
  node->dtype = dtype;
  node->buffer_var = std::move(buffer_var);
  node->index = std::move(index);
  node->predicate = std::move(predicate);
  node->span = std::move(span);

  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.Load").set_body([](TVMArgs args, TVMRetValue* ret) {
  DataType t = args[0];
  if (args.size() == 3) {
    *ret = Load(t, args[1], args[2], const_true(t.lanes()), Span());
  } else if (args.size() == 4) {
    *ret = Load(t, args[1], args[2], args[3], Span());
  } else {
    *ret = Load(t, args[1], args[2], args[3], args[4]);
  }
});

TVM_REGISTER_NODE_TYPE(LoadNode);

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

// Ramp
Ramp::Ramp(PrimExpr base, PrimExpr stride, int lanes, Span span) {
  ICHECK(base.defined());
  ICHECK(stride.defined());
  ICHECK(base.dtype().is_scalar());
  ICHECK(stride.dtype().is_scalar());
  ICHECK_GT(lanes, 1);
  ICHECK_EQ(stride.dtype(), base.dtype());

  ObjectPtr<RampNode> node = make_object<RampNode>();
  node->dtype = base.dtype().with_lanes(lanes);
  node->base = base;
  node->stride = stride;
  node->lanes = lanes;
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.Ramp")
    .set_body_typed([](PrimExpr base, PrimExpr stride, int lanes, Span span) {
      return Ramp(base, stride, lanes, span);
    });

TVM_REGISTER_NODE_TYPE(RampNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<RampNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const RampNode*>(node.get());
      p->stream << "ramp(";
      p->Print(op->base);
      p->stream << ", ";
      p->Print(op->stride);
      p->stream << ", " << op->lanes << ")";
    });

// Broadcast
Broadcast::Broadcast(PrimExpr value, int lanes, Span span) {
  ICHECK(value.defined());
  ICHECK(value.dtype().is_scalar());
  ICHECK_GT(lanes, 1);

  ObjectPtr<BroadcastNode> node = make_object<BroadcastNode>();
  node->dtype = value.dtype().with_lanes(lanes);
  node->value = std::move(value);
  node->lanes = lanes;
  node->span = std::move(span);
  data_ = node;
}

TVM_REGISTER_GLOBAL("tir.Broadcast").set_body_typed([](PrimExpr value, int lanes, Span span) {
  return Broadcast(value, lanes, span);
});

TVM_REGISTER_NODE_TYPE(BroadcastNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BroadcastNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const BroadcastNode*>(node.get());
      p->stream << "x" << op->lanes << "(";
      p->Print(op->value);
      p->stream << ")";
    });

// Let
Let::Let(Var var, PrimExpr value, PrimExpr body, Span span) {
  ICHECK(value.defined());
  ICHECK(body.defined());
  ICHECK_EQ(value.dtype(), var.dtype());

  ObjectPtr<LetNode> node = make_object<LetNode>();
  node->dtype = body.dtype();
  node->var = std::move(var);
  node->value = std::move(value);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.Let").set_body_typed([](Var var, PrimExpr value, PrimExpr body,
                                                 Span span) {
  return Let(var, value, body, span);
});

TVM_REGISTER_NODE_TYPE(LetNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<LetNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const LetNode*>(node.get());
      p->stream << "(let " << op->var << " = ";
      p->Print(op->value);
      p->stream << " in ";
      p->Print(op->body);
      p->stream << ")";
    });

// Call
Call::Call(DataType dtype, RelayExpr op, Array<PrimExpr> args, Span span) {
  for (size_t i = 0; i < args.size(); ++i) {
    ICHECK(args[i].defined());
  }

  ObjectPtr<CallNode> node = make_object<CallNode>();
  node->dtype = dtype;
  node->op = std::move(op);
  node->args = std::move(args);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.Call")
    .set_body_typed([](DataType type, RelayExpr op, Array<ObjectRef> args, Span span) {
      Array<PrimExpr> prim_expr_args;
      for (const auto& it : args) {
        ICHECK(it->IsInstance<runtime::StringObj>() || it->IsInstance<PrimExprNode>())
            << "Argument " << it << " is not a string or primexpr";
        if (const auto* str = it.as<runtime::StringObj>()) {
          prim_expr_args.push_back(StringImm(str->data));
        } else {
          prim_expr_args.push_back(Downcast<PrimExpr>(it));
        }
      }
      return Call(type, op, prim_expr_args, span);
    });

TVM_REGISTER_NODE_TYPE(CallNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<CallNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const CallNode*>(node.get());
      if (auto* ptr_op = op->op.as<OpNode>()) {
        p->stream << ptr_op->name << "(";
      } else {
        auto* ptr_gvar = op->op.as<GlobalVarNode>();
        ICHECK(ptr_gvar != nullptr);
        p->stream << "@" << ptr_gvar->name_hint << "(";
      }
      for (size_t i = 0; i < op->args.size(); ++i) {
        p->Print(op->args[i]);
        if (i < op->args.size() - 1) {
          p->stream << ", ";
        }
      }
      p->stream << ")";
    });

// Shuffle
Shuffle::Shuffle(Array<PrimExpr> vectors, Array<PrimExpr> indices, Span span) {
  ICHECK_NE(vectors.size(), 0U);
  ICHECK_NE(indices.size(), 0U);

  DataType base_type = vectors[0].dtype().element_of();
  int total_lanes = 0;

  for (PrimExpr val : vectors) {
    ICHECK(val.dtype().element_of() == base_type);
    total_lanes += val.dtype().lanes();
  }
  ICHECK_LE(indices.size(), static_cast<size_t>(total_lanes));

  ObjectPtr<ShuffleNode> node = make_object<ShuffleNode>();
  node->dtype = base_type.with_lanes(static_cast<int>(indices.size()));
  node->vectors = std::move(vectors);
  node->indices = std::move(indices);
  node->span = std::move(span);
  data_ = node;
}

PrimExpr Shuffle::Concat(Array<PrimExpr> vectors, Span span) {
  ICHECK_NE(vectors.size(), 0);
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
  return Shuffle(vectors, indices, span);
}

PrimExpr Shuffle::ExtractElement(PrimExpr vector, int index, Span span) {
  return Shuffle({vector}, {Integer(index)}, span);
}

TVM_REGISTER_GLOBAL("tir.Shuffle")
    .set_body_typed([](Array<PrimExpr> vectors, Array<PrimExpr> indices, Span span) {
      return Shuffle(vectors, indices, span);
    });

TVM_REGISTER_NODE_TYPE(ShuffleNode);

template <typename T>
void PrintList(const Array<T>& exprs, ReprPrinter* p) {
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

// CommReducer
CommReducer::CommReducer(Array<Var> lhs, Array<Var> rhs, Array<PrimExpr> result,
                         Array<PrimExpr> identity_element, Span span) {
  auto node = make_object<CommReducerNode>();
  node->lhs = lhs;
  node->rhs = rhs;
  node->result = result;
  node->identity_element = identity_element;
  node->span = std::move(span);
  data_ = std::move(node);
}

Array<PrimExpr> CommReducerNode::operator()(Array<PrimExpr> a, Array<PrimExpr> b) const {
  ICHECK_EQ(a.size(), b.size());
  ICHECK_EQ(lhs.size(), a.size());
  ICHECK_EQ(rhs.size(), b.size());
  Map<Var, PrimExpr> value_map;
  for (size_t i = 0; i < a.size(); ++i) {
    value_map.Set(lhs[i], a[i]);
    value_map.Set(rhs[i], b[i]);
  }
  auto ret = this->result;
  ret.MutateByApply([&value_map](const PrimExpr& e) { return Substitute(e, value_map); });
  return ret;
}

TVM_REGISTER_GLOBAL("tir.CommReducer")
    .set_body_typed([](Array<Var> lhs, Array<Var> rhs, Array<PrimExpr> result,
                       Array<PrimExpr> identity_element, Span span) {
      return CommReducer(lhs, rhs, result, identity_element, span);
    });

TVM_REGISTER_GLOBAL("tir.CommReducerCombine")
    .set_body_method<tir::CommReducer>(&tir::CommReducerNode::operator());

TVM_REGISTER_NODE_TYPE(CommReducerNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<CommReducerNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const CommReducerNode*>(node.get());
      p->stream << "comm_reducer(result=" << op->result << ", lhs=" << op->lhs
                << ", rhs=" << op->rhs << ", identity_element=" << op->identity_element << ")";
    });

// Reduce
Reduce::Reduce(CommReducer combiner, Array<PrimExpr> source, Array<IterVar> axis,
               PrimExpr condition, int value_index, Array<PrimExpr> init, Span span) {
  for (size_t i = 0; i < axis.size(); ++i) {
    ICHECK_EQ(axis[i]->iter_type, kCommReduce) << "Can only take axis created by reduce_axis";
  }
  if (!condition.defined()) {
    condition = const_true();
  }
  auto n = make_object<ReduceNode>();
  ICHECK(source.defined());
  for (size_t i = 0; i < axis.size(); ++i) {
    ICHECK(axis[i].defined());
  }
  if (!init.empty()) {
    ICHECK_EQ(init.size(), source.size()) << "Number of inits should match number of exprs";
    for (size_t i = 0; i < init.size(); i++) {
      ICHECK(init[i]->IsInstance<ProducerLoadNode>() || init[i]->IsInstance<IntImmNode>() ||
             init[i]->IsInstance<FloatImmNode>())
          << "init can only be a IntImm, FloatImm or ProducerLoad";
    }
  }
  n->dtype = source[value_index].dtype();
  n->combiner = std::move(combiner);
  n->source = std::move(source);
  n->init = std::move(init);
  n->axis = std::move(axis);
  n->condition = condition;
  n->value_index = value_index;
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("tir.Reduce")
    .set_body_typed([](CommReducer combiner, Array<PrimExpr> source, Array<IterVar> axis,
                       PrimExpr condition, int value_index, Array<PrimExpr> init, Span span) {
      return Reduce(combiner, source, axis, condition, value_index, init, span);
    });

TVM_REGISTER_NODE_TYPE(ReduceNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ReduceNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ReduceNode*>(node.get());
      p->stream << "reduce(combiner=" << op->combiner;
      p->stream << ", source=" << op->source;
      p->stream << ", init=" << op->init;
      p->stream << ", axis=" << op->axis;
      p->stream << ", where=" << op->condition;
      p->stream << ", value_index=" << op->value_index;
      p->stream << ")";
    });

// Any
Any::Any(Span span) {
  auto n = make_object<AnyNode>();
  n->dtype = DataType::Int(32);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("tir.Any").set_body_typed([](Span span) { return Any(span); });

TVM_REGISTER_NODE_TYPE(AnyNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AnyNode>([](const ObjectRef& node, ReprPrinter* p) { p->stream << "?"; });

// BufferLoad
BufferLoad::BufferLoad(Buffer buffer, Array<PrimExpr> indices, Span span) {
  ObjectPtr<BufferLoadNode> node = make_object<BufferLoadNode>();
  node->dtype = buffer->dtype;
  node->buffer = std::move(buffer);
  node->indices = std::move(indices);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.BufferLoad")
    .set_body_typed([](Buffer buffer, Array<PrimExpr> indices, Span span) {
      return BufferLoad(buffer, indices, span);
    });

TVM_REGISTER_NODE_TYPE(BufferLoadNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BufferLoadNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const BufferLoadNode*>(node.get());
      p->stream << op->buffer->name << "[";
      for (size_t i = 0; i < op->indices.size(); ++i) {
        p->Print(op->indices[i]);
        if (i < op->indices.size() - 1) {
          p->stream << ", ";
        }
      }
      p->stream << "]";
    });

// ProducerLoad
ProducerLoad::ProducerLoad(DataProducer producer, Array<PrimExpr> indices, Span span) {
  ObjectPtr<ProducerLoadNode> node = make_object<ProducerLoadNode>();
  node->dtype = producer->GetDataType();
  node->producer = std::move(producer);
  node->indices = std::move(indices);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.ProducerLoad")
    .set_body_typed([](DataProducer producer, Array<PrimExpr> indices, Span span) {
      return ProducerLoad(producer, indices, span);
    });

TVM_REGISTER_NODE_TYPE(ProducerLoadNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ProducerLoadNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ProducerLoadNode*>(node.get());
      p->stream << op->producer->GetNameHint() << "[";
      for (size_t i = 0; i < op->indices.size(); ++i) {
        p->Print(op->indices[i]);
        if (i < op->indices.size() - 1) {
          p->stream << ", ";
        }
      }
      p->stream << "]";
    });
}  // namespace tir
}  // namespace tvm
