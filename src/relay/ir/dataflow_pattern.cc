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
 * \file src/tvm/relay/dataflow_pattern.cc
 * \brief The dataflow pattern language for Relay.
 */
#include <tvm/relay/dataflow_pattern.h>
#include <tvm/runtime/data_type.h>

namespace tvm {
namespace relay {

ExprPattern::ExprPattern(Expr expr) {
  ObjectPtr<ExprPatternNode> n = make_object<ExprPatternNode>();
  n->expr = std::move(expr);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ExprPatternNode);

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.ExprPattern").set_body_typed([](Expr e) {
  return ExprPattern(e);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ExprPatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const ExprPatternNode*>(ref.get());
      p->Print(node->expr);
    });

VarPattern::VarPattern(String name_hint) {
  ObjectPtr<VarPatternNode> n = make_object<VarPatternNode>();
  n->name = std::move(name_hint);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(VarPatternNode);

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.VarPattern").set_body_typed([](String name_hint) {
  return VarPattern(name_hint);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<VarPatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const VarPatternNode*>(ref.get());
      p->stream << "VarPattern(" << node->name_hint() << ")";
    });

TVM_REGISTER_NODE_TYPE(ConstantPatternNode);

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.ConstantPattern").set_body_typed([]() {
  auto c = ConstantPattern(make_object<ConstantPatternNode>());
  return c;
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ConstantPatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
      p->stream << "ConstantPattern()";
    });

CallPattern::CallPattern(DFPattern op, Array<DFPattern> args) {
  ObjectPtr<CallPatternNode> n = make_object<CallPatternNode>();
  n->op = std::move(op);
  n->args = std::move(args);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(CallPatternNode);

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.CallPattern")
    .set_body_typed([](DFPattern op, Array<DFPattern> args) { return CallPattern(op, args); });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<CallPatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const CallPatternNode*>(ref.get());
      p->stream << "CallPatternNode(" << node->op << ", " << node->args << ")";
    });

FunctionPattern::FunctionPattern(Array<DFPattern> params, DFPattern body) {
  ObjectPtr<FunctionPatternNode> n = make_object<FunctionPatternNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  data_ = std::move(n);
}
TVM_REGISTER_NODE_TYPE(FunctionPatternNode);

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.FunctionPattern")
    .set_body_typed([](Array<DFPattern> params, DFPattern body) {
      return FunctionPattern(params, body);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<FunctionPatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const FunctionPatternNode*>(ref.get());
      p->stream << "FunctionPatternNode(" << node->params << ", " << node->body << ")";
    });

LetPattern::LetPattern(DFPattern var, DFPattern value, DFPattern body) {
  ObjectPtr<LetPatternNode> n = make_object<LetPatternNode>();
  n->var = std::move(var);
  n->value = std::move(value);
  n->body = std::move(body);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(LetPatternNode);

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.LetPattern")
    .set_body_typed([](DFPattern var, DFPattern value, DFPattern body) {
      return LetPattern(var, value, body);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<LetPatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const LetPatternNode*>(ref.get());
      p->stream << "LetPatternNode(" << node->var << ", " << node->value << ", " << node->body
                << ")";
    });

IfPattern::IfPattern(DFPattern cond, DFPattern true_branch, DFPattern false_branch) {
  ObjectPtr<IfPatternNode> n = make_object<IfPatternNode>();
  n->cond = std::move(cond);
  n->true_branch = std::move(true_branch);
  n->false_branch = std::move(false_branch);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(IfPatternNode);

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.IfPattern")
    .set_body_typed([](DFPattern cond, DFPattern true_branch, DFPattern false_branch) {
      return IfPattern(cond, true_branch, false_branch);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IfPatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const IfPatternNode*>(ref.get());
      p->stream << "IfPattern(" << node->cond << ", " << node->true_branch << ", "
                << node->false_branch << ")";
    });

TuplePattern::TuplePattern(tvm::Array<DFPattern> fields) {
  ObjectPtr<TuplePatternNode> n = make_object<TuplePatternNode>();
  n->fields = std::move(fields);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TuplePatternNode);

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.TuplePattern")
    .set_body_typed([](tvm::Array<DFPattern> fields) { return TuplePattern(fields); });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TuplePatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const TuplePatternNode*>(ref.get());
      p->stream << "TuplePattern(" << node->fields << ")";
    });

TupleGetItemPattern::TupleGetItemPattern(DFPattern tuple, int index) {
  ObjectPtr<TupleGetItemPatternNode> n = make_object<TupleGetItemPatternNode>();
  n->tuple = std::move(tuple);
  n->index = index;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TupleGetItemPatternNode);

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.TupleGetItemPattern")
    .set_body_typed([](DFPattern tuple, int index) { return TupleGetItemPattern(tuple, index); });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TupleGetItemPatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const TupleGetItemPatternNode*>(ref.get());
      p->stream << "TupleGetItemPatternNode(" << node->tuple << ", " << node->index << ")";
    });

AltPattern::AltPattern(DFPattern left, DFPattern right) {
  ObjectPtr<AltPatternNode> n = make_object<AltPatternNode>();
  n->left = std::move(left);
  n->right = std::move(right);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(AltPatternNode);

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.AltPattern")
    .set_body_typed([](DFPattern left, DFPattern right) { return AltPattern(left, right); });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AltPatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const AltPatternNode*>(ref.get());
      p->stream << "AltPattern(" << node->left << " | " << node->right << ")";
    });

TVM_REGISTER_NODE_TYPE(WildcardPatternNode);

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.WildcardPattern").set_body_typed([]() {
  auto w = WildcardPattern(make_object<WildcardPatternNode>());
  return w;
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<WildcardPatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
      p->stream << "*";
    });

TypePattern::TypePattern(DFPattern pattern, Type type) {
  ObjectPtr<TypePatternNode> n = make_object<TypePatternNode>();
  n->pattern = std::move(pattern);
  n->type = std::move(type);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TypePatternNode);

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.TypePattern")
    .set_body_typed([](DFPattern pattern, Type type) { return TypePattern(pattern, type); });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TypePatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const TypePatternNode*>(ref.get());
      p->stream << "TypePattern(" << node->pattern << " has type " << node->type << ")";
    });

ShapePattern::ShapePattern(DFPattern pattern, Array<PrimExpr> shape) {
  ObjectPtr<ShapePatternNode> n = make_object<ShapePatternNode>();
  n->pattern = std::move(pattern);
  n->shape = std::move(shape);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ShapePatternNode);

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.ShapePattern")
    .set_body_typed([](DFPattern pattern, Array<PrimExpr> shape) {
      return ShapePattern(pattern, shape);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ShapePatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const ShapePatternNode*>(ref.get());
      p->stream << "ShapePattern(" << node->pattern << " has shape " << node->shape << ")";
    });

DataTypePattern::DataTypePattern(DFPattern pattern, DataType dtype) {
  ObjectPtr<DataTypePatternNode> n = make_object<DataTypePatternNode>();
  n->pattern = std::move(pattern);
  n->dtype = std::move(dtype);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DataTypePatternNode);

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.DataTypePattern")
    .set_body_typed([](DFPattern pattern, DataType dtype) {
      return DataTypePattern(pattern, dtype);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<DataTypePatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const DataTypePatternNode*>(ref.get());
      p->stream << "TypePattern(" << node->pattern << " has dtype " << node->dtype << ")";
    });

AttrPattern::AttrPattern(DFPattern pattern, DictAttrs attrs) {
  ObjectPtr<AttrPatternNode> n = make_object<AttrPatternNode>();
  n->pattern = std::move(pattern);
  n->attrs = std::move(attrs);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(AttrPatternNode);

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.AttrPattern")
    .set_body_typed([](DFPattern pattern, DictAttrs attrs) { return AttrPattern(pattern, attrs); });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AttrPatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const AttrPatternNode*>(ref.get());
      p->stream << "AttrPattern(" << node->pattern << " has attributes " << node->attrs << ")";
    });

DominatorPattern::DominatorPattern(DFPattern parent, DFPattern path, DFPattern child) {
  ObjectPtr<DominatorPatternNode> n = make_object<DominatorPatternNode>();
  n->parent = std::move(parent);
  n->path = std::move(path);

  n->child = std::move(child);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DominatorPatternNode);

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.DominatorPattern")
    .set_body_typed([](DFPattern parent, DFPattern path, DFPattern child) {
      return DominatorPattern(parent, path, child);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<DominatorPatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const DominatorPatternNode*>(ref.get());
      p->stream << "DominatorPattern(" << node->parent << ", " << node->path << ", " << node->child
                << ")";
    });

// Syntatic Sugar
DFPattern DFPattern::operator()(const std::vector<DFPattern>& args) {
  return CallPattern(GetRef<DFPattern>(this->get()), Array<DFPattern>(args));
}
DFPattern DFPattern::operator+(const DFPattern& other) {
  return IsOp("add")({GetRef<DFPattern>(this->get()), other});
}
DFPattern DFPattern::operator-(const DFPattern& other) {
  return IsOp("subtract")({GetRef<DFPattern>(this->get()), other});
}
DFPattern DFPattern::operator*(const DFPattern& other) {
  return IsOp("multiply")({GetRef<DFPattern>(this->get()), other});
}
DFPattern DFPattern::operator/(const DFPattern& other) {
  return IsOp("divide")({GetRef<DFPattern>(this->get()), other});
}
DFPattern DFPattern::operator||(const DFPattern& other) {
  return AltPattern(GetRef<DFPattern>(this->get()), other);
}

DFPattern DFPattern::HasAttr(const Map<String, ObjectRef>& attrs) {
  return AttrPattern(GetRef<DFPattern>(this->get()), DictAttrs(attrs));
}
DFPattern DFPattern::HasType(const Type& type) {
  return TypePattern(GetRef<DFPattern>(this->get()), type);
}
DFPattern DFPattern::HasDtype(const DataType& dtype) {
  return DataTypePattern(GetRef<DFPattern>(this->get()), dtype);
}
DFPattern DFPattern::HasDtype(const std::string& dtype) {
  return HasDtype(DataType(runtime::String2DLDataType(dtype)));
}
DFPattern DFPattern::HasShape(const Array<PrimExpr> shape) {
  return ShapePattern(GetRef<DFPattern>(this->get()), shape);
}
DFPattern IsVar(const String& name) { return VarPattern(name); }
DFPattern IsConstant() { return ConstantPattern(make_object<ConstantPatternNode>()); }
DFPattern IsWildcard() { return WildcardPattern(make_object<WildcardPatternNode>()); }
DFPattern IsExpr(const Expr& expr) { return ExprPattern(expr); }
DFPattern IsOp(const String& op_name) { return IsExpr(Op::Get(op_name)); }
DFPattern IsTuple(const Array<DFPattern>& fields) { return TuplePattern(fields); }
DFPattern IsTupleGetItem(const DFPattern tuple, int index) {
  return TupleGetItemPattern(tuple, index);
}

}  // namespace relay
}  // namespace tvm
