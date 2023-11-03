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

DFPatternPrinter::FType& DFPatternPrinter::vtable() {
  static FType inst;
  return inst;
}

String PrettyPrint(const DFPattern& pattern) {
  std::stringstream string_stream{};
  string_stream << pattern;
  return string_stream.str();
}

void DFPatternPrinter::Print(const ObjectRef& node) {
  ICHECK(node.as<DFPatternNode>());
  DFPattern pat = Downcast<DFPattern>(node);
  static const FType& f = vtable();
  string_stream.str("");
  if (!node.defined()) {
    string_stream << "(nullptr)";
  } else if (memo_.find(pat) != memo_.end()) {
    string_stream << "(invoke pattern id " << memo_[pat].first << ")";
    auxiliary_patterns.push_back(pat);
  } else {
    if (f.can_dispatch(node)) {
      memo_.insert({pat, {memo_.size(), ""}});
      f(node, this);
      memo_[pat].second = string_stream.str();
    } else {
      // default value, output type key and addr.
      string_stream << node->GetTypeKey() << "(" << node.get() << ")";
    }
  }
}

ExprPattern::ExprPattern(Expr expr) {
  ObjectPtr<ExprPatternNode> n = make_object<ExprPatternNode>();
  n->expr = std::move(expr);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ExprPatternNode);

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.ExprPattern").set_body_typed([](Expr e) {
  return ExprPattern(e);
});

TVM_STATIC_IR_FUNCTOR(DFPatternPrinter, vtable)
    .set_dispatch<ExprPatternNode>([](const ObjectRef& ref, DFPatternPrinter* p) {
      ExprPattern pattern = Downcast<ExprPattern>(ref);
      p->string_stream.str("");
      p->string_stream << pattern->expr;
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

TVM_STATIC_IR_FUNCTOR(DFPatternPrinter, vtable)
    .set_dispatch<VarPatternNode>([](const ObjectRef& ref, DFPatternPrinter* p) {
      VarPattern pattern = Downcast<VarPattern>(ref);
      p->string_stream.str("");
      p->string_stream << "VarPattern(" << pattern->name_hint() << ")";
    });

TVM_REGISTER_NODE_TYPE(ConstantPatternNode);

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.ConstantPattern").set_body_typed([]() {
  auto c = ConstantPattern(make_object<ConstantPatternNode>());
  return c;
});

TVM_STATIC_IR_FUNCTOR(DFPatternPrinter, vtable)
    .set_dispatch<ConstantPatternNode>([](const ObjectRef& ref, DFPatternPrinter* p) {
      p->string_stream.str("");
      p->string_stream << "ConstantPattern()";
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

TVM_STATIC_IR_FUNCTOR(DFPatternPrinter, vtable)
    .set_dispatch<CallPatternNode>([](const ObjectRef& ref, DFPatternPrinter* p) {
      CallPattern pattern = Downcast<CallPattern>(ref);

      p->Print(pattern->op);
      std::string op_pattern_string{p->string_stream.str()};

      std::vector<std::string> args_pattern_string{};
      for (const DFPattern& arg : pattern->args) {
        p->Print(arg);
        args_pattern_string.push_back(p->string_stream.str());
      }

      p->string_stream.str("");
      p->string_stream << "(id " << p->memo_[pattern].first << "): ";
      p->string_stream << "CallPatternNode(" << op_pattern_string << ", [";
      for (size_t i = 0; i < args_pattern_string.size(); ++i) {
        if (i != 0) {
          p->string_stream << ", ";
        }
        p->string_stream << args_pattern_string[i];
      }
      p->string_stream << "])";
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

TVM_STATIC_IR_FUNCTOR(DFPatternPrinter, vtable)
    .set_dispatch<FunctionPatternNode>([](const ObjectRef& ref, DFPatternPrinter* p) {
      FunctionPattern pattern = Downcast<FunctionPattern>(ref);

      std::vector<std::string> params_pattern_string{};
      for (const DFPattern& param : pattern->params) {
        p->Print(param);
        params_pattern_string.push_back(p->string_stream.str());
      }

      p->Print(pattern->body);
      std::string body_pattern_string{p->string_stream.str()};

      p->string_stream.str("");
      p->string_stream << "(id " << p->memo_[pattern].first << "): ";

      p->string_stream << "FunctionPatternNode([";
      for (size_t i = 0; i < params_pattern_string.size(); ++i) {
        if (i != 0) {
          p->string_stream << ", ";
        }
        p->string_stream << params_pattern_string[i];
      }
      p->string_stream << "]";
      p->string_stream << ", " << body_pattern_string << ")";
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

TVM_STATIC_IR_FUNCTOR(DFPatternPrinter, vtable)
    .set_dispatch<LetPatternNode>([](const ObjectRef& ref, DFPatternPrinter* p) {
      LetPattern pattern = Downcast<LetPattern>(ref);

      p->Print(pattern->var);
      std::string var_pattern_string{p->string_stream.str()};

      p->Print(pattern->value);
      std::string value_pattern_string{p->string_stream.str()};

      p->Print(pattern->body);
      std::string body_pattern_string{p->string_stream.str()};

      p->string_stream.str("");
      p->string_stream << "(id " << p->memo_[pattern].first << "): ";
      p->string_stream << "LetPatternNode(" << var_pattern_string << ", " << value_pattern_string
                       << ", " << body_pattern_string << ")";
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

TVM_STATIC_IR_FUNCTOR(DFPatternPrinter, vtable)
    .set_dispatch<IfPatternNode>([](const ObjectRef& ref, DFPatternPrinter* p) {
      IfPattern pattern = Downcast<IfPattern>(ref);

      p->Print(pattern->cond);
      std::string cond_pattern_string{p->string_stream.str()};

      p->Print(pattern->true_branch);
      std::string true_branch_pattern_string{p->string_stream.str()};

      p->Print(pattern->false_branch);
      std::string false_branch_pattern_string{p->string_stream.str()};

      p->string_stream.str("");
      p->string_stream << "(id " << p->memo_[pattern].first << "): ";
      p->string_stream << "IfPattern(" << cond_pattern_string << ", " << true_branch_pattern_string
                       << ", " << false_branch_pattern_string << ")";
    });

TuplePattern::TuplePattern(tvm::Array<DFPattern> fields) {
  ObjectPtr<TuplePatternNode> n = make_object<TuplePatternNode>();
  n->fields = std::move(fields);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TuplePatternNode);

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.TuplePattern")
    .set_body_typed([](tvm::Array<DFPattern> fields) { return TuplePattern(fields); });

TVM_STATIC_IR_FUNCTOR(DFPatternPrinter, vtable)
    .set_dispatch<TuplePatternNode>([](const ObjectRef& ref, DFPatternPrinter* p) {
      TuplePattern pattern = Downcast<TuplePattern>(ref);

      std::vector<std::string> fields_pattern_string{};
      for (const DFPattern& field : pattern->fields) {
        p->Print(field);
        fields_pattern_string.push_back(p->string_stream.str());
      }

      p->string_stream.str("");
      p->string_stream << "(id " << p->memo_[pattern].first << "): ";
      p->string_stream << "TuplePattern(";
      p->string_stream << "[";
      for (size_t i = 0; i < fields_pattern_string.size(); ++i) {
        if (i != 0) {
          p->string_stream << ", ";
        }
        p->string_stream << fields_pattern_string[i];
      }
      p->string_stream << "]";
      p->string_stream << ")";
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

TVM_STATIC_IR_FUNCTOR(DFPatternPrinter, vtable)
    .set_dispatch<TupleGetItemPatternNode>([](const ObjectRef& ref, DFPatternPrinter* p) {
      TupleGetItemPattern pattern = Downcast<TupleGetItemPattern>(ref);

      p->Print(pattern->tuple);
      std::string tuple_pattern_string{p->string_stream.str()};

      p->string_stream.str("");
      p->string_stream << "(id " << p->memo_[pattern].first << "): ";
      p->string_stream << "TupleGetItemPatternNode(";
      p->string_stream << tuple_pattern_string << ", " << pattern->index << ")";
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

TVM_STATIC_IR_FUNCTOR(DFPatternPrinter, vtable)
    .set_dispatch<AltPatternNode>([](const ObjectRef& ref, DFPatternPrinter* p) {
      AltPattern pattern = Downcast<AltPattern>(ref);

      p->Print(pattern->left);
      std::string left_pattern_string{p->string_stream.str()};

      p->Print(pattern->right);
      std::string right_pattern_string{p->string_stream.str()};

      p->string_stream.str("");
      p->string_stream << "(id " << p->memo_[pattern].first << "): ";
      p->string_stream << "AltPattern(" << left_pattern_string << " | " << right_pattern_string
                       << ")";
    });

void WildcardPattern::redirect_to(DFPattern pat) const {
  WildcardPatternNode* ptr = static_cast<WildcardPatternNode*>(get_mutable());
  ptr->pattern = pat;
}

TVM_REGISTER_NODE_TYPE(WildcardPatternNode);

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.WildcardPattern_redirect_to")
    .set_body_typed([](WildcardPattern wildcard, DFPattern pat) {
      return wildcard.redirect_to(pat);
    });

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.WildcardPattern").set_body_typed([]() {
  auto w = WildcardPattern(make_object<WildcardPatternNode>());
  return w;
});

TVM_STATIC_IR_FUNCTOR(DFPatternPrinter, vtable)
    .set_dispatch<WildcardPatternNode>([](const ObjectRef& ref, DFPatternPrinter* p) {
      p->string_stream.str("");
      p->string_stream << "*";
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

TVM_STATIC_IR_FUNCTOR(DFPatternPrinter, vtable)
    .set_dispatch<TypePatternNode>([](const ObjectRef& ref, DFPatternPrinter* p) {
      TypePattern pattern = Downcast<TypePattern>(ref);

      p->Print(pattern->pattern);
      std::string pattern_pattern_string{p->string_stream.str()};

      p->string_stream.str("");
      p->string_stream << "(id " << p->memo_[pattern].first << "): ";
      p->string_stream << "TypePattern(" << pattern_pattern_string << " has type " << pattern->type
                       << ")";
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

TVM_STATIC_IR_FUNCTOR(DFPatternPrinter, vtable)
    .set_dispatch<ShapePatternNode>([](const ObjectRef& ref, DFPatternPrinter* p) {
      ShapePattern pattern = Downcast<ShapePattern>(ref);

      p->Print(pattern->pattern);
      std::string pattern_pattern_string{p->string_stream.str()};

      p->string_stream.str("");
      p->string_stream << "(id " << p->memo_[pattern].first << "): ";
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

TVM_STATIC_IR_FUNCTOR(DFPatternPrinter, vtable)
    .set_dispatch<DataTypePatternNode>([](const ObjectRef& ref, DFPatternPrinter* p) {
      DataTypePattern pattern = Downcast<DataTypePattern>(ref);

      p->Print(pattern->pattern);
      std::string pattern_pattern_string{p->string_stream.str()};

      p->string_stream.str("");
      p->string_stream << "(id " << p->memo_[pattern].first << "): ";
      p->string_stream << "DataTypePattern(" << pattern_pattern_string << " has dtype "
                       << pattern->dtype << ")";
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

TVM_STATIC_IR_FUNCTOR(DFPatternPrinter, vtable)
    .set_dispatch<AttrPatternNode>([](const ObjectRef& ref, DFPatternPrinter* p) {
      AttrPattern pattern = Downcast<AttrPattern>(ref);

      p->Print(pattern->pattern);
      std::string pattern_pattern_string{p->string_stream.str()};

      p->string_stream.str("");
      p->string_stream << "(id " << p->memo_[pattern].first << "): ";
      p->string_stream << "AttrPattern(" << pattern_pattern_string << " has attributes "
                       << pattern->attrs << ")";
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

TVM_STATIC_IR_FUNCTOR(DFPatternPrinter, vtable)
    .set_dispatch<DominatorPatternNode>([](const ObjectRef& ref, DFPatternPrinter* p) {
      DominatorPattern pattern = Downcast<DominatorPattern>(ref);

      p->Print(pattern->parent);
      std::string parent_pattern_string{p->string_stream.str()};

      p->Print(pattern->path);
      std::string path_pattern_string{p->string_stream.str()};

      p->Print(pattern->child);
      std::string child_pattern_string{p->string_stream.str()};

      p->string_stream.str("");
      p->string_stream << "(id " << p->memo_[pattern].first << "): ";
      p->string_stream << "DominatorPattern(" << parent_pattern_string << ", "
                       << path_pattern_string << ", " << child_pattern_string << ")";
    });

// Syntatic Sugar
DFPattern DFPattern::operator()(const std::vector<DFPattern>& args) const {
  return CallPattern(GetRef<DFPattern>(this->get()), Array<DFPattern>(args));
}
DFPattern DFPattern::operator+(const DFPattern& other) const {
  return IsOp("add")({GetRef<DFPattern>(this->get()), other});
}
DFPattern DFPattern::operator-(const DFPattern& other) const {
  return IsOp("subtract")({GetRef<DFPattern>(this->get()), other});
}
DFPattern DFPattern::operator*(const DFPattern& other) const {
  return IsOp("multiply")({GetRef<DFPattern>(this->get()), other});
}
DFPattern DFPattern::operator/(const DFPattern& other) const {
  return IsOp("divide")({GetRef<DFPattern>(this->get()), other});
}
DFPattern DFPattern::operator||(const DFPattern& other) const {
  return AltPattern(GetRef<DFPattern>(this->get()), other);
}

DFPattern DFPattern::Optional(const std::function<DFPattern(const DFPattern&)>& func) const {
  DFPattern current = GetRef<DFPattern>(this->get());
  return current || func(current);
}

DFPattern DFPattern::HasAttr(const Map<String, ObjectRef>& attrs) const {
  return AttrPattern(GetRef<DFPattern>(this->get()), DictAttrs(attrs));
}
DFPattern DFPattern::HasType(const Type& type) const {
  return TypePattern(GetRef<DFPattern>(this->get()), type);
}
DFPattern DFPattern::HasDtype(const DataType& dtype) const {
  return DataTypePattern(GetRef<DFPattern>(this->get()), dtype);
}
DFPattern DFPattern::HasDtype(const std::string& dtype) const {
  return HasDtype(DataType(runtime::String2DLDataType(dtype)));
}
DFPattern DFPattern::HasShape(const Array<PrimExpr> shape) const {
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
