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
 * \file src/relax/ir/dataflow_pattern.cc
 * \brief The dataflow pattern language for Relax (inherited from Relay).
 */

#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/dataflow_pattern_functor.h>

#include <memory>
#include <stack>
#include <string>

#define RELAX_PATTERN_PRINTER_DEF(NODE_TYPE, REPR_LAMBDA)                 \
  TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)                              \
      .set_dispatch<NODE_TYPE>([](const ObjectRef& ref, ReprPrinter* p) { \
        auto* node = static_cast<const NODE_TYPE*>(ref.get());            \
        REPR_LAMBDA(p, node);                                             \
      })

namespace tvm {
namespace relax {

TVM_REGISTER_NODE_TYPE(ExternFuncPatternNode);
ExternFuncPattern::ExternFuncPattern(String global_symbol) {
  ObjectPtr<ExternFuncPatternNode> n = make_object<ExternFuncPatternNode>();
  n->global_symbol_ = std::move(global_symbol);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dpl.ExternFuncPattern").set_body_typed([](String global_symbol) {
  return ExternFuncPattern(global_symbol);
});
RELAX_PATTERN_PRINTER_DEF(ExternFuncPatternNode, [](auto p, auto node) {
  p->stream << "ExternFuncPattern(" << node->global_symbol() << ")";
});

TVM_REGISTER_NODE_TYPE(VarPatternNode);
VarPattern::VarPattern(String name_hint) {
  ObjectPtr<VarPatternNode> n = make_object<VarPatternNode>();
  n->name = std::move(name_hint);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dpl.VarPattern").set_body_typed([](String name_hint) {
  return VarPattern(name_hint);
});
RELAX_PATTERN_PRINTER_DEF(VarPatternNode, [](auto p, auto node) {
  p->stream << "VarPattern(" << node->name_hint() << ")";
});

TVM_REGISTER_NODE_TYPE(DataflowVarPatternNode);
TVM_REGISTER_GLOBAL("relax.dpl.DataflowVarPattern").set_body_typed([](String name_hint) {
  return DataflowVarPattern(name_hint);
});
DataflowVarPattern::DataflowVarPattern(String name_hint) {
  ObjectPtr<DataflowVarPatternNode> n = make_object<DataflowVarPatternNode>();
  n->name = std::move(name_hint);
  data_ = std::move(n);
}
RELAX_PATTERN_PRINTER_DEF(DataflowVarPatternNode, [](auto p, auto node) {
  p->stream << "DataflowVarPattern(" << node->name_hint() << ")";
});

TVM_REGISTER_NODE_TYPE(GlobalVarPatternNode);
GlobalVarPattern::GlobalVarPattern(String name_hint) {
  ObjectPtr<GlobalVarPatternNode> n = make_object<GlobalVarPatternNode>();
  n->name = std::move(name_hint);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dpl.GlobalVarPattern").set_body_typed([](String name_hint) {
  return GlobalVarPattern(name_hint);
});
RELAX_PATTERN_PRINTER_DEF(GlobalVarPatternNode, [](auto p, auto node) {
  p->stream << "GlobalVarPattern(" << node->name_hint() << ")";
});

TVM_REGISTER_NODE_TYPE(ExprPatternNode);
ExprPattern::ExprPattern(Expr expr) {
  ObjectPtr<ExprPatternNode> n = make_object<ExprPatternNode>();
  n->expr = std::move(expr);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dpl.ExprPattern").set_body_typed([](Expr e) { return ExprPattern(e); });
RELAX_PATTERN_PRINTER_DEF(ExprPatternNode, [](auto p, auto node) { p->Print(node->expr); });

TVM_REGISTER_NODE_TYPE(ConstantPatternNode);
TVM_REGISTER_GLOBAL("relax.dpl.ConstantPattern").set_body_typed([]() {
  auto c = ConstantPattern(make_object<ConstantPatternNode>());
  return c;
});
RELAX_PATTERN_PRINTER_DEF(ConstantPatternNode,
                          [](auto p, auto node) { p->stream << "ConstantPattern()"; });

TVM_REGISTER_NODE_TYPE(CallPatternNode);
CallPattern::CallPattern(DFPattern op, Array<DFPattern> args, bool varg_default_wildcard) {
  ObjectPtr<CallPatternNode> n = make_object<CallPatternNode>();
  n->op = std::move(op);
  n->args = std::move(args);
  n->varg_default_wildcard = varg_default_wildcard;
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dpl.CallPattern")
    .set_body_typed([](DFPattern op, Array<DFPattern> args, bool varg_default_wildcard) {
      return CallPattern(op, args, varg_default_wildcard);
    });
RELAX_PATTERN_PRINTER_DEF(CallPatternNode, [](auto p, auto node) {
  p->stream << node->op << "(";
  for (size_t i = 0; i < node->args.size(); ++i) {
    if (i != 0) p->stream << ", ";
    p->stream << node->args[i];
  }
  if (node->varg_default_wildcard) {
    if (node->args.size() != 0) p->stream << ", ";
    p->stream << "...";
  }
  p->stream << ")";
});

TVM_REGISTER_NODE_TYPE(PrimArrPatternNode);
PrimArrPattern::PrimArrPattern(Array<PrimExpr> arr) {
  ObjectPtr<PrimArrPatternNode> n = make_object<PrimArrPatternNode>();
  n->fields = std::move(arr);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dpl.PrimArrPattern").set_body_typed([](Array<PrimExpr> arr) {
  return PrimArrPattern(std::move(arr));
});
RELAX_PATTERN_PRINTER_DEF(PrimArrPatternNode, [](auto p, auto node) {
  p->stream << "PrimArrPattern(" << node->fields << ")";
});

TVM_REGISTER_NODE_TYPE(FunctionPatternNode);
FunctionPattern::FunctionPattern(Array<DFPattern> params, DFPattern body) {
  ObjectPtr<FunctionPatternNode> n = make_object<FunctionPatternNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dpl.FunctionPattern")
    .set_body_typed([](Array<DFPattern> params, DFPattern body) {
      return FunctionPattern(params, body);
    });
RELAX_PATTERN_PRINTER_DEF(FunctionPatternNode, [](auto p, auto node) {
  p->stream << "FunctionPattern(" << node->params << ", " << node->body << ")";
});

TVM_REGISTER_NODE_TYPE(TuplePatternNode);
TuplePattern::TuplePattern(tvm::Array<DFPattern> fields) {
  ObjectPtr<TuplePatternNode> n = make_object<TuplePatternNode>();
  n->fields = std::move(fields);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dpl.TuplePattern").set_body_typed([](tvm::Array<DFPattern> fields) {
  return TuplePattern(fields);
});
RELAX_PATTERN_PRINTER_DEF(TuplePatternNode, [](auto p, auto node) {
  p->stream << "TuplePattern(" << node->fields << ")";
});

TVM_REGISTER_NODE_TYPE(UnorderedTuplePatternNode);
UnorderedTuplePattern::UnorderedTuplePattern(tvm::Array<DFPattern> fields) {
  ObjectPtr<UnorderedTuplePatternNode> n = make_object<UnorderedTuplePatternNode>();
  n->fields = std::move(fields);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dpl.UnorderedTuplePattern")
    .set_body_typed([](tvm::Array<DFPattern> fields) { return UnorderedTuplePattern(fields); });
RELAX_PATTERN_PRINTER_DEF(UnorderedTuplePatternNode, [](auto p, auto node) {
  p->stream << "UnorderedTuplePattern(" << node->fields << ")";
});

TVM_REGISTER_NODE_TYPE(TupleGetItemPatternNode);
TupleGetItemPattern::TupleGetItemPattern(DFPattern tuple, int index) {
  ObjectPtr<TupleGetItemPatternNode> n = make_object<TupleGetItemPatternNode>();
  n->tuple = std::move(tuple);
  n->index = index;
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dpl.TupleGetItemPattern").set_body_typed([](DFPattern tuple, int index) {
  return TupleGetItemPattern(tuple, index);
});
RELAX_PATTERN_PRINTER_DEF(TupleGetItemPatternNode, [](auto p, auto node) {
  p->stream << "TupleGetItemPattern(" << node->tuple << ", " << node->index << ")";
});

TVM_REGISTER_NODE_TYPE(AndPatternNode);
AndPattern::AndPattern(DFPattern left, DFPattern right) {
  ObjectPtr<AndPatternNode> n = make_object<AndPatternNode>();
  n->left = std::move(left);
  n->right = std::move(right);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dpl.AndPattern").set_body_typed([](DFPattern left, DFPattern right) {
  return AndPattern(left, right);
});
RELAX_PATTERN_PRINTER_DEF(AndPatternNode, [](auto p, auto node) {
  p->stream << "AndPattern(" << node->left << " & " << node->right << ")";
});

TVM_REGISTER_NODE_TYPE(OrPatternNode);
OrPattern::OrPattern(DFPattern left, DFPattern right) {
  ObjectPtr<OrPatternNode> n = make_object<OrPatternNode>();
  n->left = std::move(left);
  n->right = std::move(right);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dpl.OrPattern").set_body_typed([](DFPattern left, DFPattern right) {
  return OrPattern(left, right);
});
RELAX_PATTERN_PRINTER_DEF(OrPatternNode, [](auto p, auto node) {
  p->stream << "OrPattern(" << node->left << " | " << node->right << ")";
});

TVM_REGISTER_NODE_TYPE(NotPatternNode);
NotPattern::NotPattern(DFPattern reject) {
  ObjectPtr<NotPatternNode> n = make_object<NotPatternNode>();
  n->reject = std::move(reject);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dpl.NotPattern").set_body_typed([](DFPattern reject) {
  return NotPattern(reject);
});
RELAX_PATTERN_PRINTER_DEF(NotPatternNode,
                          [](auto p, auto node) { p->stream << "!(" << node->reject << ")"; });

TVM_REGISTER_NODE_TYPE(WildcardPatternNode);
WildcardPattern::WildcardPattern() { data_ = make_object<WildcardPatternNode>(); }
TVM_REGISTER_GLOBAL("relax.dpl.WildcardPattern").set_body_typed([]() { return WildcardPattern(); });
RELAX_PATTERN_PRINTER_DEF(WildcardPatternNode, [](auto p, auto node) { p->stream << "*"; });

TVM_REGISTER_NODE_TYPE(TypePatternNode);
TypePattern::TypePattern(DFPattern pattern, Type type) {
  ObjectPtr<TypePatternNode> n = make_object<TypePatternNode>();
  n->pattern = std::move(pattern);
  n->type = std::move(type);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dpl.TypePattern").set_body_typed([](DFPattern pattern, Type type) {
  return TypePattern(pattern, type);
});
RELAX_PATTERN_PRINTER_DEF(TypePatternNode, [](auto p, auto node) {
  p->stream << "TypePattern(" << node->pattern << " has type " << node->type << ")";
});

TVM_REGISTER_NODE_TYPE(StructInfoPatternNode);
StructInfoPattern::StructInfoPattern(DFPattern pattern, StructInfo struct_info) {
  ObjectPtr<StructInfoPatternNode> n = make_object<StructInfoPatternNode>();
  n->pattern = std::move(pattern);
  n->struct_info = std::move(struct_info);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dpl.StructInfoPattern")
    .set_body_typed([](DFPattern pattern, StructInfo struct_info) {
      return StructInfoPattern(pattern, struct_info);
    });
RELAX_PATTERN_PRINTER_DEF(StructInfoPatternNode, [](auto p, auto node) {
  p->stream << "StructInfoPattern(" << node->pattern << " has relax StructInfo "
            << node->struct_info << ")";
});

TVM_REGISTER_NODE_TYPE(ShapePatternNode);
ShapePattern::ShapePattern(DFPattern pattern, Array<PrimExpr> shape) {
  ObjectPtr<ShapePatternNode> n = make_object<ShapePatternNode>();
  n->pattern = std::move(pattern);
  n->shape = std::move(shape);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dpl.ShapePattern")
    .set_body_typed([](DFPattern pattern, Array<PrimExpr> shape) {
      return ShapePattern(pattern, shape);
    });
RELAX_PATTERN_PRINTER_DEF(ShapePatternNode, [](auto p, auto node) {
  p->stream << "ShapePattern(" << node->pattern << " has shape " << node->shape << ")";
});

TVM_REGISTER_NODE_TYPE(SameShapeConstraintNode);
SameShapeConstraint::SameShapeConstraint(Array<DFPattern> args) {
  ObjectPtr<SameShapeConstraintNode> n = make_object<SameShapeConstraintNode>();
  n->args = std::move(args);
  data_ = std::move(n);

  if (auto ctx = PatternContext::Current()) {
    ctx.value().add_constraint(*this);
  }
}
TVM_REGISTER_GLOBAL("relax.dpl.SameShapeConstraint").set_body_typed([](Array<DFPattern> args) {
  return SameShapeConstraint(args);
});
RELAX_PATTERN_PRINTER_DEF(SameShapeConstraintNode, [](auto p, auto node) {
  p->stream << "SameShapeConstraint(";
  for (size_t i = 0; i < node->args.size(); i++) {
    if (i) {
      p->stream << ", ";
    }
    p->stream << node->args;
  }
  p->stream << ")";
});

TVM_REGISTER_NODE_TYPE(DataTypePatternNode);
DataTypePattern::DataTypePattern(DFPattern pattern, DataType dtype) {
  ObjectPtr<DataTypePatternNode> n = make_object<DataTypePatternNode>();
  n->pattern = std::move(pattern);
  n->dtype = std::move(dtype);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dpl.DataTypePattern")
    .set_body_typed([](DFPattern pattern, DataType dtype) {
      return DataTypePattern(pattern, dtype);
    });
RELAX_PATTERN_PRINTER_DEF(DataTypePatternNode, [](auto p, auto node) {
  p->stream << "DataTypePattern(" << node->pattern << " has dtype " << node->dtype << ")";
});

TVM_REGISTER_NODE_TYPE(AttrPatternNode);
AttrPattern::AttrPattern(DFPattern pattern, DictAttrs attrs) {
  ObjectPtr<AttrPatternNode> n = make_object<AttrPatternNode>();
  n->pattern = std::move(pattern);
  n->attrs = std::move(attrs);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dpl.AttrPattern").set_body_typed([](DFPattern pattern, DictAttrs attrs) {
  return AttrPattern(pattern, attrs);
});
RELAX_PATTERN_PRINTER_DEF(AttrPatternNode, [](auto p, auto node) {
  p->stream << "AttrPattern(" << node->pattern << " has attributes " << node->attrs << ")";
});

class DFPatternDuplicator : public DFPatternFunctor<DFPattern(const DFPattern&)> {
 public:
  DFPattern VisitDFPattern(const DFPattern& pattern) override {
    return DFPatternFunctor::VisitDFPattern(pattern);
  }
  DFPattern VisitDFPattern_(const OrPatternNode* op) override {
    return OrPattern(op->left, op->right);
  }
  DFPattern VisitDFPattern_(const AndPatternNode* op) override {
    return AndPattern(op->left, op->right);
  }
  DFPattern VisitDFPattern_(const NotPatternNode* op) override { return NotPattern(op->reject); }
  DFPattern VisitDFPattern_(const VarPatternNode* op) override { return VarPattern(op->name); }
  DFPattern VisitDFPattern_(const ConstantPatternNode* op) override {
    return ConstantPattern(make_object<ConstantPatternNode>());
  }
  DFPattern VisitDFPattern_(const WildcardPatternNode* op) override {
    return WildcardPattern(make_object<WildcardPatternNode>());
  }
  DFPattern VisitDFPattern_(const ExprPatternNode* op) override { return ExprPattern(op->expr); }
  DFPattern VisitDFPattern_(const GlobalVarPatternNode* op) override {
    return GlobalVarPattern(op->name);
  }
  DFPattern VisitDFPattern_(const TuplePatternNode* op) override {
    return TuplePattern(op->fields);
  }
  DFPattern VisitDFPattern_(const UnorderedTuplePatternNode* op) override {
    return UnorderedTuplePattern(op->fields);
  }
  DFPattern VisitDFPattern_(const TupleGetItemPatternNode* op) override {
    return TupleGetItemPattern(op->tuple, op->index);
  }
  DFPattern VisitDFPattern_(const CallPatternNode* op) override {
    return CallPattern(op->op, op->args);
  }
  DFPattern VisitDFPattern_(const DataTypePatternNode* op) override {
    return DataTypePattern(op->pattern, op->dtype);
  }
  DFPattern VisitDFPattern_(const FunctionPatternNode* op) override {
    return FunctionPattern(op->params, op->body);
  }
  DFPattern VisitDFPattern_(const ShapePatternNode* op) override {
    return ShapePattern(op->pattern, op->shape);
  }
  DFPattern VisitDFPattern_(const StructInfoPatternNode* op) override {
    return StructInfoPattern(op->pattern, op->struct_info);
  }
  DFPattern VisitDFPattern_(const TypePatternNode* op) override {
    return TypePattern(op->pattern, op->type);
  }
  DFPattern VisitDFPattern_(const DataflowVarPatternNode* op) override {
    return DataflowVarPattern(op->name);
  }
  DFPattern VisitDFPattern_(const ExternFuncPatternNode* op) override {
    return ExternFuncPattern(op->global_symbol());
  }
  DFPattern VisitDFPattern_(const PrimArrPatternNode* op) override {
    return PrimArrPattern(op->fields);
  }
};

// Syntatic Sugar
CallPattern DFPattern::operator()(const std::vector<DFPattern>& args) const {
  return CallPattern(*this, Array<DFPattern>(args));
}
OrPattern DFPattern::operator|(const DFPattern& other) const { return OrPattern(*this, other); }

AndPattern DFPattern::operator&(const DFPattern& other) const { return AndPattern(*this, other); }

NotPattern DFPattern::operator~() const { return NotPattern(*this); }

AttrPattern DFPattern::HasAttr(const Map<String, ObjectRef>& attrs) const {
  return AttrPattern(*this, DictAttrs(attrs));
}
StructInfoPattern DFPattern::HasStructInfo(const StructInfo& struct_info) const {
  return StructInfoPattern(*this, struct_info);
}
TypePattern DFPattern::HasType(const Type& type) const { return TypePattern(*this, type); }
DataTypePattern DFPattern::HasDtype(const DataType& dtype) const {
  return DataTypePattern(*this, dtype);
}
DataTypePattern DFPattern::HasDtype(const std::string& dtype) const {
  return HasDtype(DataType(runtime::String2DLDataType(dtype)));
}
ShapePattern DFPattern::HasShape(const Array<PrimExpr>& shape) const {
  return ShapePattern(*this, shape);
}

DFPattern::operator PatternSeq() const { return PatternSeq{{*this}}; }

std::stack<PatternContext>& pattern_ctx_stack() {
  thread_local std::stack<PatternContext> graph_pattern_managers;
  return graph_pattern_managers;
}

Optional<PatternContext> PatternContext::Current() {
  if (pattern_ctx_stack().empty()) return NullOpt;
  return pattern_ctx_stack().top();
}

PatternContext::PatternContext(bool incremental) {
  auto n = make_object<PatternContextNode>();
  if (incremental) {
    ICHECK(!pattern_ctx_stack().empty())
        << "Incremental context needs to be built inside a existing context.";
    n->allow_extern_use = pattern_ctx_stack().top()->allow_extern_use;
    n->edge_constraints = pattern_ctx_stack().top()->edge_constraints;
    n->src_ordered = pattern_ctx_stack().top()->src_ordered;
  }

  data_ = std::move(n);
}

void PatternContext::EnterWithScope() const { pattern_ctx_stack().push(*this); }

void PatternContext::ExitWithScope() const {
  ICHECK(pattern_ctx_stack().top().same_as(*this));
  pattern_ctx_stack().pop();
}

static void sync_graph_constraints(const DFPattern& lhs, const DFPattern& rhs, PairCons pcon) {
  if (auto ctx = PatternContext::Current()) {
    ctx.value().add_constraint(lhs, rhs, pcon);
  }
}

TVM_REGISTER_NODE_TYPE(PatternSeqNode);
PatternSeq::PatternSeq(DFPattern init_pattern) {
  ObjectPtr<PatternSeqNode> n = make_object<PatternSeqNode>();
  n->patterns = {init_pattern};
  n->pair_constraints = {};
  data_ = std::move(n);
}
PatternSeq::PatternSeq(tvm::Array<DFPattern> patterns, bool only_used_by) {
  ICHECK_GE(patterns.size(), 1) << "PatternSeq must have at least one pattern";
  const auto cons = PairCons(only_used_by ? PairCons::kOnlyUsedBy : PairCons::kUsedBy);

  ObjectPtr<PatternSeqNode> n = make_object<PatternSeqNode>();
  n->patterns = std::move(patterns);
  n->pair_constraints = std::vector<PairCons>(n->patterns.size() - 1, cons);
  data_ = std::move(n);
}

PatternSeq PatternSeq::UsedBy(PatternSeq other, int index) const {
  return relax::UsedBy(*this, other, index);
}

PatternSeq PatternSeq::OnlyUsedBy(PatternSeq other, int index) const {
  return relax::OnlyUsedBy(*this, other, index);
}

PatternSeq PatternSeq::dup() const {
  PatternSeq ret;

  ObjectPtr<PatternSeqNode> n = make_object<PatternSeqNode>();
  n->patterns = Array<DFPattern>{};
  n->patterns.reserve(get()->patterns.size());
  n->pair_constraints = this->get()->pair_constraints;

  for (size_t i = 0; i < get()->patterns.size(); ++i) {
    n->patterns.push_back(get()->patterns[i].dup());
    if (i >= 1)
      sync_graph_constraints(n->patterns[i - 1], n->patterns[i], n->pair_constraints[i - 1]);
  }

  ret.data_ = std::move(n);

  return ret;
}
TVM_REGISTER_GLOBAL("relax.dpl.PatternSeq")
    .set_body_typed([](Array<DFPattern> patterns, bool only_used_by) {
      return PatternSeq(std::move(patterns), only_used_by);
    });
RELAX_PATTERN_PRINTER_DEF(PatternSeqNode, [](auto p, auto node) {
  p->stream << "[";
  for (size_t i = 0; i < node->patterns.size(); ++i) {
    if (i != 0)
      p->stream << (PairCons::kOnlyUsedBy == node->pair_constraints[i].type ? " >> " : " ^ ");
    p->stream << node->patterns[i];
  }
  p->stream << "]";
});

TVM_REGISTER_GLOBAL("relax.dpl.used_by")
    .set_body_typed([](PatternSeq lhs, PatternSeq rhs, int index) {
      return lhs.UsedBy(rhs, index);
    });

TVM_REGISTER_GLOBAL("relax.dpl.only_used_by")
    .set_body_typed([](PatternSeq lhs, PatternSeq rhs, int index) {
      return lhs.OnlyUsedBy(rhs, index);
    });

PatternSeq UsedBy(const PatternSeq& lhs, const PatternSeq& rhs, int index) {
  PatternSeq ret;

  const auto constraint = PairCons{PairCons::kOnlyUsedBy, index};

  sync_graph_constraints(lhs->patterns.back(), rhs->patterns.front(),
                         PairCons{PairCons::kUsedBy, index});

  Array<DFPattern> patterns;
  patterns.reserve(lhs->patterns.size() + rhs->patterns.size());
  patterns.insert(patterns.end(), lhs->patterns.begin(), lhs->patterns.end());
  patterns.insert(patterns.end(), rhs->patterns.begin(), rhs->patterns.end());

  std::vector<PairCons> pair_constraints = lhs->pair_constraints;
  pair_constraints.reserve(pair_constraints.size() + rhs->pair_constraints.size() + 1);
  pair_constraints.push_back(constraint);
  pair_constraints.insert(pair_constraints.end(), rhs->pair_constraints.begin(),
                          rhs->pair_constraints.end());

  ObjectPtr<PatternSeqNode> n = make_object<PatternSeqNode>();
  n->patterns = std::move(patterns);
  n->pair_constraints = std::move(pair_constraints);
  ret.data_ = std::move(n);

  return ret;
}
PatternSeq operator^(const PatternSeq& lhs, const PatternSeq& rhs) { return lhs.UsedBy(rhs); }

PatternSeq OnlyUsedBy(const PatternSeq& lhs, const PatternSeq& rhs, int index) {
  PatternSeq ret;

  const auto constraint = PairCons{PairCons::kOnlyUsedBy, index};

  sync_graph_constraints(lhs->patterns.back(), rhs->patterns.front(), constraint);

  Array<DFPattern> patterns;
  patterns.reserve(lhs->patterns.size() + rhs->patterns.size());
  patterns.insert(patterns.end(), lhs->patterns.begin(), lhs->patterns.end());
  patterns.insert(patterns.end(), rhs->patterns.begin(), rhs->patterns.end());

  std::vector<PairCons> pair_constraints = lhs->pair_constraints;
  pair_constraints.reserve(pair_constraints.size() + rhs->pair_constraints.size() + 1);
  pair_constraints.push_back(constraint);
  pair_constraints.insert(pair_constraints.end(), rhs->pair_constraints.begin(),
                          rhs->pair_constraints.end());

  ObjectPtr<PatternSeqNode> n = make_object<PatternSeqNode>();
  n->patterns = std::move(patterns);
  n->pair_constraints = std::move(pair_constraints);
  ret.data_ = std::move(n);

  return ret;
}
PatternSeq operator>>(const PatternSeq& lhs, const PatternSeq& rhs) { return lhs.OnlyUsedBy(rhs); }

VarPattern IsVar(const String& name) { return VarPattern(name); }
ConstantPattern IsConst() { return ConstantPattern(make_object<ConstantPatternNode>()); }
WildcardPattern Wildcard() { return WildcardPattern(make_object<WildcardPatternNode>()); }
ExprPattern IsExpr(const Expr& expr) { return ExprPattern(expr); }
ExprPattern IsOp(const String& op_name) { return IsExpr(Op::Get(op_name)); }
CallPattern IsCallTIR(const String& name, Optional<TuplePattern> var_args,
                      Optional<DFPattern> tir_vars) {
  DFPattern arg_pattern;
  if (!var_args.defined()) {
    arg_pattern = Wildcard();
  } else {
    arg_pattern = var_args.value();
  }

  if (tir_vars.defined()) {
    return IsOp("relax.call_tir")(GlobalVarPattern(name), arg_pattern, tir_vars.value());
  }
  return IsOp("relax.call_tir")(GlobalVarPattern(name), arg_pattern);
}

CallPattern IsCallTIR(const String& name, TuplePattern var_args) {
  return IsOp("relax.call_tir")(GlobalVarPattern(name), var_args);
}
CallPattern IsCallDPSPacked(const String& name, Optional<TuplePattern> var_args) {
  DFPattern arg_pattern;
  if (!var_args.defined()) {
    arg_pattern = Wildcard();
  } else {
    arg_pattern = var_args.value();
  }

  return IsOp("relax.call_dps_packed")(GlobalVarPattern(name), arg_pattern);
}

CallPattern IsCallDPSPacked(const String& name, TuplePattern var_args) {
  return IsOp("relax.call_dps_packed")(GlobalVarPattern(name), var_args);
}

DFPattern IsTuple(const Array<DFPattern>& fields, bool unordered) {
  if (unordered)
    return UnorderedTuplePattern(fields);
  else
    return TuplePattern(fields);
}
TupleGetItemPattern IsTupleGetItem(const DFPattern tuple, int index) {
  return TupleGetItemPattern(tuple, index);
}

DFPattern DFPattern::dup() const {
  auto pattern = DFPatternDuplicator().VisitDFPattern(*this);
  return pattern;
}

TVM_REGISTER_GLOBAL("relax.dpl.dup_pattern").set_body_typed([](DFPattern pattern) {
  return pattern.dup();
});

TVM_REGISTER_GLOBAL("relax.dpl.dup_seq").set_body_typed([](PatternSeq seq) { return seq.dup(); });

TVM_REGISTER_GLOBAL("relax.dpl.PatternContext").set_body_typed([](bool incre) {
  return PatternContext(incre);
});

TVM_REGISTER_GLOBAL("relax.dpl.current_context").set_body_typed([] {
  return PatternContext::Current();
});

TVM_REGISTER_GLOBAL("relax.dpl.enter_context").set_body_typed([](const PatternContext& ctx) {
  ctx.EnterWithScope();
});

TVM_REGISTER_GLOBAL("relax.dpl.exit_context").set_body_typed([](const PatternContext& ctx) {
  ctx.ExitWithScope();
});

}  // namespace relax
}  // namespace tvm
