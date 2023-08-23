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
 * \file tvm/relay/dataflow_pattern.h
 * \brief A pattern language for matching dataflow properties.
 */
#ifndef TVM_RELAY_DATAFLOW_PATTERN_H_
#define TVM_RELAY_DATAFLOW_PATTERN_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/type.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {

/*!
 * \brief Base type of all dataflow patterns.
 * \sa DFPattern
 */
class DFPatternNode : public Object {
 public:
  static constexpr const char* _type_key = "DFPatternNode";
  TVM_DECLARE_BASE_OBJECT_INFO(DFPatternNode, Object);
};

/*!
 * \brief Managed reference to dataflow patterns.
 * \sa DFPatternNode
 */
class DFPattern : public ObjectRef {
 public:
  /*! \brief Syntatic Sugar for creating a CallPattern */
  DFPattern operator()(const std::vector<DFPattern>& args) const;
  /*! \brief Syntatic Sugar for creating a CallPattern with an "add" op */
  DFPattern operator+(const DFPattern& other) const;
  /*! \brief Syntatic Sugar for creating a CallPattern with a "subtract" op */
  DFPattern operator-(const DFPattern& other) const;
  /*! \brief Syntatic Sugar for creating a CallPattern with a "multiply" op */
  DFPattern operator*(const DFPattern& other) const;
  /*! \brief Syntatic Sugar for creating a CallPattern with a "divide" op */
  DFPattern operator/(const DFPattern& other) const;
  /*! \brief Syntatic Sugar for creating an AltPattern */
  DFPattern operator||(const DFPattern& other) const;
  /*! \brief Syntatic Sugar for creating an Optional Pattern */
  DFPattern Optional(const std::function<DFPattern(const DFPattern&)>& func) const;
  /*! \brief Syntatic Sugar for creating an AttrPattern */
  DFPattern HasAttr(const Map<String, ObjectRef>& attrs) const;
  /*! \brief Syntatic Sugar for creating a TypePattern */
  DFPattern HasType(const Type& type) const;
  /*! \brief Syntatic Sugar for creating a DataTypePattern with a DataType */
  DFPattern HasDtype(const DataType& dtype) const;
  /*! \brief Syntatic Sugar for creating a DataTypePattern with a data type's name */
  DFPattern HasDtype(const std::string& dtype) const;
  /*! \brief Syntatic Sugar for creating a ShapePattern */
  DFPattern HasShape(const Array<PrimExpr> shape) const;

  TVM_DEFINE_OBJECT_REF_METHODS(DFPattern, ObjectRef, DFPatternNode);
};

/*!
 * \brief Pattern for Relay Expression.
 */
class ExprPatternNode : public DFPatternNode {
 public:
  /*! \brief The expression to match. */
  Expr expr;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("expr", &expr); }

  static constexpr const char* _type_key = "relay.dataflow_pattern.ExprPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(ExprPatternNode, DFPatternNode);
};

/*!
 * \brief A pattern which matches a literal expression.
 *
 * \note Uses structural equality on expressions to check equality.
 *
 */
class ExprPattern : public DFPattern {
 public:
  TVM_DLL explicit ExprPattern(Expr expr);
  TVM_DEFINE_OBJECT_REF_METHODS(ExprPattern, DFPattern, ExprPatternNode);
};

/*!
 * \brief A Pattern to Match a Relay Variable
 */
class VarPattern;
/*! \brief Container for Var */
class VarPatternNode : public DFPatternNode {
 public:
  /*!
   * \brief The name of the Var (optional).
   */
  String name;

  /*! \return The name hint of the variable */
  const String& name_hint() const { return name; }

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("name", &name); }

  static constexpr const char* _type_key = "relay.dataflow_pattern.VarPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(VarPatternNode, DFPatternNode);
};

class VarPattern : public DFPattern {
 public:
  TVM_DLL VarPattern(String name_hint);
  TVM_DEFINE_OBJECT_REF_METHODS(VarPattern, DFPattern, VarPatternNode);
};

/*!
 * \brief A Pattern to Match a Relay Constant
 */
class ConstantPattern;
/*! \brief Container for Constant */
class ConstantPatternNode : public DFPatternNode {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "relay.dataflow_pattern.ConstantPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(ConstantPatternNode, DFPatternNode);
};

class ConstantPattern : public DFPattern {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(ConstantPattern, DFPattern, ConstantPatternNode);
};

/*!
 * \brief Call corresponds to operator invocation.
 *  Corresponds to the operator in computational graph terminology.
 */
class CallPattern;
/*! \brief CallPattern container. */
class CallPatternNode : public DFPatternNode {
 public:
  /*!
   * \brief The operator(function) being invoked
   *
   *  - It can be relay::Op which corresponds to the primitive operators.
   *  - It can also be user defined functions (Function, GlobalVar, Var).
   */
  DFPattern op;

  /*! \brief The arguments(inputs) of the call */
  tvm::Array<relay::DFPattern> args;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("op", &op);
    v->Visit("args", &args);
  }

  static constexpr const char* _type_key = "relay.dataflow_pattern.CallPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(CallPatternNode, DFPatternNode);
};

class CallPattern : public DFPattern {
 public:
  TVM_DLL CallPattern(DFPattern op, Array<DFPattern> args);
  TVM_DEFINE_OBJECT_REF_METHODS(CallPattern, DFPattern, CallPatternNode);
};

/*!
 * \brief Relay Function container
 * \sa Function
 */
class FunctionPatternNode : public DFPatternNode {
 public:
  /*! \brief Function parameters */
  tvm::Array<DFPattern> params;
  /*!
   * \brief
   * The expression which represents the computation of the function,
   * the expression may reference the parameters, and the type of it
   * or sub-expressions may reference the type variables.
   */
  DFPattern body;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("params", &params);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "relay.dataflow_pattern.FunctionPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionPatternNode, DFPatternNode);
};

/*!
 * \brief Managed reference to FunctionNode.
 * \sa FunctionNode
 */
class FunctionPattern : public DFPattern {
 public:
  /*!
   * \brief Constructor
   * \param params The parameters of the function.
   * \param body The body of the function.
   */
  TVM_DLL FunctionPattern(tvm::Array<DFPattern> params, DFPattern body);

  TVM_DEFINE_OBJECT_REF_METHODS(FunctionPattern, DFPattern, FunctionPatternNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FunctionPatternNode);
};

/*! \brief A binding of a sub-network. */
class LetPatternNode : public DFPatternNode {
 public:
  /*! \brief The variable we bind to */
  DFPattern var;
  /*! \brief The value we bind var to */
  DFPattern value;
  /*! \brief The body of the let binding */
  DFPattern body;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("var", &var);
    v->Visit("value", &value);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "relay.dataflow_pattern.LetPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(LetPatternNode, DFPatternNode);
};

/*!
 * \brief Let binding that binds a local var
 */
class LetPattern : public DFPattern {
 public:
  /*!
   * \brief The constructor
   * \param var The variable that is bound to.
   * \param value The value used to bind to the variable.
   * \param body The body of the let binding.
   */
  TVM_DLL LetPattern(DFPattern var, DFPattern value, DFPattern body);

  TVM_DEFINE_OBJECT_REF_METHODS(LetPattern, DFPattern, LetPatternNode);
};

/*! \brief Tuple of multiple Exprs */
class TuplePattern;
/*! \brief Tuple container */
class TuplePatternNode : public DFPatternNode {
 public:
  /*! \brief the fields of the tuple */
  tvm::Array<DFPattern> fields;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("fields", &fields); }

  static constexpr const char* _type_key = "relay.dataflow_pattern.TuplePattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(TuplePatternNode, DFPatternNode);
};

class TuplePattern : public DFPattern {
 public:
  TVM_DLL explicit TuplePattern(tvm::Array<DFPattern> fields);
  TVM_DEFINE_OBJECT_REF_METHODS(TuplePattern, DFPattern, TuplePatternNode);
};

/*! \brief Get index-th field out of a tuple. */
class TupleGetItemPattern;
class TupleGetItemPatternNode : public DFPatternNode {
 public:
  /*! \brief The tuple Expression */
  DFPattern tuple;
  /*! \brief which value to get */
  int index;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("tuple", &tuple);
    v->Visit("index", &index);
  }

  static constexpr const char* _type_key = "relay.dataflow_pattern.TupleGetItemPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleGetItemPatternNode, DFPatternNode);
};

class IfPatternNode : public DFPatternNode {
 public:
  DFPattern cond, true_branch, false_branch;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("cond", &cond);
    v->Visit("true_branch", &true_branch);
    v->Visit("false_branch", &false_branch);
  }

  static constexpr const char* _type_key = "relay.dataflow_pattern.IfPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(IfPatternNode, DFPatternNode);
};

class IfPattern : public DFPattern {
 public:
  TVM_DLL IfPattern(DFPattern cond, DFPattern then_clause, DFPattern else_clause);
  TVM_DEFINE_OBJECT_REF_METHODS(IfPattern, DFPattern, IfPatternNode);
};

class TupleGetItemPattern : public DFPattern {
 public:
  TVM_DLL TupleGetItemPattern(DFPattern tuple, int index);
  TVM_DEFINE_OBJECT_REF_METHODS(TupleGetItemPattern, DFPattern, TupleGetItemPatternNode);
};

class AltPattern;
/*!
 * \brief Pattern for Alternate Expressions.
 */
class AltPatternNode : public DFPatternNode {
 public:
  /*! \brief The left optional pattern. */
  DFPattern left;
  /*! \brief The right optional pattern. */
  DFPattern right;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("left", &left);
    v->Visit("right", &right);
  }

  static constexpr const char* _type_key = "relay.dataflow_pattern.AltPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(AltPatternNode, DFPatternNode);
};

/*!
 * \brief A pattern which matches either of two patterns
 */
class AltPattern : public DFPattern {
 public:
  TVM_DLL AltPattern(DFPattern left, DFPattern right);
  TVM_DEFINE_OBJECT_REF_METHODS(AltPattern, DFPattern, AltPatternNode);
};

/*!
 * \brief Wildcard Pattern.
 */
class WildcardPatternNode : public DFPatternNode {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*! \brief If the wildcard is redirected, then pattern is not nullptr, and the wildcard
   * redirects to the pattern. */
  Optional<DFPattern> pattern{nullptr};

  static constexpr const char* _type_key = "relay.dataflow_pattern.WildcardPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(WildcardPatternNode, DFPatternNode);
};

/*!
 * \brief A pattern which matches anything.
 */
class WildcardPattern : public DFPattern {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(WildcardPattern, DFPattern, WildcardPatternNode);

  void redirect_to(DFPattern pat) const;
};

class TypePattern;
/*!
 * \brief Pattern for Types.
 */
class TypePatternNode : public DFPatternNode {
 public:
  /*! \brief The pattern. */
  DFPattern pattern;
  /*! \brief The type to match */
  Type type;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pattern", &pattern);
    v->Visit("type", &type);
  }

  static constexpr const char* _type_key = "relay.dataflow_pattern.TypePattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(TypePatternNode, DFPatternNode);
};

/*!
 * \brief A pattern which matches a type in another pattern
 */
class TypePattern : public DFPattern {
 public:
  TVM_DLL TypePattern(DFPattern pattern, Type type);
  TVM_DEFINE_OBJECT_REF_METHODS(TypePattern, DFPattern, TypePatternNode);
};

class ShapePattern;
/*!
 * \brief Pattern for Shapes.
 */
class ShapePatternNode : public DFPatternNode {
 public:
  /*! \brief The pattern. */
  DFPattern pattern;
  /*! \brief The type to match */
  Array<PrimExpr> shape;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pattern", &pattern);
    v->Visit("shape", &shape);
  }

  static constexpr const char* _type_key = "relay.dataflow_pattern.ShapePattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(ShapePatternNode, DFPatternNode);
};

/*!
 * \brief A pattern which matches a type in another pattern
 */
class ShapePattern : public DFPattern {
 public:
  TVM_DLL ShapePattern(DFPattern pattern, Array<PrimExpr> type);
  TVM_DEFINE_OBJECT_REF_METHODS(ShapePattern, DFPattern, ShapePatternNode);
};

class DataTypePattern;
/*!
 * \brief Pattern for Types.
 */
class DataTypePatternNode : public DFPatternNode {
 public:
  /*! \brief The pattern. */
  DFPattern pattern;
  /*! \brief The type to match */
  DataType dtype;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pattern", &pattern);
    v->Visit("dtype", &dtype);
  }

  static constexpr const char* _type_key = "relay.dataflow_pattern.DataTypePattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(DataTypePatternNode, DFPatternNode);
};

/*!
 * \brief A pattern which matches a type in another pattern
 */
class DataTypePattern : public DFPattern {
 public:
  TVM_DLL DataTypePattern(DFPattern pattern, DataType dtype);
  TVM_DEFINE_OBJECT_REF_METHODS(DataTypePattern, DFPattern, DataTypePatternNode);
};

class AttrPattern;
/*!
 * \brief Pattern for Attributes.
 */
class AttrPatternNode : public DFPatternNode {
 public:
  /*! \brief The pattern. */
  DFPattern pattern;
  /*! \brief The attribute to match */
  DictAttrs attrs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pattern", &pattern);
    v->Visit("attrs", &attrs);
  }

  static constexpr const char* _type_key = "relay.dataflow_pattern.AttrPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(AttrPatternNode, DFPatternNode);
};

/*!
 * \brief A pattern which matches attributes in another pattern
 */
class AttrPattern : public DFPattern {
 public:
  TVM_DLL AttrPattern(DFPattern pattern, DictAttrs attrs);
  TVM_DEFINE_OBJECT_REF_METHODS(AttrPattern, DFPattern, AttrPatternNode);
};

class DominatorPattern;
/*!
 * \brief Dominated Graph Pattern
 * Pattern for fuzzy subgraphs where all outputs of the parent are used finally by the child, and
 * every operation between the parent and the child matches the path.
 */
class DominatorPatternNode : public DFPatternNode {
 public:
  /*! \brief The parent. */
  DFPattern parent;
  /*! \brief The path. */
  DFPattern path;
  /*! \brief The child. */
  DFPattern child;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("parent", &parent);
    v->Visit("path", &path);
    v->Visit("child", &child);
  }

  static constexpr const char* _type_key = "relay.dataflow_pattern.DominatorPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(DominatorPatternNode, DFPatternNode);
};

/*!
 * \brief A pattern which matches a variable length dominator path
 */
class DominatorPattern : public DFPattern {
 public:
  TVM_DLL DominatorPattern(DFPattern parent, DFPattern path, DFPattern child);
  TVM_DEFINE_OBJECT_REF_METHODS(DominatorPattern, DFPattern, DominatorPatternNode);
};

/*! \brief Syntatic Sugar for creating a VarPattern with a name */
DFPattern IsVar(const String& name);
/*! \brief Syntatic Sugar for creating a ConstantPattern */
DFPattern IsConstant();
/*! \brief Syntatic Sugar for creating a WildcardPattern */
DFPattern IsWildcard();
/*! \brief Syntatic Sugar for creating a ExprPattern */
DFPattern IsExpr(const Expr& expr);
/*! \brief Syntatic Sugar for creating a ExprPattern base on an Op*/
DFPattern IsOp(const String& op_name);
/*! \brief Syntatic Sugar for creating a TuplePattern*/
DFPattern IsTuple(const Array<DFPattern>& fields);
/*! \brief Syntatic Sugar for creating a TupleGetItemPattern*/
DFPattern IsTupleGetItem(const DFPattern tuple, int index = -1);

/*! \brief A printer class to print pattern. */
class DFPatternPrinter : public ReprPrinter {
 public:
  std::stringstream string_stream{};

  std::unordered_map<DFPattern, std::pair<size_t, std::string>, ObjectPtrHash, ObjectPtrEqual>
      memo_{};
  /*! \brief Subpatterns that are encountered more than once during printing. If a subpattern has
   * already printed, only the pattern ID will be printed in the next encounter of the same pattern.
   * This avoids printing a subpattern infinitely many times is the considered pattern involves
   * recursion.*/
  std::vector<DFPattern> auxiliary_patterns{};

  DFPatternPrinter(std::ostream& stream)  // NOLINT(*)
      : ReprPrinter(stream) {}
  TVM_DLL void Print(const ObjectRef& node);
  using FType = NodeFunctor<void(const ObjectRef&, DFPatternPrinter*)>;
  TVM_DLL static FType& vtable();
};

inline std::ostream& operator<<(std::ostream& os,
                                const DFPattern& n) {  // NOLINT(*)
  std::stringstream string_stream{}, tmp_stream{};
  DFPatternPrinter printer{tmp_stream};
  printer.Print(n);
  string_stream << "Main pattern:" << std::endl;
  string_stream << printer.string_stream.str();
  string_stream << std::endl;
  string_stream << "Auxiliary patterns:";
  for (const DFPattern& pat : printer.auxiliary_patterns) {
    string_stream << std::endl;
    string_stream << printer.memo_[pat].second;
  }
  os << string_stream.str();
  return os;
}

String PrettyPrint(const DFPattern& pattern);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_DATAFLOW_PATTERN_H_
