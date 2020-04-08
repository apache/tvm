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
  TVM_DEFINE_OBJECT_REF_METHODS(DFPattern, ObjectRef, DFPatternNode);
};

/*!
 * \brief Pattern for Relay Expression.
 */
class ExprPatternNode : public DFPatternNode {
 public:
  /*! \brief The expression to match. */
  Expr expr;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("expr", &expr);
  }

  static constexpr const char* _type_key = "relay.df_pattern.ExprPattern";
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
  TVM_DLL ExprPattern(Expr expr);
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
  std::string name;
  /*!
   * \brief type annotaion of the variable.
   * This field records user provided type annotation of the Var.
   * This field is optional and can be None.
   */
  Type type_annotation;

  /*! \return The name hint of the variable */
  const std::string& name_hint() const {
    return name;
  }

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("type_annotation", &type_annotation);
  }

  TVM_DLL static VarPattern make(std::string name_hint, Type type_annotation);

  static constexpr const char* _type_key = "relay.df_pattern.VarPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(VarPatternNode, DFPatternNode);
};

class VarPattern : public DFPattern {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(VarPattern, DFPattern, VarPatternNode);
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

  /*! \brief The additional attributes */
  Attrs attrs;

  /*!
   * \brief The type arguments passed to polymorphic(template) function.
   *
   * This is the advance feature that is only used when the function is
   * polymorphic. It is safe to be ignored in most cases. For example, in the
   * following code, the type_args of addone call is [int].
   *
   * \code
   *
   * template<typename T>
   * T addone(T a) { return a + 1; }
   *
   * void main() {
   *   int x = addone<int>(10);
   * }
   *
   * \endcode
   */
  tvm::Array<Type> type_args;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("op", &op);
    v->Visit("args", &args);
    v->Visit("attrs", &attrs);
    v->Visit("type_args", &type_args);
  }

  TVM_DLL static CallPattern make(DFPattern op, Array<DFPattern> args, Attrs attrs,
                                  Array<Type> type_args);

  static constexpr const char* _type_key = "relay.df_pattern.CallPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(CallPatternNode, DFPatternNode);
};

class CallPattern : public DFPattern {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(CallPattern, DFPattern, CallPatternNode);
};

/*! \brief Tuple of multiple Exprs */
class TuplePattern;
/*! \brief Tuple container */
class TuplePatternNode : public DFPatternNode {
 public:
  /*! \brief the fields of the tuple */
  tvm::Array<DFPattern> fields;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("fields", &fields);
  }

  TVM_DLL static TuplePattern make(tvm::Array<DFPattern> fields);

  static constexpr const char* _type_key = "relay.df_pattern.TuplePattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(TuplePatternNode, DFPatternNode);
};

class TuplePattern : public DFPattern {
 public:
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
    v->Visit("tuple_value", &tuple);
  }

  TVM_DLL static TupleGetItemPattern make(DFPattern tuple, int index);

  static constexpr const char* _type_key = "relay.df_pattern.TupleGetItemPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleGetItemPatternNode, DFPatternNode);
};

class TupleGetItemPattern : public DFPattern {
 public:
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

  TVM_DLL static AltPattern make(DFPattern left, DFPattern right);

  static constexpr const char* _type_key = "relay.df_pattern.AltPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(AltPatternNode, DFPatternNode);
};

/*!
 * \brief A pattern which matches either of two patterns
 */
class AltPattern : public DFPattern {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(AltPattern, DFPattern, AltPatternNode);
};


/*!
 * \brief Wildcard Pattern.
 */
class WildcardPatternNode : public DFPatternNode {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "relay.df_pattern.WildcardPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(WildcardPatternNode, DFPatternNode);
};

/*!
 * \brief A pattern which matches anything.
 */
class WildcardPattern : public DFPattern {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(WildcardPattern, DFPattern, WildcardPatternNode);
};

/*!
 * \brief Null Pattern.
 */
class NullPatternNode : public DFPatternNode {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "relay.df_pattern.NullPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(NullPatternNode, DFPatternNode);
};

/*!
 * \brief A pattern which matches anything.
 */
class NullPattern : public DFPattern {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(NullPattern, DFPattern, NullPatternNode);
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

  TVM_DLL static TypePattern make(DFPattern pattern, Type type);

  static constexpr const char* _type_key = "relay.df_pattern.TypePattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(TypePatternNode, DFPatternNode);
};

/*!
 * \brief A pattern which matches a type in another pattern
 */
class TypePattern : public DFPattern {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(TypePattern, DFPattern, TypePatternNode);
};

class AttrPattern;
/*!
 * \brief Pattern for Types.
 */
class AttrPatternNode : public DFPatternNode {
 public:
  /*! \brief The pattern. */
  DFPattern pattern;
  /*! \brief The attribute to match */
  Attrs attrs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pattern", &pattern);
    v->Visit("attrs", &attrs);
  }

  TVM_DLL static AttrPattern make(DFPattern pattern, Attrs attrs);

  static constexpr const char* _type_key = "relay.df_pattern.AttrPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(AttrPatternNode, DFPatternNode);
};

/*!
 * \brief A pattern which matches a type in another pattern
 */
class AttrPattern : public DFPattern {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(AttrPattern, DFPattern, AttrPatternNode);
};

class DominatorPattern;
/*!
 * \brief Pattern for Types.
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

  TVM_DLL static DominatorPattern make(DFPattern parent, DFPattern path, DFPattern child);

  static constexpr const char* _type_key = "relay.df_pattern.DominatorPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(DominatorPatternNode, DFPatternNode);
};

/*!
 * \brief A pattern which matches a variable length dominator path
 */
class DominatorPattern : public DFPattern {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(DominatorPattern, DFPattern, DominatorPatternNode);
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_DATAFLOW_PATTERN_H_
