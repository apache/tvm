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
 * \file src/contrib/msc/core/printer/msc_doc.h
 * \brief Extra docs for MSC
 */
#ifndef TVM_CONTRIB_MSC_CORE_PRINTER_MSC_DOC_H_
#define TVM_CONTRIB_MSC_CORE_PRINTER_MSC_DOC_H_

#include <tvm/script/printer/doc.h>

#include <string>

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::script::printer;

/*!
 * \brief Doc that declare a var with type.
 *
 * \sa DeclareDoc
 */
class DeclareDocNode : public ExprDocNode {
 public:
  /*! \brief The type of the variable */
  Optional<ExprDoc> type;
  /*! \brief The variable */
  ExprDoc variable{nullptr};
  /*! \brief The init arguments for the variable. */
  Array<ExprDoc> init_args;
  /*! \brief Whether to use constructor(otherwise initializer) */
  bool use_constructor{true};

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("type", &type);
    v->Visit("variable", &variable);
    v->Visit("init_args", &init_args);
    v->Visit("use_constructor", &use_constructor);
  }

  static constexpr const char* _type_key = "script.printer.DeclareDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(DeclareDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of DeclareDocNode.
 *
 * \sa DeclareDocNode
 */
class DeclareDoc : public ExprDoc {
 public:
  /*!
   * \brief Constructor of DeclareDoc.
   * \param type The type of the variable.
   * \param variable The variable.
   * \param init_args The init arguments of the variable.
   * \param use_constructor Whether to use constructor(otherwise initializer).
   */
  explicit DeclareDoc(Optional<ExprDoc> type, ExprDoc variable, Array<ExprDoc> init_args,
                      bool use_constructor);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(DeclareDoc, ExprDoc, DeclareDocNode);
};

/*!
 * \brief Doc that build a strict list, which check the empty.
 *
 * \sa StrictListDoc
 */
class StrictListDocNode : public ExprDocNode {
 public:
  /*! \brief The inner list doc */
  ListDoc list;
  /*! \brief Whether to allow empty */
  bool allow_empty{true};

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("list", &list);
    v->Visit("allow_empty", &allow_empty);
  }

  static constexpr const char* _type_key = "script.printer.StrictListDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(StrictListDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of StrictListDocNode.
 *
 * \sa StrictListDocNode
 */
class StrictListDoc : public ExprDoc {
 public:
  /*!
   * \brief Constructor of StrictListDoc.
   * \param list The inner list doc.
   * \param allow_empty Whether to allow empty.
   */
  explicit StrictListDoc(ListDoc list, bool allow_empty);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(StrictListDoc, ExprDoc, StrictListDocNode);
};

/*!
 * \brief Doc that represents pointer.
 *
 * \sa PointerDoc
 */
class PointerDocNode : public ExprDocNode {
 public:
  /*! \brief The name of the identifier */
  String name;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("name", &name);
  }

  static constexpr const char* _type_key = "script.printer.PointerDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(PointerDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of PointerDocNode.
 *
 * \sa PointerDocNode
 */
class PointerDoc : public ExprDoc {
 public:
  /*!
   * \brief Constructor of PointerDoc.
   * \param name The name of identifier.
   */
  explicit PointerDoc(String name);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(PointerDoc, ExprDoc, PointerDocNode);
};

/*!
 * \brief Doc that represents struct definition.
 *
 * \sa StructDoc
 */
class StructDocNode : public StmtDocNode {
 public:
  /*! \brief The name of class. */
  IdDoc name{nullptr};
  /*! \brief Decorators of class. */
  Array<ExprDoc> decorators;
  /*! \brief The body of class. */
  Array<StmtDoc> body;

  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
    v->Visit("name", &name);
    v->Visit("decorators", &decorators);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "script.printer.StructDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(StructDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of StructDocNode.
 *
 * \sa StructDocNode
 */
class StructDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of StructDoc.
   * \param name The name of class.
   * \param decorators The decorator of class.
   * \param body The body of class.
   */
  explicit StructDoc(IdDoc name, Array<ExprDoc> decorators, Array<StmtDoc> body);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(StructDoc, StmtDoc, StructDocNode);
};

/*!
 * \brief Doc that represents constructor definition.
 *
 * \sa ConstructorDoc
 */
class ConstructorDocNode : public StmtDocNode {
 public:
  /*! \brief The name of function. */
  IdDoc name{nullptr};
  /*!
   * \brief The arguments of function.
   *
   * The `lhs` means argument name,
   * `annotation` means argument type,
   * and `rhs` means default value.
   */
  Array<AssignDoc> args;
  /*! \brief The body of function. */
  Array<StmtDoc> body;

  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
    v->Visit("name", &name);
    v->Visit("args", &args);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "script.printer.ConstructorDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(ConstructorDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of ConstructorDocNode.
 *
 * \sa ConstructorDocNode
 */
class ConstructorDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of ConstructorDoc.
   * \param name The name of function..
   * \param args The arguments of function.
   * \param body The body of function.
   */
  explicit ConstructorDoc(IdDoc name, Array<AssignDoc> args, Array<StmtDoc> body);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ConstructorDoc, StmtDoc, ConstructorDocNode);
};

/*!
 * \brief Doc that represent switch statement.
 *
 * \sa SwitchDoc
 */
class SwitchDocNode : public StmtDocNode {
 public:
  /*! \brief The predicates of the switch statement. */
  Array<ExprDoc> predicates;
  /*! \brief The branchs of the switch statement. */
  Array<Array<StmtDoc>> branchs;
  /*! \brief The default_branch of the switch statement. */
  Array<StmtDoc> default_branch;

  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
    v->Visit("predicates", &predicates);
    v->Visit("branchs", &branchs);
    v->Visit("default_branch", &default_branch);
  }

  static constexpr const char* _type_key = "script.printer.SwitchDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(SwitchDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of SwitchDocNode.
 *
 * \sa SwitchDocNode
 */
class SwitchDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of SwitchDoc.
   * \param predicates The predicates of the switch statement.
   * \param branchs The branchs of the switch statement.
   * \param default_branch The default_branch of the switch statement.
   */
  explicit SwitchDoc(Array<ExprDoc> predicates, Array<Array<StmtDoc>> branchs,
                     Array<StmtDoc> default_branch);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(SwitchDoc, StmtDoc, SwitchDocNode);
};

/*!
 * \brief Doc that represents lambda definition.
 *
 * \sa LambdaDoc
 */
class LambdaDocNode : public StmtDocNode {
 public:
  /*! \brief The name of lambda. */
  IdDoc name{nullptr};
  /*!
   * \brief The arguments of lambda.
   *
   * The `lhs` means argument name,
   * `annotation` means argument type,
   * and `rhs` means default value.
   */
  Array<AssignDoc> args;
  /*! \brief References of lambda. */
  Array<ExprDoc> refs;
  /*! \brief The body of lambda. */
  Array<StmtDoc> body;

  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
    v->Visit("name", &name);
    v->Visit("args", &args);
    v->Visit("refs", &refs);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "script.printer.LambdaDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(LambdaDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of LambdaDocNode.
 *
 * \sa LambdaDoc
 */
class LambdaDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of LambdaDoc.
   * \param name The name of lambda.
   * \param args The arguments of lambda.
   * \param refs The references of lambda.
   * \param body The body of lambda.
   */
  explicit LambdaDoc(IdDoc name, Array<AssignDoc> args, Array<ExprDoc> refs, Array<StmtDoc> body);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(LambdaDoc, StmtDoc, LambdaDocNode);
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_PRINTER_MSC_DOC_H_
