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

#include <tvm/ffi/reflection/registry.h>
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
  ffi::Optional<ExprDoc> type;
  /*! \brief The variable */
  ExprDoc variable{ffi::UnsafeInit{}};
  /*! \brief The init arguments for the variable. */
  ffi::Array<ExprDoc> init_args;
  /*! \brief Whether to use constructor(otherwise initializer) */
  bool use_constructor{true};

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DeclareDocNode>()
        .def_ro("type", &DeclareDocNode::type)
        .def_ro("variable", &DeclareDocNode::variable)
        .def_ro("init_args", &DeclareDocNode::init_args)
        .def_ro("use_constructor", &DeclareDocNode::use_constructor);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("msc.script.printer.DeclareDoc", DeclareDocNode, ExprDocNode);
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
  explicit DeclareDoc(ffi::Optional<ExprDoc> type, ExprDoc variable, ffi::Array<ExprDoc> init_args,
                      bool use_constructor);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DeclareDoc, ExprDoc, DeclareDocNode);
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

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StrictListDocNode>()
        .def_ro("list", &StrictListDocNode::list)
        .def_ro("allow_empty", &StrictListDocNode::allow_empty);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("msc.script.printer.StrictListDoc", StrictListDocNode,
                                    ExprDocNode);
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
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StrictListDoc, ExprDoc, StrictListDocNode);
};

/*!
 * \brief Doc that represents pointer.
 *
 * \sa PointerDoc
 */
class PointerDocNode : public ExprDocNode {
 public:
  /*! \brief The name of the identifier */
  ffi::String name;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PointerDocNode>().def_ro("name", &PointerDocNode::name);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("msc.script.printer.PointerDoc", PointerDocNode, ExprDocNode);
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
  explicit PointerDoc(ffi::String name);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PointerDoc, ExprDoc, PointerDocNode);
};

/*!
 * \brief Doc that represents struct definition.
 *
 * \sa StructDoc
 */
class StructDocNode : public StmtDocNode {
 public:
  /*! \brief The name of class. */
  IdDoc name{ffi::UnsafeInit{}};
  /*! \brief Decorators of class. */
  ffi::Array<ExprDoc> decorators;
  /*! \brief The body of class. */
  ffi::Array<StmtDoc> body;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StructDocNode>()
        .def_ro("name", &StructDocNode::name)
        .def_ro("decorators", &StructDocNode::decorators)
        .def_ro("body", &StructDocNode::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("msc.script.printer.StructDoc", StructDocNode, StmtDocNode);
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
  explicit StructDoc(IdDoc name, ffi::Array<ExprDoc> decorators, ffi::Array<StmtDoc> body);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StructDoc, StmtDoc, StructDocNode);
};

/*!
 * \brief Doc that represents constructor definition.
 *
 * \sa ConstructorDoc
 */
class ConstructorDocNode : public StmtDocNode {
 public:
  /*! \brief The name of function. */
  IdDoc name{ffi::UnsafeInit{}};
  /*!
   * \brief The arguments of function.
   *
   * The `lhs` means argument name,
   * `annotation` means argument type,
   * and `rhs` means default value.
   */
  ffi::Array<AssignDoc> args;
  /*! \brief The body of function. */
  ffi::Array<StmtDoc> body;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ConstructorDocNode>()
        .def_ro("name", &ConstructorDocNode::name)
        .def_ro("args", &ConstructorDocNode::args)
        .def_ro("body", &ConstructorDocNode::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("msc.script.printer.ConstructorDoc", ConstructorDocNode,
                                    StmtDocNode);
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
  explicit ConstructorDoc(IdDoc name, ffi::Array<AssignDoc> args, ffi::Array<StmtDoc> body);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ConstructorDoc, StmtDoc, ConstructorDocNode);
};

/*!
 * \brief Doc that represent switch statement.
 *
 * \sa SwitchDoc
 */
class SwitchDocNode : public StmtDocNode {
 public:
  /*! \brief The predicates of the switch statement. */
  ffi::Array<ExprDoc> predicates;
  /*! \brief The branchs of the switch statement. */
  ffi::Array<ffi::Array<StmtDoc>> branchs;
  /*! \brief The default_branch of the switch statement. */
  ffi::Array<StmtDoc> default_branch;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SwitchDocNode>()
        .def_ro("predicates", &SwitchDocNode::predicates)
        .def_ro("branchs", &SwitchDocNode::branchs)
        .def_ro("default_branch", &SwitchDocNode::default_branch);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("msc.script.printer.SwitchDoc", SwitchDocNode, StmtDocNode);
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
  explicit SwitchDoc(ffi::Array<ExprDoc> predicates, ffi::Array<ffi::Array<StmtDoc>> branchs,
                     ffi::Array<StmtDoc> default_branch);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SwitchDoc, StmtDoc, SwitchDocNode);
};

/*!
 * \brief Doc that represents lambda definition.
 *
 * \sa LambdaDoc
 */
class LambdaDocNode : public StmtDocNode {
 public:
  /*! \brief The name of lambda. */
  IdDoc name{ffi::UnsafeInit{}};
  /*!
   * \brief The arguments of lambda.
   *
   * The `lhs` means argument name,
   * `annotation` means argument type,
   * and `rhs` means default value.
   */
  ffi::Array<AssignDoc> args;
  /*! \brief References of lambda. */
  ffi::Array<ExprDoc> refs;
  /*! \brief The body of lambda. */
  ffi::Array<StmtDoc> body;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LambdaDocNode>()
        .def_ro("name", &LambdaDocNode::name)
        .def_ro("args", &LambdaDocNode::args)
        .def_ro("refs", &LambdaDocNode::refs)
        .def_ro("body", &LambdaDocNode::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("msc.script.printer.LambdaDoc", LambdaDocNode, StmtDocNode);
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
  explicit LambdaDoc(IdDoc name, ffi::Array<AssignDoc> args, ffi::Array<ExprDoc> refs,
                     ffi::Array<StmtDoc> body);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LambdaDoc, StmtDoc, LambdaDocNode);
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_PRINTER_MSC_DOC_H_
