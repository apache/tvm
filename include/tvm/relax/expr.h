/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#ifndef TVM_RELAX_EXPR_H_
#define TVM_RELAX_EXPR_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/cow.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/function.h>
#include <tvm/ir/source_map.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/tensor.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>

#include <functional>

namespace tvm {
namespace relax {

/*!
 * \brief The unique identifier of variables.
 *
 * Id is like name to the variables,
 * except that id is unique for each Var.
 *
 * \note Do not create Id directly, they are created in Var.
 */
class IdNode : public ffi::Object {
 public:
  /*!
   * \brief The name of the variable,
   *  this only acts as a hint to the user,
   *  and is not used for equality.
   */
  ffi::String name_hint;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IdNode>().def_ro("name_hint", &IdNode::name_hint,
                                     refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindFreeVar;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.Id", IdNode, ffi::Object);
};

class Id : public ffi::ObjectRef {
 public:
  /*!
   * \brief The constructor
   * \param name_hint The name of the variable.
   */
  TVM_DLL explicit Id(ffi::String name_hint);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Id, ffi::ObjectRef, IdNode);
};

/*!
 * \brief Call corresponds to callable invocation.
 *  Corresponds to operation in computational graph terminology.
 */
class CallNode : public ExprNode {
 public:
  /*!
   * \brief The operator(function) being invoked
   *
   *  - It can be tvm::Op which corresponds to the primitive operators.
   *  - It can also be user defined functions (Function, GlobalVar, Var).
   */
  Expr op;

  /*! \brief The arguments(inputs) of the call */
  tvm::ffi::Array<Expr> args;

  /*! \brief The additional attributes */
  Attrs attrs;

  /*!
   * \brief The type information arguments of a CallNode.
   * ty_args is by default designed to be non-empty only for intrinsic op (e.g.,
   * call_tir, call_builtin_with_ctx, etc.) and calls to ExternFuncs, with the main
   * usage of type information inference.
   *
   * Regular ops also at times may have ty_args defined to specialize partial
   * or complete type information. Like VDevice customization with mixed input memory_scopes.
   * The customized pass can set this info and operator specific inference will respect it.
   */
  ffi::Array<Type> ty_args;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CallNode>()
        .def_ro("op", &CallNode::op)
        .def_ro("args", &CallNode::args)
        .def_ro("attrs", &CallNode::attrs)
        .def_ro("ty_args", &CallNode::ty_args);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.Call", CallNode, ExprNode);
};

class Call : public Expr {
 public:
  /*!
   * \brief The constructor
   * \param op The operator to be invoked.
   * \param args The arguments of the call.
   * \param attrs The attributes of the call node.
   * \param ty_args The type information arguments passed to a function.
   * \param span The source span of the expression.
   */
  TVM_DLL Call(Expr op, ffi::Array<Expr> args, Attrs attrs = Attrs(),
               ffi::Array<Type> ty_args = ffi::Array<Type>(), Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Call, Expr, CallNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(CallNode);
};

/*!
 * \brief Returns \p call with the given properties. A null property denotes 'no change'.
 * Returns \p call if all properties are unchanged. Otherwise, returns a copy with the new
 * fields.
 */
Call WithFields(Call call, ffi::Optional<Expr> opt_op = ffi::Optional<Expr>(),
                ffi::Optional<ffi::Array<Expr>> opt_args = ffi::Optional<ffi::Array<Expr>>(),
                ffi::Optional<Attrs> opt_attrs = ffi::Optional<Attrs>(),
                ffi::Optional<ffi::Array<Type>> opt_ty_args = ffi::Optional<ffi::Array<Type>>(),
                ffi::Optional<Span> opt_span = ffi::Optional<Span>());

/*! \brief Tuple container */
class TupleNode : public ExprNode {
 public:
  /*! \brief the fields of the tuple */
  tvm::ffi::Array<Expr> fields;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TupleNode>().def_ro("fields", &TupleNode::fields);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.Tuple", TupleNode, ExprNode);
};

class Tuple : public Expr {
 public:
  /*!
   * \brief The constructor
   * \param fields The fields of a tuple.
   * \param span The source span of the expression.
   */
  TVM_DLL explicit Tuple(tvm::ffi::Array<Expr> fields, Span span = Span());

  /*!
   * \brief Utility constructor to handle conversion to relax::Expr
   *
   * If the calling scope already has an array of a specific type of
   * relax expression (e.g. `ffi::Array<relax::Var>`), it must be converted
   * into an array of base type.  This constructor handles the
   * conversion to the base `ffi::Array<relax::Expr>`.
   *
   * \tparam ExprType The type of relax expression passed in as an argument.
   *
   * \param fields The fields of a tuple.
   *
   * \param span The source span of the expression.
   */
  template <typename ExprType, typename = std::enable_if_t<std::is_base_of_v<Expr, ExprType>>>
  TVM_DLL explicit Tuple(tvm::ffi::Array<ExprType> fields, Span span = Span())
      : Tuple(fields.Map([](const ExprType& expr) -> Expr { return expr; }), span) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Tuple, Expr, TupleNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TupleNode);
};

/*!
 * \brief Returns \p tuple with the given properties. A null property denotes 'no change'.
 * Returns \p tuple if all properties are unchanged. Otherwise, returns a copy with the new
 * fields.
 */
Tuple WithFields(Tuple tuple,
                 ffi::Optional<ffi::Array<Expr>> opt_fields = ffi::Optional<ffi::Array<Expr>>(),
                 ffi::Optional<Span> opt_span = ffi::Optional<Span>());

/*! \brief Get index-th field out of a tuple. */
class TupleGetItemNode : public ExprNode {
 public:
  /*! \brief The tuple Expression */
  Expr tuple;
  /*! \brief which value to get */
  int index;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TupleGetItemNode>()
        .def_ro("tuple_value", &TupleGetItemNode::tuple)
        .def_ro("index", &TupleGetItemNode::index);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.TupleGetItem", TupleGetItemNode, ExprNode);
};

class TupleGetItem : public Expr {
 public:
  /*!
   * \brief The constructor
   * \param tuple The tuple to get an element from.
   * \param index The index for extracting a value in the tuple.
   * \param span The source span of the expression.
   */
  TVM_DLL TupleGetItem(Expr tuple, int index, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TupleGetItem, Expr, TupleGetItemNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TupleGetItemNode);
};

/*!
 * \brief Returns \p tuple_get_item with the given properties. A null property denotes 'no change'.
 * Returns \p tuple_get_item if all properties are unchanged. Otherwise, returns a copy with the new
 * fields.
 */
TupleGetItem WithFields(TupleGetItem tuple_get_item,
                        ffi::Optional<Expr> opt_tuple = ffi::Optional<Expr>(),
                        ffi::Optional<int64_t> opt_index = ffi::Optional<int64_t>(),
                        ffi::Optional<Span> opt_span = ffi::Optional<Span>());

/*! \brief A shape expression which allows users to construct a shape containing PrimExpr.
 */
class ShapeExprNode : public ExprNode {
 public:
  /*! The values of the shape expression. */
  ffi::Array<PrimExpr> values;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ShapeExprNode>().def_ro("values", &ShapeExprNode::values);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.ShapeExpr", ShapeExprNode, ExprNode);
};

class ShapeExpr : public Expr {
 public:
  TVM_DLL explicit ShapeExpr(ffi::Array<PrimExpr> values, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ShapeExpr, Expr, ShapeExprNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ShapeExprNode);
};

/*! \brief The variable class for all Relax bindings. */
class VarNode : public ExprNode {
 public:
  /*! \brief The identifier of the variable, which is used for comparing stable equality across
   * transformations. */
  Id vid;

  /*! \return The name hint of the variable */
  const ffi::String& name_hint() const { return vid->name_hint; }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<VarNode>().def_ro("vid", &VarNode::vid);
    // customize structural equal and hash to include ty
    refl::TypeAttrDef<VarNode>()
        .def("__s_equal__", &VarNode::SEqual)
        .def("__s_hash__", &VarNode::SHash);
  }

  bool SEqual(const VarNode* other,
              ffi::TypedFunction<bool(AnyView, AnyView, bool, AnyView)> equal) const {
    return equal(vid, other->vid, false, "vid") && equal(ty, other->ty, false, "ty");
  }

  int64_t SHash(int64_t init_hash, ffi::TypedFunction<int64_t(AnyView, int64_t, bool)> hash) const {
    int64_t hash_value = init_hash;
    hash_value = hash(vid, hash_value, false);
    hash_value = hash(ty, hash_value, false);
    return hash_value;
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindDAGNode;
  static constexpr const uint32_t _type_child_slots = 1;
  TVM_FFI_DECLARE_OBJECT_INFO("relax.expr.Var", VarNode, ExprNode);
};

class Var : public Expr {
 public:
  TVM_DLL explicit Var(ffi::String name_hint, ffi::Optional<Type> ty_annotation, Span span = Span())
      : Var(Id(name_hint), ty_annotation, span) {}

  TVM_DLL explicit Var(Id vid, ffi::Optional<Type> ty_annotation, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Var, Expr, VarNode);

  VarNode* CopyOnWrite();
};

/*! \brief A sub-type of the variable node used to mark dataflow variables from
 * normal visible "function local" bindings.
 */
class DataflowVarNode : public VarNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DataflowVarNode>();
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindDAGNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.DataflowVar", DataflowVarNode, VarNode);
};

class DataflowVar : public Var {
 public:
  TVM_DLL explicit DataflowVar(ffi::String name_hint, ffi::Optional<Type> ty_annotation,
                               Span span = Span())
      : DataflowVar(Id(name_hint), ty_annotation, span) {}

  TVM_DLL explicit DataflowVar(Id vid, ffi::Optional<Type> ty_annotation, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DataflowVar, Var, DataflowVarNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DataflowVarNode);
};

/*!
 * \brief Constant tensor.
 *
 * \note Scalar constants are represented by ndim-0 constant tensors.
 */
class ConstantNode : public ExprNode {
 public:
  /*! \brief The data of the tensor */
  runtime::Tensor data;

  /*! \return The corresponding tensor type of the data */
  TensorType tensor_type() const;

  /*! \return Whether it is scalar(ndim-0 tensor) */
  bool is_scalar() const { return data->ndim == 0; }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ConstantNode>().def_ro("data", &ConstantNode::data);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.Constant", ConstantNode, ExprNode);
};

class Constant : public Expr {
 public:
  /*!
   * \brief The constructor
   * \param data The data of the constant tensor.
   * \param ty_annotation The type of the constant tensor.
   *        If not specified, infer it from data.
   * \param span The source span of the expression.
   */
  TVM_DLL explicit Constant(runtime::Tensor data, ffi::Optional<Type> ty_annotation = std::nullopt,
                            Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Constant, Expr, ConstantNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ConstantNode);
};

/*!
 * \brief Represent a string literal constant.
 */
class StringImmNode : public ExprNode {
 public:
  /*! \brief The data value. */
  ffi::String value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StringImmNode>().def_ro("value", &StringImmNode::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.StringImm", StringImmNode, ExprNode);
};

/*!
 * \brief Managed reference to StringImm
 * \sa StringImmNode
 */
class StringImm : public Expr {
 public:
  /*!
   * \brief The constructor
   * \param value The value input.
   * \param span The source span of the expression.
   */
  TVM_DLL explicit StringImm(ffi::String value, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(StringImm, Expr, StringImmNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(StringImmNode);
};

/*!
 * \brief Represent a data type constant.
 */
class DataTypeImmNode : public ExprNode {
 public:
  /*! \brief The data value. */
  DLDataType value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DataTypeImmNode>().def_ro("value", &DataTypeImmNode::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.DataTypeImm", DataTypeImmNode, ExprNode);
};

/*!
 * \brief Managed reference to DataTypeImm
 * \sa DataTypeImmNode
 */
class DataTypeImm : public Expr {
 public:
  /*!
   * \brief The constructor
   * \param value The value input.
   * \param span The source span of the expression.
   */
  TVM_DLL explicit DataTypeImm(DLDataType value, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DataTypeImm, Expr, DataTypeImmNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DataTypeImmNode);
};

/*! \brief The base class of a variable binding in Relax. */
class BindingNode : public ffi::Object {
 public:
  mutable Span span;
  /*! \brief The return variable to bound to. */
  Var var;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BindingNode>()
        .def_ro("span", &BindingNode::span, refl::AttachFieldFlag::SEqHashIgnore())
        // TODO(tqchen): use SEqHashDefNonRecursive after the next pypi tvm-ffi release
        .def_ro("var", &BindingNode::var, refl::AttachFieldFlag::SEqHashDefRecursive());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;

  TVM_FFI_DECLARE_OBJECT_INFO("relax.expr.Binding", BindingNode, ffi::Object);
};

class Binding : public ffi::ObjectRef {
 protected:
  Binding() = default;

 public:
  explicit Binding(ffi::ObjectPtr<BindingNode> n) : ffi::ObjectRef(n) {}
  explicit Binding(ffi::UnsafeInit tag) : ffi::ObjectRef(tag) {}
  Binding(const Binding&) = default;
  Binding(Binding&&) = default;
  Binding& operator=(const Binding&) = default;
  Binding& operator=(Binding&&) = default;
  const BindingNode* operator->() const { return static_cast<const BindingNode*>(data_.get()); }
  const BindingNode* get() const { return operator->(); }
  using ContainerType = BindingNode;
};

/*!
 * \brief Runtime-match the value to the type.
 *
 * This operation does runtime check, populates the un-defined symbolic shape vars
 * and vars in ty in first occurance, and insert equality assertions in
 * other cases.
 */
class MatchCastNode : public BindingNode {
 public:
  /*! \brief The input value to match cast. */
  Expr value;
  /*! \brief The type pattern to match to. */
  Type ty;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MatchCastNode>()
        .def_ro("value", &MatchCastNode::value)
        // TODO(tqchen): use SEqHashDefNonRecursive after the next pypi tvm-ffi release
        .def_ro("ty", &MatchCastNode::ty, refl::AttachFieldFlag::SEqHashDefRecursive());
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.MatchCast", MatchCastNode, BindingNode);
};

/*!
 * \brief Managed reference to MatchCastNode.
 * \sa MatchCastNode
 */
class MatchCast : public Binding {
 public:
  TVM_DLL explicit MatchCast(Var var, Expr value, Type ty, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(MatchCast, Binding, MatchCastNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MatchCastNode);
};

class VarBindingNode : public BindingNode {
 public:
  /*! \brief The binding value. */
  Expr value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<VarBindingNode>().def_ro("value", &VarBindingNode::value);
    // customize the SEqual and SHash methods for better error messages
    refl::TypeAttrDef<VarBindingNode>()
        .def("__s_equal__", &VarBindingNode::SEqual)
        .def("__s_hash__", &VarBindingNode::SHash);
  }

  bool SEqual(const VarBindingNode* other,
              ffi::TypedFunction<bool(AnyView, AnyView, bool, AnyView)> equal) const;
  int64_t SHash(int64_t init_hash, ffi::TypedFunction<int64_t(AnyView, int64_t, bool)> hash) const;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.VarBinding", VarBindingNode, BindingNode);
};

class VarBinding : public Binding {
 public:
  TVM_DLL explicit VarBinding(Var var, Expr value, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(VarBinding, Binding, VarBindingNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(VarBindingNode);
};

class BindingBlockNode : public ffi::Object {
 public:
  ffi::Array<Binding> bindings;
  mutable Span span;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BindingBlockNode>()
        .def_ro("bindings", &BindingBlockNode::bindings)
        .def_ro("span", &BindingBlockNode::span, refl::AttachFieldFlag::SEqHashIgnore(),
                refl::DefaultValue(Span()));
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO("relax.expr.BindingBlock", BindingBlockNode, ffi::Object);
};

class BindingBlock : public ffi::ObjectRef {
 public:
  TVM_DLL explicit BindingBlock(ffi::Array<Binding> bindings, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(BindingBlock, ffi::ObjectRef, BindingBlockNode);

  BindingBlockNode* CopyOnWrite();
};

class DataflowBlockNode : public BindingBlockNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DataflowBlockNode>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.DataflowBlock", DataflowBlockNode,
                                    BindingBlockNode);
};

class DataflowBlock : public BindingBlock {
 public:
  TVM_DLL explicit DataflowBlock(ffi::Array<Binding> bindings, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DataflowBlock, BindingBlock, DataflowBlockNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DataflowBlockNode);
};

/*! \brief A sequence of blocks followed by an expression.
 *
 * The order of blocks enforces scoping and ordering.
 */
class SeqExprNode : public ExprNode {
 public:
  ffi::Array<BindingBlock> blocks;
  Expr body;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SeqExprNode>()
        .def_ro("blocks", &SeqExprNode::blocks)
        .def_ro("body", &SeqExprNode::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.SeqExpr", SeqExprNode, ExprNode);
};

class SeqExpr : public Expr {
 public:
  /* \brief Implicit conversion constructor
   *
   * Relax nodes that introduce a new scope (e.g. `relax::Function`)
   * are required to be held as SeqExpr.  This implicit conversion
   * provides allows callsites to use these member variables when the
   * C++ compile-time type is a `relax::Expr`.  For example,
   * a transform may use `func.CopyOnWrite()->body = expr;`.
   *
   * If the expression is already a `relax::SeqExpr`, the same
   * underlying `relax::SeqExprNode` is used, and no copies are made.
   */
  TVM_DLL SeqExpr(Expr body);  // NOLINT(*)

  TVM_DLL explicit SeqExpr(ffi::Array<BindingBlock> blocks, Expr body, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SeqExpr, Expr, SeqExprNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SeqExprNode);
};

/*!
 * \brief Condition expression
 *
 * Unlike traditional statement `if`s, the if evalutes
 * to the result of the branch taken.
 *
 * x = if (true) { 1 } else { 0 }; // x is 1
 * y = if (false) { 1 } else { 0 }; // y is 0
 *
 * \note This is similar to C's ternary operator.
 */
class IfNode : public ExprNode {
 public:
  /*! \brief The condition. */
  Expr cond;
  /*! \brief The expression evaluated when condition is true. */
  SeqExpr true_branch;
  /*! \brief The expression evaluated when condition is false */
  SeqExpr false_branch;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IfNode>()
        .def_ro("cond", &IfNode::cond)
        .def_ro("true_branch", &IfNode::true_branch)
        .def_ro("false_branch", &IfNode::false_branch);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindDAGNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.If", IfNode, ExprNode);
};

class If : public Expr {
 public:
  /*!
   * \brief The constructor
   *
   * \param cond The condition of a if node.
   *
   * \param true_branch The fall through branch.  If this is not a
   *     SeqExpr, it will be wrapped in a SeqExpr, to satisfy the
   *     Relax IR requirement that all scopes be contained in a
   *     SeqExpr.
   *
   * \param false_branch The branch for execution when condition is
   *     false.  If this is not a SeqExpr, it will be wrapped in a
   *     SeqExpr, to satisfy the Relax IR requirement that all scopes
   *     be contained in a SeqExpr.
   *
   * \param span The source span of the expression.
   */
  TVM_DLL If(Expr cond, Expr true_branch, Expr false_branch, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(If, Expr, IfNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(IfNode);
};

/*!
 * \brief Returns \p if_expr with the given properties. A null property denotes 'no change'.
 * Returns \p if_expr if all properties are unchanged. Otherwise, returns a copy with the new
 * fields.
 */
If WithFields(If if_expr, ffi::Optional<Expr> opt_cond = ffi::Optional<Expr>(),
              ffi::Optional<Expr> opt_true_branch = ffi::Optional<Expr>(),
              ffi::Optional<Expr> opt_false_branch = ffi::Optional<Expr>(),
              ffi::Optional<Span> opt_span = ffi::Optional<Span>());

/*! \brief A Relax function. */
class FunctionNode : public BaseFuncNode {
 public:
  /*! \brief The parameters to the function. */
  ffi::Array<Var> params;
  /*! \brief The body of the function. */
  SeqExpr body;
  /*! \brief The return type of the function. */
  Type ret_ty;
  /*! \brief Whether the function is annotated as pure or not. */
  bool is_pure;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FunctionNode>()
        .def_ro("params", &FunctionNode::params, refl::AttachFieldFlag::SEqHashDefRecursive())
        .def_ro("body", &FunctionNode::body)
        .def_ro("ret_ty", &FunctionNode::ret_ty)
        .def_ro("is_pure", &FunctionNode::is_pure);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindDAGNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.Function", FunctionNode, BaseFuncNode);
};

class Function : public BaseFunc {
 public:
  /*!
   * \brief Construct a Relax Function
   *
   * \param params The parameters accepted by the function
   *
   * \param body The body of the function.  If this is not a
   *     SeqExpr, it will be wrapped in a SeqExpr, to satisfy the
   *     Relax IR requirement that all scopes be contained in a
   *     SeqExpr.
   *
   * \param ret_ty The Type returned by the function.
   *     If std::nullopt, will be inferred from the Type of the
   *     function's body.
   *
   * \param is_pure The purity of the function.
   *
   * \param attrs Any attributes associated with the function.
   *     Defaults to an empty dictionary.
   *
   * \param span The source span of the expression.
   */
  TVM_DLL explicit Function(ffi::Array<Var> params, Expr body, ffi::Optional<Type> ret_ty,
                            bool is_pure = true, DictAttrs attrs = DictAttrs(), Span span = Span());

  /*!
   * \brief Mimics the constructor but without body Expr.
   * \note ret_ty is required, since it can not deduced by the body.
   */
  TVM_DLL static Function CreateEmpty(ffi::Array<Var> params, Type ret_ty, bool is_pure = true,
                                      DictAttrs attrs = DictAttrs(), Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Function, BaseFunc, FunctionNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FunctionNode);
};

// TODO(@sunggg): Investigate the exact usage of kComposite, kPartitionedFromPattern, and
// kPrimitive.
namespace attr {
/*! \brief Mark the function as a primitive function. */
constexpr const char* kPrimitive = "Primitive";
/*!
 * \brief Indicate the codegen that should be used for building this function.
 * When this is unset or set to "default", the default compilation pipeline will be used.
 */
constexpr const char* kCodegen = "Codegen";
/*! \brief Treat the function as a composite operator. */
constexpr const char* kComposite = "Composite";
/*! \brief Indicate the function was created by the Pattern Partitioning Pass. */
constexpr const char* kPartitionedFromPattern = "PartitionedFromPattern";
/*! \brief The required workspace for an external function. */
constexpr const char* kWorkspaceSize = "WorkspaceSize";

// Note: in the future, we prefer snake_case instead of CamelCase for attributes.
// Past ones will be kept for backwards compatibility.
/*! \brief Override checking purity for this function and treat as pure
 * (is_pure must be set to true) */
constexpr const char* kForcePure = "relax.force_pure";

/*!
 * \brief The number of inputs of a function.
 * If a function has the num_input attribute, the last func->params.size() - num_inputs
 * arguments are assumed to be weights that are fixed across invocations.
 */
constexpr const char* kNumInput = "num_input";
}  // namespace attr

/*! \brief The extern function, which can represent packed function. */
class ExternFuncNode : public BaseFuncNode {
 public:
  /*! \brief The name of global symbol. */
  ffi::String global_symbol;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ExternFuncNode>().def_ro("global_symbol", &ExternFuncNode::global_symbol);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.expr.ExternFunc", ExternFuncNode, BaseFuncNode);
};

class ExternFunc : public BaseFunc {
 public:
  TVM_DLL ExternFunc(ffi::String global_symbol, Span span = Span());
  TVM_DLL ExternFunc(ffi::String global_symbol, Type ty, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ExternFunc, BaseFunc, ExternFuncNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ExternFuncNode);
};

/*!
 * \brief Get the shape of Expr.
 * \param expr The input expr.
 * \return The corresonding shape.
 *
 * \note This function requires expr to be normalized.
 *       The function will report an error if expr's Type is not TensorType.
 *       It will try to return symbolic function when possible. If the tensor do not
 *       have a compile-time symbolic shape, the function will then choose to return
 *       Call(relax.op.shape_of, [expr]).
 */
TVM_DLL Expr GetShapeOf(const Expr& expr);

}  // namespace relax
}  // namespace tvm

/* \brief Allow relax.Var as key in STL tables
 *
 * For most Relax expressions, it would be ambiguous whether the
 * expression should follow reference equality or structural equality.
 * This is not the case for variables, which do not contain nested
 * internal structure, and are frequently used as keys in lookup
 * tables.
 *
 * Providing `std::hash` and `std::equal_to` specializations for
 * `relax::Var` allows it to be used as a key in STL tables.  For
 * `relax::Expr`, the user must specify the type of equality used
 * (e.g. `std::unordered_set<T, StructuralHash, StructuralEqual>` or
 * `std::unordered_set<T, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>`).
 */
template <>
struct std::hash<tvm::relax::Var> {
  std::size_t operator()(const tvm::relax::Var& var) const {
    return tvm::ffi::ObjectPtrHash()(var);
  }
};

template <>
struct std::equal_to<tvm::relax::Var> {
  bool operator()(const tvm::relax::Var& var_a, const tvm::relax::Var& var_b) const {
    return tvm::ffi::ObjectPtrEqual()(var_a, var_b);
  }
};

#endif  // TVM_RELAX_EXPR_H_
