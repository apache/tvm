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

#include <tvm/ir/expr.h>
#include <tvm/ir/source_map.h>
#include <tvm/node/node.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/object.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <functional>

namespace tvm {
namespace relax {

using Expr = RelayExpr;
using ExprNode = RelayExprNode;
/*!
 * \brief The unique identifier of variables.
 *
 * Id is like name to the variables,
 * except that id is unique for each Var.
 *
 * \note Do not create Id directly, they are created in Var.
 */
class IdNode : public Object {
 public:
  /*!
   * \brief The name of the variable,
   *  this only acts as a hint to the user,
   *  and is not used for equality.
   */
  String name_hint;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("name_hint", &name_hint); }

  bool SEqualReduce(const IdNode* other, SEqualReducer equal) const {
    return equal.FreeVarEqualImpl(this, other);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce.FreeVarHashImpl(this); }

  static constexpr const char* _type_key = "relax.Id";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(IdNode, Object);
};

class Id : public ObjectRef {
 public:
  /*!
   * \brief The constructor
   * \param name_hint The name of the variable.
   */
  TVM_DLL explicit Id(String name_hint);

  TVM_DEFINE_OBJECT_REF_METHODS(Id, ObjectRef, IdNode);
};

/*!
 * \brief Base type of all structure information.
 *
 * StructInfo stores possible structure information
 * deduced during compile-time. It encapsulates
 * both static type and runtime information such
 * as shape.
 *
 * StructInfo of each non-primitive Expr can be
 * deduced during compilation in a "best-effort" manner.
 *
 * When struct_info appears in function parameter and return
 * signatures. They will imply a runtime check that matches
 * the structure information with the value.
 *
 * When it appears in Expr, they follow "assume-semantics",
 * which means the compiler will take the deduced information as it is
 * and only do best effort prove and checks.
 *
 * Each struct info can be uniquely erased to a static-type.
 * The compiler will still compile the code(with less information)
 * when we erase to the static type.
 *
 * If an StructInfo contains an Expr field, then that field
 * must be normalized already through NormalizeArg.
 * This invariant will be checked in constructors
 * and help us to simplify our assumption
 * during struct info deduction.
 */
class StructInfoNode : public Object {
 public:
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;

  static constexpr const char* _type_key = "StructInfo";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const uint32_t _type_child_slots = 5;
  TVM_DECLARE_BASE_OBJECT_INFO(StructInfoNode, Object);
};

/*!
 * \brief Managed reference to StructInfoNode.
 * \sa StructInfoNode
 */
class StructInfo : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(StructInfo, ObjectRef, StructInfoNode);
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
  tvm::Array<Expr> args;

  /*! \brief The additional attributes */
  Attrs attrs;

  /*!
   * \brief The structure info arguments of a CallNode.
   * sinfo_args is designed to be non-empty only for intrinsic op (e.g.,
   * call_tir, call_builtin_with_ctx, etc.) and calls to ExternFuncs, with the main
   * usage of structure info inference.
   */
  Array<StructInfo> sinfo_args;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("op", &op);
    v->Visit("args", &args);
    v->Visit("attrs", &attrs);
    v->Visit("sinfo_args", &sinfo_args);
    v->Visit("struct_info_", &struct_info_);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const CallNode* other, SEqualReducer equal) const {
    // skip sinfo_args check for primitive ops.
    return equal(op, other->op) && equal(args, other->args) && equal(attrs, other->attrs) &&
           equal(sinfo_args, other->sinfo_args) && equal(struct_info_, other->struct_info_);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(op);
    hash_reduce(args);
    hash_reduce(attrs);
    hash_reduce(sinfo_args);
    hash_reduce(struct_info_);
  }

  static constexpr const char* _type_key = "relax.expr.Call";
  TVM_DECLARE_FINAL_OBJECT_INFO(CallNode, ExprNode);
};

class Call : public Expr {
 public:
  /*!
   * \brief The constructor
   * \param op The operator to be invoked.
   * \param args The arguments of the call.
   * \param attrs The attributes of the call node.
   * \param sinfo_args The structure info arguments passed to a function.
   * \param span The source span of the expression.
   */
  TVM_DLL Call(Expr op, Array<Expr> args, Attrs attrs = Attrs(),
               Array<StructInfo> sinfo_args = Array<StructInfo>(), Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(Call, Expr, CallNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(CallNode);
};

/*!
 * \brief Returns \p call with the given properties. A null property denotes 'no change'.
 * Returns \p call if all properties are unchanged. Otherwise, returns a copy with the new
 * fields.
 */
Call WithFields(Call call, Optional<Expr> opt_op = Optional<Expr>(),
                Optional<Array<Expr>> opt_args = Optional<Array<Expr>>(),
                Optional<Attrs> opt_attrs = Optional<Attrs>(),
                Optional<Array<StructInfo>> opt_sinfo_args = Optional<Array<StructInfo>>(),
                Optional<Span> opt_span = Optional<Span>());

/*! \brief Tuple container */
class TupleNode : public ExprNode {
 public:
  /*! \brief the fields of the tuple */
  tvm::Array<Expr> fields;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("fields", &fields);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("struct_info_", &struct_info_);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const TupleNode* other, SEqualReducer equal) const {
    // struct info can be deterministically derived from fields.
    return equal(fields, other->fields);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(fields); }

  static constexpr const char* _type_key = "relax.expr.Tuple";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleNode, ExprNode);
};

class Tuple : public Expr {
 public:
  /*!
   * \brief The constructor
   * \param fields The fields of a tuple.
   * \param span The source span of the expression.
   */
  TVM_DLL explicit Tuple(tvm::Array<Expr> fields, Span span = Span());

  /*!
   * \brief Utility constructor to handle conversion to relax::Expr
   *
   * If the calling scope already has an array of a specific type of
   * relax expression (e.g. `Array<relax::Var>`), it must be converted
   * into an array of base type.  This constructor handles the
   * conversion to the base `Array<relax::Expr>`.
   *
   * \tparam RelaxExpr The type of relax expression passed in as an argument.
   *
   * \param fields The fields of a tuple.
   *
   * \param span The source span of the expression.
   */
  template <typename RelaxExpr, typename = std::enable_if_t<std::is_base_of_v<Expr, RelaxExpr>>>
  TVM_DLL explicit Tuple(tvm::Array<RelaxExpr> fields, Span span = Span())
      : Tuple(fields.Map([](const RelaxExpr& expr) -> Expr { return expr; }), span) {}

  TVM_DEFINE_OBJECT_REF_METHODS(Tuple, Expr, TupleNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TupleNode);
};

/*!
 * \brief Returns \p tuple with the given properties. A null property denotes 'no change'.
 * Returns \p tuple if all properties are unchanged. Otherwise, returns a copy with the new
 * fields.
 */
Tuple WithFields(Tuple tuple, Optional<Array<Expr>> opt_fields = Optional<Array<Expr>>(),
                 Optional<Span> opt_span = Optional<Span>());

/*! \brief Get index-th field out of a tuple. */
class TupleGetItemNode : public ExprNode {
 public:
  /*! \brief The tuple Expression */
  Expr tuple;
  /*! \brief which value to get */
  int index;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("tuple_value", &tuple);
    v->Visit("index", &index);
    v->Visit("struct_info_", &struct_info_);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const TupleGetItemNode* other, SEqualReducer equal) const {
    // struct info can be deterministically tuple and index.
    return equal(tuple, other->tuple) && equal(index, other->index);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(tuple);
    hash_reduce(index);
  }

  static constexpr const char* _type_key = "relax.expr.TupleGetItem";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleGetItemNode, ExprNode);
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

  TVM_DEFINE_OBJECT_REF_METHODS(TupleGetItem, Expr, TupleGetItemNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TupleGetItemNode);
};

/*!
 * \brief Returns \p tuple_get_item with the given properties. A null property denotes 'no change'.
 * Returns \p tuple_get_item if all properties are unchanged. Otherwise, returns a copy with the new
 * fields.
 */
TupleGetItem WithFields(TupleGetItem tuple_get_item, Optional<Expr> opt_tuple = Optional<Expr>(),
                        Optional<Integer> opt_index = Optional<Integer>(),
                        Optional<Span> opt_span = Optional<Span>());

/*!
 * \brief Base type of all (non-function) leaf Exprs.
 * \sa Expr
 */
class LeafExprNode : public ExprNode {
 public:
  static constexpr const char* _type_key = "relax.expr.LeafExpr";
  static constexpr const uint32_t _type_child_slots = 7;
  TVM_DECLARE_BASE_OBJECT_INFO(LeafExprNode, ExprNode);
};

/*!
 * \brief Managed reference to BaseExprNode.
 * \sa LeafExprNode
 */
class LeafExpr : public Expr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(LeafExpr, Expr, LeafExprNode);
};

/*! \brief A shape expression which allows users to construct a shape containing PrimExpr.
 */
class ShapeExprNode : public LeafExprNode {
 public:
  /*! The values of the shape expression. */
  Array<PrimExpr> values;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("values", &values);
    v->Visit("struct_info_", &struct_info_);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const ShapeExprNode* other, SEqualReducer equal) const {
    // struct info can be deterministically derived from values.
    return equal(values, other->values);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(values); }

  static constexpr const char* _type_key = "relax.expr.ShapeExpr";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(ShapeExprNode, LeafExprNode);
};

class ShapeExpr : public LeafExpr {
 public:
  TVM_DLL explicit ShapeExpr(Array<PrimExpr> values, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(ShapeExpr, LeafExpr, ShapeExprNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ShapeExprNode);
};

/*! \brief The variable class for all Relax bindings. */
class VarNode : public LeafExprNode {
 public:
  /*! \brief The identifier of the variable, which is used for comparing stable equality across
   * transformations. */
  Id vid;

  /*! \return The name hint of the variable */
  const String& name_hint() const { return vid->name_hint; }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("vid", &vid);
    v->Visit("struct_info_", &struct_info_);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const VarNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(vid, other->vid) && equal(struct_info_, other->struct_info_);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(vid);
    hash_reduce(struct_info_);
  }

  static constexpr const char* _type_key = "relax.expr.Var";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const uint32_t _type_child_slots = 2;
  TVM_DECLARE_BASE_OBJECT_INFO(VarNode, LeafExprNode);
};

class Var : public LeafExpr {
 public:
  TVM_DLL explicit Var(String name_hint, Optional<StructInfo> struct_info_annotation,
                       Span span = Span())
      : Var(Id(name_hint), struct_info_annotation, span) {}

  TVM_DLL explicit Var(Id vid, Optional<StructInfo> struct_info_annotation, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(Var, LeafExpr, VarNode);

  VarNode* CopyOnWrite();
};

/*! \brief A sub-type of the variable node used to mark dataflow variables from
 * normal visible "function local" bindings.
 */
class DataflowVarNode : public VarNode {
 public:
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("vid", &vid);
    v->Visit("struct_info_", &struct_info_);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const DataflowVarNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(vid, other->vid) && equal(struct_info_, other->struct_info_);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(vid);
    hash_reduce(struct_info_);
  }

  static constexpr const char* _type_key = "relax.expr.DataflowVar";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(DataflowVarNode, VarNode);
};

class DataflowVar : public Var {
 public:
  TVM_DLL explicit DataflowVar(String name_hint, Optional<StructInfo> struct_info_annotation,
                               Span span = Span())
      : DataflowVar(Id(name_hint), struct_info_annotation, span) {}

  TVM_DLL explicit DataflowVar(Id vid, Optional<StructInfo> struct_info_annotation,
                               Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(DataflowVar, Var, DataflowVarNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DataflowVarNode);
};

/*!
 * \brief Constant tensor.
 *
 * \note Scalar constants are represented by ndim-0 constant tensors.
 */
class ConstantNode : public LeafExprNode {
 public:
  /*! \brief The data of the tensor */
  runtime::NDArray data;

  /*! \return The corresponding tensor type of the data */
  TensorType tensor_type() const;

  /*! \return Whether it is scalar(ndim-0 tensor) */
  bool is_scalar() const { return data->ndim == 0; }

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("data", &data);
    v->Visit("struct_info_", &struct_info_);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const ConstantNode* other, SEqualReducer equal) const {
    // struct info can be deterministically derived from data.
    return equal(data, other->data) && equal(struct_info_, other->struct_info_);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(data);
    hash_reduce(struct_info_);
  }

  static constexpr const char* _type_key = "relax.expr.Constant";
  TVM_DECLARE_FINAL_OBJECT_INFO(ConstantNode, LeafExprNode);
};

class Constant : public LeafExpr {
 public:
  /*!
   * \brief The constructor
   * \param data The data of the constant tensor.
   * \param struct_info_annotation The struct info of the constant tensor.
   *        If not specified, infer it from data.
   * \param span The source span of the expression.
   */
  TVM_DLL explicit Constant(runtime::NDArray data,
                            Optional<StructInfo> struct_info_annotation = NullOpt,
                            Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(Constant, LeafExpr, ConstantNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ConstantNode);
};

/*!
 * \brief PrimValue.
 *
 * Expression representing a TIR POD expression.
 */
class PrimValueNode : public LeafExprNode {
 public:
  /*! \brief The prim expr representing the value */
  PrimExpr value;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("value", &value);
    v->Visit("struct_info_", &struct_info_);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const PrimValueNode* other, SEqualReducer equal) const {
    // struct info can be deterministically derived from data.
    return equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(value); }

  static constexpr const char* _type_key = "relax.expr.PrimValue";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrimValueNode, LeafExprNode);
};

/*!
 * \brief Managed reference to PrimValueNode
 * \sa PrimValeNode
 */
class PrimValue : public LeafExpr {
 public:
  /*!
   * \brief The constructor
   * \param value The value input.
   * \param span The source span of the expression.
   */
  TVM_DLL explicit PrimValue(PrimExpr value, Span span = Span());

  /*!
   * \brief Create a int64 prim value.
   * \param value The input value.
   * \param span The source span of the expression.
   * \return The created prim value.
   */
  TVM_DLL static PrimValue Int64(int64_t value, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(PrimValue, LeafExpr, PrimValueNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(PrimValueNode);
};

/*!
 * \brief Represent a string literal constant.
 */
class StringImmNode : public LeafExprNode {
 public:
  /*! \brief The data value. */
  String value;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("value", &value);
    v->Visit("struct_info_", &struct_info_);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const StringImmNode* other, SEqualReducer equal) const {
    // struct info can be deterministically derived from data.
    return equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(value); }

  static constexpr const char* _type_key = "relax.expr.StringImm";
  TVM_DECLARE_FINAL_OBJECT_INFO(StringImmNode, LeafExprNode);
};

/*!
 * \brief Managed reference to StringImm
 * \sa StringImmNode
 */
class StringImm : public LeafExpr {
 public:
  /*!
   * \brief The constructor
   * \param value The value input.
   * \param span The source span of the expression.
   */
  TVM_DLL explicit StringImm(String value, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(StringImm, LeafExpr, StringImmNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(StringImmNode);
};

/*!
 * \brief Represent a data type constant.
 */
class DataTypeImmNode : public LeafExprNode {
 public:
  /*! \brief The data value. */
  DataType value;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("value", &value);
    v->Visit("struct_info_", &struct_info_);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const DataTypeImmNode* other, SEqualReducer equal) const {
    // struct info can be deterministically derived from data.
    return equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(value); }

  static constexpr const char* _type_key = "relax.expr.DataTypeImm";
  TVM_DECLARE_FINAL_OBJECT_INFO(DataTypeImmNode, LeafExprNode);
};

/*!
 * \brief Managed reference to DataTypeImm
 * \sa DataTypeImmNode
 */
class DataTypeImm : public LeafExpr {
 public:
  /*!
   * \brief The constructor
   * \param value The value input.
   * \param span The source span of the expression.
   */
  TVM_DLL explicit DataTypeImm(DataType value, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(DataTypeImm, LeafExpr, DataTypeImmNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DataTypeImmNode);
};

/*! \brief The base class of a variable binding in Relax. */
class BindingNode : public Object {
 public:
  /*! \brief The return variable to bound to. */
  Var var;
  mutable Span span;

  static constexpr const char* _type_key = "relax.expr.Binding";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_BASE_OBJECT_INFO(BindingNode, Object);
};

class Binding : public ObjectRef {
 protected:
  Binding() = default;

 public:
  explicit Binding(ObjectPtr<Object> n) : ObjectRef(n) {}
  TVM_DEFINE_DEFAULT_COPY_MOVE_AND_ASSIGN(Binding);
  const BindingNode* operator->() const { return static_cast<const BindingNode*>(data_.get()); }
  const BindingNode* get() const { return operator->(); }
  using ContainerType = BindingNode;
};

/*!
 * \brief Runtime-match the value to the struct info.
 *
 * This operation does runtime check, populates the un-defined symbolic shape vars
 * and vars in struct_info in first occurance, and insert equality assertions in
 * other cases.
 */
class MatchCastNode : public BindingNode {
 public:
  /*! \brief The input value to match cast. */
  Expr value;
  /*! \brief The struct info pattern to match to. */
  StructInfo struct_info;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("var", &var);
    v->Visit("value", &value);
    v->Visit("struct_info", &struct_info);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const MatchCastNode* other, SEqualReducer equal) const;
  void SHashReduce(SHashReducer hash_reduce) const;

  static constexpr const char* _type_key = "relax.expr.MatchCast";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(MatchCastNode, BindingNode);
};

/*!
 * \brief Managed reference to MatchCastNode.
 * \sa MatchCastNode
 */
class MatchCast : public Binding {
 public:
  TVM_DLL explicit MatchCast(Var var, Expr value, StructInfo struct_info, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(MatchCast, Binding, MatchCastNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MatchCastNode);
};

class VarBindingNode : public BindingNode {
 public:
  /*! \brief The binding value. */
  Expr value;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("var", &var);
    v->Visit("value", &value);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const VarBindingNode* other, SEqualReducer equal) const;
  void SHashReduce(SHashReducer hash_reduce) const;

  static constexpr const char* _type_key = "relax.expr.VarBinding";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(VarBindingNode, BindingNode);
};

class VarBinding : public Binding {
 public:
  TVM_DLL explicit VarBinding(Var var, Expr value, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(VarBinding, Binding, VarBindingNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(VarBindingNode);
};

class BindingBlockNode : public Object {
 public:
  mutable Span span;
  Array<Binding> bindings;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("span", &span);
    v->Visit("bindings", &bindings);
  }

  bool SEqualReduce(const BindingBlockNode* other, SEqualReducer equal) const {
    return equal(bindings, other->bindings);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(bindings); }

  static constexpr const char* _type_key = "relax.expr.BindingBlock";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_BASE_OBJECT_INFO(BindingBlockNode, Object);
};

class BindingBlock : public ObjectRef {
 public:
  TVM_DLL explicit BindingBlock(Array<Binding> bindings, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(BindingBlock, ObjectRef, BindingBlockNode);

  BindingBlockNode* CopyOnWrite();
};

class DataflowBlockNode : public BindingBlockNode {
 public:
  bool SEqualReduce(const DataflowBlockNode* other, SEqualReducer equal) const {
    return equal(bindings, other->bindings);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(bindings); }

  static constexpr const char* _type_key = "relax.expr.DataflowBlock";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(DataflowBlockNode, BindingBlockNode);
};

class DataflowBlock : public BindingBlock {
 public:
  TVM_DLL explicit DataflowBlock(Array<Binding> bindings, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(DataflowBlock, BindingBlock, DataflowBlockNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DataflowBlockNode);
};

/*! \brief A sequence of blocks followed by an expression.
 *
 * The order of blocks enforces scoping and ordering.
 */
class SeqExprNode : public ExprNode {
 public:
  Array<BindingBlock> blocks;
  Expr body;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("blocks", &blocks);
    v->Visit("body", &body);
    v->Visit("struct_info_", &struct_info_);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const SeqExprNode* other, SEqualReducer equal) const {
    return equal(blocks, other->blocks) && equal(body, other->body) &&
           equal(struct_info_, other->struct_info_);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(blocks);
    hash_reduce(body);
    hash_reduce(struct_info_);
  }

  static constexpr const char* _type_key = "relax.expr.SeqExpr";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(SeqExprNode, ExprNode);
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

  TVM_DLL explicit SeqExpr(Array<BindingBlock> blocks, Expr body, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(SeqExpr, Expr, SeqExprNode);
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

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("cond", &cond);
    v->Visit("true_branch", &true_branch);
    v->Visit("false_branch", &false_branch);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("struct_info_", &struct_info_);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const IfNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(cond, other->cond) && equal(true_branch, other->true_branch) &&
           equal(false_branch, other->false_branch) && equal(struct_info_, other->struct_info_);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(cond);
    hash_reduce(true_branch);
    hash_reduce(false_branch);
    hash_reduce(struct_info_);
  }

  static constexpr const char* _type_key = "relax.expr.If";
  TVM_DECLARE_FINAL_OBJECT_INFO(IfNode, ExprNode);
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

  TVM_DEFINE_OBJECT_REF_METHODS(If, Expr, IfNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(IfNode);
};

/*!
 * \brief Returns \p if_expr with the given properties. A null property denotes 'no change'.
 * Returns \p if_expr if all properties are unchanged. Otherwise, returns a copy with the new
 * fields.
 */
If WithFields(If if_expr, Optional<Expr> opt_cond = Optional<Expr>(),
              Optional<Expr> opt_true_branch = Optional<Expr>(),
              Optional<Expr> opt_false_branch = Optional<Expr>(),
              Optional<Span> opt_span = Optional<Span>());

/*! \brief A Relax function. */
class FunctionNode : public BaseFuncNode {
 public:
  /*! \brief The parameters to the function. */
  Array<Var> params;
  /*! \brief The body of the function. */
  SeqExpr body;
  /*! \brief The return type of the function. */
  StructInfo ret_struct_info;
  /*! \brief Whether the function is annotated as pure or not. */
  bool is_pure;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("params", &params);
    v->Visit("body", &body);
    v->Visit("is_pure", &is_pure);
    v->Visit("ret_struct_info", &ret_struct_info);
    v->Visit("attrs", &attrs);
    v->Visit("struct_info_", &struct_info_);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const FunctionNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal.DefEqual(params, other->params) && equal(body, other->body) &&
           equal(ret_struct_info, other->ret_struct_info) && equal(is_pure, other->is_pure) &&
           equal(attrs, other->attrs) && equal(struct_info_, other->struct_info_);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce.DefHash(params);
    hash_reduce(body);
    hash_reduce(ret_struct_info);
    hash_reduce(is_pure);
    hash_reduce(attrs);
    hash_reduce(struct_info_);
  }

  static constexpr const char* _type_key = "relax.expr.Function";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionNode, BaseFuncNode);
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
   * \param ret_struct_info The StructInfo returned by the function.
   *     If NullOpt, will be inferred from the StructInfo of the
   *     function's body.
   *
   * \param is_pure The purity of the function.
   *
   * \param attrs Any attributes associated with the function.
   *     Defaults to an empty dictionary.
   *
   * \param span The source span of the expression.
   */
  TVM_DLL explicit Function(Array<Var> params, Expr body, Optional<StructInfo> ret_struct_info,
                            bool is_pure = true, DictAttrs attrs = DictAttrs(), Span span = Span());

  /*!
   * \brief Mimics the constructor but without body Expr.
   * \note ret_struct_info is required, since it can not deduced by the body.
   */
  TVM_DLL static Function CreateEmpty(Array<Var> params, StructInfo ret_struct_info,
                                      bool is_pure = true, DictAttrs attrs = DictAttrs(),
                                      Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(Function, BaseFunc, FunctionNode);
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
  String global_symbol;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("global_symbol", &global_symbol);
    v->Visit("struct_info_", &struct_info_);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const ExternFuncNode* other, SEqualReducer equal) const {
    return equal(global_symbol, other->global_symbol) && equal(struct_info_, other->struct_info_);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(global_symbol);
    hash_reduce(struct_info_);
  }

  static constexpr const char* _type_key = "relax.expr.ExternFunc";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(ExternFuncNode, BaseFuncNode);
};

class ExternFunc : public BaseFunc {
 public:
  TVM_DLL ExternFunc(String global_symbol, Span span = Span());
  TVM_DLL ExternFunc(String global_symbol, StructInfo struct_info, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(ExternFunc, BaseFunc, ExternFuncNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ExternFuncNode);
};

/*!
 * \brief Get the shape of Expr.
 * \param expr The input expr.
 * \return The corresonding shape.
 *
 * \note This function requires expr to be normalized.
 *       The function will report an error if expr's StructInfo is not TensorStructInfo.
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
 * `std::unordered_set<T, ObjectPtrHash, ObjectPtrEqual>`).
 */
template <>
struct std::hash<tvm::relax::Var> {
  std::size_t operator()(const tvm::relax::Var& var) const {
    return tvm::runtime::ObjectPtrHash()(var);
  }
};

template <>
struct std::equal_to<tvm::relax::Var> {
  bool operator()(const tvm::relax::Var& var_a, const tvm::relax::Var& var_b) const {
    return tvm::runtime::ObjectPtrEqual()(var_a, var_b);
  }
};

#endif  // TVM_RELAX_EXPR_H_
