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
 * \file tvm/relay/expr.h
 * \brief Relay expression language.
 */
#ifndef TVM_RELAY_EXPR_H_
#define TVM_RELAY_EXPR_H_

#include <tvm/ir/attrs.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/module.h>
#include <tvm/ir/op.h>
#include <tvm/target/virtual_device.h>

#include <functional>
#include <stack>
#include <string>
#include <utility>

#include "./base.h"
#include "./type.h"

namespace tvm {

/*!
 * \brief Returns \p global_var with the given properties. A null property denotes 'no change'.
 * Returns \p global_var if all properties are unchanged. Otherwise, returns a copy with the new
 * fields.
 */
GlobalVar WithFields(GlobalVar global_var, Optional<String> opt_name_hint = {},
                     Optional<Type> opt_type = {}, Optional<VirtualDevice> opt_virtual_device = {},
                     Optional<Span> opt_span = {});

namespace relay {

using Expr = tvm::RelayExpr;
using ExprNode = tvm::RelayExprNode;
using BaseFunc = tvm::BaseFunc;
using BaseFuncNode = tvm::BaseFuncNode;
using GlobalVar = tvm::GlobalVar;
using GlobalVarNode = tvm::GlobalVarNode;

/*!
 * \brief Constant tensor, backed by an NDArray on the cpu(0) device.
 *
 * \note Scalar constants are represented by rank-0 const tensor.
 *  Constant folding are handled uniformly via Tensor types.
 */
class Constant;
/*!
 * \brief Constant tensor type.
 */
class ConstantNode : public ExprNode {
 public:
  /*! \brief The data of the tensor */
  runtime::NDArray data;

  /*! \return The corresponding tensor type of the data */
  TensorType tensor_type() const;

  /*! \return Whether it is scalar(rank-0 tensor) */
  bool is_scalar() const { return data->ndim == 0; }

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("data", &data);
    v->Visit("virtual_device_", &virtual_device_);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const ConstantNode* other, SEqualReducer equal) const {
    return equal(data, other->data);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(data); }

  static constexpr const char* _type_key = "relay.Constant";
  TVM_DECLARE_FINAL_OBJECT_INFO(ConstantNode, ExprNode);
};

class Constant : public Expr {
 public:
  /*!
   * \brief The constructor
   * \param data The data of the constant tensor.
   * \param span The source span of the expression.
   */
  TVM_DLL explicit Constant(runtime::NDArray data, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(Constant, RelayExpr, ConstantNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ConstantNode);
};

/*!
 * \brief Returns \p constant with the given properties. A null property denotes 'no change'.
 * Returns \p constant if all properties are unchanged. Otherwise, returns a copy with the new
 * fields.
 */
Constant WithFields(Constant constant, Optional<runtime::NDArray> opt_data = {},
                    Optional<VirtualDevice> opt_virtual_device = {}, Optional<Span> opt_span = {});

/*! \brief Tuple of multiple Exprs */
class Tuple;
/*! \brief Tuple container */
class TupleNode : public ExprNode {
 public:
  /*! \brief the fields of the tuple */
  tvm::Array<relay::Expr> fields;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("fields", &fields);
    v->Visit("virtual_device_", &virtual_device_);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const TupleNode* other, SEqualReducer equal) const {
    // specially handle empty tuple as a constant is not a graph node.
    if (fields.size() == other->fields.size() && fields.size() == 0) {
      return true;
    } else {
      equal->MarkGraphNode();
      return equal(fields, other->fields);
    }
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    if (fields.size() != 0) {
      hash_reduce->MarkGraphNode();
      hash_reduce(fields);
    }
  }

  static constexpr const char* _type_key = "relay.Tuple";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleNode, ExprNode);
};

class Tuple : public Expr {
 public:
  /*!
   * \brief The constructor
   * \param fields The fields of a tuple.
   * \param span The source span of the expression.
   */
  TVM_DLL explicit Tuple(tvm::Array<relay::Expr> fields, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(Tuple, RelayExpr, TupleNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TupleNode);
};

/*!
 * \brief Returns \p tuple with the given properties. A null property denotes 'no change'.
 * Returns \p tuple if all properties are unchanged. Otherwise, returns a copy with the new
 * fields.
 */
Tuple WithFields(Tuple tuple, Optional<Array<Expr>> opt_fields = Optional<Array<Expr>>(),
                 Optional<VirtualDevice> opt_virtual_device = Optional<VirtualDevice>(),
                 Optional<Span> opt_span = Optional<Span>());

/*!
 * \brief Local variables used in the let expression.
 *
 * Its semantics are similar to tvm.Var node used in TVM's low level
 * tensor expression language.
 *
 * \note Each Var is bind only once and is immutable.
 */
class Var;
/*! \brief Container for Var */
class VarNode : public ExprNode {
 public:
  /*!
   * \brief The unique identifier of the Var.
   *
   * vid will be preserved for the same Var during type inference
   * and other rewritings, while the VarNode might be recreated
   * to attach additional information.
   * This property can be used to keep track of parameter Var
   * information across passes.
   */
  Id vid;
  /*!
   * \brief type annotaion of the variable.
   * This field records user provided type annotation of the Var.
   * This field is optional and can be None.
   */
  Type type_annotation;

  /*! \return The name hint of the variable */
  const String& name_hint() const { return vid->name_hint; }

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("vid", &vid);
    v->Visit("type_annotation", &type_annotation);
    v->Visit("virtual_device_", &virtual_device_);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const VarNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(type_annotation, other->type_annotation) && equal(vid, other->vid) &&
           equal(virtual_device_, other->virtual_device_);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(type_annotation);
    hash_reduce(vid);
  }

  static constexpr const char* _type_key = "relay.Var";
  TVM_DECLARE_FINAL_OBJECT_INFO(VarNode, ExprNode);
};

class Var : public Expr {
 public:
  /*!
   * \brief The constructor
   * \param name_hint The name hint of a variable.
   * \param type_annotation The type annotation of a variable.
   * \param span The source span of the expression.
   */
  TVM_DLL Var(String name_hint, Type type_annotation, Span span = Span())
      : Var(Id(name_hint), type_annotation, span) {}

  /*!
   * \brief The constructor
   * \param vid The unique id of a variable.
   * \param type_annotation The type annotation of a variable.
   * \param span The source span of the expression.
   */
  TVM_DLL Var(Id vid, Type type_annotation, Span span = Span());

  /*!
   * \brief Return a globally fresh name. Helps with debugging to follow the same
   * variable between passes and sub-expressions.
   *
   * TODO(mbs): Replace with name creation w.r.t. scopes once available as part of
   * name gen overhaul.
   */
  static Var GenSym(Type type_annotation = {}, Span span = {});

  TVM_DEFINE_OBJECT_REF_METHODS(Var, RelayExpr, VarNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(VarNode);
};

/*!
 * \brief Returns \p var with the given properties. A null property denotes 'no change'.
 * Returns \p var if all properties are unchanged. Otherwise, returns a copy with the new
 * fields.
 */
Var WithFields(Var var, Optional<Id> opt_vid = Optional<Id>(),
               Optional<Type> opt_type_annotation = Optional<Type>(),
               Optional<VirtualDevice> opt_virtual_device = Optional<VirtualDevice>(),
               Optional<Span> opt_span = Optional<Span>());

/*!
 * \brief Call corresponds to operator invocation.
 *  Corresponds to the operator in computational graph terminology.
 */
class Call;
/*! \brief Call container. */
class CallNode : public ExprNode {
 protected:
  // CallNode uses own deleter to indirectly call non-recursive destructor
  Object::FDeleter saved_deleter_;
  static void Deleter_(Object* ptr);

 public:
  /*!
   * \brief The operator(function) being invoked
   *
   *  - It can be tvm::Op which corresponds to the primitive operators.
   *  - It can also be user defined functions (Function, GlobalVar, Var).
   */
  Expr op;

  /*! \brief The arguments(inputs) of the call */
  tvm::Array<relay::Expr> args;

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
    v->Visit("virtual_device_", &virtual_device_);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const CallNode* other, SEqualReducer equal) const {
    // skip type_args check for primitive ops.
    equal->MarkGraphNode();
    return equal(op, other->op) && equal(args, other->args) && equal(attrs, other->attrs) &&
           (IsPrimitiveOp(op) || equal(type_args, other->type_args));
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(op);
    hash_reduce(args);
    hash_reduce(attrs);
    if (!IsPrimitiveOp(op)) {
      hash_reduce(type_args);
    }
  }

  static constexpr const char* _type_key = "relay.Call";
  TVM_DECLARE_FINAL_OBJECT_INFO(CallNode, ExprNode);
  template <typename>
  friend class runtime::ObjAllocatorBase;
  friend class Call;
};

class Call : public Expr {
 public:
  /*!
   * \brief The destructor
   */
  ~Call();

  /*!
   * \brief The constructor
   * \param op The operator will be invoked.
   * \param args The arguments of the call.
   * \param attrs The attributes of the call node.
   * \param type_args The type arguments passed to a polymorphic function.
   * \param span The source span of the expression.
   */
  TVM_DLL Call(Expr op, Array<Expr> args, Attrs attrs = Attrs(),
               Array<Type> type_args = Array<Type>(), Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(Call, RelayExpr, CallNode);
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
                Optional<Array<Type>> opt_type_args = Optional<Array<Type>>(),
                Optional<VirtualDevice> opt_virtual_device = Optional<VirtualDevice>(),
                Optional<Span> opt_span = Optional<Span>());

/*!
 * \brief Let binding that binds a local var and optionally a type annotation.
 *
 * \note Let is useful to transform the program to be A-normal form.
 *  where each of the expression corresponds to a let binding.
 *
 *  For developers who are familar with the computational graph.
 *  Each of the let can be viewed as a operator node in the computational graph.
 *  Traversing the list of let bindings is similar to running
 * PostDFS-order(topo-order) traversal on the computational graph.
 */
class Let;
/*! \brief A binding of a sub-network. */
class LetNode : public ExprNode {
 protected:
  // LetNode uses own deleter to indirectly call non-recursive destructor
  Object::FDeleter saved_deleter_;
  static void Deleter_(Object* ptr);

 public:
  /*! \brief The variable we bind to */
  Var var;
  /*! \brief The value we bind var to */
  Expr value;
  /*! \brief The body of the let binding */
  Expr body;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("var", &var);
    v->Visit("value", &value);
    v->Visit("body", &body);
    v->Visit("virtual_device_", &virtual_device_);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const LetNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal.DefEqual(var, other->var) && equal(value, other->value) &&
           equal(body, other->body);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce.DefHash(var);
    hash_reduce(value);
    hash_reduce(body);
  }

  static constexpr const char* _type_key = "relay.Let";
  TVM_DECLARE_FINAL_OBJECT_INFO(LetNode, ExprNode);
  template <typename>
  friend class runtime::ObjAllocatorBase;
  friend class Let;
};

class Let : public Expr {
 public:
  /*!
   * \brief The destructor
   */
  ~Let();

  /*!
   * \brief The constructor
   * \param var The variable that is bound to.
   * \param value The value used to bind to the variable.
   * \param body The body of the let binding.
   * \param span The source span of the expression.
   */
  TVM_DLL Let(Var var, Expr value, Expr body, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(Let, RelayExpr, LetNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LetNode);
};

/*!
 * \brief Returns \p let with the given properties. A null property denotes 'no change'.
 * Returns \p let if all properties are unchanged. Otherwise, returns a copy with the new
 * fields.
 */
Let WithFields(Let let, Optional<Var> opt_var = Optional<Var>(),
               Optional<Expr> opt_value = Optional<Expr>(),
               Optional<Expr> opt_body = Optional<Expr>(),
               Optional<VirtualDevice> opt_virtual_device = Optional<VirtualDevice>(),
               Optional<Span> opt_span = Optional<Span>());

/*!
 * \brief Condition expression
 *
 * Unlike traditional statement `if`s, the if evalutes
 * to the result of the branch taken.
 *
 * let x = if (true) { 1 } else { 0 }; // x is 1
 * let y = if (false) { 1 } else { 0 }; // y is 0
 *
 * \note This is similar to C's ternary operator.
 */
class If;
/*! \brief container of If */
class IfNode : public ExprNode {
 public:
  /*! \brief The condition */
  Expr cond;
  /*! \brief The expression evaluated when condition is true. */
  Expr true_branch;
  /*! \brief The expression evaluated when condition is false */
  Expr false_branch;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("cond", &cond);
    v->Visit("true_branch", &true_branch);
    v->Visit("false_branch", &false_branch);
    v->Visit("virtual_device_", &virtual_device_);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const IfNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(cond, other->cond) && equal(true_branch, other->true_branch) &&
           equal(false_branch, other->false_branch);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(cond);
    hash_reduce(true_branch);
    hash_reduce(false_branch);
  }

  static constexpr const char* _type_key = "relay.If";
  TVM_DECLARE_FINAL_OBJECT_INFO(IfNode, ExprNode);
};

class If : public Expr {
 public:
  /*!
   * \brief The constructor
   * \param cond The condition of a if node.
   * \param true_branch The fall through branch
   * \param false_branch The branch for execution when condition is false.
   * \param span The source span of the expression.
   */
  TVM_DLL If(Expr cond, Expr true_branch, Expr false_branch, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(If, RelayExpr, IfNode);
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
              Optional<VirtualDevice> opt_virtual_device = Optional<VirtualDevice>(),
              Optional<Span> opt_span = Optional<Span>());

/*! \brief Get index-th field out of a tuple. */
class TupleGetItem;
class TupleGetItemNode : public ExprNode {
 public:
  /*! \brief The tuple Expression */
  Expr tuple;
  /*! \brief which value to get */
  int index;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("tuple_value", &tuple);
    v->Visit("index", &index);
    v->Visit("virtual_device_", &virtual_device_);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const TupleGetItemNode* other, SEqualReducer equal) const {
    return equal(tuple, other->tuple) && equal(index, other->index);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(tuple);
    hash_reduce(index);
  }

  static constexpr const char* _type_key = "relay.TupleGetItem";
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

  TVM_DEFINE_OBJECT_REF_METHODS(TupleGetItem, RelayExpr, TupleGetItemNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TupleGetItemNode);
};

/*!
 * \brief Returns \p tuple_get_item with the given properties. A null property denotes 'no change'.
 * Returns \p tuple_get_item if all properties are unchanged. Otherwise, returns a copy with the new
 * fields.
 */
TupleGetItem WithFields(TupleGetItem tuple_get_item, Optional<Expr> opt_tuple = Optional<Expr>(),
                        Optional<Integer> opt_index = Optional<Integer>(),
                        Optional<VirtualDevice> opt_virtual_device = Optional<VirtualDevice>(),
                        Optional<Span> opt_span = Optional<Span>());

/*! \brief Create a new Reference out of initial value. */
class RefCreate;
class RefCreateNode : public ExprNode {
 public:
  /*! \brief The initial value of the Reference. */
  Expr value;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("value", &value);
    v->Visit("virtual_device_", &virtual_device_);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const RefCreateNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(value);
  }

  static constexpr const char* _type_key = "relay.RefCreate";
  TVM_DECLARE_FINAL_OBJECT_INFO(RefCreateNode, ExprNode);
};

class RefCreate : public Expr {
 public:
  /*!
   * \brief The constructor
   * \param value The initial value of the reference.
   * \param span The source span of the expression.
   */
  TVM_DLL explicit RefCreate(Expr value, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(RefCreate, RelayExpr, RefCreateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(RefCreateNode);
};

/*!
 * \brief Returns \p ref_create with the given properties. A null property denotes 'no change'.
 * Returns \p ref_crete if all properties are unchanged. Otherwise, returns a copy with the new
 * fields.
 */
RefCreate WithFields(RefCreate ref_create, Optional<Expr> opt_value = Optional<Expr>(),
                     Optional<VirtualDevice> opt_virtual_device = Optional<VirtualDevice>(),
                     Optional<Span> opt_span = Optional<Span>());

/*! \brief Get value out of Reference. */
class RefRead;
class RefReadNode : public ExprNode {
 public:
  /*! \brief The Reference Expression. */
  Expr ref;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("ref", &ref);
    v->Visit("virtual_device_", &virtual_device_);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const RefReadNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(ref, other->ref);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(ref);
  }

  static constexpr const char* _type_key = "relay.RefRead";
  TVM_DECLARE_FINAL_OBJECT_INFO(RefReadNode, ExprNode);
};

class RefRead : public Expr {
 public:
  /*!
   * \brief The constructor
   * \param ref The reference where to read data.
   * \param span The source span of the expression.
   */
  TVM_DLL explicit RefRead(Expr ref, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(RefRead, RelayExpr, RefReadNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(RefReadNode);
};

/*!
 * \brief Returns \p ref_read with the given properties. A null property denotes 'no change'.
 * Returns \p ref_read if all properties are unchanged. Otherwise, returns a copy with the new
 * fields.
 */
RefRead WithFields(RefRead ref_read, Optional<Expr> opt_ref = Optional<Expr>(),
                   Optional<VirtualDevice> opt_virtual_device = Optional<VirtualDevice>(),
                   Optional<Span> opt_span = Optional<Span>());

/*! \brief Set value of Reference. The whole expression evaluates to an Empty Tuple. */
class RefWrite;
class RefWriteNode : public ExprNode {
 public:
  /*! \brief The Reference Expression. */
  Expr ref;
  /*! \brief The value to write into. */
  Expr value;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("ref", &ref);
    v->Visit("value", &value);
    v->Visit("virtual_device_", &virtual_device_);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const RefWriteNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(ref, other->ref) && equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(ref);
    hash_reduce(value);
  }

  static constexpr const char* _type_key = "relay.RefWrite";
  TVM_DECLARE_FINAL_OBJECT_INFO(RefWriteNode, ExprNode);
};

class RefWrite : public Expr {
 public:
  /*!
   * \brief The constructor
   * \param ref The reference where data is write to.
   * \param value The value to write.
   * \param span The source span of the expression.
   */
  TVM_DLL RefWrite(Expr ref, Expr value, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(RefWrite, RelayExpr, RefWriteNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(RefWriteNode);
};

/*!
 * \brief Returns \p ref_write with the given properties. A null property denotes 'no change'.
 * Returns \p ref_write if all properties are unchanged. Otherwise, returns a copy with the new
 * fields.
 */
RefWrite WithFields(RefWrite ref_write, Optional<Expr> opt_ref = Optional<Expr>(),
                    Optional<Expr> opt_value = Optional<Expr>(),
                    Optional<VirtualDevice> opt_virtual_device = Optional<VirtualDevice>(),
                    Optional<Span> opt_span = Optional<Span>());

/*!
 * \brief Base class of the temporary expression.
 *
 * TempExprs are pass specific expression that can be
 * useful to define intermediate result in the
 * rewriting pass such as layout or type transformation.
 *
 * Subclass TempExprNode allows us to pattern match on
 * specific kind of TempExpr and use them for expression rewriting.
 *
 * TempExpr should only be used within a pass,
 */
class TempExprNode : public ExprNode {
 public:
  /*! \brief virtual destructor */
  virtual ~TempExprNode() {}
  /*!
   * \brief Convert the expression to a normal(non-temp) Expr.
   * \return The corresponding normal(non-temp) expression.
   */
  virtual Expr Realize() const = 0;

  static constexpr const char* _type_key = "relay.TempExpr";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  static constexpr const uint32_t _type_child_slots = 0;
  TVM_DECLARE_BASE_OBJECT_INFO(TempExprNode, ExprNode);
};

class TempExpr : public Expr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(TempExpr, RelayExpr, TempExprNode);
};

}  // namespace relay

namespace runtime {

template <>
template <>
inline ObjectPtr<relay::LetNode>
ObjAllocatorBase<SimpleObjAllocator>::make_object<relay::LetNode>() {
  using Derived = SimpleObjAllocator;
  using T = relay::LetNode;
  using Handler = typename Derived::template Handler<T>;
  static_assert(std::is_base_of<Object, T>::value, "make can only be used to create Object");
  T* ptr = Handler::New(static_cast<Derived*>(this));
  ptr->type_index_ = T::RuntimeTypeIndex();
  ptr->saved_deleter_ = Handler::Deleter();
  ptr->deleter_ = relay::LetNode::Deleter_;
  return ObjectPtr<T>(ptr);
}

template <>
template <>
inline ObjectPtr<relay::CallNode>
ObjAllocatorBase<SimpleObjAllocator>::make_object<relay::CallNode>() {
  using Derived = SimpleObjAllocator;
  using T = relay::CallNode;
  using Handler = typename Derived::template Handler<T>;
  static_assert(std::is_base_of<Object, T>::value, "make can only be used to create Object");
  T* ptr = Handler::New(static_cast<Derived*>(this));
  ptr->type_index_ = T::RuntimeTypeIndex();
  ptr->saved_deleter_ = Handler::Deleter();
  ptr->deleter_ = relay::CallNode::Deleter_;
  return ObjectPtr<T>(ptr);
}

}  // namespace runtime

}  // namespace tvm
#endif  // TVM_RELAY_EXPR_H_
