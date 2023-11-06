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
 * \file tvm/tir/stmt.h
 * \brief TIR statements.
 */
// Acknowledgement: Many low-level stmts originate from Halide.
#ifndef TVM_TIR_STMT_H_
#define TVM_TIR_STMT_H_

#include <tvm/tir/expr.h>

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tvm {
namespace tir {

/*! \brief Base node of all statements. */
class StmtNode : public Object {
 public:
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;

  StmtNode() = default;
  explicit StmtNode(Span span) : span(span) {}

  TVM_OBJECT_ENABLE_SCRIPT_PRINTER();

  static constexpr const char* _type_key = "tir.Stmt";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const uint32_t _type_child_slots = 15;
  TVM_DECLARE_BASE_OBJECT_INFO(StmtNode, Object);
};

/*! \brief Container of all statements */
class Stmt : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Stmt, ObjectRef, StmtNode);
};

/*!
 * \brief Let binding, bind var to value, then run body.
 */
class LetStmtNode : public StmtNode {
 public:
  /*! \brief The variable. */
  Var var;
  /*! \brief The value to be bound. */
  PrimExpr value;
  /*! \brief The body block. */
  Stmt body;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("var", &var);
    v->Visit("value", &value);
    v->Visit("body", &body);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const LetStmtNode* other, SEqualReducer equal) const {
    return equal.DefEqual(var, other->var) && equal(value, other->value) &&
           equal(body, other->body);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(var);
    hash_reduce(value);
    hash_reduce(body);
  }

  static constexpr const char* _type_key = "tir.LetStmt";
  TVM_DECLARE_FINAL_OBJECT_INFO(LetStmtNode, StmtNode);
};

/*!
 * \brief Managed reference to LetStmtNode.
 * \sa LetStmtNode
 */
class LetStmt : public Stmt {
 public:
  TVM_DLL LetStmt(Var var, PrimExpr value, Stmt body, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(LetStmt, Stmt, LetStmtNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LetStmtNode);
};

/*!
 * \brief Define certain auxiliary attribute for the body to be a symbolic value.
 *  This provide auxiliary information for IR passes that transforms body.
 *
 *  In terms of effect, this is equivalent to Block(Evaluate(value), body).
 *
 *  Examples of possible usage:
 *    - Bound of function, variables.
 *    - Hint which block corresponds to a parallel region.
 */
class AttrStmtNode : public StmtNode {
 public:
  /*! \brief this is attribute about certain node */
  ObjectRef node;
  /*! \brief the type key of the attribute */
  String attr_key;
  /*! \brief The attribute value, value is well defined at current scope. */
  PrimExpr value;
  /*! \brief The body statement to be executed */
  Stmt body;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("node", &node);
    v->Visit("attr_key", &attr_key);
    v->Visit("value", &value);
    v->Visit("body", &body);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const AttrStmtNode* other, SEqualReducer equal) const {
    return equal(node, other->node) && equal(attr_key, other->attr_key) &&
           equal(value, other->value) && equal(body, other->body);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(node);
    hash_reduce(attr_key);
    hash_reduce(value);
    hash_reduce(body);
  }

  static constexpr const char* _type_key = "tir.AttrStmt";
  TVM_DECLARE_FINAL_OBJECT_INFO(AttrStmtNode, StmtNode);
};

/*!
 * \brief Managed reference to AttrStmtNode.
 * \sa AttrStmtNode
 */
class AttrStmt : public Stmt {
 public:
  TVM_DLL AttrStmt(ObjectRef node, String attr_key, PrimExpr value, Stmt body, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(AttrStmt, Stmt, AttrStmtNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AttrStmtNode);
};

/*!
 * \brief Assert condition, if an error occurs, return the error message.
 */
class AssertStmtNode : public StmtNode {
 public:
  /*! \brief Condition to be checked. */
  PrimExpr condition;
  /*! \brief Error message when assertion failed. */
  PrimExpr message;
  /*!
   * \brief Body which this assertion holds true.
   *  Will be executed after the assertion.
   */
  Stmt body;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("condition", &condition);
    v->Visit("message", &message);
    v->Visit("body", &body);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const AssertStmtNode* other, SEqualReducer equal) const {
    return equal(condition, other->condition) && equal(message, other->message) &&
           equal(body, other->body);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(condition);
    hash_reduce(message);
    hash_reduce(body);
  }

  static constexpr const char* _type_key = "tir.AssertStmt";
  TVM_DECLARE_FINAL_OBJECT_INFO(AssertStmtNode, StmtNode);
};

/*!
 * \brief Managed reference to AssertStmtNode.
 * \sa AssertStmtNode
 */
class AssertStmt : public Stmt {
 public:
  TVM_DLL AssertStmt(PrimExpr condition, PrimExpr message, Stmt body, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(AssertStmt, Stmt, AssertStmtNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AssertStmtNode);
};

/*!
 * \brief Store value to the high dimension buffer.
 *
 * \code
 *
 *  buffer[i, j] = value;
 *
 * \endcode
 * \sa BufferLoad
 */
class BufferStoreNode : public StmtNode {
 public:
  /*! \brief The buffer variable. */
  Buffer buffer;
  /*! \brief The value to be stored. */
  PrimExpr value;
  /*! \brief The indices location to be stored. */
  Array<PrimExpr> indices;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("buffer", &buffer);
    v->Visit("value", &value);
    v->Visit("indices", &indices);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const BufferStoreNode* other, SEqualReducer equal) const {
    return equal(buffer, other->buffer) && equal(value, other->value) &&
           equal(indices, other->indices);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(buffer);
    hash_reduce(value);
    hash_reduce(indices);
  }

  static constexpr const char* _type_key = "tir.BufferStore";
  TVM_DECLARE_FINAL_OBJECT_INFO(BufferStoreNode, StmtNode);
};

/*!
 * \brief Managed reference to BufferStoreNode.
 * \sa BufferStoreNode
 */
class BufferStore : public Stmt {
 public:
  TVM_DLL explicit BufferStore(Buffer buffer, PrimExpr value, Array<PrimExpr> indices,
                               Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(BufferStore, Stmt, BufferStoreNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BufferStoreNode);
};

/*!
 * \brief Annotate the region where the buffer need to
 *  be read and write in the body.
 *  We only need to allocate the space for the corresponding region.
 *
 * \note There should be at most one BufferRealize for each buffer.
 *       BufferRealize is not necessary for external buffers,
 *       since they are assumed to be fully allocated.
 *
 * \sa BufferLoad, BufferStore
 */
class BufferRealizeNode : public StmtNode {
 public:
  /*! \brief The buffer variable. */
  Buffer buffer;
  /*! \brief Bounds to be realized */
  Array<Range> bounds;
  /*! \brief Only realize if condition holds. */
  PrimExpr condition;
  /*! \brief The body of realization. */
  Stmt body;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("buffer", &buffer);
    v->Visit("bounds", &bounds);
    v->Visit("condition", &condition);
    v->Visit("body", &body);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const BufferRealizeNode* other, SEqualReducer equal) const {
    return equal(buffer, other->buffer) && equal(bounds, other->bounds) &&
           equal(condition, other->condition) && equal(body, other->body);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(buffer);
    hash_reduce(bounds);
    hash_reduce(condition);
    hash_reduce(body);
  }

  BufferRealizeNode() = default;
  BufferRealizeNode(Buffer buffer, Array<Range> bounds, PrimExpr condition, Stmt body,
                    Span span = Span())
      : StmtNode(span), buffer(buffer), bounds(bounds), condition(condition), body(body) {}

  static constexpr const char* _type_key = "tir.BufferRealize";
  TVM_DECLARE_FINAL_OBJECT_INFO(BufferRealizeNode, StmtNode);
};

/*!
 * \brief Managed reference to BufferRealizeNode.
 * \sa BufferRealizeNode
 */
class BufferRealize : public Stmt {
 public:
  TVM_DLL explicit BufferRealize(Buffer buffer, Array<Range> bounds, PrimExpr condition, Stmt body,
                                 Span span = Span());

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BufferRealize, Stmt, BufferRealizeNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BufferRealizeNode);
};

/*!
 * \brief Store value into mult-dimensional array that will be read by the consumer
 *        of the producer.
 *
 * \note This node only appears in high-level DSLs that are built on top of the TIR.
 *       It should not appear in a valid TIR PrimFunc. A high-level DSL needs to lower
 *       this node before TIR transformations.
 *
 * \sa DataProducer
 */
class ProducerStoreNode : public StmtNode {
 public:
  /*! \brief The producer to store the results into. */
  DataProducer producer;
  /*! \brief The value to be stored. */
  PrimExpr value;
  /*! \brief The index arguments of the function. */
  Array<PrimExpr> indices;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("producer", &producer);
    v->Visit("value", &value);
    v->Visit("indices", &indices);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const ProducerStoreNode* other, SEqualReducer equal) const {
    return equal(producer, other->producer) && equal(value, other->value) &&
           equal(indices, other->indices);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(producer);
    hash_reduce(value);
    hash_reduce(indices);
  }

  static constexpr const char* _type_key = "tir.ProducerStore";
  TVM_DECLARE_FINAL_OBJECT_INFO(ProducerStoreNode, StmtNode);
};

/*!
 * \brief Managed reference to ProducerStoreNode.
 * \sa ProducerStoreNode
 */
class ProducerStore : public Stmt {
 public:
  TVM_DLL ProducerStore(DataProducer producer, PrimExpr value, Array<PrimExpr> indices,
                        Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(ProducerStore, Stmt, ProducerStoreNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ProducerStoreNode);
};

/*!
 * \brief Annotate the bounds where the data produced by the producer
 *  need to be written and read in body.
 *  We will need to allocate space for the corresponding regions.
 *
 * \note This node only appears in high-level DSLs that are built on top of the TIR.
 *       It should not appear in a valid TIR PrimFunc. A high-level DSL needs to lower
 *       this node before TIR transformations.
 *
 * \sa DataProducer
 */
class ProducerRealizeNode : public StmtNode {
 public:
  /*! \brief The producer that produces the data. */
  DataProducer producer;
  /*! \brief Bounds to be realized. */
  Region bounds;
  /*! \brief Only realize if condition holds. */
  PrimExpr condition;
  /*! \brief The body of realization. */
  Stmt body;
  /*! \brief The storage scope associated with this realization. */
  String storage_scope;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("producer", &producer);
    v->Visit("bounds", &bounds);
    v->Visit("condition", &condition);
    v->Visit("body", &body);
    v->Visit("storage_scope", &storage_scope);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const ProducerRealizeNode* other, SEqualReducer equal) const {
    return equal(producer, other->producer) && equal(bounds, other->bounds) &&
           equal(condition, other->condition) && equal(body, other->body) &&
           equal(storage_scope, other->storage_scope);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(producer);
    hash_reduce(bounds);
    hash_reduce(condition);
    hash_reduce(body);
    hash_reduce(storage_scope);
  }

  static constexpr const char* _type_key = "tir.ProducerRealize";
  TVM_DECLARE_FINAL_OBJECT_INFO(ProducerRealizeNode, StmtNode);
};

/*!
 * \brief Managed reference to ProducerRealizeNode.
 * \sa ProducerRealizeNode
 */
class ProducerRealize : public Stmt {
 public:
  TVM_DLL ProducerRealize(DataProducer producer, Region bounds, PrimExpr condition, Stmt body,
                          String storage_scope = "", Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(ProducerRealize, Stmt, ProducerRealizeNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ProducerRealizeNode);
};

/*!
 * \brief Allocate a buffer that can be used in body.
 */
class AllocateNode : public StmtNode {
 public:
  /*! \brief The buffer variable. */
  Var buffer_var;
  /*! \brief The type of the buffer. */
  DataType dtype;
  /*! \brief The extents of the buffer. */
  Array<PrimExpr> extents;
  /*! \brief Only allocate buffer when condition is satisfied. */
  PrimExpr condition;
  /*! \brief The body to be executed. */
  Stmt body;
  /*!
   * \brief Additional annotations about the allocation.
   *
   *  These annotations can be used as auxiliary hint
   *  to future transformations.
   */
  Map<String, ObjectRef> annotations;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("buffer_var", &buffer_var);
    v->Visit("dtype", &dtype);
    v->Visit("extents", &extents);
    v->Visit("condition", &condition);
    v->Visit("body", &body);
    v->Visit("annotations", &annotations);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const AllocateNode* other, SEqualReducer equal) const {
    return equal.DefEqual(buffer_var, other->buffer_var) && equal(dtype, other->dtype) &&
           equal(extents, other->extents) && equal(condition, other->condition) &&
           equal(body, other->body) && equal(annotations, other->annotations);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(buffer_var);
    hash_reduce(dtype);
    hash_reduce(extents);
    hash_reduce(condition);
    hash_reduce(body);
    hash_reduce(annotations);
  }

  /*!
   * \brief If the buffer size is constant, return the size.
   *        Otherwise return 0.
   * \return The result.
   */
  int64_t ConstantAllocationSize() const { return ConstantAllocationSize(extents); }
  /*!
   * \brief If the buffer size is constant, return the size.
   *        Otherwise return 0.
   * \param extents The extents of the buffer.
   * \return The result.
   */
  TVM_DLL static int64_t ConstantAllocationSize(const Array<PrimExpr>& extents);

  static constexpr const char* _type_key = "tir.Allocate";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(AllocateNode, StmtNode);
};

/*!
 * \brief Managed reference to AllocateNode.
 * \sa AllocateNode
 */
class Allocate : public Stmt {
 public:
  TVM_DLL Allocate(Var buffer_var, DataType dtype, Array<PrimExpr> extents, PrimExpr condition,
                   Stmt body, Map<String, ObjectRef> annotations = Map<String, ObjectRef>(),
                   Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(Allocate, Stmt, AllocateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AllocateNode);
};

/*!
 * \brief Allocate a buffer that can be used in body.
 */
class AllocateConstNode : public StmtNode {
 public:
  /*! \brief The buffer variable. */
  Var buffer_var;
  /*! \brief The optional data associated to the constant.
   */
  Optional<runtime::NDArray> data;
  /*!
   * \brief If the PrimFunc containing the Stmt is added to IRModule, this is an optional index
   * to indicate the index within "constants" attribute, that is a Array<NDArray> of IRModule.
   */
  Optional<Integer> irmod_storage_idx;
  /*! \brief The type of the buffer. */
  DataType dtype;
  /*! \brief The extents of the buffer. */
  Array<PrimExpr> extents;
  /*! \brief The body to be executed. */
  Stmt body;
  /*!
   * \brief Additional annotations about the allocation.
   *
   *  These annotations can be used as auxiliary hint
   *  to future transformations.
   */
  Map<String, ObjectRef> annotations;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("buffer_var", &buffer_var);
    v->Visit("data", &data);
    v->Visit("irmod_storage_idx", &irmod_storage_idx);
    v->Visit("dtype", &dtype);
    v->Visit("extents", &extents);
    v->Visit("body", &body);
    v->Visit("annotations", &annotations);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const AllocateConstNode* other, SEqualReducer equal) const {
    return equal.DefEqual(buffer_var, other->buffer_var) && equal(dtype, other->dtype) &&
           equal(extents, other->extents) && equal(data, other->data) && equal(body, other->body) &&
           equal(annotations, other->annotations);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(buffer_var);
    hash_reduce(dtype);
    hash_reduce(extents);
    hash_reduce(body);
    hash_reduce(annotations);
    hash_reduce(data);
  }

  /*!
   * \brief If the buffer size is constant, return the size.
   *        Otherwise return 0.
   * \return The result.
   */
  int64_t ConstantAllocationSize() const { return ConstantAllocationSize(extents); }
  /*!
   * \brief If the buffer size is constant, return the size.
   *        Otherwise return 0.
   * \param extents The extents of the buffer.
   * \return The result.
   */
  TVM_DLL static int64_t ConstantAllocationSize(const Array<PrimExpr>& extents);

  static constexpr const char* _type_key = "tir.AllocateConst";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(AllocateConstNode, StmtNode);
};

/*!
 * \brief Managed reference to AllocateConstNode.
 * \sa AllocateConstNode
 */
class AllocateConst : public Stmt {
 public:
  /* The constructor to create a IRNode with constant data
   * depending on the type of ObjectRef, it will either
   * create AllocateConstNode with irmod_storage_idx or data
   */
  TVM_DLL AllocateConst(Var buffer_var, DataType dtype, Array<PrimExpr> extents,
                        ObjectRef data_or_idx, Stmt body,
                        Map<String, ObjectRef> annotations = Map<String, ObjectRef>(),
                        Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(AllocateConst, Stmt, AllocateConstNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AllocateConstNode);
};

/*! \brief Declare a buffer that can be used in the body */
class DeclBufferNode : public StmtNode {
 public:
  /*! \brief The buffer being declared */
  Buffer buffer;
  /*! \brief The body to be executed */
  Stmt body;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("buffer", &buffer);
    v->Visit("body", &body);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const DeclBufferNode* other, SEqualReducer equal) const {
    return equal(buffer, other->buffer) && equal(body, other->body);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(buffer);
    hash_reduce(body);
  }

  static constexpr const char* _type_key = "tir.DeclBuffer";
  TVM_DECLARE_FINAL_OBJECT_INFO(DeclBufferNode, StmtNode);
};

/*! \brief Managed reference to DeclBufferNode */
class DeclBuffer : public Stmt {
 public:
  TVM_DLL DeclBuffer(Buffer buffer, Stmt body, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(DeclBuffer, Stmt, DeclBufferNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DeclBufferNode);
};

/*!
 * \brief The container of seq statement.
 *        Represent a sequence of statements.
 */
class SeqStmtNode : public StmtNode {
 public:
  /*! \brief internal sequence content. */
  Array<Stmt> seq;

  /*! \return get the size of the sequence */
  size_t size() const { return seq.size(); }
  /*!
   * \brief Get the index-th element in the sequence.
   */
  Stmt operator[](size_t index) const { return seq[index]; }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("seq", &seq);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const SeqStmtNode* other, SEqualReducer equal) const {
    return equal(seq, other->seq);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(seq); }

  static constexpr const char* _type_key = "tir.SeqStmt";
  TVM_DECLARE_FINAL_OBJECT_INFO(SeqStmtNode, StmtNode);
};

/*!
 * \brief Evaluates an expression.
 *  This is mostly used for putting a Call node into Stmt.
 *
 *  If value do not have side-effect, this node can be safely removed.
 */
class EvaluateNode : public StmtNode {
 public:
  /*! \brief The expression to be evaluated. */
  PrimExpr value;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("value", &value);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const EvaluateNode* other, SEqualReducer equal) const {
    return equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(value); }

  static constexpr const char* _type_key = "tir.Evaluate";
  TVM_DECLARE_FINAL_OBJECT_INFO(EvaluateNode, StmtNode);
};

/*!
 * \brief Managed reference to EvaluateNode.
 * \sa EvaluateNode
 */
class Evaluate : public Stmt {
 public:
  TVM_DLL explicit Evaluate(PrimExpr value, Span span = Span());

  explicit Evaluate(int value, Span span = Span()) : Evaluate(PrimExpr(value), span) {}

  TVM_DEFINE_OBJECT_REF_METHODS(Evaluate, Stmt, EvaluateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(EvaluateNode);
};

/*! \brief Sequence statement. */
class SeqStmt : public Stmt {
 public:
  /*!
   * \brief Construct SeqStmt.
   * \param seq The sequence.
   * \param span The location of this object in the source code.
   */
  TVM_DLL explicit SeqStmt(Array<Stmt> seq, Span span = Span());

  /*! \return get the size of the sequence */
  size_t size() const { return operator->()->size(); }
  /*!
   * \brief Get the index-th element in the sequence.
   */
  Stmt operator[](size_t index) const { return (*(operator->()))[index]; }
  /*!
   * \brief Construct a sequence statement by flattening
   *        all the arrays and sequences in the arguments
   *        recursively.
   *
   * - When an argument is nullptr, it will be ignored.
   * - When an argument is an array or a SeqStmt, it will be flattened recursively.
   * - A normal Stmt will be appended to the end of the sequence.
   *
   * \note This function can directly return an element
   *       if it is the only element in the sequence.
   *
   * \note If the only argument to this function is a SeqStmt, and if
   *       no flattening of the SeqStmt is required, then the SeqStmt
   *       will be returned as-is.
   *
   * \param seq_args The list of arguments to be flattened.
   * \tparam Args arguments
   * \return The constructed statement
   */
  template <typename... Args>
  static Stmt Flatten(Args&&... seq_args) {
    Array<Stmt> seq;
    runtime::detail::for_each(Flattener(&seq), std::forward<Args>(seq_args)...);

    if (seq.empty()) {
      return Evaluate(0);
    } else if (seq.size() == 1) {
      return seq[0];
    }

    // If the argument is a single SeqStmt argument with no
    // flattening or unwrapping required, then we may
    // return the SeqStmt as-is.
    if constexpr (sizeof...(seq_args) == 1) {
      if (auto opt = Flattener::AsSeqStmt(std::forward<Args>(seq_args)...)) {
        SeqStmt original = opt.value();
        bool all_same = [&]() {
          if (original->seq.size() != seq.size()) {
            return false;
          }
          for (size_t i = 0; i < seq.size(); i++) {
            if (!original->seq[i].same_as(seq[i])) {
              return false;
            }
          }
          return true;
        }();
        if (all_same) {
          return original;
        }
      }
    }

    return SeqStmt(seq);
  }
  /*! \brief Helper class to flatten sequence of arguments into Array. */
  class Flattener {
   public:
    explicit Flattener(Array<Stmt>* seq) : seq_(seq) {}

    template <typename T>
    static Optional<SeqStmt> AsSeqStmt(const T& t) {
      if constexpr (std::is_same_v<T, SeqStmt>) {
        return t;
      } else if constexpr (!std::is_base_of_v<T, SeqStmt>) {
        return NullOpt;
      } else if (auto* ptr = t.template as<SeqStmtNode>()) {
        return GetRef<SeqStmt>(ptr);
      } else {
        return NullOpt;
      }
    }

    template <typename T>
    void operator()(size_t i, const T& stmt_or_seq) const {
      if constexpr (std::is_base_of_v<ObjectRef, T>) {
        // Early bail-out, applicable to any ObjectRef
        if (!stmt_or_seq.defined()) {
          return;
        }
      }

      if constexpr (std::is_same_v<T, SeqStmt>) {
        // Static type-checking for a SeqStmt that could be flattened.
        (*this)(0, stmt_or_seq->seq);
        return;
      }

      if constexpr (std::is_base_of_v<T, SeqStmt>) {
        // Dynamic type-checking for a SeqStmt that could be
        // flattened.
        if (auto* op = stmt_or_seq.template as<SeqStmtNode>()) {
          operator()(0, op->seq);
          return;
        }
      }

      if constexpr (std::is_base_of_v<T, Evaluate>) {
        // Evaluate(0) is used to represent a no-op, and may be
        // generated by previous calls to SeqStmt::Flatten().  These
        // should be removed to ensure that Flatten(a+b) is equivalent
        // to Flatten(Flatten(a), Flatten(b)).
        if (auto* op = stmt_or_seq.template as<EvaluateNode>()) {
          if (auto* as_int = op->value.template as<IntImmNode>(); as_int && as_int->value == 0) {
            return;
          }
        }
      }

      if constexpr (std::is_base_of_v<Stmt, T>) {
        // Any other Stmt type just gets appended.
        seq_->push_back(stmt_or_seq);
      } else {
        // Anything else is treated as an iterable of Stmt.
        for (auto v : stmt_or_seq) {
          this->operator()(0, v);
        }
      }
    }

   private:
    Array<Stmt>* seq_;
  };

  TVM_DEFINE_OBJECT_REF_METHODS(SeqStmt, Stmt, SeqStmtNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SeqStmtNode);
};

/*!
 * \brief IfThenElse statement.
 */
class IfThenElseNode : public StmtNode {
 public:
  /*! \brief The condition. */
  PrimExpr condition;
  /*! \brief The branch to be executed when condition is true. */
  Stmt then_case;
  /*! \brief The branch to be executed when condition is false, can be null. */
  Optional<Stmt> else_case;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("condition", &condition);
    v->Visit("then_case", &then_case);
    v->Visit("else_case", &else_case);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const IfThenElseNode* other, SEqualReducer equal) const {
    return equal(condition, other->condition) && equal(then_case, other->then_case) &&
           equal(else_case, other->else_case);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(condition);
    hash_reduce(then_case);
    hash_reduce(else_case);
  }

  static constexpr const char* _type_key = "tir.IfThenElse";
  TVM_DECLARE_FINAL_OBJECT_INFO(IfThenElseNode, StmtNode);
};

/*!
 * \brief Managed reference to IfThenElseNode.
 * \sa IfThenElseNode
 */
class IfThenElse : public Stmt {
 public:
  TVM_DLL IfThenElse(PrimExpr condition, Stmt then_case, Optional<Stmt> else_case = NullOpt,
                     Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(IfThenElse, Stmt, IfThenElseNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(IfThenElseNode);
};

/*!
 * \brief The kind of the loop.
 *
 *  ForKind can change the control flow semantics
 *  of the loop. So the kind field needs to be considered
 *  in all TIR passes.
 */
enum class ForKind : int {
  /*! \brief default semantics -- serial execution. */
  kSerial = 0,
  /*! \brief Parallel execution on CPU. */
  kParallel = 1,
  /*!
   * \brief Vector SIMD loop.
   *  The loop body will be vectorized.
   */
  kVectorized = 2,
  /*! \brief The loop body must be unrolled. */
  kUnrolled = 3,
  /*!
   * \brief The loop variable is bound to a thread in
   * an environment. In the final stage of lowering,
   * the loop is simply removed and the loop variable is
   * mapped to the corresponding context thread.
   */
  kThreadBinding = 4
};

/*!
 * \brief A for loop, with possible type annotations.
 *
 * \code
 *
 *  for (loop_var = min; loop_var < min + extent; ++loop_var) {
 *    // body
 *  }
 * \endcode
 */
class ForNode : public StmtNode {
 public:
  /*! \brief The loop variable. */
  Var loop_var;
  /*! \brief The minimum value of iteration. */
  PrimExpr min;
  /*! \brief The extent of the iteration. */
  PrimExpr extent;
  /*! \brief The kind of the for loop. */
  ForKind kind;
  /*! \brief The body of the for loop. */
  Stmt body;
  /*!
   * \brief Only valid when kind == ForKind::kThreadBinding
   * The context thread that this loop variable bounds to.
   */
  Optional<IterVar> thread_binding;
  /*!
   * \brief Additional annotations about the loop.
   *
   *  These annotations can be used as auxiliary hint
   *  to future transformations. An annotation should
   *  not change the control flow semantics of the loop
   *  and can be ignored in most passes.
   */
  Map<String, ObjectRef> annotations;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("loop_var", &loop_var);
    v->Visit("min", &min);
    v->Visit("extent", &extent);
    v->Visit("kind", &kind);
    v->Visit("body", &body);
    v->Visit("thread_binding", &thread_binding);
    v->Visit("annotations", &annotations);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const ForNode* other, SEqualReducer equal) const {
    return equal.DefEqual(loop_var, other->loop_var) && equal(min, other->min) &&
           equal(extent, other->extent) && equal(kind, other->kind) && equal(body, other->body) &&
           equal(thread_binding, other->thread_binding) && equal(annotations, other->annotations);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(loop_var);
    hash_reduce(min);
    hash_reduce(extent);
    hash_reduce(kind);
    hash_reduce(body);
    hash_reduce(thread_binding);
    hash_reduce(annotations);
  }

  static constexpr const char* _type_key = "tir.For";
  TVM_DECLARE_FINAL_OBJECT_INFO(ForNode, StmtNode);
};

/*!
 * \brief Managed reference to ForNode.
 * \sa ForNode
 */
class For : public Stmt {
 public:
  TVM_DLL For(Var loop_var, PrimExpr min, PrimExpr extent, ForKind kind, Stmt body,
              Optional<IterVar> thread_binding = NullOpt,
              Map<String, ObjectRef> annotations = Map<String, ObjectRef>(), Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(For, Stmt, ForNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ForNode);
};

/*!
 * \brief A While loop
 *
 * \code
 *
 *  while (condition)
 *    body
 *
 * \endcode
 */
class WhileNode : public StmtNode {
 public:
  /*! \brief The termination condition. */
  PrimExpr condition;
  /*! \brief The body of the while loop. */
  Stmt body;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("condition", &condition);
    v->Visit("body", &body);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const WhileNode* other, SEqualReducer equal) const {
    return equal(condition, other->condition) && equal(body, other->body);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(condition);
    hash_reduce(body);
  }

  static constexpr const char* _type_key = "tir.While";
  TVM_DECLARE_FINAL_OBJECT_INFO(WhileNode, StmtNode);
};

/*!
 * \brief Managed reference to WhileNode.
 * \sa WhileNode
 */
class While : public Stmt {
 public:
  TVM_DLL While(PrimExpr condition, Stmt body, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(While, Stmt, WhileNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(WhileNode);
};

/*!
 * \brief A prefetch hint for a buffer
 */
class PrefetchNode : public StmtNode {
 public:
  /*! \brief The function to be prefetched. */
  Buffer buffer;
  /*! \brief Bounds to be prefetched. */
  Array<Range> bounds;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("buffer", &buffer);
    v->Visit("bounds", &bounds);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const PrefetchNode* other, SEqualReducer equal) const {
    return equal(buffer, other->buffer) && equal(bounds, other->bounds);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(buffer);
    hash_reduce(bounds);
  }

  PrefetchNode() = default;
  PrefetchNode(Buffer buffer, Array<Range> bounds, Span span = Span())
      : StmtNode(span), buffer(buffer), bounds(bounds) {}

  static constexpr const char* _type_key = "tir.Prefetch";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrefetchNode, StmtNode);
};

/*!
 * \brief Managed reference to PrefetchNode.
 * \sa PrefetchNode
 */
class Prefetch : public Stmt {
 public:
  TVM_DLL explicit Prefetch(Buffer buffer, Array<Range> bounds, Span span = Span());

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Prefetch, Stmt, PrefetchNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(PrefetchNode);
};

/*!
 * \brief Representing the region of multi-dimensional buffer access.
 */
class BufferRegionNode : public Object {
 public:
  /*! \brief The buffer of the buffer region. */
  Buffer buffer;
  /*! \brief The region array of the buffer region. */
  Array<Range> region;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("buffer", &buffer);
    v->Visit("region", &region);
  }

  bool SEqualReduce(const BufferRegionNode* other, SEqualReducer equal) const {
    return equal(buffer, other->buffer) && equal(region, other->region);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(buffer);
    hash_reduce(region);
  }

  static constexpr const char* _type_key = "tir.BufferRegion";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(BufferRegionNode, Object);
};

/*!
 * \brief Managed reference to BufferRegionNode.
 * \sa BufferRegionNode
 */
class BufferRegion : public ObjectRef {
 public:
  TVM_DLL explicit BufferRegion(Buffer buffer, Array<Range> region);

  /*!
   * \brief Create a BufferRegion which is full region of the given buffer.
   * \param buffer The buffer to generate full BufferRegion.
   * \return The BufferRegion which covers all region of the given buffer
   */
  TVM_DLL static BufferRegion FullRegion(Buffer buffer);

  /*!
   * \brief Create a BufferRegion which is a single point of the given buffer.
   * \param buffer The buffer to generate single point BufferRegion.
   * \param indices The access point indices of the buffer
   * \return The BufferRegion which is the single point of the given buffer.
   */
  TVM_DLL static BufferRegion FromPoint(Buffer buffer, Array<PrimExpr> indices);

  TVM_DEFINE_OBJECT_REF_METHODS(BufferRegion, ObjectRef, BufferRegionNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BufferRegionNode);
};

/*!
 * \brief Match introduces a constraint that the source buffer region can be remapped to the data
 * layout specified by the buffer field. The constraint can be checked in later part of lowering (or
 * optionally during runtime).
 *
 * MatchBufferRegion provides a mechanism to represent data layout and compactness constraints in
 * low-level hardware primitives in the IR and defer the check after the sequence of
 * transformations.
 */
class MatchBufferRegionNode : public Object {
 public:
  /*! \brief The target buffer. */
  Buffer buffer;
  /*! \brief The source buffer region. */
  BufferRegion source;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("buffer", &buffer);
    v->Visit("source", &source);
  }

  bool SEqualReduce(const MatchBufferRegionNode* other, SEqualReducer equal) const {
    return equal(buffer, other->buffer) && equal(source, other->source);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(buffer);
    hash_reduce(source);
  }

  static constexpr const char* _type_key = "tir.MatchBufferRegion";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(MatchBufferRegionNode, Object);
};

/*!
 * \brief Managed reference to MatchBufferRegionNode.
 * \sa MatchBufferRegionNode
 */
class MatchBufferRegion : public ObjectRef {
 public:
  TVM_DLL explicit MatchBufferRegion(Buffer buffer, BufferRegion source);

  TVM_DEFINE_OBJECT_REF_METHODS(MatchBufferRegion, ObjectRef, MatchBufferRegionNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MatchBufferRegionNode);
};

/*!
 * \brief A block is a basic schedule unit in TIR.
 * \note Block's body is parameterized by iter vars.
 * \code
 *
 *  with T.block(name):
 *      v0 = T.axis.S(domain, value0)
 *      v1 = T.axis.R(domain, value1)
 *      ...
 *      T.reads([buffer0[start:end, ...], ...])
 *      T.writes([buffer1[start:end, ...], ...])
 *      T.where(predicate)
 *      buffer2 = T.alloc_buffer(shape, dtype)
 *      buffer3 = T.match_buffer(source_buffer[start:end, ...])
 *      T.attr({attr_key: attr_value, ...})
 *      with T.init():
 *          // init body
 *      // body
 *
 * \endcode
 */
class BlockNode : public StmtNode {
 public:
  /*! \brief The variables of the block. */
  Array<IterVar> iter_vars;
  /*! \brief The read buffer regions of the block. */
  Array<BufferRegion> reads;
  /*! \brief The write buffer regions of the block. */
  Array<BufferRegion> writes;
  /*! \brief The name_hint of the block. */
  String name_hint;
  /*! \brief The body of the block. */
  Stmt body;
  /*!
   * \brief The init statement is executed during the first iteration of reduction loops in a
   *  reduction block. The optional init field allows us to represent initialization and
   *  reduction update in a single block and transform them collectively.
   *  We also provide primitives to decompose the init into a separate block during scheduling.
   *  Init field is `NullOpt` if there is no reduction iter_vars
   */
  Optional<Stmt> init;
  /*! \brief The buffer allocated in the block. */
  Array<Buffer> alloc_buffers;
  /*! \brief The match buffer regions. */
  Array<MatchBufferRegion> match_buffers;
  /*! \brief The annotation of the block. */
  Map<String, ObjectRef> annotations;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("iter_vars", &iter_vars);
    v->Visit("reads", &reads);
    v->Visit("writes", &writes);
    v->Visit("name_hint", &name_hint);
    v->Visit("body", &body);
    v->Visit("init", &init);
    v->Visit("alloc_buffers", &alloc_buffers);
    v->Visit("match_buffers", &match_buffers);
    v->Visit("annotations", &annotations);
  }

  bool SEqualReduce(const BlockNode* other, SEqualReducer equal) const {
    // Need first reduce iter_vars, alloc_buffers and match_buffers to define new vars
    return equal.DefEqual(iter_vars, other->iter_vars) &&
           equal(alloc_buffers, other->alloc_buffers) &&
           equal(match_buffers, other->match_buffers) && equal(reads, other->reads) &&
           equal(writes, other->writes) && equal(body, other->body) && equal(init, other->init) &&
           equal(annotations, other->annotations);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(iter_vars);
    hash_reduce(alloc_buffers);
    hash_reduce(match_buffers);
    hash_reduce(reads);
    hash_reduce(writes);
    hash_reduce(body);
    hash_reduce(init);
    hash_reduce(annotations);
  }

  static constexpr const char* _type_key = "tir.Block";
  TVM_DECLARE_FINAL_OBJECT_INFO(BlockNode, StmtNode);
};

/*!
 * \brief Managed reference to BlockNode.
 * \sa BlockNode
 */
class Block : public Stmt {
 public:
  TVM_DLL explicit Block(Array<IterVar> iter_vars, Array<BufferRegion> reads,
                         Array<BufferRegion> writes, String name_hint, Stmt body,
                         Optional<Stmt> init = NullOpt,
                         Array<Buffer> alloc_buffers = Array<Buffer>(),
                         Array<MatchBufferRegion> match_buffers = Array<MatchBufferRegion>(),
                         Map<String, ObjectRef> annotations = Map<String, ObjectRef>(),
                         Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(Block, Stmt, BlockNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BlockNode);
};

/*!
 * \brief A block realization node represents execution of the block at the binding values.
 */
class BlockRealizeNode : public StmtNode {
 public:
  /*! \brief The corresponding values of the iter vars. */
  Array<PrimExpr> iter_values;
  /*!
   * \brief The predicate of the block realization, the block will only be executed when the
   * predicate is true.
   */
  PrimExpr predicate;
  /*! \brief The block to be realized. */
  Block block;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("iter_values", &iter_values);
    v->Visit("predicate", &predicate);
    v->Visit("block", &block);
  }

  bool SEqualReduce(const BlockRealizeNode* other, SEqualReducer equal) const {
    return equal(iter_values, other->iter_values) && equal(predicate, other->predicate) &&
           equal(block, other->block);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(iter_values);
    hash_reduce(predicate);
    hash_reduce(block);
  }

  static constexpr const char* _type_key = "tir.BlockRealize";
  TVM_DECLARE_FINAL_OBJECT_INFO(BlockRealizeNode, StmtNode);
};

/*!
 * \brief Managed reference to BlockRealizeNode
 * \sa BlockRealizeNode
 */
class BlockRealize : public Stmt {
 public:
  TVM_DLL explicit BlockRealize(Array<PrimExpr> iter_values, PrimExpr predicate, Block block,
                                Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(BlockRealize, Stmt, BlockRealizeNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BlockRealizeNode);
};

/*! \brief namespace of possible attributes in AttrStmt.attr_key */
namespace attr {
// The above attr does not pass to ir stage.
/*! \brief Mark launching extent of thread, used by device API. */
constexpr const char* thread_extent = "thread_extent";
/*! \brief Mark launching of a virtual thread. */
constexpr const char* virtual_thread = "virtual_thread";
/*! \brief Mark region is processed by a co-processor */
constexpr const char* coproc_scope = "coproc_scope";
/*!
 * \brief Mark region creates coprocessor micro ops,
 *  can be reused if corresponding variable is independent.
 */
constexpr const char* coproc_uop_scope = "coproc_uop_scope";
/*! \brief Mark the scope as volatile access for certain handle. */
constexpr const char* volatile_scope = "volatile_scope";
/*!
 * \brief Mark the scope as generated by extern primitive.
 *  such scope can contain arbitrary ir program and we need to be careful
 *  when make certain assumptions about the structure of the program.
 */
constexpr const char* extern_scope = "extern_scope";
/*!
 * \brief Mark the scope as when computation start to happen
 *  This can hint some code generator to create a new function for compute.
 */
constexpr const char* compute_scope = "compute_scope";
/*! \brief Mark storage alignment requirement of buffers */
constexpr const char* storage_alignment = "storage_alignment";
/*! \brief Mark storage scope of realization */
constexpr const char* realize_scope = "realize_scope";
/*! \brief The allocation device for global malloc in host. */
constexpr const char* device_id = "device_id";
/*! \brief The device type. */
constexpr const char* device_type = "device_type";
/*! \brief Mark of loop scope */
constexpr const char* loop_scope = "loop_scope";
/*! \brief Mark of reduce scope */
constexpr const char* reduce_scope = "reduce_scope";
/*! \brief Pragma: auto-unroll, max_step */
constexpr const char* pragma_auto_unroll_max_step = "pragma_auto_unroll_max_step";
/*! \brief Pragma: unroll explicit */
constexpr const char* pragma_unroll_explicit = "pragma_unroll_explicit";
/*! \brief Mark region is guarded by the pragma extension */
constexpr const char* pragma_scope_prefix = "pragma_";
/*! \brief Import C source or file into the final code gen module */
constexpr const char* pragma_import_c = "pragma_import_c";
/*! \brief Import llvm source or file into the final code gen module */
constexpr const char* pragma_import_llvm = "pragma_import_llvm";
/*! \brief Try to modify the AST to support Tensor Core */
constexpr const char* pragma_tensor_core = "pragma_tensor_core";
/*!
 * \brief Mark of prefetch scope, value=offset,
 *  run prefetch of Tensor on the current loop scope
 */
constexpr const char* prefetch_scope = "prefetch_scope";
/*!
 * \brief Marks the layout transforms to be used for a tensor.
 *
 * Only applies to a DataProducer, as it should be made part of the
 * PrimFunc attributes for TIR.
 */
constexpr const char* layout_transforms = "layout_transforms";
/*!
 * \brief Marks the physical axis separators
 *
 * Only applies to a DataProducer, as it should be made part of the
 * Buffer definition in a PrimFunc.  See `BufferNode::axis_separators`
 * for more details.
 */
constexpr const char* axis_separators = "axis_separators";
/*!
 * \brief Marks production of double buffer data
 */
constexpr const char* double_buffer_scope = "double_buffer_scope";
/*!
 * \brief Marks region used by double buffer write
 */
constexpr const char* double_buffer_write = "double_buffer_write";
/*! \brief Mark realization for rolling buffer optimization */
constexpr const char* rolling_buffer_scope = "rolling_buffer_scope";
/*! \brief Mark of scan update scope */
constexpr const char* scan_update_scope = "scan_update_scope";
/*! \brief Mark of scan init scope */
constexpr const char* scan_init_scope = "scan_init_scope";
/*!
 * \brief Mark alignment of buffer dimension
 *  stmt.node is Tensor
 *  stmt.value is tvm_tuple(dim, align, offset)
 *  This gives hint to require stride of dim to be k * align + offset.
 */
constexpr const char* buffer_dim_align = "buffer_dim_align";
/*! \brief Mark stores/loads with theirs bounds.  */
constexpr const char* buffer_bound = "buffer_bound";
/*!
 * \brief Bind the buffer specification to the region of the op
 *  When this scope occurs, the stmt.node is a Array<NodeRef> = [buffer, tensor]
 *  stmt.value is a tvm_tuple(min0, extent0, min1, extent1, ...).
 *  The scope represents that we need to bind the storage region of tensor to buffer.
 *  This will affect replacement of some variables inside the scope that
 *  corresponds to field of buffer to be the actual expressions of tensor during
 *  storage flattening phase.
 */
constexpr const char* buffer_bind_scope = "buffer_bind_scope";
// Pipeline related attributes
/*! \brief channel read scope */
constexpr const char* channel_read_scope = "channel_read_scope";
/*! \brief Advance step of channel after end of scope */
constexpr const char* channel_read_advance = "channel_read_advance";
/*! \brief channel write scope */
constexpr const char* channel_write_scope = "channel_write_scope";
/*! \brief Advance step of channel after end of scope */
constexpr const char* channel_write_advance = "channel_write_advance";
/*! \brief pipeline stage scope, implies always execution */
constexpr const char* pipeline_stage_scope = "pipeline_stage_scope";
/*! \brief pipeline execution scope, implies the scope can be pipelined. */
constexpr const char* pipeline_exec_scope = "pipeline_exec_scope";

/*!
 * \brief Mark that it is in the device scope.
 */
constexpr const char* device_scope = "device_scope";

/*!
 * \brief Mark that the attached statement runs asynchronously.
 */
constexpr const char* async_scope = "async_scope";

/*!
 * \brief Annotations for invoking and synchronizing asynchronous operations.

 * Synchronization is done in terms of "queue": It is an abstract entity associated
 * with each asynchronous unit, and it tracks invocations and completions of asynchronous
 * operations in the FIFO order.
 *
 * Similarly to PTX instructions commit_group and wait_group, these annotations express
 * synchronization by "counting":
 *
 * async_commit_queue(i): Group one or more invocations of async operations in the given scope,
 * and "commit" (or push) them to the queue i. A group of operations committed together is
 * awaited as one chunk. Groups committed to the same queue complete in the FIFO order.
 *
 * async_wait_queue(i, N): Block until only N most recent committed groups are still in-flight at
 * the queue i. N does not have to be a constant, but some backends may require a constant count.
*/
constexpr const char* async_commit_queue_scope = "async_commit_queue_scope";
constexpr const char* async_wait_queue_scope = "async_wait_queue_scope";
constexpr const char* async_wait_inflight_count = "async_wait_inflight_count";

/*!
 * \brief Mark that the shape of TensorCore fragment
 */
constexpr const char* fragment_shape = "fragment_shape";

/*!
 * \brief Mark that the layout of TensorCore fragment
 */
constexpr const char* fragment_layout = "fragment_layout";

/*!
 * \brief Mark that the kernel is hand threaded and doesn't need syncs inserted
 */
constexpr const char* hand_threaded = "hand_threaded";

/*!
 * \brief Mark whether the script-completer need to fill in missing access region
 *        during script parsing.
 * \note The result should be a integer mask with range [0, 4).
 *       if (mask & 1) the read region should be detected,
 *       if (mask & 2) the write region should be detected.
 */
constexpr const char* script_parsing_detect_access = "tir.script_parsing_detect_access";

/*!
 * \brief Mark that the loop should be partitioned.
 */
constexpr const char* pragma_loop_partition_hint = "pragma_loop_partition_hint";

/*! \brief Mark the stage of a statement in the software pipeline */
constexpr const char* software_pipeline_stage = "software_pipeline_stage";

/*! \brief Mark the order of a statement in the software pipeline */
constexpr const char* software_pipeline_order = "software_pipeline_order";

/*! \brief List stages in the software pipeline that should run asynchronously
 * \note All statements in the provided stages are assumed to have asynchronous
 *       semantics (e.g. CUDA async global to shared memory copy).
 */
constexpr const char* software_pipeline_async_stages = "software_pipeline_async_stages";

/*! \brief Mark the buffers which is const access and can be transformed layout. */
constexpr const char* layout_free_buffers = "layout_free_buffers";

/*! \brief Mark the local stage for the shared memory access should be added. */
constexpr const char* manifest_shared_memory_local_stage = "tir.manifest_shared_memory_local_stage";

/*! \brief Mark the tiling structure of blocks that are applied by rule Multi-Level-Tiling */
constexpr const char* meta_schedule_tiling_structure = "meta_schedule.tiling_structure";

/*!
 * \brief Mark that the loop should be further skip and bound to environment threads to enable
 * cooperative fetching.
 */
constexpr const char* meta_schedule_cooperative_fetch = "meta_schedule.cooperative_fetch";

/*! \brief The allowed range of thread extent in thread bindings */
constexpr const char* meta_schedule_thread_extent_low_inclusive =
    "meta_schedule.thread_extent_low_inclusive";

/*! \brief The allowed range of thread extent in thread bindings */
constexpr const char* meta_schedule_thread_extent_high_inclusive =
    "meta_schedule.thread_extent_high_inclusive";

/*! \brief Mark the block whose producer needs to be applied by rule Random-Compute-Location */
constexpr const char* meta_schedule_random_compute_producer =
    "meta_schedule.random_compute_producer";

/*! \brief Mark auto-parallel setting on the block. */
constexpr const char* meta_schedule_parallel = "meta_schedule.parallel";

/*! \brief Mark auto-vectorize setting on the block. */
constexpr const char* meta_schedule_vectorize = "meta_schedule.vectorize";

/*! \brief Mark auto-unroll setting on the block. */
constexpr const char* meta_schedule_unroll_explicit = "meta_schedule.unroll_explicit";

/*! \brief Mark auto-unroll setting on the block. */
constexpr const char* meta_schedule_unroll_implicit = "meta_schedule.unroll_implicit";

/*! \brief Mark that a block should be further rewritten using tensorization. */
constexpr const char* meta_schedule_auto_tensorize = "meta_schedule.auto_tensorize";

/*! \brief Mark that a block is a preprocessor block for layout rewrite. */
constexpr const char* meta_schedule_layout_rewrite_preproc = "meta_schedule.layout_rewrite_preproc";
/*!
 * \brief Mark that the init statement of a block should be further rewritten using tensorization.
 */
constexpr const char* meta_schedule_auto_tensorize_init = "meta_schedule.auto_tensorize_init";

/*!
 * \brief Mark that the block need to add predicate for block var bounds during lowering
 */
constexpr const char* require_block_var_bound_predicate = "require_bound_predicate";

/*! \brief Mark that tensor core is enabled in the PrimExpr */
constexpr const char* meta_schedule_tensor_core_enabled = "meta_schedule.tensor_core_enabled";

/*!
 * \brief Mark a block as generated by cache_read or cache_write block.
 * 0 means cache_read; 1 means cache_write.
 * \sa meta_schedule_cache_type_read
 * \sa meta_schedule_cache_type_write
 */
constexpr const char* meta_schedule_cache_type = "meta_schedule.cache_type";

/*! \sa meta_schedule_cache_type */
constexpr const int meta_schedule_cache_type_read = 0;

/*! \sa meta_schedule_cache_type */
constexpr const int meta_schedule_cache_type_write = 1;

/*! \brief Mark auto copy for memhammer */
constexpr const char* auto_copy = "auto_copy";

/*! \brief Mark local stage constraint on data copy */
constexpr const char* local_stage = "local_stage";

/*! \brief Mark vectorization length constraint on block */
constexpr const char* vector_bytes = "vector_bytes";

/*!
 * \brief Mark that a block is executed by a warp. This implies the extend of threadIdx.x is
 * warp size.
 */
constexpr const char* warp_execution = "warp_execution";

/*! \brief Mark that a block is disallowed in auto inline. */
constexpr const char* meta_schedule_inline_rule = "meta_schedule.inline_rule";

/*!
 * \brief Check if attr_key is a pragma key extension
 * \param attr_key The attr key to be compared
 * \return true if it is a pragma key
 */
inline bool IsPragmaKey(const std::string& attr_key) {
  return attr_key.compare(0, 7, "pragma_") == 0;
}

}  // namespace attr
/*!
 * \brief Create a type annotation expression
 * \param dtype The data type
 * \param span The location of this object in the source code.
 * \return Expr a expression with dtype.
 */
TVM_DLL PrimExpr TypeAnnotation(DataType dtype, Span span = Span());

// overload printing of for type.
TVM_DLL std::ostream& operator<<(std::ostream& os, ForKind kind);

// inline implementations
inline const char* ForKind2String(ForKind t) {
  switch (t) {
    case ForKind::kSerial:
      return "serial";
    case ForKind::kParallel:
      return "parallel";
    case ForKind::kVectorized:
      return "vectorized";
    case ForKind::kUnrolled:
      return "unroll";
    case ForKind::kThreadBinding:
      return "thread_binding";
  }
  LOG(FATAL) << "Unknown ForKind" << t;
}

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_STMT_H_
