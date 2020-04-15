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
// Acknowledgement: Mnay low-level stmts originate from Halide.
#ifndef TVM_TIR_STMT_H_
#define TVM_TIR_STMT_H_

#include <tvm/tir/expr.h>

#include <type_traits>
#include <string>
#include <vector>
#include <utility>

namespace tvm {
namespace tir {

/*! \brief Base node of all statements. */
class StmtNode : public Object {
 public:
  static constexpr const char* _type_key = "Stmt";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
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
  /*! \brief The value to be binded. */
  PrimExpr value;
  /*! \brief The body block. */
  Stmt body;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("var", &var);
    v->Visit("value", &value);
    v->Visit("body", &body);
  }

  bool SEqualReduce(const LetStmtNode* other, SEqualReducer equal) const {
    return
        equal.DefEqual(var, other->var) &&
        equal(value, other->value) &&
        equal(body, other->body);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(var);
    hash_reduce(value);
    hash_reduce(body);
  }

  TVM_DLL static Stmt make(Var var, PrimExpr value, Stmt body);

  static constexpr const char* _type_key = "LetStmt";
  TVM_DECLARE_FINAL_OBJECT_INFO(LetStmtNode, StmtNode);
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
  std::string attr_key;
  /*! \brief The attribute value, value is well defined at current scope. */
  PrimExpr value;
  /*! \brief The body statement to be executed */
  Stmt body;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("node", &node);
    v->Visit("attr_key", &attr_key);
    v->Visit("value", &value);
    v->Visit("body", &body);
  }

  bool SEqualReduce(const AttrStmtNode* other, SEqualReducer equal) const {
    return
        equal(node, other->node) &&
        equal(attr_key, other->attr_key) &&
        equal(value, other->value) &&
        equal(body, other->body);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(node);
    hash_reduce(attr_key);
    hash_reduce(value);
    hash_reduce(body);
  }

  TVM_DLL static Stmt make(ObjectRef node,
                           std::string type_key,
                           PrimExpr value,
                           Stmt body);

  static constexpr const char* _type_key = "AttrStmt";
  TVM_DECLARE_FINAL_OBJECT_INFO(AttrStmtNode, StmtNode);
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
  }

  bool SEqualReduce(const AssertStmtNode* other, SEqualReducer equal) const {
    return
        equal(condition, other->condition) &&
        equal(message, other->message) &&
        equal(body, other->body);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(condition);
    hash_reduce(message);
    hash_reduce(body);
  }

  TVM_DLL static Stmt make(PrimExpr condition, PrimExpr message, Stmt body);

  static constexpr const char* _type_key = "AssertStmt";
  TVM_DECLARE_FINAL_OBJECT_INFO(AssertStmtNode, StmtNode);
};

/*!
 * \brief Store value to the buffer.
 *
 *  Equivalent to ((DType*)buffer_var)[index] = value.
 *  where DType is the type specified by type().element_of().
 *
 *  For example, if type = float32x3, then the store will corresponds to
 *
 * \code
 *
 *  auto buffer = static_cast<float*>(buffer_var);
 *  buffer[index.v0] = value.v0;
 *  buffer[index.v1] = value.v1;
 *  buffer[index.v2] = value.v2;
 *
 * \endcode
 * \sa LoadNode
 */
class StoreNode : public StmtNode {
 public:
  /*! \brief The buffer variable. */
  Var buffer_var;
  /*! \brief The value to be stored. */
  PrimExpr value;
  /*! \brief The index locations to be stored. */
  PrimExpr index;
  /*! \brief The predicate to mask which lanes would be stored. */
  PrimExpr predicate;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("buffer_var", &buffer_var);
    v->Visit("value", &value);
    v->Visit("index", &index);
    v->Visit("predicate", &predicate);
  }

  bool SEqualReduce(const StoreNode* other, SEqualReducer equal) const {
    return
        equal(buffer_var, other->buffer_var) &&
        equal(value, other->value) &&
        equal(index, other->index) &&
        equal(predicate, other->predicate);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(buffer_var);
    hash_reduce(value);
    hash_reduce(index);
    hash_reduce(predicate);
  }

  TVM_DLL static Stmt make(Var buffer_var,
                           PrimExpr value,
                           PrimExpr index,
                           PrimExpr predicate);

  static constexpr const char* _type_key = "Store";
  TVM_DECLARE_FINAL_OBJECT_INFO(StoreNode, StmtNode);
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
class BufferStore;
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
  }

  bool SEqualReduce(const BufferStoreNode* other, SEqualReducer equal) const {
    return
        equal(buffer, other->buffer) &&
        equal(value, other->value) &&
        equal(indices, other->indices);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(buffer);
    hash_reduce(value);
    hash_reduce(indices);
  }

  static constexpr const char* _type_key = "BufferStore";
  TVM_DECLARE_FINAL_OBJECT_INFO(BufferStoreNode, StmtNode);
};

class BufferStore : public Stmt {
 public:
  TVM_DLL explicit BufferStore(Buffer buffer,
                               PrimExpr value,
                               Array<PrimExpr> indices);
  TVM_DEFINE_OBJECT_REF_METHODS(BufferStore, Stmt, BufferStoreNode);
};

/*!
 * \brief Store value into mult-dimensional array defined by func.
 */
class ProvideNode : public StmtNode {
 public:
  /*! \brief The function to be updated. */
  FunctionRef func;
  /*! \brief The output value index if func's value is a tuple. */
  int value_index{0};
  /*! \brief The value to be stored. */
  PrimExpr value;
  /*! \brief The index arguments of the function. */
  Array<PrimExpr> args;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("func", &func);
    v->Visit("value_index", &value_index);
    v->Visit("value", &value);
    v->Visit("args", &args);
  }

  bool SEqualReduce(const ProvideNode* other, SEqualReducer equal) const {
    return
        equal(func, other->func) &&
        equal(value_index, other->value_index) &&
        equal(value, other->value) &&
        equal(args, other->args);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(func);
    hash_reduce(value_index);
    hash_reduce(value);
    hash_reduce(args);
  }

  TVM_DLL static Stmt make(FunctionRef func,
                           int value_index,
                           PrimExpr value,
                           Array<PrimExpr> args);

  static constexpr const char* _type_key = "Provide";
  TVM_DECLARE_FINAL_OBJECT_INFO(ProvideNode, StmtNode);
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

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("buffer_var", &buffer_var);
    v->Visit("dtype", &dtype);
    v->Visit("extents", &extents);
    v->Visit("condition", &condition);
    v->Visit("body", &body);
  }

  bool SEqualReduce(const AllocateNode* other, SEqualReducer equal) const {
    return
        equal.DefEqual(buffer_var, other->buffer_var) &&
        equal(dtype, other->dtype) &&
        equal(extents, other->extents) &&
        equal(condition, other->condition) &&
        equal(body, other->body);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(buffer_var);
    hash_reduce(dtype);
    hash_reduce(extents);
    hash_reduce(condition);
    hash_reduce(body);
  }

  TVM_DLL static Stmt make(Var buffer_var,
                           DataType dtype,
                           Array<PrimExpr> extents,
                           PrimExpr condition,
                           Stmt body);

  /*!
   * \brief If the buffer size is constant, return the size.
   *        Otherwise return 0.
   * \return The result.
   */
  int32_t constant_allocation_size() const {
    return constant_allocation_size(extents);
  }
  /*!
   * \brief If the buffer size is constant, return the size.
   *        Otherwise return 0.
   * \param extents The extents of the buffer.
   * \return The result.
   */
  TVM_DLL static int32_t constant_allocation_size(
      const Array<PrimExpr>& extents);

  static constexpr const char* _type_key = "Allocate";
  TVM_DECLARE_FINAL_OBJECT_INFO(AllocateNode, StmtNode);
};

/*! \brief Free the resources in the buffer before the scope ends. */
class FreeNode : public StmtNode {
 public:
  /*! \brief The buffer variable. */
  Var buffer_var;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("buffer_var", &buffer_var);
  }

  bool SEqualReduce(const FreeNode* other, SEqualReducer equal) const {
    return
        equal(buffer_var, other->buffer_var);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(buffer_var);
  }

  TVM_DLL static Stmt make(Var buffer_var);

  static constexpr const char* _type_key = "Free";
  TVM_DECLARE_FINAL_OBJECT_INFO(FreeNode, StmtNode);
};

/*!
 * \brief Annotate the bounds where func need to be written and read in body.
 *  We will need to allocate space for the corresponding regions.
 */
class RealizeNode : public StmtNode {
 public:
  /*! \brief The function to be realized. */
  FunctionRef func;
  /*! \brief The output value index if func's value is a tuple. */
  int value_index;
  /*! \brief The data type of the array. */
  DataType dtype;
  /*! \brief Bounds to be realized. */
  Region bounds;
  /*! \brief Only realize if condition holds. */
  PrimExpr condition;
  /*! \brief The body of realization. */
  Stmt body;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("func", &func);
    v->Visit("value_index", &value_index);
    v->Visit("dtype", &dtype);
    v->Visit("bounds", &bounds);
    v->Visit("condition", &condition);
    v->Visit("body", &body);
  }

  TVM_DLL static Stmt make(FunctionRef func,
                           int value_index,
                           DataType dtype,
                           Region bounds,
                           PrimExpr condition,
                           Stmt body);

  bool SEqualReduce(const RealizeNode* other, SEqualReducer equal) const {
    return
        equal(func, other->func) &&
        equal(value_index, other->value_index) &&
        equal(dtype, other->dtype) &&
        equal(bounds, other->bounds) &&
        equal(condition, other->condition) &&
        equal(body, other->body);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(func);
    hash_reduce(value_index);
    hash_reduce(dtype);
    hash_reduce(bounds);
    hash_reduce(condition);
    hash_reduce(body);
  }

  static constexpr const char* _type_key = "Realize";
  TVM_DECLARE_FINAL_OBJECT_INFO(RealizeNode, StmtNode);
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
  size_t size() const {
    return seq.size();
  }
  /*!
   * \brief Get the index-th element in the sequence.
   */
  Stmt operator[](size_t index) const {
    return seq[index];
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("seq", &seq);
  }

  bool SEqualReduce(const SeqStmtNode* other, SEqualReducer equal) const {
    return equal(seq, other->seq);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(seq);
  }

  static constexpr const char* _type_key = "SeqStmt";
  TVM_DECLARE_FINAL_OBJECT_INFO(SeqStmtNode, StmtNode);
};

/*! \brief Sequence statement. */
class SeqStmt : public Stmt {
 public:
  /*!
   * \brief Construct SeqStmt.
   * \param seq The sequence.
   */
  TVM_DLL explicit SeqStmt(Array<Stmt> seq);

  /*! \return get the size of the sequence */
  size_t size() const {
    return operator->()->size();
  }
  /*!
   * \brief Get the index-th element in the sequence.
   */
  Stmt operator[](size_t index) const {
    return (*(operator->()))[index];
  }
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
   * \param seq_args The list of arguments to be flattened.
   * \tparam Args arguments
   * \return The constructed statement
   */
  template<typename ...Args>
  static Stmt Flatten(Args&&... seq_args) {
    Array<Stmt> seq;
    runtime::detail::for_each(
        Flattener(&seq), std::forward<Args>(seq_args)...);
    if (seq.size() == 1) return seq[0];
    return SeqStmt(seq);
  }
  /*! \brief Helper class to flatten sequence of arguments into Array. */
  class Flattener {
   public:
    explicit Flattener(Array<Stmt>* seq)
        : seq_(seq) {}

    void operator()(size_t i, const Stmt& stmt) const {
      if (!stmt.defined()) return;
      if (auto* op = stmt.as<SeqStmtNode>()) {
        operator()(0, op->seq);
      } else {
        seq_->push_back(stmt);
      }
    }

    template<typename T>
    void operator()(size_t i, const T& seq) const {
      for (auto v : seq) {
        this->operator()(0, v);
      }
    }

   private:
    Array<Stmt>* seq_;
  };

  TVM_DEFINE_OBJECT_REF_METHODS(SeqStmt, Stmt, SeqStmtNode);
};

/*!
 * \brief IfThenElse statment.
 */
class IfThenElseNode : public StmtNode {
 public:
  /*! \brief The condition. */
  PrimExpr condition;
  /*! \brief The branch to be executed when condition is true. */
  Stmt then_case;
  /*! \brief The branch to be executed when condition is false, can be null. */
  Stmt else_case;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("condition", &condition);
    v->Visit("then_case", &then_case);
    v->Visit("else_case", &else_case);
  }

  bool SEqualReduce(const IfThenElseNode* other, SEqualReducer equal) const {
    return
        equal(condition, other->condition) &&
        equal(then_case, other->then_case) &&
        equal(else_case, other->else_case);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(condition);
    hash_reduce(then_case);
    hash_reduce(else_case);
  }

  TVM_DLL static Stmt make(PrimExpr condition, Stmt then_case, Stmt else_case = Stmt());

  static constexpr const char* _type_key = "IfThenElse";
  TVM_DECLARE_FINAL_OBJECT_INFO(IfThenElseNode, StmtNode);
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
  }

  bool SEqualReduce(const EvaluateNode* other, SEqualReducer equal) const {
    return equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(value);
  }

  TVM_DLL static Stmt make(PrimExpr v);

  static constexpr const char* _type_key = "Evaluate";
  TVM_DECLARE_FINAL_OBJECT_INFO(EvaluateNode, StmtNode);
};

/*! \brief Additional annotation of for loop. */
enum class ForType : int {
  /*! \brief serial execution. */
  Serial = 0,
  /*! \brief parallel execution on CPU. */
  Parallel = 1,
  /*! \brief Vector SIMD loop annotaion. */
  Vectorized = 2,
  /*! \brief Unroll annotation. */
  Unrolled = 3
};

// Kevice api of for loop
// kept for backward compatibility
// consider refactor and remove later.
enum class DeviceAPI: int {
  None = 0
};

/*!
 * \brief A for loop, with poissible type annotations.
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
  /*! \brief The type of the for loop. */
  ForType for_type;
  /*!
   * \brief Deprecated, reserved for backward compatibility.
   *  Consider refactor and remove later.
   */
  DeviceAPI device_api;
  /*! \brief The body of the for loop. */
  Stmt body;

  TVM_DLL static Stmt make(Var loop_var,
                           PrimExpr min,
                           PrimExpr extent,
                           ForType for_type,
                           DeviceAPI device_api,
                           Stmt body);

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("loop_var", &loop_var);
    v->Visit("min", &min);
    v->Visit("extent", &extent);
    v->Visit("for_type", &for_type);
    v->Visit("device_api", &device_api);
    v->Visit("body", &body);
  }

  bool SEqualReduce(const ForNode* other, SEqualReducer equal) const {
    return
        equal.DefEqual(loop_var, other->loop_var) &&
        equal(min, other->min) &&
        equal(extent, other->extent) &&
        equal(for_type, other->for_type) &&
        equal(device_api, other->device_api) &&
        equal(body, other->body);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(loop_var);
    hash_reduce(min);
    hash_reduce(extent);
    hash_reduce(for_type);
    hash_reduce(device_api);
    hash_reduce(body);
  }


  static constexpr const char* _type_key = "For";
  TVM_DECLARE_FINAL_OBJECT_INFO(ForNode, StmtNode);
};

/*!
 * \brief A prefetch hint of func.
 */
class PrefetchNode : public StmtNode {
 public:
  /*! \brief The function to be prefetched. */
  FunctionRef func;
  /*! \brief The output value index if func's value is a tuple. */
  int value_index;
  /*! \brief The data type of the array. */
  DataType dtype;
  /*! \brief Bounds to be prefetched. */
  Region bounds;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("func", &func);
    v->Visit("value_index", &value_index);
    v->Visit("dtype", &dtype);
    v->Visit("bounds", &bounds);
  }

  bool SEqualReduce(const PrefetchNode* other, SEqualReducer equal) const {
    return
        equal(func, other->func) &&
        equal(value_index, other->value_index) &&
        equal(dtype, other->dtype) &&
        equal(bounds, other->bounds);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(func);
    hash_reduce(value_index);
    hash_reduce(dtype);
    hash_reduce(bounds);
  }

  TVM_DLL static Stmt make(FunctionRef func,
                           int value_index,
                           DataType dtype,
                           Region bounds);

  static constexpr const char* _type_key = "Prefetch";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrefetchNode, StmtNode);
};

/*!
 * \brief Auxiliary data structure used in IR Pass to indicate a tensor.
 */
struct TensorKey {
  FunctionRef f;
  int value_index;

  inline bool operator==(const TensorKey& other) const {
    return f == other.f && value_index == other.value_index;
  }
  inline std::string GetName() const {
    if (f->num_outputs() == 1) return f->func_name();
    std::ostringstream os;
    os << f->func_name() << ".v" << value_index;
    return os.str();
  }
};

/*! \brief namespace of possible attribute sin AttrStmt.attr_key */
namespace attr {
// The above attr does not pass to ir stage.
/*! \brief Mark launching extent of thread, used by device API. */
constexpr const char* thread_extent = "thread_extent";
/*! \brief Mark launching of a virtual thread. */
constexpr const char* virtual_thread = "virtual_thread";
/*! \brief Mark region is processed by a co-proccesor */
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
/*! \brief Mark storage scope of buffers */
constexpr const char* storage_scope = "storage_scope";
/*! \brief Mark storage alignement requirement of buffers */
constexpr const char* storage_alignment = "storage_alignment";
/*! \brief Mark storage scope of realization */
constexpr const char* realize_scope = "realize_scope";
/*! \brief The allocation context for global malloc in host. */
constexpr const char* device_context_id = "device_context_id";
/*! \brief The device type. */
constexpr const char* device_context_type = "device_context_type";
/*! \brief Mark of loop scope */
constexpr const char* loop_scope = "loop_scope";
/*! \brief Mark of reduce scope */
constexpr const char* reduce_scope = "reduce_scope";
/*! \brief Mark region is guarded by the pragma extension */
constexpr const char* pragma_scope_prefix = "pragma_";
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
 * \brief Marks production of double buffer data
 */
constexpr const char* double_buffer_scope = "double_buffer_scope";
/*!
 * \brief Marks region used by double buffer write
 */
constexpr const char* double_buffer_write = "double_buffer_write";
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
 * \brief Mark that this stage is an OpenGL shader. Since OpenGL shader only
 * allows writing out to one element of the output texture, the Provide node
 * gets translated to a special Call::glsl_texture_store statement instead of a
 * Store statement.
 */
constexpr const char* opengl_stage_scope = "opengl_stage_scope";

/*!
 * \brief Mark that it is in the device scope.
 */
constexpr const char* device_scope = "device_scope";

/*!
 * \brief Mark that the shape of TensorCore fragment
 */
constexpr const char* fragment_shape = "fragment_shape";

/*!
 * \brief Mark that the layout of TensorCore fragment
 */
constexpr const char* fragment_layout = "fragment_layout";

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
 * \return Expr a expression with dtype.
 */
inline PrimExpr TypeAnnotation(DataType dtype) {
  return tir::CallNode::make(dtype,
                        "type_annotation", {},
                        tir::CallNode::PureIntrinsic);
}

// overload printing of for type.
TVM_DLL std::ostream& operator<<(std::ostream& os, ForType for_type);

}  // namespace tir
}  // namespace tvm

namespace std {
template <>
struct hash<::tvm::tir::TensorKey> {
  std::size_t operator()(const ::tvm::tir::TensorKey& k) const {
    size_t lhs = ::tvm::ObjectHash()(k.f);
    size_t rhs = static_cast<size_t>(k.value_index);
    lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
    return lhs;
  }
};
}  // namespace std

#endif  // TVM_TIR_STMT_H_
