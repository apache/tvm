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
 * \file tvm/tensor.h
 * \brief Dataflow tensor object
 */
#ifndef TVM_TENSOR_H_
#define TVM_TENSOR_H_

#include <tvm/node/container.h>
#include <string>
#include <vector>
#include <utility>
#include <type_traits>

#include "base.h"
#include "expr.h"
#include "expr_operator.h"
#include "arithmetic.h"

namespace tvm {

// Internal node container of Tensor
class TensorNode;
// internal node container for Operation
class OperationNode;

/*!
 * \brief Tensor structure representing a possible input,
 *  or intermediate computation result.
 */
class Tensor : public NodeRef {
 public:
  /*! \brief default constructor, used internally */
  Tensor() {}
  explicit Tensor(NodePtr<Node> n) : NodeRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const TensorNode* operator->() const;
  /*!
   * \brief check if two tensors equals each other.
   * \param other tensor to be checked.
   * \return whether the two tensors equals each other.
   */
  inline bool operator==(const Tensor& other) const;
  /*!
   * \brief check if two tensors are different.
   * \param other tensor to be checked.
   * \return whether the two tensors are different.
   */
  inline bool operator!=(const Tensor& other) const;
  /*! \return The dimension of the tensor */
  inline size_t ndim() const;
  /*!
   * \brief Take elements from the tensor
   * \param args The indices
   * \return the result expression representing tensor read.
   */
  template<typename... Args>
  inline Expr operator()(Args&& ...args) const {
    Array<Expr> indices{std::forward<Args>(args)...};
    return operator()(indices);
  }
  /*!
   * \brief Take elements from the tensor
   * \param indices the indices.
   * \return the result expression representing tensor read.
   */
  TVM_DLL Expr operator()(Array<Expr> indices) const;
  /*!
   * \brief Take elements from the tensor
   * \param indices the indices.
   * \return the result expression representing tensor read.
   */
  TVM_DLL Expr operator()(Array<Var> indices) const;
  /*!
   * \brief data structure to represent a slice that fixes first k coordinates.
   *  This is used to enable syntax sugar of Tensor[x][y][z] to get the element.
   */
  class Slice {
   public:
    // construct via tensor and indices
    Slice(const Tensor& tensor, std::vector<Expr> indices)
        : tensor_(tensor), indices_(indices) {}
    /*!
     * \brief get i-th slice from the current slice.
     * \param i the index of the coordinate
     * \return the subsequent slice.
     */
    inline Slice operator[](Expr i) {
      std::vector<Expr> other = indices_;
      other.emplace_back(i);
      return Slice(tensor_, other);
    }
    /*!
     * \brief Convert slice to expression.
     *  This is only valid when all the coordinates are fully specified.
     * \return the corresponding expression of this slice.
     */
    inline operator Expr() const {
      return tensor_(indices_);
    }

   private:
    const Tensor& tensor_;
    std::vector<Expr> indices_;
  };
  /*!
   * \brief get i-th slice from the current Tensor.
   * \param i the index of the coordinate
   * \return the subsequent slice.
   */
  inline Slice operator[](Expr i) const {
    return Slice(*this, {i});
  }
  /*! \brief specify container node */
  using ContainerType = TensorNode;
};

/*! \brief Operation that produces tensors */
class Operation : public ir::FunctionRef {
 public:
  /*! \brief default constructor  */
  Operation() {}
  explicit Operation(NodePtr<Node> n) : FunctionRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const OperationNode* operator->() const;
  /*!
   * \brief get the i-th output of the operation.
   * \param i the output index.
   * \return The i-th output.
   */
  TVM_DLL Tensor output(size_t i) const;
  /*! \brief specify container node */
  using ContainerType = OperationNode;
};

/*! \brief Node to represent a tensor */
class TensorNode : public Node {
 public:
  /*! \brief The shape of the tensor */
  Array<Expr> shape;
  /*! \brief data type in the content of the tensor */
  Type dtype;
  /*! \brief the source operation, can be None */
  Operation op;
  /*! \brief the output index from source operation */
  int value_index{0};
  /*! \brief constructor */
  TensorNode() {}

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("shape", &shape);
    v->Visit("dtype", &dtype);
    v->Visit("op", &op);
    v->Visit("value_index", &value_index);
  }
  TVM_DLL static Tensor make(Array<Expr> shape,
                             Type dtype,
                             Operation op,
                             int value_index);

  static constexpr const char* _type_key = "Tensor";
  TVM_DECLARE_NODE_TYPE_INFO(TensorNode, Node);
};


// Implementations of inline functions
inline const TensorNode* Tensor::operator->() const {
  return static_cast<const TensorNode*>(node_.get());
}

inline size_t Tensor::ndim() const {
  return (*this)->shape.size();
}

inline bool Tensor::operator==(const Tensor& other) const {
  if (get() == other.get()) return true;
  if (get() == nullptr || other.get() == nullptr) return false;
  if ((*this)->op.defined() || other->op.defined()) {
    return (*this)->op == other->op &&
        (*this)->value_index == other->value_index;
  } else {
    return false;
  }
}

inline bool Tensor::operator!=(const Tensor& other) const {
  return !(*this == other);
}

// macro to turn every operation of slice to expression
#define DEFINE_OVERLOAD_SLICE_UNARY_OP(Op)                              \
  inline Expr operator Op (const Tensor::Slice& a) {                    \
    return Op a.operator Expr() ;                                       \
  }                                                                     \

#define DEFINE_OVERLOAD_SLICE_BINARY_OP(Op)                             \
  template<typename T>                                                  \
  inline Expr operator Op (const Tensor::Slice& a, const T& b) {        \
    return a.operator Expr() Op b;                                      \
  }                                                                     \
  template<typename T>                                                  \
  inline Expr operator Op (const T& a, const Tensor::Slice& b) {        \
    return a Op b.operator Expr();                                      \
  }                                                                     \
  inline Expr operator Op (const Tensor::Slice& a, const Tensor::Slice& b) { \
    return a.operator Expr() Op b.operator Expr();                      \
  }

DEFINE_OVERLOAD_SLICE_UNARY_OP(!);
DEFINE_OVERLOAD_SLICE_UNARY_OP(-);
DEFINE_OVERLOAD_SLICE_BINARY_OP(+);
DEFINE_OVERLOAD_SLICE_BINARY_OP(-);
DEFINE_OVERLOAD_SLICE_BINARY_OP(*);
DEFINE_OVERLOAD_SLICE_BINARY_OP(/);
DEFINE_OVERLOAD_SLICE_BINARY_OP(%);
DEFINE_OVERLOAD_SLICE_BINARY_OP(==);
DEFINE_OVERLOAD_SLICE_BINARY_OP(<=);
DEFINE_OVERLOAD_SLICE_BINARY_OP(>=);
DEFINE_OVERLOAD_SLICE_BINARY_OP(!=);
DEFINE_OVERLOAD_SLICE_BINARY_OP(&&);
DEFINE_OVERLOAD_SLICE_BINARY_OP(||);
DEFINE_OVERLOAD_SLICE_BINARY_OP(>>);
DEFINE_OVERLOAD_SLICE_BINARY_OP(<<);
DEFINE_OVERLOAD_SLICE_BINARY_OP(>);  // NOLINT(*)
DEFINE_OVERLOAD_SLICE_BINARY_OP(<);  // NOLINT(*)

}  // namespace tvm

namespace std {
template <>
struct hash<::tvm::Operation> {
  std::size_t operator()(const ::tvm::Operation& k) const {
    return k.hash();
  }
};

template <>
struct hash<::tvm::Tensor> {
  std::size_t operator()(const ::tvm::Tensor& k) const {
    if (k.defined() && k->op.defined()) {
      return k->op.hash();
    } else{
      return k.hash();
    }
  }
};
}  // namespace std
#endif  // TVM_TENSOR_H_
