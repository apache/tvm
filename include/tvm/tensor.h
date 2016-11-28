/*!
 *  Copyright (c) 2016 by Contributors
 * \file tensor.h
 * \brief Dataflow tensor object
 */
#ifndef TVM_TENSOR_H_
#define TVM_TENSOR_H_

#include <tvm/array.h>
#include <ir/FunctionBase.h>
#include <string>
#include <vector>
#include <type_traits>

#include "./base.h"
#include "./expr.h"
#include "./domain.h"

namespace tvm {

// Internal node container of Tensor
class TensorNode;
// internal node container for Operation
class OperationNode;

using Halide::IR::FunctionRef;

/*!
 * \brief Tensor structure representing a possible input,
 *  or intermediate computation result.
 */
class Tensor : public FunctionRef {
 public:
  /*! \brief default constructor, used internally */
  Tensor() {}
  explicit Tensor(std::shared_ptr<Node> n) : FunctionRef(n) {}
  /*!
   * \brief constructor of input tensor
   * \param shape Shape of the tensor.
   * \param name optional name of the Tensor.
   * \param dtype The data type of the input tensor.
   */
  explicit Tensor(Array<Expr> shape,
                  std::string name = "tensor",
                  Type dtype = Float(32));
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const TensorNode* operator->() const;
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
  Expr operator()(Array<Expr> indices) const;
  // overload print function
  friend std::ostream& operator<<(std::ostream &os, const Tensor& t);
};

/*! \brief Operation that produces tensors */
class Operation : public NodeRef {
 public:
  /*! \brief default constructor  */
  Operation() {}
  explicit Operation(std::shared_ptr<Node> n) : NodeRef(n) {}
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
  Tensor output(size_t i) const;
};

/*! \brief Node to represent a tensor */
class TensorNode : public FunctionBaseNode {
 public:
  /*! \brief The shape of the tensor */
  Array<Expr> shape;
  /*! \brief optional name of the tensor */
  std::string name;
  /*! \brief data type in the content of the tensor */
  Type dtype;
  /*! \brief the source operation, can be None */
  Operation op;
  /*! \brief the output index from source operation */
  int value_index{0};
  /*! \brief constructor */
  TensorNode() {}
  const char* type_key() const final {
    return "Tensor";
  }
  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("shape", &shape);
    v->Visit("name", &name);
    v->Visit("dtype", &dtype);
    v->Visit("op", &op);
    v->Visit("value_index", &value_index);
  }
  const std::string& func_name() const final {
    return name;
  }
  int outputs() const final {
    return 1;
  }
  static Tensor make(Array<Expr> shape,
                     std::string name,
                     Type dtype,
                     Operation op,
                     int value_index);
};

/*!
 * \brief base class of operation node.
 */
class OperationNode : public Node {
 public:
  /*! \brief The domain of iteration of this op. */
  Domain domain;
  /*! \brief iter-Var over the dimensions */
  Array<Var> dim_var;
  /*! \brief optional name of the operation */
  std::string name;
  /*! \return number of outputs of this op */
  virtual size_t num_outputs() const = 0;
  /*! \return name of i-th output */
  virtual std::string output_name(size_t i) const = 0;
  /*! \return type of i-th output */
  virtual Type output_dtype(size_t i) const = 0;
  /*! \return shape of i-th output */
  virtual Array<Expr> output_shape(size_t i) const = 0;
};

// Implementations of inline functions
inline const OperationNode* Operation::operator->() const {
  return static_cast<const OperationNode*>(node_.get());
}

inline const TensorNode* Tensor::operator->() const {
  return static_cast<const TensorNode*>(node_.get());
}

inline size_t Tensor::ndim() const {
  return (*this)->shape.size();
}

inline std::ostream& operator<<(std::ostream &os, const Tensor& t) {  // NOLINT(*)
  os << "Tensor(shape=" << t->shape
     << ", name=" << t->name << ')';
  return os;
}

}  // namespace tvm
#endif  // TVM_TENSOR_H_
