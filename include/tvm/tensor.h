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
#include "./operation.h"

namespace tvm {

// Internal node container of Tensor
class TensorNode;

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

/*! \brief The compute function to specify the input source of a Tensor */
using FCompute = std::function<Expr (const Array<Var>& i)>;

// converters from other functions into fcompute
inline FCompute GetFCompute(std::function<Expr(Var x)> f) {
  return [f] (const Array<Var>& i) { return f(i[0]); };
}
inline FCompute GetFCompute(std::function<Expr(Var, Var)> f) {
  return [f] (const Array<Var>& i) { return f(i[0], i[1]); };
}
inline FCompute GetFCompute(std::function<Expr(Var, Var, Var)> f) {
  return [f] (const Array<Var>& i) { return f(i[0], i[1], i[2]); };
}
inline FCompute GetFCompute(std::function<Expr(Var, Var, Var, Var)> f) {
  return [f] (const Array<Var>& i) { return f(i[0], i[1], i[2], i[3]); };
}

/*!
 * \brief Construct a new tensor by computing over shape,
 *  using the computation rule: result_tensor[axis] = fcompute(axis)
 * \param shape Shape of the tensor.
 * \param fcompute The compute function to create the tensor.
 * \param name The optional name of the tensor.
 */
Tensor Compute(Array<Expr> shape, FCompute fcompute, std::string name = "tensor");

// same as compute, specialized for different fcompute function
inline Tensor Compute(Array<Expr> shape,
                      std::function<Expr(Var)> f,
                      std::string name = "tensor") {
  FCompute fc = [f] (const Array<Var>& i) { return f(i[0]); };
  return Compute(shape, fc, name);
}
inline Tensor Compute(Array<Expr> shape,
                      std::function<Expr(Var, Var)> f,
                      std::string name = "tensor") {
  FCompute fc = [f] (const Array<Var>& i) { return f(i[0], i[1]); };
  return Compute(shape, fc, name);
}
inline Tensor Compute(Array<Expr> shape,
                      std::function<Expr(Var, Var, Var)> f,
                      std::string name = "tensor") {
  FCompute fc = [f] (const Array<Var>& i) { return f(i[0], i[1], i[2]); };
  return  Compute(shape, fc, name);
}
inline Tensor Compute(Array<Expr> shape,
                      std::function<Expr(Var, Var, Var, Var)> f,
                      std::string name = "tensor") {
  FCompute fc = [f] (const Array<Var>& i) { return f(i[0], i[1], i[2], i[3]); };
  return Compute(shape, fc, name);
}

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
  Operation source_op;
  /*! \brief the output index from source operation */
  int source_index{0};
  /*! \brief constructor */
  TensorNode() {}
  const char* type_key() const final {
    return "Tensor";
  }
  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("shape", &shape);
    v->Visit("name", &name);
    v->Visit("dtype", &dtype);
    v->Visit("source_op", &source_op);
    v->Visit("source_index", &source_index);
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
                     Operation source_op,
                     int source_index);
};

// implementations

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
