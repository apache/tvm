/*!
 *  Copyright (c) 2016 by Contributors
 * \file tensor.h
 * \brief Dataflow tensor object
 */
#ifndef TVM_TENSOR_H_
#define TVM_TENSOR_H_

#include <string>
#include <vector>
#include <type_traits>
#include <tvm/array.h>
#include <ir/FunctionBase.h>
#include "./base.h"
#include "./expr.h"

namespace tvm {

// Internal node container of Tensor
class TensorNode;

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

using Halide::IR::FunctionRef;

/*!
 * \brief Tensor structure representing a possible input,
 *  or intermediate computation result.
 */
class Tensor : public FunctionRef {
 public:
  /*! \brief default constructor, used internally */
  Tensor() {}
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
   * \brief constructor of intermediate result.
   * \param shape Shape of the tensor.
   * \param fcompute The compute function to create the tensor.
   * \param name The optional name of the tensor.
   */
  Tensor(Array<Expr> shape, FCompute fcompute, std::string name = "tensor");
  // same constructor, specialized for different fcompute function
  Tensor(Array<Expr> shape, std::function<Expr(Var)> f, std::string name = "tensor")
      :Tensor(shape, GetFCompute(f), name) {}
  Tensor(Array<Expr> shape, std::function<Expr(Var, Var)> f, std::string name = "tensor")
      :Tensor(shape, GetFCompute(f), name) {}
  Tensor(Array<Expr> shape, std::function<Expr(Var, Var, Var)> f, std::string name = "tensor")
      :Tensor(shape, GetFCompute(f), name) {}
  Tensor(Array<Expr> shape, std::function<Expr(Var, Var, Var, Var)> f, std::string name = "tensor")
        :Tensor(shape, GetFCompute(f), name) {}
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

/*! \brief Node to represent a tensor */
class TensorNode : public Node {
 public:
  /*! \brief optional name of the tensor */
  std::string name;
  /*! \brief data type in the content of the tensor */
  Type dtype;
  /*! \brief The index representing each dimension, used by source expression. */
  Array<Var> dim_var;
  /*! \brief The shape of the tensor */
  Array<Expr> shape;
  /*! \brief source expression */
  Expr source;
  /*! \brief constructor */
  TensorNode() {}
  const char* type_key() const override {
    return "Tensor";
  }
  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("dtype", &dtype);
    v->Visit("dim_var", &dim_var);
    v->Visit("shape", &shape);
    v->Visit("source", &source);

  }
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
     << ", source=" << t->source
     << ", name=" << t->name << ')';
  return os;
}

}  // namespace tvm
#endif  // TVM_TENSOR_H_
