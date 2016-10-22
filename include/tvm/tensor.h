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
#include "./base.h"
#include "./expr.h"
#include "./array.h"

namespace tvm {

/*! \brief Node to represent a tensor */
class TensorNode : public Node {
 public:
  /*! \brief optional name of the tensor */
  std::string name;
  /*! \brief data type in the content of the tensor */
  DataType dtype;
  /*! \brief The index representing each dimension, used by source expression. */
  Array<Var> dim_index;
  /*! \brief The shape of the tensor */
  Array<Expr> shape;
  /*! \brief source expression */
  Expr source;
  /*! \brief constructor */
  TensorNode() {}
  const char* type_key() const override {
    return "TensorNode";
  }
  void VisitAttrs(AttrVisitor* visitor) override {
    visitor->Visit("name", &name);
    visitor->Visit("dtype", &dtype);
  }
  void VisitNodeRefFields(FNodeRefVisit fvisit) override {
    fvisit("dim_index", &dim_index);
    fvisit("shape", &shape);
    fvisit("source", &source);
  }
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
 * \brief Tensor structure representing a possible input,
 *  or intermediate computation result.
 */
class Tensor : public NodeRef {
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
                  DataType dtype = kFloat32);
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
  /*! \return The dimension of the tensor */
  inline size_t ndim() const {
    return static_cast<const TensorNode*>(node_.get())->shape.size();
  }
  /*! \return The name of the tensor */
  inline const std::string& name() const {
    return static_cast<const TensorNode*>(node_.get())->name;
  }
  /*! \return The data type tensor */
  inline DataType dtype() const {
    return static_cast<const TensorNode*>(node_.get())->dtype;
  }
  /*! \return The source expression of intermediate tensor */
  inline const Expr& source() const {
    return static_cast<const TensorNode*>(node_.get())->source;
  }
  /*! \return The internal dimension index used by source expression */
  inline const Array<Var>& dim_index() const {
    return static_cast<const TensorNode*>(node_.get())->dim_index;
  }
  /*! \return The shape of the tensor */
  inline const Array<Expr>& shape() const {
    return static_cast<const TensorNode*>(node_.get())->shape;
  }
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
  /*! \return list of input tensors to this tensor */
  std::vector<Tensor> InputTensors() const;
  /*! \return whether the tensor stores a result of reduction */
  bool IsRTensor() const;
  // printt function
  friend std::ostream& operator<<(std::ostream &os, const Tensor& t) {  // NOLINT(*)
    os << "Tensor(shape=" << t.shape()
       << ", source=" << t.source()
       << ", name=" << t.name() << ')';
    return os;
  }
};

}  // namespace tvm
#endif  // TVM_TENSOR_H_
