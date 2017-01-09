/*!
 *  Copyright (c) 2016 by Contributors
 * \file operation.h
 * \brief Operation node can generate one or multiple Tensors
 */
#ifndef TVM_OPERATION_H_
#define TVM_OPERATION_H_

#include <string>
#include "./expr.h"
#include "./tensor.h"

namespace tvm {

/*!
 * \brief A Compute op that compute a tensor on certain domain.
 */
class ComputeOpNode : public OperationNode {
 public:
  /*! \brief IterVar on each axis */
  Array<IterVar> axis;
  /*! \brief the compute expression */
  Expr body;
  /*! \brief constructor */
  ComputeOpNode() {}

  size_t num_outputs() const final {
    return 1;
  }
  Array<IterVar> root_iter_vars() const final;
  std::string output_name(size_t i) const final;
  Type output_dtype(size_t i) const final;
  Array<Expr> output_shape(size_t i) const final;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("axis", &axis);
    v->Visit("body", &body);
  }
  static Operation make(std::string name,
                        Array<IterVar> axis,
                        Expr body);

  static constexpr const char* _type_key = "ComputeOp";
  TVM_DECLARE_NODE_TYPE_INFO(ComputeOpNode);
};


/*! \brief The compute function to specify the input source of a Tensor */
using FCompute = std::function<Expr (const Array<Var>& i)>;

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

}  // namespace tvm


namespace std {
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
#endif  // TVM_OPERATION_H_
