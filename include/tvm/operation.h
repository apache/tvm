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
 * \brief A placeholder op represents an input placeholder.
 */
class PlaceholderOpNode : public OperationNode {
 public:
  /*! \brief The shape of the input */
  Array<Expr> shape;
  /*! \brief The data type of the input. */
  Type dtype;

  int num_outputs() const final {
    return 1;
  }
  Array<IterVar> root_iter_vars() const final;
  Type output_dtype(size_t i) const final;
  Array<Expr> output_shape(size_t i) const final;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("shape", &shape);
    v->Visit("dtype", &dtype);
  }
  static Operation make(std::string name,
                        Array<Expr> shape,
                        Type dtype);

  static constexpr const char* _type_key = "PlaceholderOp";
  TVM_DECLARE_NODE_TYPE_INFO(PlaceholderOpNode);
};

/*!
 * \brief A Compute op that compute a tensor on certain domain.
 */
class ComputeOpNode : public OperationNode {
 public:
  /*! \brief IterVar on each axis */
  Array<IterVar> axis;
  /*! \brief IterVar on each reduction axis, if the body is a Reduce */
  Array<IterVar> reduce_axis;
  /*! \brief the compute expression */
  Expr body;
  /*! \brief constructor */
  ComputeOpNode() {}

  int num_outputs() const final {
    return 1;
  }
  Array<IterVar> root_iter_vars() const final;
  Type output_dtype(size_t i) const final;
  Array<Expr> output_shape(size_t i) const final;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("axis", &axis);
    v->Visit("reduce_axis", &reduce_axis);
    v->Visit("body", &body);
  }
  static Operation make(std::string name,
                        Array<IterVar> axis,
                        Expr body);

  static constexpr const char* _type_key = "ComputeOp";
  TVM_DECLARE_NODE_TYPE_INFO(ComputeOpNode);
};

/*!
 * \brief Symbolic scan.
 */
class ScanOpNode : public OperationNode {
 public:
  /*! \brief IterVar to scan over */
  IterVar scan_axis;
  /*! \brief the initialization tensors */
  Array<Tensor> init;
  /*! \brief the update function represented by tensor */
  Array<Tensor> update;
  /*! \brief The placeholder to refer as states in update. */
  Array<Tensor> state_placeholder;
  /*!
   * \brief Spatial axis to indicate spatial dimension of each output.
   *  They corresponds to flattened spatial axis of the outputs.
   *
   *  [output[0].axis[1], output[0].axis[2]... output[k].axis[j]...]
   *  These are auxiliary data structure for storing result of bound inference.
   *  They do not corresponds to splittable iterations, thus the name comes
   *  with underscore.
   */
  Array<IterVar> spatial_axis_;
  /*! \brief constructor */
  ScanOpNode() {}
  // override behavior.
  int num_outputs() const final;
  Array<IterVar> root_iter_vars() const final;
  Type output_dtype(size_t i) const final;
  Array<Expr> output_shape(size_t i) const final;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("scan_axis", &scan_axis);
    v->Visit("init", &init);
    v->Visit("update", &update);
    v->Visit("state_placeholder", &state_placeholder);
    v->Visit("spatial_axis_", &spatial_axis_);
  }
  static Operation make(std::string name,
                        IterVar axis,
                        Array<Tensor> init,
                        Array<Tensor> update,
                        Array<Tensor> state_placeholder);

  static constexpr const char* _type_key = "ScanOp";
  TVM_DECLARE_NODE_TYPE_INFO(ScanOpNode);
};


/*! \brief The compute function to specify the input source of a Tensor */
using FCompute = std::function<Expr (const Array<Var>& i)>;

/*!
 * \brief create a place holder tensor.
 * \param shape The shape of the tensor.
 * \param dtype the data type of the tensor.
 * \param name The name of the Tensor.
 */
Tensor Placeholder(Array<Expr> shape,
                   Type dtype = Float(32),
                   std::string name = "placeholder");

/*!
 * \brief Construct a new tensor by computing over shape,
 *  using the computation rule: result_tensor[axis] = fcompute(axis)
 * \param shape Shape of the tensor.
 * \param fcompute The compute function to create the tensor.
 * \param name The optional name of the tensor.
 */
Tensor Compute(Array<Expr> shape, FCompute fcompute, std::string name = "tensor");

/*!
 * \brief Construct new tensors by scan over scan_axis.
 *
 * \param scan_axis The iteration representing the scan.
 * \param init The intialize tensor of first K steps.
 * \param update The update tensor indicated the updated result after each timestamp.
 * \param state_placeholder The placeholder for the states.
 * \param name The optional name of the tensor.
 */
Array<Tensor> Scan(IterVar scan_axis,
                   Array<Tensor> init,
                   Array<Tensor> update,
                   Array<Tensor> state_placeholder,
                   std::string name = "scan");

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
