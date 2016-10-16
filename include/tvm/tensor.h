/*!
 *  Copyright (c) 2016 by Contributors
 * \file tensor.h
 * \brief Dataflow tensor object
 */
#ifndef TVM_TENSOR_H_
#define TVM_TENSOR_H_

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
  /*! \brief The index on each dimension */
  Array<Var> dim_index;
  /*! \brief The shape of the tensor */
  Array<Expr> shape;
  /*! \brief source expression */
  Expr source;
  /*! \brief constructor */
  TensorNode() {
  }
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

class Tensor : public NodeRef {
 public:
  Tensor(Array<Expr> shape);
  Tensor(Array<Expr> shape, std::function<Expr (Var, Var, Var)> f3) {
  }
  inline size_t ndim() const;

  template<typename... Args>
  inline Expr operator()(Args&& ...args) const {
    Array<Expr> indices{std::forward<Args>(args)...};
    CHECK_EQ(ndim(), indices.size())
        << "Tensor dimension mismatch in read";
    return Expr{};
  }
};


}  // namespace tvm
#endif  // TVM_TENSOR_H_
