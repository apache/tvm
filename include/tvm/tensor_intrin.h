/*!
 *  Copyright (c) 2017 by Contributors
 * \file tvm/tensor_intrin.h
 * \brief Tensor intrinsic operations.
 */
#ifndef TVM_TENSOR_INTRIN_H_
#define TVM_TENSOR_INTRIN_H_

#include <string>
#include "tensor.h"
#include "buffer.h"

namespace tvm {

// Internal node container of tensor intrinsics.
class TensorIntrinNode;

/*! \brief Tensor intrinsic node. */
class TensorIntrin : public NodeRef {
 public:
  TensorIntrin() {}
  explicit TensorIntrin(NodePtr<Node> n) : NodeRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const TensorIntrinNode* operator->() const;

  /*! \brief specify container node */
  using ContainerType = TensorIntrinNode;
};

/*! \brief Node to represent a Tensor intrinsic operator */
class TensorIntrinNode : public Node {
 public:
  /*! \brief The name of the intrinsic */
  std::string name;
  /*! \brief The operation this intrinsics is carrying out */
  Operation op;
  /*! \brief List of inputs of operator, placeholder in postdfs order */
  Array<Tensor> inputs;
  /*!
   * \brief Symbolic buffers of each output/input tensor
   *  buffers[0:len(inputs)] are buffers of the inputs.
   *  buffers[len(inputs):] are buffers of each output.
   *
   * \note When a field in Buffer is Var, it means we can be flexible
   *  wrt that field and Var can occur in body.
   *  When it is a constant, it means we can only take data in that shape.
   */
  Array<Buffer> buffers;
  /*! \brief The normal statement to execute the intrinsic */
  Stmt body;
  /*!
   * \brief Special statement for reduction op, can be None
   *  reset the value of output buffer to identity value.
   */
  Stmt reduce_init;
  /*!
   * \brief Special statement for reduction op, can be None
   *  Reduce: do a reduction of current output buffer with the result.
   */
  Stmt reduce_update;
  /*! \brief constructor */
  TensorIntrinNode() {}

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("op", &op);
    v->Visit("inputs", &inputs);
    v->Visit("buffers", &buffers);
    v->Visit("body", &body);
    v->Visit("reduce_init", &reduce_init);
    v->Visit("reduce_update", &reduce_update);
  }

  static TensorIntrin make(std::string name,
                           Operation op,
                           Array<Tensor> inputs,
                           Array<Buffer> buffers,
                           Stmt body,
                           Stmt reduce_init,
                           Stmt reduce_update);

  static constexpr const char* _type_key = "TensorIntrin";
  TVM_DECLARE_NODE_TYPE_INFO(TensorIntrinNode, Node);
};

inline const TensorIntrinNode* TensorIntrin::operator->() const {
  return static_cast<const TensorIntrinNode*>(node_.get());
}


// Internal node container of tensor intrinsic calling.
class TensorIntrinCallNode;

/*! \brief Tensor intrinsic calling node. */
class TensorIntrinCall : public NodeRef {
 public:
  TensorIntrinCall() {}
  explicit TensorIntrinCall(NodePtr<Node> n) : NodeRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const TensorIntrinCallNode* operator->() const;

  /*! \brief specify container node */
  using ContainerType = TensorIntrinCallNode;
};

class TensorIntrinCallNode : public Node {
 public:
  /*! \brief the tensor intrinsic */
  TensorIntrin intrin;
  /*! \brief input tensors of the intrinsic */
  Array<Tensor> tensors;
  /*! \brief regions of input tensors */
  Array<Region> regions;
  /*!
   * \brief IterVar on each reduction axis, if the
   * intrin will use the reduce axis
   */
  Array<IterVar> reduce_axis;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("intrin", &intrin);
    v->Visit("tensors", &tensors);
    v->Visit("regions", &regions);
    v->Visit("reduce_axis", &reduce_axis);
  }
  static TensorIntrinCall make(TensorIntrin intrin,
                               Array<Tensor> tensors,
                               Array<Region> regions,
                               Array<IterVar> reduce_axis);

  static constexpr const char* _type_key = "TensorIntrinCall";
  TVM_DECLARE_NODE_TYPE_INFO(TensorIntrinCallNode, Node);
};

inline const TensorIntrinCallNode* TensorIntrinCall::operator->() const {
  return static_cast<const TensorIntrinCallNode*>(node_.get());
}

}  // namespace tvm
#endif  // TVM_TENSOR_INTRIN_H_
