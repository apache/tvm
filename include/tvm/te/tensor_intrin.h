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
 * \file tvm/te/tensor_intrin.h
 * \brief Tensor intrinsic operations.
 */
#ifndef TVM_TE_TENSOR_INTRIN_H_
#define TVM_TE_TENSOR_INTRIN_H_

#include <tvm/te/tensor.h>
#include <tvm/tir/buffer.h>

#include <string>

namespace tvm {
namespace te {

/*! \brief Node to represent a Tensor intrinsic operator */
class TensorIntrinNode : public Object {
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
  /*! \brief List of scalar variables, used in body. These placeholders
   *  will be bound to expressions passed in when the TensorIntrin is called
   * from a TensorComputeOp.
   */
  Array<Var> scalar_params;
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

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("op", &op);
    v->Visit("inputs", &inputs);
    v->Visit("buffers", &buffers);
    v->Visit("scalar_params", &scalar_params);
    v->Visit("body", &body);
    v->Visit("reduce_init", &reduce_init);
    v->Visit("reduce_update", &reduce_update);
  }

  static constexpr const char* _type_key = "TensorIntrin";
  TVM_DECLARE_FINAL_OBJECT_INFO(TensorIntrinNode, Object);
};

/*!
 * \brief Managed reference to TensorIntrinNode
 * \sa TensorIntrinNode
 */
class TensorIntrin : public ObjectRef {
 public:
  TVM_DLL TensorIntrin(std::string name, Operation op, Array<Tensor> inputs, Array<Buffer> buffers,
                       Array<Var> scalar_params, Stmt body, Stmt reduce_init, Stmt reduce_update);

  TVM_DEFINE_OBJECT_REF_METHODS(TensorIntrin, ObjectRef, TensorIntrinNode);
};

class TensorIntrinCallNode : public Object {
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

  /*! \brief scalar expression inputs */
  Array<PrimExpr> scalar_inputs;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("intrin", &intrin);
    v->Visit("tensors", &tensors);
    v->Visit("regions", &regions);
    v->Visit("reduce_axis", &reduce_axis);
    v->Visit("scalar_inputs", &scalar_inputs);
  }

  static constexpr const char* _type_key = "TensorIntrinCall";
  TVM_DECLARE_FINAL_OBJECT_INFO(TensorIntrinCallNode, Object);
};

/*!
 * \brief Managed reference to TensorIntrinCallNode
 * \sa TensorIntrinCallNode
 */
class TensorIntrinCall : public ObjectRef {
 public:
  TVM_DLL TensorIntrinCall(TensorIntrin intrin, Array<Tensor> tensors, Array<Region> regions,
                           Array<IterVar> reduce_axis, Array<PrimExpr> scalar_inputs);

  TVM_DEFINE_OBJECT_REF_METHODS(TensorIntrinCall, ObjectRef, TensorIntrinCallNode);
};

}  // namespace te
}  // namespace tvm
#endif  // TVM_TE_TENSOR_INTRIN_H_
