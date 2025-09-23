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
 * \file tvm/te/operation.h
 * \brief Operation node can generate one or multiple Tensors
 */
#ifndef TVM_TE_OPERATION_H_
#define TVM_TE_OPERATION_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
/*! \brief Tensor expression language DSL. */
namespace te {

/*!
 * \brief Temporary data structure to store union
 *  of bounds of each axis of Tensor.
 */
struct TensorDom {
  // constructor
  explicit TensorDom(int ndim) : data(ndim) {}
  /*! \brief The domain data */
  std::vector<std::vector<IntSet>> data;
};

/*!
 * \brief Base class of all operation nodes
 */
class TVM_DLL OperationNode : public Object {
 public:
  /*! \brief optional name of the operation */
  std::string name;
  /*! \brief optional tag of the operation */
  std::string tag;
  /*! \brief additional attributes of the operation*/
  ffi::Map<ffi::String, ffi::Any> attrs;
  // virtual destructor.
  virtual ~OperationNode() {}
  /*! \return number of outputs */
  virtual int num_outputs() const = 0;
  /*!
   * \brief Get data type. i-th output tensor.
   * \param i The output index.
   * \return type of i-th output.
   */
  virtual DataType output_dtype(size_t i) const = 0;
  /*!
   * \brief Get shape of i-th output tensor.
   * \param i The output index.
   * \return shape of i-th output.
   */
  virtual ffi::Array<PrimExpr> output_shape(size_t i) const = 0;
  /*!
   * \brief List all the input Tensors.
   * \return List of input tensors.
   */
  virtual ffi::Array<Tensor> InputTensors() const = 0;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<OperationNode>()
        .def_ro("name", &OperationNode::name)
        .def_ro("tag", &OperationNode::tag)
        .def_ro("attrs", &OperationNode::attrs);
  }
  TVM_FFI_DECLARE_OBJECT_INFO("te.Operation", OperationNode, Object);
};

/*!
 * \brief A placeholder op represents an input placeholder.
 */
class PlaceholderOpNode : public OperationNode {
 public:
  /*! \brief The shape of the input */
  ffi::Array<PrimExpr> shape;
  /*! \brief The data type of the input. */
  DataType dtype;
  // override behavior.
  int num_outputs() const final;
  DataType output_dtype(size_t i) const final;
  ffi::Array<PrimExpr> output_shape(size_t i) const final;
  ffi::Array<Tensor> InputTensors() const final;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PlaceholderOpNode>()
        .def_ro("shape", &PlaceholderOpNode::shape)
        .def_ro("dtype", &PlaceholderOpNode::dtype);
  }
  TVM_FFI_DECLARE_OBJECT_INFO("te.PlaceholderOp", PlaceholderOpNode, OperationNode);
};

/*!
 * \brief Managed reference to PlaceholderOpNode
 * \sa PlaceholderOpNode
 */
class PlaceholderOp : public Operation {
 public:
  TVM_DLL PlaceholderOp(std::string name, ffi::Array<PrimExpr> shape, DataType dtype);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PlaceholderOp, Operation, PlaceholderOpNode);
};

/*!
 * \brief A Compute op that compute a tensor on certain domain.
 * This is the base class for ComputeOp (operating on a scalar at a time)
 */
class TVM_DLL BaseComputeOpNode : public OperationNode {
 public:
  /*! \brief IterVar on each axis */
  ffi::Array<IterVar> axis;
  /*! \brief IterVar on each reduction axis, if the body is a Reduce */
  ffi::Array<IterVar> reduce_axis;
  // override functions
  ffi::Array<PrimExpr> output_shape(size_t idx) const final;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BaseComputeOpNode>()
        .def_ro("axis", &BaseComputeOpNode::axis)
        .def_ro("reduce_axis", &BaseComputeOpNode::reduce_axis);
  }
  TVM_FFI_DECLARE_OBJECT_INFO("te.BaseComputeOp", BaseComputeOpNode, OperationNode);
};

/*!
 * \brief A Compute op that compute a tensor on certain domain.
 */
class TVM_DLL ComputeOpNode : public BaseComputeOpNode {
 public:
  /*! \brief the compute expression */
  ffi::Array<PrimExpr> body;
  /*! \brief constructor */
  ComputeOpNode() {}
  // override functions
  int num_outputs() const final;
  DataType output_dtype(size_t i) const final;
  ffi::Array<Tensor> InputTensors() const final;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ComputeOpNode>().def_ro("body", &ComputeOpNode::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("te.ComputeOp", ComputeOpNode, BaseComputeOpNode);
};

/*!
 * \brief Managed reference to ComputeOpNode
 * \sa ComputeOpNode
 */
class ComputeOp : public Operation {
 public:
  TVM_DLL ComputeOp(std::string name, std::string tag, ffi::Map<ffi::String, ffi::Any> attrs,
                    ffi::Array<IterVar> axis, ffi::Array<PrimExpr> body);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ComputeOp, Operation, ComputeOpNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ComputeOpNode);
};

/*!
 * \brief Symbolic scan.
 */
class ScanOpNode : public OperationNode {
 public:
  /*! \brief IterVar to scan over */
  IterVar scan_axis;
  /*! \brief the initialization tensors */
  ffi::Array<Tensor> init;
  /*! \brief the update function represented by tensor */
  ffi::Array<Tensor> update;
  /*! \brief The placeholder to refer as states in update. */
  ffi::Array<Tensor> state_placeholder;
  /*!
   * \brief the inputs to the scan, these are optionally provided
   *  But they can be helpful to provide hints to speedup get of scan body.
   */
  ffi::Array<Tensor> inputs;
  /*!
   * \brief Spatial axis to indicate spatial dimension of each output.
   *  They corresponds to flattened spatial axis of the outputs.
   *
   *  [output[0].axis[1], output[0].axis[2]... output[k].axis[j]...]
   *  These are auxiliary data structure for storing result of bound inference.
   *  They do not corresponds to splittable iterations, thus the name comes
   *  with underscore.
   */
  ffi::Array<IterVar> spatial_axis_;
  /*! \brief constructor */
  ScanOpNode() {}
  // override behavior.
  int num_outputs() const final;
  DataType output_dtype(size_t i) const final;
  ffi::Array<PrimExpr> output_shape(size_t i) const final;
  ffi::Array<Tensor> InputTensors() const final;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ScanOpNode>()
        .def_ro("scan_axis", &ScanOpNode::scan_axis)
        .def_ro("init", &ScanOpNode::init)
        .def_ro("update", &ScanOpNode::update)
        .def_ro("state_placeholder", &ScanOpNode::state_placeholder)
        .def_ro("inputs", &ScanOpNode::inputs)
        .def_ro("spatial_axis_", &ScanOpNode::spatial_axis_);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("te.ScanOp", ScanOpNode, OperationNode);
};

/*!
 * \brief Managed reference to ScanOpNode
 * \sa ScanOpNode
 */
class ScanOp : public Operation {
 public:
  TVM_DLL ScanOp(std::string name, std::string tag,
                 ffi::Optional<ffi::Map<ffi::String, ffi::Any>> attrs, IterVar axis,
                 ffi::Array<Tensor> init, ffi::Array<Tensor> update,
                 ffi::Array<Tensor> state_placeholder, ffi::Array<Tensor> input);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ScanOp, Operation, ScanOpNode);
};

/*!
 * \brief External computation that cannot be splitted.
 */
class ExternOpNode : public OperationNode {
 public:
  /*! \brief The input tensors */
  ffi::Array<Tensor> inputs;
  /*! \brief Symbolic placeholder representation of inputs */
  ffi::Array<Buffer> input_placeholders;
  /*! \brief Symbolic placeholder representation of outputs */
  ffi::Array<Buffer> output_placeholders;
  /*! \brief the statement that generates the computation. */
  Stmt body;

  /*! \brief constructor */
  ExternOpNode() {}
  // override functions
  int num_outputs() const final;
  DataType output_dtype(size_t i) const final;
  ffi::Array<PrimExpr> output_shape(size_t i) const final;
  ffi::Array<Tensor> InputTensors() const final;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ExternOpNode>()
        .def_ro("inputs", &ExternOpNode::inputs)
        .def_ro("input_placeholders", &ExternOpNode::input_placeholders)
        .def_ro("output_placeholders", &ExternOpNode::output_placeholders)
        .def_ro("body", &ExternOpNode::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("te.ExternOp", ExternOpNode, OperationNode);
};

/*!
 * \brief Managed reference to ExternOpNode
 * \sa ExternOpNode
 */
class ExternOp : public Operation {
 public:
  TVM_DLL ExternOp(std::string name, std::string tag, ffi::Map<ffi::String, ffi::Any> attrs,
                   ffi::Array<Tensor> inputs, ffi::Array<Buffer> input_placeholders,
                   ffi::Array<Buffer> output_placeholders, Stmt body);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ExternOp, Operation, ExternOpNode);
};

/*!
 * \brief Construct a new Var expression
 * \param name_hint The name hint for the expression
 * \param t The type of the expression
 */
TVM_DLL Var var(std::string name_hint, DataType t = DataType::Int(32));

/*!
 * \brief Create a new IterVar that represents an axis in thread.
 *
 * \param dom Optional, domain of the thread axis.
 * \param tag The thread tag of the axis.
 */
TVM_DLL IterVar thread_axis(Range dom, std::string tag);

/*!
 * \brief Create a new IterVar for reduction operations.
 *
 * \param dom The domain of the reduction axis.
 * \param name The name of the reduction axis.
 */
TVM_DLL IterVar reduce_axis(Range dom, std::string name = "rv");

/*! \brief The compute function to specify the input source of a Tensor */
using FCompute = std::function<PrimExpr(const ffi::Array<Var>& i)>;

/*! \brief The compute function to specify the inputs source of Tensors */
using FBatchCompute = std::function<ffi::Array<PrimExpr>(const ffi::Array<Var>& i)>;

/*!
 * \brief create a place holder tensor.
 * \param shape The shape of the tensor.
 * \param dtype the data type of the tensor.
 * \param name The name of the Tensor.
 */
TVM_DLL Tensor placeholder(ffi::Array<PrimExpr> shape, DataType dtype = DataType::Float(32),
                           std::string name = "placeholder");

/*!
 * \brief Construct a new tensor by computing over shape,
 *  using the computation rule: result_tensor[axis] = fcompute(axis)
 * \param shape Shape of the tensor.
 * \param fcompute The compute function to create the tensor.
 * \param name The optional name of the tensor.
 * \param tag The optional tag of the tensor.
 * \param attrs Optional additional attributes of the compute.
 */
TVM_DLL Tensor compute(ffi::Array<PrimExpr> shape, FCompute fcompute, std::string name = "tensor",
                       std::string tag = "", ffi::Map<ffi::String, ffi::Any> attrs = {});

/*!
 * \brief Construct a new tensor by computing over shape,
 *  using the computation rule: result_tensor[axis] = fcompute(axis)
 * \param shape Shape of the tensor.
 * \param fcompute The compute function to create the tensors.
 * \param name The optional name of the tensor.
 * \param tag The optional tag of the tensor.
 * \param attrs Optional additional attributes of the compute.
 */
TVM_DLL ffi::Array<Tensor> compute(ffi::Array<PrimExpr> shape, FBatchCompute fcompute,
                                   std::string name = "tensor", std::string tag = "",
                                   ffi::Map<ffi::String, ffi::Any> attrs = {});

/*!
 * \brief Construct new tensors by scan.
 *
 * \param init The intialize tensor of first K steps.
 * \param update The update tensor indicated the updated result after each timestamp.
 * \param state_placeholder The placeholder for the states.
 * \param inputs The inputs to the scan body, this is optional,
 *    but recommended to provide concrete information about scan body.
 * \param name The optional name of the tensor.
 * \param tag The optional tag of the tensor.
 * \param attrs Optional additional attributes of the compute.
 */
TVM_DLL ffi::Array<Tensor> scan(ffi::Array<Tensor> init, ffi::Array<Tensor> update,
                                ffi::Array<Tensor> state_placeholder,
                                ffi::Array<Tensor> inputs = ffi::Array<Tensor>(),
                                std::string name = "scan", std::string tag = "",
                                ffi::Map<ffi::String, ffi::Any> attrs = {});

// same as compute, specialized for different fcompute function
inline Tensor compute(ffi::Array<PrimExpr> shape, std::function<PrimExpr(Var)> f,
                      std::string name = "tensor", std::string tag = "",
                      ffi::Map<ffi::String, ffi::Any> attrs = {}) {
  FCompute fc = [f](const ffi::Array<Var>& i) { return f(i[0]); };
  return compute(shape, fc, name, tag, attrs);
}
inline Tensor compute(ffi::Array<PrimExpr> shape, std::function<PrimExpr(Var, Var)> f,
                      std::string name = "tensor", std::string tag = "",
                      ffi::Map<ffi::String, ffi::Any> attrs = {}) {
  FCompute fc = [f](const ffi::Array<Var>& i) { return f(i[0], i[1]); };
  return compute(shape, fc, name, tag, attrs);
}
inline Tensor compute(ffi::Array<PrimExpr> shape, std::function<PrimExpr(Var, Var, Var)> f,
                      std::string name = "tensor", std::string tag = "",
                      ffi::Map<ffi::String, ffi::Any> attrs = {}) {
  FCompute fc = [f](const ffi::Array<Var>& i) { return f(i[0], i[1], i[2]); };
  return compute(shape, fc, name, tag, attrs);
}
inline Tensor compute(ffi::Array<PrimExpr> shape, std::function<PrimExpr(Var, Var, Var, Var)> f,
                      std::string name = "tensor", std::string tag = "",
                      ffi::Map<ffi::String, ffi::Any> attrs = {}) {
  FCompute fc = [f](const ffi::Array<Var>& i) { return f(i[0], i[1], i[2], i[3]); };
  return compute(shape, fc, name, tag, attrs);
}

// inline function.
inline const OperationNode* Operation::operator->() const {
  return static_cast<const OperationNode*>(get());
}
}  // namespace te
}  // namespace tvm
#endif  // TVM_TE_OPERATION_H_
