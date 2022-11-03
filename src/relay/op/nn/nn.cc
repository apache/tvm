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
 * \file nn.cc
 * \brief Property def of nn operators.
 */

#include "nn.h"

#include <tvm/auto_scheduler/compute_dag.h>
#include <tvm/relay/attrs/image.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>
#include <tvm/topi/nn.h>
#include <tvm/topi/nn/bias_add.h>
#include <tvm/topi/nn/flatten.h>
#include <tvm/topi/nn/softmax.h>

#include <algorithm>
#include <string>
#include <vector>

#include "../../transforms/infer_layout_utils.h"
#include "../make_op.h"
#include "../op_common.h"
#include "../type_relations.h"

namespace tvm {
namespace relay {

// relay.nn.bias_add
TVM_REGISTER_NODE_TYPE(BiasAddAttrs);

bool BiasAddRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const BiasAddAttrs* param = attrs.as<BiasAddAttrs>();
  ICHECK(param != nullptr);
  int axis = param->axis;
  if (axis < 0) {
    axis = data->shape.size() + axis;
  }
  if (axis >= static_cast<int>(data->shape.size()) || axis < 0) {
    reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                     << "The axis in bias_add must be in range for the shape; "
                                     << "attempted to access index " << param->axis << " of "
                                     << PrettyPrint(data->shape));
    return false;
  }

  // assign output type
  reporter->Assign(types[1], TensorType({data->shape[axis]}, data->dtype));
  reporter->Assign(types[2], types[0]);
  return true;
}

// Positional relay function to create dense operator used by frontend FFI.
Expr MakeBiasAdd(Expr data, Expr bias, int axis) {
  auto attrs = make_object<BiasAddAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("nn.bias_add");
  return Call(op, {data, bias}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.bias_add").set_body_typed(MakeBiasAdd);

RELAY_REGISTER_OP("nn.bias_add")
    .describe(R"code(Add bias to an axis of the input.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<BiasAddAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "nD Tensor", "Input data.")
    .add_argument("bias", "1D Tensor", "Bias.")
    .set_support_level(1)
    .add_type_rel("BiasAdd", BiasAddRel)
    .set_attr<TOpPattern>("TOpPattern", kBroadcast)
    .set_attr<FTVMCompute>("FTVMCompute", [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                                             const Type& out_type) {
      const auto* param = attrs.as<BiasAddAttrs>();
      return tvm::Array<tvm::te::Tensor>{topi::nn::bias_add(inputs[0], inputs[1], param->axis)};
    });

// relay.nn.fifo_buffer
TVM_REGISTER_NODE_TYPE(FIFOBufferAttrs);

Expr MakeFIFOBuffer(Expr input, Expr buffer, int axis) {
  auto attrs = make_object<FIFOBufferAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("nn.fifo_buffer");
  return Call(op, {input, buffer}, Attrs(attrs), {});
}

bool FIFOBufferRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* input = types[0].as<TensorTypeNode>();
  const auto* buffer = types[1].as<TensorTypeNode>();
  const FIFOBufferAttrs* param = attrs.as<FIFOBufferAttrs>();
  if (input == nullptr || buffer == nullptr) {
    return false;
  }
  ICHECK(param != nullptr);
  ICHECK_EQ(input->shape.size(), buffer->shape.size());

  const size_t buffer_axis = static_cast<size_t>(
      param->axis < 0 ? static_cast<int>(buffer->shape.size()) + param->axis : param->axis);

  reporter->Assert(buffer_axis < buffer->shape.size());
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    if (i != buffer_axis) {
      reporter->AssertEQ(input->shape[i], buffer->shape[i]);
    }
  }
  reporter->Assert(input->shape[buffer_axis] < buffer->shape[buffer_axis]);

  Array<tvm::PrimExpr> oshape = buffer->shape;

  reporter->Assign(types[2], TensorType(oshape, buffer->dtype));
  return true;
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.fifo_buffer").set_body_typed(MakeFIFOBuffer);

RELAY_REGISTER_OP("nn.fifo_buffer")
    .describe(R"code(FIFO buffer
Compute equivalent of

```
concat(buffer, data, axis=axis) \
.slice_axis(axis=axis, begin=data.shape[axis], end=data.shape[axis]+buffer.shape[axis])
```

Useful for
* Encoding explicit re-use of computation in convolution ops operated on a sliding window input
* Implementing a FIFO queue to cache intermediate results, e.g. as in Fast WaveNet.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<FIFOBufferAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "Latest input")
    .add_argument("buffer", "Tensor", "Buffer storing latest [length_buffer] inputs")
    .set_support_level(3)
    .add_type_rel("FIFOBuffer", FIFOBufferRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

// ------------------- relay.nn.matmul
TVM_REGISTER_NODE_TYPE(MatmulAttrs);

Expr MakeMatmul(Expr tensor_a, Expr tensor_b, IndexExpr units, DataType out_dtype, bool transpose_a,
                bool transpose_b) {
  auto attrs = make_object<MatmulAttrs>();
  attrs->units = units;
  attrs->out_dtype = out_dtype;
  attrs->transpose_a = transpose_a;
  attrs->transpose_b = transpose_b;
  static const Op& matmul_op = Op::Get("nn.matmul");
  return Call(matmul_op, {tensor_a, tensor_b}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.matmul").set_body_typed(MakeMatmul);

RELAY_REGISTER_OP("nn.matmul")
    .describe(R"code(Applies a linear transformation: :math:`C = A * B`. A & B can be transposed.

- **tensor_a**: `(x1, x2, ..., xn, input_dim)` or `(x1, x2, ..., input_dim, xn)`
- **tensor_b**: `(input_dim, units)` or `(units, input_dim)`
- **out**: `(x1, x2, ..., xn, units)`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<MatmulAttrs>()
    .set_num_inputs(2)
    .add_argument("tensor_a", "nD Tensor", "The first input Tensor.")
    .add_argument("tensor_b", "2D Tensor", "The second input Tensor.")
    .set_support_level(1)
    .add_type_rel("Matmul", MatmulRel<MatmulAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

// ------------------- relay.nn.matmul

// ------------------- relay.nn.dense
TVM_REGISTER_NODE_TYPE(DenseAttrs);

// Positional relay function to create dense operator used by frontend FFI.
Expr MakeDense(Expr data, Expr weight, IndexExpr units, DataType out_dtype) {
  auto attrs = make_object<DenseAttrs>();
  attrs->units = units;
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("nn.dense");
  return Call(op, {data, weight}, Attrs(attrs), {});
}

InferCorrectLayoutOutput DenseInferCorrectLayout(const Attrs& attrs,
                                                 const Array<Layout>& new_in_layouts,
                                                 const Array<Layout>& old_in_layouts,
                                                 const Array<tvm::relay::Type>& old_in_types) {
  return InferCorrectLayoutOutput({"NC", "NC"}, {"NC"}, attrs);
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.dense").set_body_typed(MakeDense);

RELAY_REGISTER_OP("nn.dense")
    .describe(R"code(Applies a linear transformation: :math:`Y = XW^T`.

- **data**: `(x1, x2, ..., xn, input_dim)`
- **weight**: `(units, input_dim)`
- **out**: `(x1, x2, ..., xn, units)`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<DenseAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "nD Tensor", "Input data.")
    .add_argument("weight", "2D Tensor", "Weight matrix.")
    .set_support_level(1)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", DenseInferCorrectLayout)
    .add_type_rel("Dense", MatmulRel<DenseAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);
// ------------------- relay.nn.dense

// ------------------- relay.nn.contrib_dense_pack
TVM_REGISTER_NODE_TYPE(DensePackAttrs);

// Positional relay function to create dense_pack operator used by frontend FFI.
Expr MakeDensePack(Expr data, Expr weight, tvm::String weight_layout, IndexExpr units,
                   DataType out_dtype) {
  auto attrs = make_object<DensePackAttrs>();
  attrs->units = units;
  attrs->out_dtype = out_dtype;
  attrs->weight_layout = std::move(weight_layout);
  static const Op& op = Op::Get("nn.contrib_dense_pack");
  return Call(op, {data, weight}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.contrib_dense_pack").set_body_typed(MakeDensePack);

bool DensePackRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr || weight == nullptr) return false;

  const DensePackAttrs* param = attrs.as<DensePackAttrs>();
  ICHECK(param != nullptr);

  ICHECK_EQ(data->shape.size(), 2) << "Only 2D data is supported";
  ICHECK(weight->shape.size() == 3 || weight->shape.size() == 4) << "Expect weight to be 3D or 4D";

  Array<tvm::PrimExpr> oshape = data->shape;
  oshape.Set(1, weight->shape[0] * weight->shape[2]);

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }
  // assign output type
  reporter->Assign(types[2], TensorType(oshape, out_dtype));
  return true;
}

InferCorrectLayoutOutput DensePackInferCorrectLayout(const Attrs& attrs,
                                                     const Array<Layout>& new_in_layouts,
                                                     const Array<Layout>& old_in_layouts,
                                                     const Array<tvm::relay::Type>& old_in_types) {
  auto params = attrs.as<DensePackAttrs>();
  ICHECK(params);
  return InferCorrectLayoutOutput({"NC", params->weight_layout}, {"NC"}, attrs);
}

RELAY_REGISTER_OP("nn.contrib_dense_pack")
    .describe(R"code(Applies a linear transformation: :math:`Y = XW^T`.

- **data**: `(batch, input_dim)`
- **weight**: `(units // pack_weight_tile, input_dim, pack_weight_tile)`
- **out**: `(batch, units)`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<DenseAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "2D Tensor", "Input data.")
    .add_argument("weight", "3D Tensor", "Packed weight matrix.")
    .set_support_level(10)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", DensePackInferCorrectLayout)
    .add_type_rel("DensePack", DensePackRel)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

// ------------------- relay.nn.contrib_dense_pack

// relay.leaky_relu
TVM_REGISTER_NODE_TYPE(LeakyReluAttrs);

// Positional relay function to create leaky relu operator used by frontend FFI.
Expr MakeLeakyRelu(Expr data, double alpha) {
  auto attrs = make_object<LeakyReluAttrs>();
  attrs->alpha = alpha;
  static const Op& op = Op::Get("nn.leaky_relu");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.leaky_relu").set_body_typed(MakeLeakyRelu);

RELAY_REGISTER_OP("nn.leaky_relu")
    .describe(R"code(Leaky version of a Rectified Linear Unit.

`y = x > 0 ? x : alpha * x`

)code" TVM_ADD_FILELINE)
    .set_attrs_type<LeakyReluAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "Input data.")
    .set_support_level(3)
    .add_type_rel("Identity", IdentityRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
    .set_attr<TOpPattern>("TOpPattern", kElemWise)
    .set_attr<FTVMCompute>("FTVMCompute", [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                                             const Type& out_type) {
      const auto* param = attrs.as<LeakyReluAttrs>();
      return Array<te::Tensor>{topi::leaky_relu(inputs[0], param->alpha)};
    });

// relay.prelu
TVM_REGISTER_NODE_TYPE(PReluAttrs);

bool PReluRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
              const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const PReluAttrs* param = attrs.as<PReluAttrs>();
  ICHECK(param != nullptr);

  ICHECK(param->axis < static_cast<int>(data->shape.size()))
      << "Wrong axis (" << param->axis << ")value.";

  // assign alpha type
  Array<IndexExpr> alpha_shape({data->shape[param->axis]});
  reporter->Assign(types[1], TensorType(alpha_shape, data->dtype));

  // assign output type
  reporter->Assign(types[2], TensorType(data->shape, data->dtype));
  return true;
}

InferCorrectLayoutOutput PReluInferCorrectLayout(const Attrs& attrs,
                                                 const Array<Layout>& new_in_layouts,
                                                 const Array<Layout>& old_in_layouts,
                                                 const Array<tvm::relay::Type>& old_in_types) {
  ICHECK_EQ(old_in_layouts.size(), 2U);
  ICHECK_EQ(old_in_types.size(), 2U);
  Layout data_layout = old_in_layouts[0];
  if (new_in_layouts.defined()) {
    ICHECK_EQ(new_in_layouts.size(), 2U);
  }
  return InferCorrectLayoutOutput({data_layout, Layout("C")}, {data_layout}, attrs);
}

// Positional relay function to create prelu operator used by frontend FFI.
Expr MakePRelu(Expr data, Expr alpha, int axis) {
  auto attrs = make_object<PReluAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("nn.prelu");
  return Call(op, {data, alpha}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.prelu").set_body_typed(MakePRelu);

RELAY_REGISTER_OP("nn.prelu")
    .describe(R"code(Parametric version of a Rectified Linear Unit.
It accepts two arguments: an input ``x`` and a channelwise slope ``alpha``
and computes the output as :math:`PReLU(x) y = x > 0 ? x : alpha * x`,
where :math:`*` is an channelwise multiplication for each sample in the batch.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<PReluAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "Input data.")
    .add_argument("alpha", "Tensor", "Input channelwise alpha.")
    .set_support_level(3)
    .add_type_rel("PRelu", PReluRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", PReluInferCorrectLayout)
    .set_attr<TOpPattern>("TOpPattern", kBroadcast)
    .set_attr<FTVMCompute>("FTVMCompute", [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                                             const Type& out_type) {
      const auto* param = attrs.as<PReluAttrs>();
      return Array<te::Tensor>{topi::prelu(inputs[0], inputs[1], param->axis)};
    });

// relay.softmax
TVM_REGISTER_NODE_TYPE(SoftmaxAttrs);

bool SoftmaxRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const SoftmaxAttrs* param = attrs.as<SoftmaxAttrs>();
  ICHECK(param != nullptr);
  int axis = param->axis;
  int ndim = static_cast<int>(data->shape.size());
  if (axis >= ndim || axis < -ndim) {
    reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                     << "Wrong axis (" << axis << ") not in expected range: ["
                                     << -ndim << ", " << ndim << ")");
    return false;
  }

  reporter->Assign(types[1], types[0]);
  return true;
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.softmax").set_body_typed([](Expr data, int axis) {
  auto attrs = make_object<SoftmaxAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("nn.softmax");
  return Call(op, {data}, Attrs(attrs), {});
});

RELAY_REGISTER_OP("nn.softmax")
    .describe(R"code(Softmax layer.

.. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}

.. note::
    This operator can be optimized away for inference.

- **data**: The input data
)code" TVM_ADD_FILELINE)
    .set_attrs_type<SoftmaxAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(1)
    .add_type_rel("Softmax", SoftmaxRel)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

// relay.fast_softmax
TVM_REGISTER_NODE_TYPE(SoftmaxAttrs);

TVM_REGISTER_GLOBAL("relay.op.nn._make.fast_softmax").set_body_typed([](Expr data, int axis) {
  auto attrs = make_object<SoftmaxAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("nn.fast_softmax");
  return Call(op, {data}, Attrs(attrs), {});
});

RELAY_REGISTER_OP("nn.fast_softmax")
    .describe(R"code(Softmax layer.
    Use approximation to compute exponent for faster speed.

.. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}

.. note::
    This operator can be optimized away for inference.

- **data**: The input data
)code" TVM_ADD_FILELINE)
    .set_attrs_type<SoftmaxAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(1)
    .add_type_rel("Softmax", SoftmaxRel)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

// relay.nn.log_softmax
TVM_REGISTER_GLOBAL("relay.op.nn._make.log_softmax").set_body_typed([](Expr data, int axis) {
  auto attrs = make_object<SoftmaxAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("nn.log_softmax");
  return Call(op, {data}, Attrs(attrs), {});
});

RELAY_REGISTER_OP("nn.log_softmax")
    .describe(R"code(Computes log softmax.

.. math:: \text{log_softmax}(x)_i = \log \frac{exp(x_i)}{\sum_j exp(x_j)}

.. note::
    This operator can be optimized away for inference.

- **data**: The input data
)code" TVM_ADD_FILELINE)
    .set_attrs_type<SoftmaxAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(1)
    .add_type_rel("Softmax", SoftmaxRel)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable)
    .set_attr<FTVMCompute>("FTVMCompute", [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                                             const Type& out_type) {
      const auto* param = attrs.as<SoftmaxAttrs>();
      ICHECK(param != nullptr);
      ICHECK(param->axis == -1 || param->axis == static_cast<int32_t>(inputs[0].ndim()) - 1)
          << "log_softmax currently only works on last dimension";
      return Array<te::Tensor>{topi::nn::log_softmax(inputs[0])};
    });

// relay.nn.batch_flatten
bool BatchFlattenRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  if (data->shape.size() == 0) return false;

  auto target_dim = tir::make_const(DataType::Int(32), 1);

  for (uint32_t i = 1; i < data->shape.size(); ++i) {
    if (!data->shape[i].as<tir::AnyNode>()) {
      target_dim = target_dim * data->shape[i];
    } else {
      target_dim = data->shape[i];
      break;
    }
  }

  std::vector<IndexExpr> oshape({data->shape[0], target_dim});

  // assign output type
  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

Expr MakeBatchFlatten(Expr data) {
  static const Op& op = Op::Get("nn.batch_flatten");
  return Call(op, {data}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.batch_flatten").set_body_typed(MakeBatchFlatten);

RELAY_REGISTER_OP("nn.batch_flatten")
    .describe(R"code(Flattens the input into a 2-D array.

For an input array with shape ``(d1, d2, ..., dk)``, `batch_flatten` operation reshapes
the input array into an output array of shape ``(d1, d2*...*dk)``.

Example::

    x = [[
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ],
    [   [1,2,3],
        [4,5,6],
        [7,8,9]
    ]],

    batch_flatten(x) = [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
       [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]]

)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(2)
    .add_type_rel("BatchFlatten", BatchFlattenRel)
    .set_attr<TOpPattern>("TOpPattern", kInjective)
    .set_attr<FTVMCompute>("FTVMCompute",
                           [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_type) {
                             return Array<te::Tensor>{topi::nn::flatten(inputs[0])};
                           })
    .set_attr<TReshapeOp>("TReshapeOp", true);

// relu
TVM_REGISTER_GLOBAL("relay.op.nn._make.relu").set_body_typed([](Expr data) {
  static const Op& op = Op::Get("nn.relu");
  return Call(op, {data}, Attrs(), {});
});

RELAY_REGISTER_OP("nn.relu")
    .describe(R"code(Returns the relu input array, computed element-wise.

.. math::
   max(x, 0)

)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(1)
    .add_type_rel("Identity", IdentityRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
    .set_attr<TOpPattern>("TOpPattern", kElemWise)
    .set_attr<FTVMCompute>("FTVMCompute", [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                                             const Type& out_type) {
      return Array<te::Tensor>{topi::relu(inputs[0], 0.0f)};
    });

// Positional relay function to create LRN operator used by frontend FFI.
TVM_REGISTER_NODE_TYPE(LRNAttrs);

Expr MakeLRN(Expr data, int size, int axis, double alpha, double beta, double bias) {
  auto attrs = make_object<LRNAttrs>();
  attrs->size = size;
  attrs->axis = axis;
  attrs->alpha = alpha;
  attrs->beta = beta;
  attrs->bias = bias;
  static const Op& op = Op::Get("nn.lrn");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.lrn").set_body_typed(MakeLRN);

RELAY_REGISTER_OP("nn.lrn")
    .describe(R"code(LRN layer.

Normalize the input in a local region across or within feature maps.
Each input value is divided by (1 + (\alpha/n) \sum_i x_i^2)^\beta,
where n is the size of each local region, and the sum is taken over the region
centered at that value (zero padding is added where necessary).

.. math::

    data / (bias + (alpha * sum_data ^2 /size))^beta

- **data**: The input tensor.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<LRNAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(2)
    .add_type_rel("Identity", IdentityRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

// Positional relay function to create L2Normalize operator used by frontend FFI.
TVM_REGISTER_NODE_TYPE(L2NormalizeAttrs);

Expr MakeL2Normalize(Expr data, double eps, Array<Integer> axis) {
  auto attrs = make_object<L2NormalizeAttrs>();
  attrs->eps = eps;
  attrs->axis = std::move(axis);
  static const Op& op = Op::Get("nn.l2_normalize");
  return Call(op, {data}, Attrs(attrs), {});
}

InferCorrectLayoutOutput L2NormalizeInferCorrectLayout(
    const Attrs& attrs, const Array<Layout>& new_in_layouts, const Array<Layout>& old_in_layouts,
    const Array<tvm::relay::Type>& old_in_types) {
  const auto* attrs_ptr = attrs.as<L2NormalizeAttrs>();
  ICHECK(attrs_ptr);
  ObjectPtr<L2NormalizeAttrs> param = make_object<L2NormalizeAttrs>(*attrs_ptr);

  Array<Array<IndexExpr>> old_in_shapes;
  for (auto old_in_t : old_in_types) {
    ICHECK(old_in_t.as<TensorTypeNode>());
    old_in_shapes.push_back(old_in_t.as<TensorTypeNode>()->shape);
  }
  std::vector<size_t> axis_list;
  for (auto i : param->axis) {
    int64_t axis = i->value;
    if (axis < 0) {
      axis = axis + static_cast<size_t>(old_in_shapes[0].size());
    }
    axis_list.emplace_back(axis);
  }

  Layout ret = Layout::Undef();
  if (new_in_layouts.defined() && old_in_layouts.defined()) {
    for (size_t i = 0; i < axis_list.size(); ++i) {
      const auto& axis_dim = old_in_layouts[0][axis_list[i]];
      auto axis_index = new_in_layouts[0].IndexOf(axis_dim);
      param->axis.Set(i, axis_index);
    }
    ret = new_in_layouts[0];
  } else if (old_in_layouts.defined()) {
    ret = old_in_layouts[0];
  }

  return InferCorrectLayoutOutput({ret}, {ret}, Attrs(param));
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.l2_normalize").set_body_typed(MakeL2Normalize);

RELAY_REGISTER_OP("nn.l2_normalize")
    .describe(R"code(L2 Normalization layer.

Normalizes along dimension axis using an L2 norm

.. math::
    output = x / sqrt(max(sum(x^2), epsilon))

- **data**: The input tensor.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<L2NormalizeAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(2)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", L2NormalizeInferCorrectLayout)
    .add_type_rel("Identity", IdentityRel);

// Dropout
TVM_REGISTER_NODE_TYPE(DropoutAttrs);

bool DropoutRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  // dropout returns the original tensor with dropout applied
  // and a mask tensor (1.0 where element not dropped, 0.0 where dropped)
  auto ret_type = TensorType(data->shape, data->dtype);
  reporter->Assign(types[1], TupleType(Array<Type>({ret_type, ret_type})));
  return true;
}

Expr MakeDropout(Expr data, double rate) {
  auto attrs = make_object<DropoutAttrs>();
  attrs->rate = rate;
  static const Op& op = Op::Get("nn.dropout");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.dropout").set_body_typed(MakeDropout);

RELAY_REGISTER_OP("nn.dropout")
    .describe(R"code(Applies the dropout operation to the input array.

During training, each element of the input is set to zero with probability ``p``.
The whole array is rescaled by ``1/(1-p)`` to keep the expected sum of the input unchanged.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<DropoutAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "Input to which dropout will be applied.")
    .set_support_level(1)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .add_type_rel("Dropout", DropoutRel)
    .set_attr<TOpIsStateful>("TOpIsStateful", true);

// batch_norm
TVM_REGISTER_NODE_TYPE(BatchNormAttrs);

InferCorrectLayoutOutput BatchNormInferCorrectLayout(const Attrs& attrs,
                                                     const Array<Layout>& new_in_layouts,
                                                     const Array<Layout>& old_in_layouts,
                                                     const Array<tvm::relay::Type>& old_in_types) {
  const auto* attrs_ptr = attrs.as<BatchNormAttrs>();
  ICHECK(attrs_ptr);
  ObjectPtr<BatchNormAttrs> param = make_object<BatchNormAttrs>(*attrs_ptr);

  Array<Array<IndexExpr>> old_in_shapes;
  for (auto old_in_t : old_in_types) {
    ICHECK(old_in_t.as<TensorTypeNode>());
    old_in_shapes.push_back(old_in_t.as<TensorTypeNode>()->shape);
  }

  size_t axis =
      param->axis < 0 ? param->axis + old_in_shapes[0].size() : static_cast<size_t>(param->axis);

  Layout ret = Layout::Undef();

  // If new_in_layouts are defined, this code tries to modify the layout.
  if (new_in_layouts.defined() && old_in_layouts.defined()) {
    // Get the new C axis. Extract the dim in old layout. Find the index of that dim in next layout.
    const auto& bn_dim = old_in_layouts[0][axis];
    auto new_index = new_in_layouts[0].IndexOf(bn_dim);
    param->axis = new_index;
    ret = new_in_layouts[0];
  } else if (old_in_layouts.defined()) {
    ret = old_in_layouts[0];
  }
  // BN has 5 inputs, 3 outputs. The last 4 inputs and last 2 outputs have "C" layout.
  Layout c_layout = Layout("C");
  return InferCorrectLayoutOutput({ret, c_layout, c_layout, c_layout, c_layout},
                                  {ret, c_layout, c_layout}, Attrs(param));
}

bool BatchNormRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 6);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const BatchNormAttrs* param = attrs.as<BatchNormAttrs>();

  // axis of -1 means use the last dimension
  ICHECK(param->axis >= -1 && param->axis < (int)data->shape.size());
  int axis = (param->axis != -1) ? param->axis : data->shape.size() - 1;
  auto axis_size = data->shape[axis];

  // if we are using beta and gamma, they need to be of shape (dim,)
  reporter->Assign(types[1], TensorType({axis_size}, data->dtype));
  reporter->Assign(types[2], TensorType({axis_size}, data->dtype));
  reporter->Assign(types[3], TensorType({axis_size}, data->dtype));
  reporter->Assign(types[4], TensorType({axis_size}, data->dtype));

  // output is a tuple of the normed data (same shape as input), new running mean,
  // new running variance, saved mean and saved variance (the latter are all
  // vectors of length dim)
  std::vector<Type> fields;
  auto vec_ty = TensorType(Array<IndexExpr>({data->shape[axis]}), data->dtype);
  fields.push_back(TensorType(data->shape, data->dtype));
  fields.push_back(vec_ty);
  fields.push_back(vec_ty);
  reporter->Assign(types[5], TupleType(Array<Type>(fields)));
  return true;
}

Expr MakeBatchNorm(Expr data, Expr gamma, Expr beta, Expr moving_mean, Expr moving_var, int axis,
                   double epsilon, bool center, bool scale) {
  auto attrs = make_object<BatchNormAttrs>();
  attrs->axis = axis;
  attrs->epsilon = epsilon;
  attrs->center = center;
  attrs->scale = scale;
  static const Op& op = Op::Get("nn.batch_norm");
  return Call(op, {data, gamma, beta, moving_mean, moving_var}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.batch_norm").set_body_typed(MakeBatchNorm);

RELAY_REGISTER_OP("nn.batch_norm")
    .describe(R"code(Batch normalization layer (Ioffe and Szegedy, 2014).
Normalizes the input at each batch, i.e. applies a transformation
that maintains the mean activation close to 0 and the activation
standard deviation close to 1.

.. math::

  data\_mean[i] = mean(data[:,i,:,...]) \\
  data\_var[i] = var(data[:,i,:,...])

Then compute the normalized output, which has the same shape as input, as following:

.. math::

  out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}} \
* gamma[i] + beta[i]

Both *mean* and *var* returns a scalar by treating the input as a vector.

Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta`` have shape *(k,)*.

Besides the inputs and the outputs, this operator accepts two auxiliary
states, ``moving_mean`` and ``moving_var``, which are *k*-length
vectors. They are global statistics for the whole dataset, which are updated
by::

  moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
  moving_var = moving_var * momentum + data_var * (1 - momentum)

The parameter ``axis`` specifies which axis of the input shape denotes
the 'channel' (separately normalized groups).  The default is 1.  Specifying -1 sets the channel
axis to be the last item in the input shape.

.. note::
    This operator can be optimized away for inference.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<BatchNormAttrs>()
    .set_num_inputs(5)
    .add_argument("data", "Tensor", "Input to which batch_norm will be applied.")
    .add_argument("gamma", "Tensor", "The gamma scale factor.")
    .add_argument("beta", "Tensor", "The beta offset factor.")
    .add_argument("moving_mean", "Tensor", "Running mean of input.")
    .add_argument("moving_var", "Tensor", "Running variance of input.")
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", BatchNormInferCorrectLayout)
    .set_support_level(1)
    .add_type_rel("BatchNorm", BatchNormRel)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

// instance_norm
TVM_REGISTER_NODE_TYPE(InstanceNormAttrs);

template <typename T>
InferCorrectLayoutOutput NormalizationInferCorrectLayout(
    const Attrs& attrs, const Array<Layout>& new_in_layouts, const Array<Layout>& old_in_layouts,
    const Array<tvm::relay::Type>& old_in_types) {
  const auto* attrs_ptr = attrs.as<T>();
  ICHECK(attrs_ptr);
  ObjectPtr<T> param = make_object<T>(*attrs_ptr);

  Array<Array<IndexExpr>> old_in_shapes;
  for (auto old_in_t : old_in_types) {
    ICHECK(old_in_t.as<TensorTypeNode>());
    old_in_shapes.push_back(old_in_t.as<TensorTypeNode>()->shape);
  }

  size_t axis =
      param->axis < 0 ? param->axis + old_in_shapes[0].size() : static_cast<size_t>(param->axis);

  Layout ret = Layout::Undef();

  // If new_in_layouts are defined, this code tries to modify the layout.
  if (new_in_layouts.defined() && old_in_layouts.defined()) {
    // Get the new C axis. Extract the dim in old layout. Find the index of that dim in next layout.
    const auto& ln_dim = old_in_layouts[0][axis];
    auto new_index = new_in_layouts[0].IndexOf(ln_dim);
    param->axis = new_index;
    ret = new_in_layouts[0];
  } else if (old_in_layouts.defined()) {
    ret = old_in_layouts[0];
  }

  // For normalization has 3 inputs, 1 outputs. The last 2 inputs have "C" layout.
  Layout c_layout = Layout("C");
  return InferCorrectLayoutOutput({ret, c_layout, c_layout}, {ret}, Attrs(param));
}

bool InstanceNormRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 4);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  ICHECK_GT(data->shape.size(), 2);
  const InstanceNormAttrs* param = attrs.as<InstanceNormAttrs>();
  int axis = param->axis >= 0 ? param->axis : param->axis + data->shape.size();
  ICHECK(axis >= 0 && axis < (int)data->shape.size());
  reporter->Assign(types[1], TensorType({data->shape[axis]}, data->dtype));
  reporter->Assign(types[2], TensorType({data->shape[axis]}, data->dtype));
  reporter->Assign(types[3], TensorType(data->shape, data->dtype));

  return true;
}

Expr MakeInstanceNorm(Expr data, Expr gamma, Expr beta, int axis, double epsilon, bool center,
                      bool scale) {
  auto attrs = make_object<InstanceNormAttrs>();
  attrs->axis = axis;
  attrs->epsilon = epsilon;
  attrs->center = center;
  attrs->scale = scale;
  static const Op& op = Op::Get("nn.instance_norm");
  return Call(op, {data, gamma, beta}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.instance_norm").set_body_typed(MakeInstanceNorm);

RELAY_REGISTER_OP("nn.instance_norm")
    .describe(R"code(Instance Normalization (Ulyanov and et al., 2016)
Applies instance normalization to the n-dimensional input array.

.. math::

    out = \frac{data - mean(data)}{\sqrt{var(data)+\epsilon}}
        * gamma + beta

The instance normalization is similar to batch normalization, but unlike
batch normalization, the mean and var are calculated per-dimension
separately for each object(instance) in a mini-batch, not over a batch.
And the same normalization is applied both at test and train time.

Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
have shape *(k,)*.

The parameter ``axis`` specifies which axis of the input shape denotes
the 'channel'.  The default is 1. Specifying -1 sets the channel axis
to be the last item in the input shape.

.. note::

    This operator can be optimized away for inference.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<InstanceNormAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "Input to which instance_norm will be applied.")
    .add_argument("gamma", "Tensor", "The gamma scale factor.")
    .add_argument("beta", "Tensor", "The beta offset factor.")
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                                   NormalizationInferCorrectLayout<InstanceNormAttrs>)
    .set_support_level(1)
    .add_type_rel("InstanceNorm", InstanceNormRel);

// layer_norm
TVM_REGISTER_NODE_TYPE(LayerNormAttrs);

bool LayerNormRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 4);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  const LayerNormAttrs* param = attrs.as<LayerNormAttrs>();
  int axis = param->axis >= 0 ? param->axis : param->axis + data->shape.size();
  ICHECK(axis >= 0 && axis < (int)data->shape.size());
  reporter->Assign(types[1], TensorType({data->shape[axis]}, data->dtype));
  reporter->Assign(types[2], TensorType({data->shape[axis]}, data->dtype));
  reporter->Assign(types[3], TensorType(data->shape, data->dtype));

  return true;
}

Expr MakeLayerNorm(Expr data, Expr gamma, Expr beta, int axis, double epsilon, bool center,
                   bool scale) {
  auto attrs = make_object<LayerNormAttrs>();
  attrs->axis = axis;
  attrs->epsilon = epsilon;
  attrs->center = center;
  attrs->scale = scale;
  static const Op& op = Op::Get("nn.layer_norm");
  return Call(op, {data, gamma, beta}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.layer_norm").set_body_typed(MakeLayerNorm);

RELAY_REGISTER_OP("nn.layer_norm")
    .describe(R"code(
)code" TVM_ADD_FILELINE)
    .set_attrs_type<LayerNormAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "Input to which layer_norm will be applied.")
    .add_argument("gamma", "Tensor", "The gamma scale factor.")
    .add_argument("beta", "Tensor", "The beta offset factor.")
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                                   NormalizationInferCorrectLayout<LayerNormAttrs>)
    .set_support_level(1)
    .add_type_rel("LayerNorm", LayerNormRel);

// group_norm
TVM_REGISTER_NODE_TYPE(GroupNormAttrs);

bool GroupNormRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 4);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  const GroupNormAttrs* param = attrs.as<GroupNormAttrs>();
  int axis = param->axis >= 0 ? param->axis : param->axis + data->shape.size();
  ICHECK(axis >= 0 && axis < (int)data->shape.size());
  reporter->Assign(types[1], TensorType({data->shape[axis]}, data->dtype));
  reporter->Assign(types[2], TensorType({data->shape[axis]}, data->dtype));
  reporter->Assign(types[3], TensorType(data->shape, data->dtype));

  return true;
}

Expr MakeGroupNorm(Expr data, Expr gamma, Expr beta, int num_groups, int axis, double epsilon,
                   bool center, bool scale) {
  auto attrs = make_object<GroupNormAttrs>();
  attrs->num_groups = num_groups;
  attrs->axis = axis;
  attrs->epsilon = epsilon;
  attrs->center = center;
  attrs->scale = scale;
  static const Op& op = Op::Get("nn.group_norm");
  return Call(op, {data, gamma, beta}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.group_norm").set_body_typed(MakeGroupNorm);

RELAY_REGISTER_OP("nn.group_norm")
    .describe(R"code(
Group normalization normalizes over group of channels for each training examples.
We can say that, Group Norm is in between Instance Norm and Layer Norm. When we put
all the channels into a single group, group normalization becomes Layer normalization.
And, when we put each channel into different groups it becomes Instance normalization

https://arxiv.org/pdf/1803.08494.pdf

Applies group normalization to the n-dimensional input array by seperating the input channels
into 'num_groups' groups, each containing 'num_channels / num_groups' channels.
The mean and standard-deviation are calculated separately over the each group. gamma and
beta are learnable per-channel affine transform parameter vectors of size num_channels.

.. math::

    out = \frac{data - mean(data, axis)}{\sqrt{var(data, axis)+\epsilon}}
        * gamma + beta

Unlike batch normalization, the mean and var are computed along a group of channels.

If the input has size k on axis 1, then both gamma and beta have shape (k,).

.. note::

    This operator can be optimized away for inference.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<GroupNormAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "Input to which group_norm will be applied.")
    .add_argument("gamma", "Tensor", "The gamma scale factor.")
    .add_argument("beta", "Tensor", "The beta offset factor.")
    .set_support_level(1)
    .add_type_rel("GroupNorm", GroupNormRel);

// ------------------- relay.nn.batch_matmul
TVM_REGISTER_NODE_TYPE(BatchMatmulAttrs);

// Positional relay function to create batch_matmul operator used by frontend FFI.
Expr MakeBatchMatmul(Expr tensor_a, Expr tensor_b, DataType out_dtype, bool transpose_a,
                     bool transpose_b) {
  auto attrs = make_object<BatchMatmulAttrs>();
  attrs->out_dtype = out_dtype;
  attrs->transpose_a = transpose_a;
  attrs->transpose_b = transpose_b;
  static const Op& op = Op::Get("nn.batch_matmul");
  return Call(op, {tensor_a, tensor_b}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.batch_matmul").set_body_typed(MakeBatchMatmul);

RELAY_REGISTER_OP("nn.batch_matmul")
    .describe(R"code(Compute batch matrix multiplication of `tensor_a` and `tensor_b`.

Both `tensor_a` and `tensor_b` can be transposed. For legacy reason, we use NT format
(transpose_a=False, transpose_b=True) by default.

.. math::

  batch\_matmul(A, B)[i, :, :] = matmul(A[i, :, :], B[i, :, :]^T)

- **tensor_a**: `(b, m, k)` or `(b, k, m)`
- **tensor_b**: `(b, k, n)` or `(b, n, k)`
- **out**: `(b, m, n)`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<BatchMatmulAttrs>()
    .set_num_inputs(2)
    .add_argument("tensor_a", "3D Tensor", "The first input.")
    .add_argument("tensor_b", "3D Tensor", "The second input.")
    .set_support_level(10)
    .add_type_rel("BatchMatmul", BatchMatmulRel<BatchMatmulAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

// ------------------- relay.nn.batch_matmul

// relay.nn.cross_entropy
bool CrossEntropyRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* x = types[0].as<TensorTypeNode>();
  const auto* y = types[1].as<TensorTypeNode>();
  if (x == nullptr || y == nullptr) return false;
  ICHECK(x->shape.size() == 2 && y->shape.size() == 2)
      << "CrossEntropy: shapes of x and y is inconsistent, "
      << "x shape = " << x->shape << ", "
      << "y shape = " << y->shape;
  ICHECK(reporter->AssertEQ(x->shape[0], y->shape[0]))
      << "CrossEntropy: shapes of x and y is inconsistent, "
      << "x shape = " << x->shape << ", "
      << "y shape = " << y->shape;
  ICHECK(reporter->AssertEQ(x->shape[1], y->shape[1]))
      << "CrossEntropy: shapes of x and y is inconsistent, "
      << "x shape = " << x->shape << ", "
      << "y shape = " << y->shape;
  // assign output type
  reporter->Assign(types[2], TensorType({}, x->dtype));
  return true;
}

// Positional relay function to create cross_entropy operator used by frontend FFI.
Expr MakeCrossEntropy(Expr predictions, Expr targets) {
  static const Op& op = Op::Get("nn.cross_entropy");
  return Call(op, {predictions, targets}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.cross_entropy").set_body_typed(MakeCrossEntropy);

RELAY_REGISTER_OP("nn.cross_entropy")
    .describe(R"code(
Computes cross entropy given predictions and targets.
Do log on the data - do not accept logits.
)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("x", "1D Tensor", "Predictions.")
    .add_argument("y", "1D Tensor", "Targets.")
    .set_support_level(10)
    .add_type_rel("CrossEntropy", CrossEntropyRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

// relay.nn.dilate
TVM_REGISTER_NODE_TYPE(DilateAttrs);

bool DilateRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* x = types[0].as<TensorTypeNode>();
  const DilateAttrs* param = attrs.as<DilateAttrs>();
  if (x == nullptr) return false;
  ICHECK_EQ(x->shape.size(), param->strides.size());

  std::vector<IndexExpr> oshape;
  for (size_t i = 0; i < param->strides.size(); ++i) {
    if (!x->shape[i].as<tir::AnyNode>()) {
      oshape.push_back((x->shape[i] - 1) * param->strides[i] + 1);
    } else {
      oshape.push_back(x->shape[i]);
    }
  }

  reporter->Assign(types[1], TensorType(Array<IndexExpr>(oshape), x->dtype));
  return true;
}

// Positional relay function to create dilate operator used by frontend FFI.
Expr MakeDilate(Expr data, Array<IndexExpr> strides, double dilation_value = 0.0) {
  auto attrs = make_object<DilateAttrs>();
  attrs->strides = std::move(strides);
  attrs->dilation_value = std::move(dilation_value);
  static const Op& op = Op::Get("nn.dilate");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.dilate").set_body_typed(MakeDilate);

RELAY_REGISTER_OP("nn.dilate")
    .describe(R"code(
Dilate data with given dilation value (0 by default).
)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("x", "1D Tensor", "Data to dilate.")
    .set_support_level(10)
    .add_type_rel("Dilate", DilateRel)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// relay.nn.cross_entropy_with_logits
// Positional relay function to create cross_entropy_with_logits operator used by frontend FFI.
Expr MakeCrossEntropyWithLogits(Expr predictions, Expr targets) {
  static const Op& op = Op::Get("nn.cross_entropy_with_logits");
  return Call(op, {predictions, targets}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.cross_entropy_with_logits")
    .set_body_typed(MakeCrossEntropyWithLogits);

RELAY_REGISTER_OP("nn.cross_entropy_with_logits")
    .describe(R"code(
Computes cross entropy given predictions and targets.
Accept logits.
)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("x", "1D Tensor", "Predictions.")
    .add_argument("y", "1D Tensor", "Targets.")
    .set_support_level(10)
    .add_type_rel("CrossEntropy", CrossEntropyRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

// Depth to space and space to depth
TVM_REGISTER_NODE_TYPE(SubPixelAttrs);

// relay.nn.nll_loss
TVM_REGISTER_NODE_TYPE(NLLLossAttrs);

bool NLLLossRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 4) << "NLLLossRel expects 4 types, but " << types.size()
                             << " were provided.";
  const auto* predictions = types[0].as<TensorTypeNode>();
  const auto* targets = types[1].as<TensorTypeNode>();
  const auto* weights = types[2].as<TensorTypeNode>();
  const NLLLossAttrs* param = attrs.as<NLLLossAttrs>();
  if (predictions == nullptr || targets == nullptr || weights == nullptr) return false;
  if (!(predictions->shape.size() - targets->shape.size() == 1)) {
    reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                     << "NLLLossRel: predictions should be one"
                                     << " dimension larger than targets,"
                                     << "predictions shape = " << predictions->shape
                                     << ", targets shape = " << targets->shape);
    return false;
  }
  if (!(weights->shape.size() == 1)) {
    reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                     << "NLLLossRel: weights should be a one dimension"
                                     << " Tensor with its length the number of classes,"
                                     << " but Tensor of dimension " << weights->shape.size()
                                     << " were provided.");
    return false;
  }
  if (!reporter->AssertEQ(predictions->shape[1], weights->shape[0])) {
    reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                     << "NLLLossRel: the second dimension of predictions"
                                     << " should be the number of classes, "
                                     << "which is the length of weights, "
                                     << "predictions shape = " << predictions->shape
                                     << ", weights shape = " << weights->shape);
    return false;
  }
  if (!(predictions->dtype == weights->dtype &&
        (predictions->dtype.is_float() || predictions->dtype.is_bfloat16()))) {
    reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                     << "NLLLossRel: predictions and weights should"
                                     << " be of the same floating type.");
    return false;
  }
  if (!targets->dtype.is_int()) {
    reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                     << "NLLLossRel: targets should be of int type.");
    return false;
  }
  // assign output type
  if (param->reduction == "none") {
    reporter->Assign(types[3], TensorType(targets->shape, predictions->dtype));
  } else {
    reporter->Assign(types[3], TensorType({}, predictions->dtype));
  }
  return true;
}

// Handler to create a call to the padding op used by front-end FFI
Expr MakeNLLLoss(Expr predictions, Expr targets, Expr weights, String reduction, int ignore_index) {
  auto attrs = make_object<NLLLossAttrs>();
  attrs->reduction = reduction;
  attrs->ignore_index = ignore_index;
  static const Op& op = Op::Get("nn.nll_loss");
  return Call(op, {predictions, targets, weights}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.nll_loss").set_body_typed(MakeNLLLoss);

RELAY_REGISTER_OP("nn.nll_loss")
    .describe(R"code(
Negative log likelihood loss for given prediction and target.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<NLLLossAttrs>()
    .set_num_inputs(3)
    .add_argument("predictions", "Tensor", "The prediction tensor.")
    .add_argument("targets", "Tensor", "The target tensor.")
    .add_argument("weights", "Tensor", "The weight of each target values.")
    .add_type_rel("NLLLoss", NLLLossRel)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

bool DepthToSpaceRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  static const Layout kNCHW("NCHW");

  const SubPixelAttrs* param = attrs.as<SubPixelAttrs>();
  ICHECK(param != nullptr);
  const int block_size = param->block_size;
  const Layout in_layout(param->layout);
  auto layout_converter = tir::BijectiveLayout(in_layout, kNCHW);
  ICHECK(layout_converter.defined())
      << "DepthToSpace only support input layouts that are convertible from NCHW."
      << " But got " << in_layout;

  auto oshape = layout_converter.ForwardShape(data->shape);
  if (!oshape[1].as<tir::AnyNode>()) {
    oshape.Set(1, indexdiv(oshape[1], (block_size * block_size)));
  }
  if (!oshape[2].as<tir::AnyNode>()) {
    oshape.Set(2, oshape[2] * block_size);
  }
  if (!oshape[3].as<tir::AnyNode>()) {
    oshape.Set(3, oshape[3] * block_size);
  }

  // Assign output type
  reporter->Assign(types[1], TensorType(layout_converter.BackwardShape(oshape), data->dtype));

  return true;
}

// Positional relay function to create DepthToSpace operator
// used by frontend FFI
Expr MakeDepthToSpace(Expr data, int block_size, String layout, String mode) {
  auto attrs = make_object<SubPixelAttrs>();
  attrs->block_size = block_size;
  attrs->layout = std::move(layout);
  attrs->mode = std::move(mode);
  static const Op& op = Op::Get("nn.depth_to_space");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.depth_to_space").set_body_typed(MakeDepthToSpace);

RELAY_REGISTER_OP("nn.depth_to_space")
    .describe(R"code(Rearrange input channels into spatial pixels.

- **data**: data is a 4D array of shape
            (batch, in_channels, in_height, in_width) for NCHW

- **out**: Output is a 4D array of shape
           (batch, in_channels / block_size * block_size, in_height * block_size, in_width * block_size) for NCHW.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<SubPixelAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_support_level(5)
    .add_type_rel("DepthToSpace", DepthToSpaceRel)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

bool SpaceToDepthRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  static const Layout kNCHW("NCHW");

  const SubPixelAttrs* param = attrs.as<SubPixelAttrs>();
  ICHECK(param != nullptr);
  const int block_size = param->block_size;
  const Layout in_layout(param->layout);
  auto layout_converter = tir::BijectiveLayout(in_layout, kNCHW);
  ICHECK(layout_converter.defined())
      << "SpaceToDepth only support input layouts that are convertible from NCHW."
      << " But got " << in_layout;

  auto oshape = layout_converter.ForwardShape(data->shape);
  if (!oshape[1].as<tir::AnyNode>()) {
    oshape.Set(1, oshape[1] * (block_size * block_size));
  }
  if (!oshape[2].as<tir::AnyNode>()) {
    oshape.Set(2, indexdiv(oshape[2], block_size));
  }
  if (!oshape[3].as<tir::AnyNode>()) {
    oshape.Set(3, indexdiv(oshape[3], block_size));
  }

  // Assign output type
  reporter->Assign(types[1], TensorType(layout_converter.BackwardShape(oshape), data->dtype));

  return true;
}

// Positional relay function to create SpaceToDepth operator
// used by frontend FFI
Expr MakeSpaceToDepth(Expr data, int block_size, String layout) {
  auto attrs = make_object<SubPixelAttrs>();
  attrs->block_size = block_size;
  attrs->layout = std::move(layout);
  static const Op& op = Op::Get("nn.space_to_depth");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.space_to_depth").set_body_typed(MakeSpaceToDepth);

RELAY_REGISTER_OP("nn.space_to_depth")
    .describe(R"code(Rearrange spatial pixels into new output channels.

- **data**: data is a 4D array of shape
            (batch, in_channels, in_height, in_width) for NCHW

- **out**: Output is a 4D array of shape
           (batch, in_channels * block_size * block_size, in_height / block_size, in_width / block_size) for NCHW.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<SubPixelAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_support_level(5)
    .add_type_rel("SpaceToDepth", SpaceToDepthRel)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// Positional relay function to create SpaceToBatchND operator
// used by frontend FFI
TVM_REGISTER_NODE_TYPE(SpaceToBatchNDAttrs);

Expr MakeSpaceToBatchND(Expr data, Array<Integer> block_shape, Array<Array<IndexExpr>> paddings,
                        double pad_value) {
  auto attrs = make_object<SpaceToBatchNDAttrs>();
  attrs->block_shape = std::move(block_shape);
  attrs->paddings = std::move(paddings);
  attrs->pad_value = pad_value;
  static const Op& op = Op::Get("nn.space_to_batch_nd");
  return Call(op, {data}, Attrs(attrs), {});
}

bool SpaceToBatchNDRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                       const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);

  auto* input = types[0].as<TensorTypeNode>();
  // Input must be a TensorType
  if (input == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "SpaceToBatchND: expect input type to be TensorType but got " << types[0];
    return false;
  }

  if (input->shape.size() <= 1) return false;

  const auto* param = attrs.as<SpaceToBatchNDAttrs>();
  CHECK(param != nullptr);

  auto block_shape = param->block_shape;
  auto paddings = param->paddings;
  const int bdims = static_cast<int>(block_shape.size());
  const int pdims = static_cast<int>(paddings.size());
  // Paddings must be provided for each spatial dim.
  CHECK(pdims == bdims) << "SpaceToBatchND: Paddings must be provided for each spatial dim";

  // Apply paddings to input
  auto in_shape = input->shape;
  std::vector<IndexExpr> padded_shape(input->shape.begin(), input->shape.end());
  for (size_t i = 0; i < paddings.size(); i++) {
    CHECK_EQ(paddings[i].size(), 2U);
    auto pad_before = tir::as_const_int(param->paddings[i][0]);
    auto pad_after = tir::as_const_int(param->paddings[i][1]);
    auto padding = tir::make_const(input->shape[i].dtype(), *pad_before + *pad_after);
    padded_shape[i + 1] = in_shape[i + 1] + padding;
  }

  auto block_shape_numele = tir::make_const(DataType::Int(32), 1);
  for (size_t i = 0; i < block_shape.size(); i++) {
    block_shape_numele *= block_shape[i];
  }

  // Construct output shape
  std::vector<IndexExpr> out_shape(padded_shape);
  out_shape[0] = in_shape[0] * block_shape_numele;
  for (size_t i = 1; i <= block_shape.size(); i++) {
    out_shape[i] = div(padded_shape[i], block_shape[i - 1]);
  }

  // Assign output shape
  reporter->Assign(types[1], TensorType(Array<IndexExpr>(out_shape), input->dtype));
  return true;
}

Array<te::Tensor> SpaceToBatchNDCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                        const Type& out_type) {
  const auto* param = attrs.as<SpaceToBatchNDAttrs>();
  CHECK(param != nullptr);

  auto b_shape = param->block_shape;
  auto paddings = param->paddings;
  Array<IndexExpr> pad_before;
  Array<IndexExpr> pad_after;

  for (size_t i = 0; i < paddings.size(); ++i) {
    pad_before.push_back(paddings[i][0]);
  }
  for (size_t i = 0; i < paddings.size(); ++i) {
    pad_after.push_back(paddings[i][1]);
  }
  const auto* out_ttype = out_type.as<TensorTypeNode>();
  return Array<te::Tensor>{
      topi::space_to_batch_nd(inputs[0], b_shape, pad_before, pad_after,
                              tvm::tir::make_const(out_ttype->dtype, param->pad_value))};
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.space_to_batch_nd").set_body_typed(MakeSpaceToBatchND);

RELAY_REGISTER_OP("nn.space_to_batch_nd")
    .describe(R"code(Divide spatial dimensions of the input into a grid of blocks
and interleave them into batch dim.

- **data**: data is a ND array of shape
            (batch, spatial_shapes, remaining_shapes) for NHWC

- **out**: Output is a ND array of shape
           (batch * prod(block_shape), padded_data[1] / block_shape[0], ..., padded_data[M] / block_shape[M-1],
            remaining_shape) for NHWC, where M is the number of spatial dimensions.

Example::

  x = [[[[1], [2]], [[3], [4]]]]

  space_to_batch_nd(x, block_shape = [2, 2]) =
    [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]

)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attrs_type<SpaceToBatchNDAttrs>()
    .set_support_level(5)
    .add_type_rel("SpaceToBatchND", SpaceToBatchNDRel)
    .set_attr<FTVMCompute>("FTVMCompute", SpaceToBatchNDCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

/*****************************************************************/

// Positional relay function to create BatchToSpaceND operator
// used by frontend FFI
TVM_REGISTER_NODE_TYPE(BatchToSpaceNDAttrs);

Expr MakeBatchToSpaceND(Expr data, Array<Integer> block_shape, Array<Array<IndexExpr>> crops) {
  auto attrs = make_object<BatchToSpaceNDAttrs>();
  attrs->block_shape = std::move(block_shape);
  attrs->crops = std::move(crops);
  static const Op& op = Op::Get("nn.batch_to_space_nd");
  return Call(op, {data}, Attrs(attrs), {});
}

bool BatchToSpaceNDRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                       const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);

  auto* input = types[0].as<TensorTypeNode>();
  // Input must be a TensorType
  if (input == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "BatchToSpaceND: expect input type to be TensorType but got " << types[0];
    return false;
  }

  if (input->shape.size() <= 1) return false;

  const auto* param = attrs.as<BatchToSpaceNDAttrs>();
  CHECK(param != nullptr);

  auto block_shape = param->block_shape;
  auto crops = param->crops;
  const int bdims = static_cast<int>(block_shape.size());
  const int cdims = static_cast<int>(crops.size());
  const int indims = static_cast<int>(input->shape.size());
  // crops must be provided for each spatial dim.
  CHECK(cdims == bdims) << "BatchToSpaceND: crops must be provided for each spatial dim";
  CHECK(bdims < indims) << "BatchToSpaceND: block_shape must be less than input shape";

  auto block_shape_numele = tir::make_const(DataType::Int(32), 1);
  for (size_t i = 0; i < block_shape.size(); i++) {
    block_shape_numele *= block_shape[i];
  }

  auto in_shape = input->shape;

  // Construct output shape
  // Start with input shape, only batch and spatial dims shapes are modified.
  std::vector<IndexExpr> out_shape(input->shape.begin(), input->shape.end());
  out_shape[0] = in_shape[0] / block_shape_numele;
  for (size_t i = 1; i <= block_shape.size(); i++) {
    out_shape[i] = (in_shape[i] * block_shape[i - 1]) - crops[i - 1][0] - crops[i - 1][1];
  }
  for (int i = bdims + 1; i < indims; i++) {
    out_shape[i] = in_shape[i];
  }

  // Assign output shape
  reporter->Assign(types[1], TensorType(Array<IndexExpr>(out_shape), input->dtype));
  return true;
}

Array<te::Tensor> BatchToSpaceNDCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                        const Type& out_type) {
  const auto* param = attrs.as<BatchToSpaceNDAttrs>();
  CHECK(param != nullptr);

  auto b_shape = param->block_shape;
  auto crops = param->crops;
  Array<IndexExpr> crop_begin_list, crop_end_list;
  for (size_t i = 0; i < crops.size(); ++i) {
    crop_begin_list.push_back(crops[i][0]);
    crop_end_list.push_back(crops[i][1]);
  }

  return Array<te::Tensor>{
      topi::batch_to_space_nd(inputs[0], b_shape, crop_begin_list, crop_end_list)};
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.batch_to_space_nd").set_body_typed(MakeBatchToSpaceND);

RELAY_REGISTER_OP("nn.batch_to_space_nd")
    .describe(R"code(Reshape the batch dimension into spatial dimensions.

Example::

  x = [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]

  batch_to_space_nd(x, block_shape = [2, 2]) =
    [[[[1], [2]], [[3], [4]]]]

)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attrs_type<BatchToSpaceNDAttrs>()
    .set_support_level(5)
    .add_type_rel("BatchToSpaceND", BatchToSpaceNDRel)
    .set_attr<FTVMCompute>("FTVMCompute", BatchToSpaceNDCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

}  // namespace relay
}  // namespace tvm
