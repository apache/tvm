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

#include <tvm/tir/data_layout.h>
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/image.h>
#include <topi/nn.h>
#include <topi/nn/bias_add.h>
#include <topi/nn/softmax.h>
#include <topi/nn/flatten.h>
#include <vector>
#include <string>
#include "../type_relations.h"
#include "../../transforms/infer_layout_util.h"
#include "../op_common.h"
#include "nn.h"

namespace tvm {
namespace relay {

// relay.nn.bias_add
TVM_REGISTER_NODE_TYPE(BiasAddAttrs);

bool BiasAddRel(const Array<Type>& types,
                int num_inputs,
                const Attrs& attrs,
                const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const BiasAddAttrs* param = attrs.as<BiasAddAttrs>();
  CHECK(param != nullptr);
  int axis = param->axis;
  if (axis < 0) {
    axis = data->shape.size() + axis;
  }
  CHECK_LE(axis, static_cast<int>(data->shape.size()))
      << "axis " << param->axis << " is out of range";

  // assign output type
  reporter->Assign(types[1], TensorType(
      {data->shape[axis]}, data->dtype));
  reporter->Assign(types[2], types[0]);
  return true;
}


// Positional relay function to create dense operator used by frontend FFI.
Expr MakeBiasAdd(Expr data,
                 Expr bias,
                 int axis) {
  auto attrs = make_object<BiasAddAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("nn.bias_add");
  return Call(op, {data, bias}, Attrs(attrs), {});
}


TVM_REGISTER_GLOBAL("relay.op.nn._make.bias_add")
.set_body_typed(MakeBiasAdd);


RELAY_REGISTER_OP("nn.bias_add")
.describe(R"code(Add bias to an axis of the input.

)code" TVM_ADD_FILELINE)
.set_attrs_type<BiasAddAttrs>()
.set_num_inputs(2)
.add_argument("data", "nD Tensor", "Input data.")
.add_argument("bias", "1D Tensor", "Bias.")
.set_support_level(1)
.add_type_rel("BiasAdd", BiasAddRel)
.set_attr<FTVMCompute>("FTVMCompute", [](const Attrs& attrs,
                                         const Array<te::Tensor>& inputs,
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

bool FIFOBufferRel(const Array<Type>& types,
                   int num_inputs,
                   const Attrs& attrs,
                   const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* input = types[0].as<TensorTypeNode>();
  const auto* buffer = types[1].as<TensorTypeNode>();
  const FIFOBufferAttrs* param = attrs.as<FIFOBufferAttrs>();
  if (input == nullptr || buffer == nullptr) {
    return false;
  }
  CHECK(param != nullptr);
  CHECK_EQ(input->shape.size(), buffer->shape.size());

  const size_t buffer_axis
    = static_cast<size_t>(param->axis < 0 ? static_cast<int>(buffer->shape.size()) + param->axis
                                          : param->axis);

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

TVM_REGISTER_GLOBAL("relay.op.nn._make.fifo_buffer")
.set_body_typed(MakeFIFOBuffer);

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
.add_argument("buffer", "Tensor",
              "Buffer storing latest [length_buffer] inputs")
.set_support_level(3)
.add_type_rel("FIFOBuffer", FIFOBufferRel);


// relay.nn.dense
TVM_REGISTER_NODE_TYPE(DenseAttrs);

// Positional relay function to create dense operator used by frontend FFI.
Expr MakeDense(Expr data,
               Expr weight,
               IndexExpr units,
               DataType out_dtype) {
  auto attrs = make_object<DenseAttrs>();
  attrs->units = units;
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("nn.dense");
  return Call(op, {data, weight}, Attrs(attrs), {});
}


TVM_REGISTER_GLOBAL("relay.op.nn._make.dense")
.set_body_typed(MakeDense);


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
.add_type_rel("Dense", DenseRel<DenseAttrs>);

// relay.leaky_relu
TVM_REGISTER_NODE_TYPE(LeakyReluAttrs);

// Positional relay function to create leaky relu operator used by frontend FFI.
Expr MakeLeakyRelu(Expr data,
                   double alpha) {
  auto attrs = make_object<LeakyReluAttrs>();
  attrs->alpha = alpha;
  static const Op& op = Op::Get("nn.leaky_relu");
  return Call(op, {data}, Attrs(attrs), {});
}


TVM_REGISTER_GLOBAL("relay.op.nn._make.leaky_relu")
.set_body_typed(MakeLeakyRelu);


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
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const Attrs& attrs,
                    const Array<te::Tensor>& inputs,
                    const Type& out_type) {
    const auto* param = attrs.as<LeakyReluAttrs>();
    return Array<te::Tensor>{ topi::leaky_relu(inputs[0], param->alpha) };
});


// relay.prelu
TVM_REGISTER_NODE_TYPE(PReluAttrs);

bool PReluRel(const Array<Type>& types,
              int num_inputs,
              const Attrs& attrs,
              const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const PReluAttrs* param = attrs.as<PReluAttrs>();
  CHECK(param != nullptr);

  CHECK(param->axis < static_cast<int>(data->shape.size()))
    << "Wrong axis ("  << param->axis << ")value.";

  // assign alpha type
  Array<IndexExpr> alpha_shape({data->shape[param->axis]});
  reporter->Assign(types[1], TensorType(alpha_shape, data->dtype));

  // assign output type
  reporter->Assign(types[2], TensorType(data->shape, data->dtype));
  return true;
}

template<typename T>
Array<Array<Layout> > PReluInferCorrectLayout(
    const Attrs& attrs,
    const Array<Layout>& new_in_layouts,
    const Array<Layout>& old_in_layouts,
    const Array<tvm::relay::Type> &old_in_types) {

  CHECK_EQ(old_in_layouts.size(), 2U);
  CHECK_EQ(old_in_types.size(), 2U);
  Layout data_layout = old_in_layouts[0];
  if (new_in_layouts.defined()) {
    CHECK_EQ(new_in_layouts.size(), 2U);
  }
  return Array<Array<Layout> >{{data_layout, Layout("C")},
                               {data_layout}};
}

// Positional relay function to create prelu operator used by frontend FFI.
Expr MakePRelu(Expr data,
               Expr alpha,
               int axis) {
  auto attrs = make_object<PReluAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("nn.prelu");
  return Call(op, {data, alpha}, Attrs(attrs), {});
}


TVM_REGISTER_GLOBAL("relay.op.nn._make.prelu")
.set_body_typed(MakePRelu);


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
.set_attr<FInferCorrectLayout>("FInferCorrectLayout", PReluInferCorrectLayout<PReluAttrs>)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const Attrs& attrs,
                    const Array<te::Tensor>& inputs,
                    const Type& out_type) {
    const auto* param = attrs.as<PReluAttrs>();
    return Array<te::Tensor>{ topi::prelu(inputs[0], inputs[1], param->axis)};
});


// relay.softmax
TVM_REGISTER_NODE_TYPE(SoftmaxAttrs);

TVM_REGISTER_GLOBAL("relay.op.nn._make.softmax")
.set_body_typed([](Expr data, int axis) {
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
.add_type_rel("Identity", IdentityRel);


// relay.nn.log_softmax
TVM_REGISTER_GLOBAL("relay.op.nn._make.log_softmax")
.set_body_typed([](Expr data, int axis) {
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
.add_type_rel("Identity", IdentityRel)
.set_attr<FTVMCompute>("FTVMCompute", [](const Attrs& attrs,
                                         const Array<te::Tensor>& inputs,
                                         const Type& out_type) {
  const auto* param = attrs.as<SoftmaxAttrs>();
  CHECK(param != nullptr);
  CHECK(param->axis == -1 || param->axis == static_cast<int32_t>(inputs[0].ndim()) - 1)
      << "log_softmax currently only works on last dimension";
  return Array<te::Tensor>{ topi::nn::log_softmax(inputs[0]) };
});


// relay.nn.batch_flatten
bool BatchFlattenRel(const Array<Type>& types,
                     int num_inputs,
                     const Attrs& attrs,
                     const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
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


TVM_REGISTER_GLOBAL("relay.op.nn._make.batch_flatten")
.set_body_typed(MakeBatchFlatten);


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
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const Attrs& attrs,
                    const Array<te::Tensor>& inputs,
                    const Type& out_type) {
    return Array<te::Tensor>{ topi::nn::flatten(inputs[0]) };
});


// relu
TVM_REGISTER_GLOBAL("relay.op.nn._make.relu")
.set_body_typed([](Expr data) {
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
.set_attr<FTVMCompute>("FTVMCompute", [](const Attrs& attrs,
                                         const Array<te::Tensor>& inputs,
                                         const Type& out_type) {
  return Array<te::Tensor>{ topi::relu(inputs[0], 0.0f) };
});


// Positional relay function to create LRN operator used by frontend FFI.
TVM_REGISTER_NODE_TYPE(LRNAttrs);

Expr MakeLRN(Expr data,
             int size,
             int axis,
             double alpha,
             double beta,
             double bias) {
  auto attrs = make_object<LRNAttrs>();
  attrs->size = size;
  attrs->axis = axis;
  attrs->alpha = alpha;
  attrs->beta = beta;
  attrs->bias = bias;
  static const Op& op = Op::Get("nn.lrn");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.lrn")
.set_body_typed(MakeLRN);

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
.set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
.add_type_rel("Identity", IdentityRel);


// Positional relay function to create L2Normalize operator used by frontend FFI.
TVM_REGISTER_NODE_TYPE(L2NormalizeAttrs);

Expr MakeL2Normalize(Expr data,
                     double eps,
                     Array<Integer> axis) {
  auto attrs = make_object<L2NormalizeAttrs>();
  attrs->eps = eps;
  attrs->axis = std::move(axis);
  static const Op& op = Op::Get("nn.l2_normalize");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.l2_normalize")
.set_body_typed(MakeL2Normalize);

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
.set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
.add_type_rel("Identity", IdentityRel);

// Dropout
TVM_REGISTER_NODE_TYPE(DropoutAttrs);

bool DropoutRel(const Array<Type>& types,
                int num_inputs,
                const Attrs& attrs,
                const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
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

TVM_REGISTER_GLOBAL("relay.op.nn._make.dropout")
.set_body_typed(MakeDropout);

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
.add_type_rel("Dropout", DropoutRel);

// batch_norm
TVM_REGISTER_NODE_TYPE(BatchNormAttrs);

Array<Array<Layout>> BatchNormInferCorrectLayout(const Attrs& attrs,
                                                 const Array<Layout>& new_in_layouts,
                                                 const Array<Layout>& old_in_layouts,
                                                 const Array<tvm::relay::Type>& old_in_types) {
  BatchNormAttrs* param = const_cast<BatchNormAttrs*>(attrs.as<BatchNormAttrs>());

  Array<Array<IndexExpr>> old_in_shapes;
  for (auto old_in_t : old_in_types) {
    CHECK(old_in_t.as<TensorTypeNode>());
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

  return Array<Array<Layout>>{{ret, c_layout, c_layout, c_layout, c_layout},
                              {ret, c_layout, c_layout}};
}

bool BatchNormRel(const Array<Type>& types,
                  int num_inputs,
                  const Attrs& attrs,
                  const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 6);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const BatchNormAttrs* param = attrs.as<BatchNormAttrs>();

  // axis of -1 means use the last dimension
  CHECK(param->axis >= -1 && param->axis < (int)data->shape.size());
  int axis = (param->axis != -1) ? param->axis : data->shape.size() - 1;
  auto axis_size = data->shape[axis];

  // if we are using beta and gamma, they need to be of shape (dim,)
  reporter->Assign(types[1], TensorType({axis_size}, data->dtype));
  reporter->Assign(types[2], TensorType({axis_size}, data->dtype));
  reporter->Assign(types[3], TensorType({axis_size}, data->dtype));
  reporter->Assign(types[4], TensorType({axis_size}, data->dtype));

  // output is a tuple of the normed data (same shape as input), new running mean,
  // and new running average (the latter two are both vectors of length dim)
  std::vector<Type> fields;
  auto vec_ty = TensorType(Array<IndexExpr>({data->shape[axis]}),
                                     data->dtype);
  fields.push_back(TensorType(data->shape, data->dtype));
  fields.push_back(vec_ty);
  fields.push_back(vec_ty);
  reporter->Assign(types[5], TupleType(Array<Type>(fields)));
  return true;
}

Expr MakeBatchNorm(Expr data, Expr gamma, Expr beta, Expr moving_mean, Expr moving_var,
                   int axis, double epsilon, bool center, bool scale) {
  auto attrs = make_object<BatchNormAttrs>();
  attrs->axis = axis;
  attrs->epsilon = epsilon;
  attrs->center = center;
  attrs->scale = scale;
  static const Op& op = Op::Get("nn.batch_norm");
  return Call(op, {data, gamma, beta, moving_mean, moving_var}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.batch_norm")
.set_body_typed(MakeBatchNorm);

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
.add_type_rel("BatchNorm", BatchNormRel);


// instance_norm
TVM_REGISTER_NODE_TYPE(InstanceNormAttrs);

bool InstanceNormRel(const Array<Type>& types,
                     int num_inputs,
                     const Attrs& attrs,
                     const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 4);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  const InstanceNormAttrs* param = attrs.as<InstanceNormAttrs>();
  int axis = param->axis >= 0 ? param->axis : param->axis + data->shape.size();
  CHECK(axis >= 0 && axis < (int)data->shape.size());
  reporter->Assign(types[1], TensorType({data->shape[axis]}, data->dtype));
  reporter->Assign(types[2], TensorType({data->shape[axis]}, data->dtype));
  reporter->Assign(types[3], TensorType(data->shape, data->dtype));

  return true;
}

Expr MakeInstanceNorm(Expr data, Expr gamma, Expr beta, int axis, double epsilon,
                      bool center, bool scale) {
  auto attrs = make_object<InstanceNormAttrs>();
  attrs->axis = axis;
  attrs->epsilon = epsilon;
  attrs->center = center;
  attrs->scale = scale;
  static const Op& op = Op::Get("nn.instance_norm");
  return Call(op, {data, gamma, beta}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.instance_norm")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 7>(MakeInstanceNorm, args, rv);
  });

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
.set_support_level(1)
.add_type_rel("InstanceNorm", InstanceNormRel);


// layer_norm
TVM_REGISTER_NODE_TYPE(LayerNormAttrs);

bool LayerNormRel(const Array<Type>& types,
                  int num_inputs,
                  const Attrs& attrs,
                  const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 4);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  const LayerNormAttrs* param = attrs.as<LayerNormAttrs>();
  int axis = param->axis >= 0 ? param->axis : param->axis + data->shape.size();
  CHECK(axis >= 0 && axis < (int)data->shape.size());
  reporter->Assign(types[1], TensorType({data->shape[axis]}, data->dtype));
  reporter->Assign(types[2], TensorType({data->shape[axis]}, data->dtype));
  reporter->Assign(types[3], TensorType(data->shape, data->dtype));

  return true;
}

Expr MakeLayerNorm(Expr data, Expr gamma, Expr beta, int axis, double epsilon,
                   bool center, bool scale) {
  auto attrs = make_object<LayerNormAttrs>();
  attrs->axis = axis;
  attrs->epsilon = epsilon;
  attrs->center = center;
  attrs->scale = scale;
  static const Op& op = Op::Get("nn.layer_norm");
  return Call(op, {data, gamma, beta}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.layer_norm")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 7>(MakeLayerNorm, args, rv);
  });

RELAY_REGISTER_OP("nn.layer_norm")
.describe(R"code(
)code" TVM_ADD_FILELINE)
.set_attrs_type<LayerNormAttrs>()
.set_num_inputs(3)
.add_argument("data", "Tensor", "Input to which layer_norm will be applied.")
.add_argument("gamma", "Tensor", "The gamma scale factor.")
.add_argument("beta", "Tensor", "The beta offset factor.")
.set_support_level(1)
.add_type_rel("LayerNorm", LayerNormRel);

// group_norm
TVM_REGISTER_NODE_TYPE(GroupNormAttrs);

bool GroupNormRel(const Array<Type>& types,
                  int num_inputs,
                  const Attrs& attrs,
                  const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 4);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  const GroupNormAttrs* param = attrs.as<GroupNormAttrs>();
  int axis = param->axis >= 0 ? param->axis : param->axis + data->shape.size();
  CHECK(axis >= 0 && axis < (int)data->shape.size());
  reporter->Assign(types[1], TensorType({data->shape[axis]}, data->dtype));
  reporter->Assign(types[2], TensorType({data->shape[axis]}, data->dtype));
  reporter->Assign(types[3], TensorType(data->shape, data->dtype));

  return true;
}

Expr MakeGroupNorm(Expr data, Expr gamma, Expr beta, int num_groups,
                   int axis, double epsilon, bool center, bool scale) {
  auto attrs = make_object<GroupNormAttrs>();
  attrs->num_groups =  num_groups;
  attrs->axis = axis;
  attrs->epsilon = epsilon;
  attrs->center = center;
  attrs->scale = scale;
  static const Op& op = Op::Get("nn.group_norm");
  return Call(op, {data, gamma, beta}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.group_norm")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 8>(MakeGroupNorm, args, rv);
  });

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


// relay.nn.batch_matmul
bool BatchMatmulRel(const Array<Type>& types,
                    int num_inputs,
                    const Attrs& attrs,
                    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* x = types[0].as<TensorTypeNode>();
  const auto* y = types[1].as<TensorTypeNode>();
  if (x == nullptr || y == nullptr) return false;
  CHECK(x->shape.size() == 3 && y->shape.size() == 3);
  CHECK(reporter->AssertEQ(x->shape[0], y->shape[0]))
      << "BatchDot: batch dimension doesn't match, "
      << " x shape=" << x->shape
      << ", y shape=" << y->shape;
  CHECK(reporter->AssertEQ(x->shape[2], y->shape[2]))
      << "BatchDot: shapes of x and y is inconsistent, "
      << " x shape=" << x->shape
      << ", y shape=" << y->shape;

  Array<tvm::PrimExpr> oshape = x->shape;
  oshape.Set(2, y->shape[1]);

  // assign output type
  reporter->Assign(types[2], TensorType(oshape, x->dtype));
  return true;
}


// Positional relay function to create batch_matmul operator used by frontend FFI.
Expr MakeBatchMatmul(Expr x,
                     Expr y) {
  static const Op& op = Op::Get("nn.batch_matmul");
  return Call(op, {x, y}, Attrs(), {});
}


TVM_REGISTER_GLOBAL("relay.op.nn._make.batch_matmul")
.set_body_typed(MakeBatchMatmul);


RELAY_REGISTER_OP("nn.batch_matmul")
.describe(R"code(Computes matrix multiplication of `x` and `y` when `x` and `y`
are data in batch.

.. math::

  batch\_matmul(x, y)[i, :, :] = matmul(x[i, :, :], y[i, :, :]^T)

- **x**: `(b, m, k)`
- **y**: `(b, n, k)`
- **out**: `(b, m, n)`.

)code" TVM_ADD_FILELINE)
.set_num_inputs(2)
.add_argument("x", "3D Tensor", "First input.")
.add_argument("y", "3D Tensor", "Second input.")
.set_support_level(10)
.add_type_rel("BatchMatmul", BatchMatmulRel);


// relay.nn.cross_entropy
bool CrossEntropyRel(const Array<Type>& types,
                    int num_inputs,
                    const Attrs& attrs,
                    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* x = types[0].as<TensorTypeNode>();
  const auto* y = types[1].as<TensorTypeNode>();
  if (x == nullptr || y == nullptr) return false;
  CHECK(x->shape.size() == 2 && y->shape.size() == 2)
    << "CrossEntropy: shapes of x and y is inconsistent, "
    << "x shape = " << x->shape << ", "
    << "y shape = " << y->shape;
  CHECK(reporter->AssertEQ(x->shape[0], y->shape[0]))
    << "CrossEntropy: shapes of x and y is inconsistent, "
    << "x shape = " << x->shape << ", "
    << "y shape = " << y->shape;
  CHECK(reporter->AssertEQ(x->shape[1], y->shape[1]))
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


TVM_REGISTER_GLOBAL("relay.op.nn._make.cross_entropy")
.set_body_typed(MakeCrossEntropy);


RELAY_REGISTER_OP("nn.cross_entropy")
.describe(R"code(
Computes cross entropy given predictions and targets.
Do log on the data - do not accept logits.
)code" TVM_ADD_FILELINE)
.set_num_inputs(2)
.add_argument("x", "1D Tensor", "Predictions.")
.add_argument("y", "1D Tensor", "Targets.")
.set_support_level(10)
.add_type_rel("CrossEntropy", CrossEntropyRel);


// relay.nn.dilate
TVM_REGISTER_NODE_TYPE(DilateAttrs);

bool DilateRel(const Array<Type>& types,
               int num_inputs,
               const Attrs& attrs,
               const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* x = types[0].as<TensorTypeNode>();
  const DilateAttrs* param = attrs.as<DilateAttrs>();
  if (x == nullptr) return false;
  CHECK_EQ(x->shape.size(), param->strides.size());

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
Expr MakeDilate(Expr data, Array<IndexExpr> strides) {
  auto attrs = make_object<DilateAttrs>();
  attrs->strides = std::move(strides);
  static const Op& op = Op::Get("nn.dilate");
  return Call(op, {data}, Attrs(attrs), {});
}


TVM_REGISTER_GLOBAL("relay.op.nn._make.dilate")
.set_body_typed(MakeDilate);


RELAY_REGISTER_OP("nn.dilate")
.describe(R"code(
Dilate data with zeros.
)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("x", "1D Tensor", "Data to dilate.")
.set_support_level(10)
.add_type_rel("Dilate", DilateRel);

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
.add_type_rel("CrossEntropy", CrossEntropyRel);

// Depth to space and space to depth
TVM_REGISTER_NODE_TYPE(SubPixelAttrs);

bool DepthToSpaceRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  static const Layout kNCHW("NCHW");

  const SubPixelAttrs* param = attrs.as<SubPixelAttrs>();
  CHECK(param != nullptr);
  const int block_size = param->block_size;
  const Layout in_layout(param->layout);
  auto layout_converter = tir::BijectiveLayout(in_layout, kNCHW);
  CHECK(layout_converter.defined())
      << "DepthToSpace only support input layouts that are convertible from NCHW."
      << " But got " << in_layout;

  auto oshape = layout_converter.ForwardShape(data->shape);
  oshape.Set(1, indexdiv(oshape[1], (block_size * block_size)));
  oshape.Set(2, oshape[2] * block_size);
  oshape.Set(3, oshape[3] * block_size);

  // Assign output type
  reporter->Assign(types[1],
                   TensorType(layout_converter.BackwardShape(oshape), data->dtype));

  return true;
}

// Positional relay function to create DepthToSpace operator
// used by frontend FFI
Expr MakeDepthToSpace(Expr data, int block_size, std::string layout, std::string mode) {
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
    .add_type_rel("DepthToSpace", DepthToSpaceRel);

bool SpaceToDepthRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  static const Layout kNCHW("NCHW");

  const SubPixelAttrs* param = attrs.as<SubPixelAttrs>();
  CHECK(param != nullptr);
  const int block_size = param->block_size;
  const Layout in_layout(param->layout);
  auto layout_converter = tir::BijectiveLayout(in_layout, kNCHW);
  CHECK(layout_converter.defined())
      << "SpaceToDepth only support input layouts that are convertible from NCHW."
      << " But got " << in_layout;

  auto oshape = layout_converter.ForwardShape(data->shape);
  oshape.Set(1, oshape[1] * (block_size * block_size));
  oshape.Set(2, indexdiv(oshape[2], block_size));
  oshape.Set(3, indexdiv(oshape[3], block_size));

  // Assign output type
  reporter->Assign(types[1],
                   TensorType(layout_converter.BackwardShape(oshape), data->dtype));

  return true;
}

// Positional relay function to create SpaceToDepth operator
// used by frontend FFI
Expr MakeSpaceToDepth(Expr data, int block_size, std::string layout) {
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
    .add_type_rel("SpaceToDepth", SpaceToDepthRel);

}  // namespace relay
}  // namespace tvm
