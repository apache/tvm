/*!
 *  Copyright (c) 2018 by Contributors
 * \file nn.cc
 * \brief Property def of nn operators.
 */

#include <tvm/relay/op.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/image.h>
#include <vector>
#include "../type_relations.h"
#include "../op_common.h"
#include "layout.h"

namespace tvm {
namespace relay {

TVM_REGISTER_API("relay.op.nn._make.softmax")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
  auto make_func = [](Expr data, int axis) {
    auto attrs = make_node<SoftmaxAttrs>();
    attrs->axis = axis;
    static const Op& op = Op::Get("nn.softmax");
    return CallNode::make(op, {data}, Attrs(attrs), {});
  };

  runtime::detail::unpack_call<Expr, 2>(make_func, args, rv);
});

RELAY_REGISTER_OP("nn.softmax")
    .describe(R"code(Softmax layer.

.. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}

.. note::
    This operator can be optimized away for inference.

- **data**: The input data
)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(1)
.add_type_rel("Identity", IdentityRel);


TVM_REGISTER_API("relay.op.nn._make.log_softmax")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
  auto make_func = [](Expr data, int axis) {
    auto attrs = make_node<SoftmaxAttrs>();
    attrs->axis = axis;
    static const Op& op = Op::Get("nn.log_softmax");
    return CallNode::make(op, {data}, Attrs(attrs), {});
  };

  runtime::detail::unpack_call<Expr, 2>(make_func, args, rv);
});

RELAY_REGISTER_OP("nn.log_softmax")
    .describe(R"code(Computes log softmax.

.. math:: \text{log_softmax}(x)_i = \log \frac{exp(x_i)}{\sum_j exp(x_j)}

.. note::
    This operator can be optimized away for inference.

- **data**: The input data
)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(1)
.add_type_rel("Identity", IdentityRel);


// BatchFlatten
bool BatchFlattenRel(const Array<Type>& types,
               int num_inputs,
               const Attrs& attrs,
               const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  if (data->shape.size() == 0) return false;

  auto target_dim = make_const(Int(32), 1);

  for (uint32_t i = 1; i < data->shape.size(); ++i) {
    target_dim = target_dim * data->shape[i];
  }

  std::vector<IndexExpr> oshape({data->shape[0], target_dim});

  // assign output type
  reporter->Assign(types[1], TensorTypeNode::make(oshape, data->dtype));
  return true;
}

Expr MakeBatchFlatten(Expr data) {
  static const Op& op = Op::Get("nn.batch_flatten");
  return CallNode::make(op, {data}, Attrs(), {});
}


TVM_REGISTER_API("relay.op.nn._make.batch_flatten")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 1>(MakeBatchFlatten, args, rv);
  });


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
.add_type_rel("BatchFlatten", BatchFlattenRel);

RELAY_REGISTER_UNARY_OP("relay.op.nn._make.", "relu")
.describe(R"code(Returns the relu input array, computed element-wise.

.. math::
   max(x, 0)

)code" TVM_ADD_FILELINE)
.set_support_level(1)
.add_type_rel("Identity", IdentityRel);


// Positional relay function to create LRN operator used by frontend FFI.
Expr MakeLRN(Expr data,
             IndexExpr size,
             IndexExpr axis,
             double alpha,
             double beta,
             double bias) {
  auto attrs = make_node<LRNAttrs>();
  attrs->size = size;
  attrs->axis = axis;
  attrs->alpha = alpha;
  attrs->beta = beta;
  attrs->bias = bias;
  static const Op& op = Op::Get("nn.lrn");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.nn._make.lrn")
  .set_body([](const TVMArgs& args, TVMRetValue* rv) {
      runtime::detail::unpack_call<Expr, 6>(MakeLRN, args, rv);
  });

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
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(2)
.add_type_rel("Identity", IdentityRel);


// Positional relay function to create L2Normalize operator used by frontend FFI.
Expr MakeL2Normalize(Expr data,
                     double eps,
                     Array<IndexExpr> axis) {
  auto attrs = make_node<L2NormalizeAttrs>();
  attrs->eps = eps;
  attrs->axis = std::move(axis);
  static const Op& op = Op::Get("nn.l2_normalize");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.nn._make.l2_normalize")
  .set_body([](const TVMArgs& args, TVMRetValue* rv) {
      runtime::detail::unpack_call<Expr, 3>(MakeL2Normalize, args, rv);
  });

RELAY_REGISTER_OP("nn.l2_normalize")
    .describe(R"code(L2 Normalization layer.

Normalizes along dimension axis using an L2 norm

.. math::
    output = x / sqrt(max(sum(x^2), epsilon))

- **data**: The input tensor.
)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(2)
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
  auto ret_type = TensorTypeNode::make(data->shape, data->dtype);
  reporter->Assign(types[1], TupleTypeNode::make(Array<Type>({ret_type, ret_type})));
  return true;
}

Expr MakeDropout(Expr data, double rate) {
  auto attrs = make_node<DropoutAttrs>();
  attrs->rate = rate;
  static const Op& op = Op::Get("nn.dropout");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.nn._make.dropout")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 2>(MakeDropout, args, rv);
  });

RELAY_REGISTER_OP("nn.dropout")
.describe(R"code(Applies the dropout operation to the input array.

During training, each element of the input is set to zero with probability ``p``.
The whole array is rescaled by ``1/(1-p)`` to keep the expected sum of the input unchanged.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "Input to which dropout will be applied.")
.set_support_level(1)
.add_type_rel("Dropout", DropoutRel);

// batch_norm
TVM_REGISTER_NODE_TYPE(BatchNormAttrs);

bool CheckVectorLength(int64_t dim, const DataType& dtype, Type vector, const char* name) {
  const auto* candidate = vector.as<TensorTypeNode>();
  CHECK(candidate != nullptr)
    << name << " should be a vector but is not a tensor type,";
  CHECK_EQ(dtype, candidate->dtype)
    << name << " should be of the same data type as the original but it is not.";
  CHECK_EQ(candidate->shape.size(), 1)
    << name << " should be a vector but has a shape of "
    << candidate->shape.size() << " dimensions instead of 1.";

  const int64_t* length = as_const_int(candidate->shape[0]);
  if (length == nullptr) return false;
  CHECK(*length == dim)
    << name << " should be as long as the channel but has length "
    << *length << " instead of " << dim << ".";
  return true;
}

bool BatchNormRel(const Array<Type>& types,
                  int num_inputs,
                  const Attrs& attrs,
                  const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 6);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  if (data->shape.size() == 0) return false;

  const BatchNormAttrs* param = attrs.as<BatchNormAttrs>();

  // axis of -1 means use the last dimension
  CHECK(param->axis >= -1 && param->axis < (int)data->shape.size());
  int axis = (param->axis != -1) ? param->axis : data->shape.size() - 1;

  auto dim = as_const_int(data->shape[axis]);
  if (dim == nullptr) return false;

  // if we are using beta and gamma, they need to be of shape (dim,)
  if (param->scale && !CheckVectorLength(*dim, data->dtype, types[1], "The gamma scale factor")) {
    return false;
  }

  if (param->center && !CheckVectorLength(*dim, data->dtype, types[2], "The beta offset factor")) {
    return false;
  }

  // the two running averages must also be vectors of length dim
  if (!CheckVectorLength(*dim, data->dtype, types[3], "The moving mean")) {
    return false;
  }
  if (!CheckVectorLength(*dim, data->dtype, types[4], "The moving variance")) {
    return false;
  }

  // output is a tuple of the normed data (same shape as input), new running mean,
  // and new running average (the latter two are both vectors of length dim)
  std::vector<Type> fields;
  auto vec_ty = TensorTypeNode::make(Array<IndexExpr>({data->shape[axis]}),
                                     data->dtype);
  fields.push_back(TensorTypeNode::make(data->shape, data->dtype));
  fields.push_back(vec_ty);
  fields.push_back(vec_ty);
  reporter->Assign(types[5], TupleTypeNode::make(Array<Type>(fields)));
  return true;
}

Expr MakeBatchNorm(Expr data, Expr gamma, Expr beta, Expr moving_mean, Expr moving_var,
                   int axis, double epsilon, bool center, bool scale) {
  auto attrs = make_node<BatchNormAttrs>();
  attrs->axis = axis;
  attrs->epsilon = epsilon;
  attrs->center = center;
  attrs->scale = scale;
  static const Op& op = Op::Get("nn.batch_norm");
  return CallNode::make(op, {data, gamma, beta, moving_mean, moving_var}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.nn._make.batch_norm")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 9>(MakeBatchNorm, args, rv);
  });

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
.set_num_inputs(5)
.add_argument("data", "Tensor", "Input to which batch_norm will be applied.")
.add_argument("gamma", "Tensor", "The gamma scale factor.")
.add_argument("beta", "Tensor", "The beta offset factor.")
.add_argument("moving_mean", "Tensor", "Running mean of input.")
.add_argument("moving_var", "Tensor", "Running variance of input.")
.set_support_level(1)
.add_type_rel("BatchNorm", BatchNormRel);

}  // namespace relay
}  // namespace tvm
