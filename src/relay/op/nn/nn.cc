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

TVM_REGISTER_NODE_TYPE(DenseAttrs);


bool DenseRel(const Array<Type>& types,
              int num_inputs,
              const Attrs& attrs,
              const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const DenseAttrs* param = attrs.as<DenseAttrs>();
  CHECK(param != nullptr);

  CHECK(static_cast<int>(data->shape.size()) != 0);

  Array<tvm::Expr> oshape = data->shape;
  if (param->units.defined()) {
    oshape.Set((oshape.size() - 1), param->units);
  } else {
    const auto* weight = types[1].as<TensorTypeNode>();
    if (weight == nullptr) return false;
    Array<tvm::Expr> wshape = weight->shape;
    oshape.Set((oshape.size() - 1), wshape[wshape.size() - 1]);
  }

  // assign output type
  reporter->Assign(types[2], TensorTypeNode::make(oshape, data->dtype));
  return true;
}


// Positional relay function to create dense operator used by frontend FFI.
Expr MakeDense(Expr data,
               Expr weight,
               IndexExpr units) {
  auto attrs = make_node<DenseAttrs>();
  attrs->units = units;
  static const Op& op = Op::Get("nn.dense");
  return CallNode::make(op, {data, weight}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.nn._make.dense")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 3>(MakeDense, args, rv);
  });


RELAY_REGISTER_OP("nn.dense")
.describe(R"code(Applies a linear transformation: :math:`Y = XW^T`.

- **data**: `(x1, x2, ..., xn, input_dim)`
- **weight**: `(units, input_dim)`
- **out**: `(x1, x2, ..., xn, units)`.

)code" TVM_ADD_FILELINE)
.set_num_inputs(2)
.add_argument("data", "nD Tensor", "Input data.")
.add_argument("weight", "2D Tensor", "Weight matrix.")
.set_support_level(2)
.add_type_rel("Dense", DenseRel);


// Positional relay function to create leaky relu operator used by frontend FFI.
Expr MakeLeakyRelu(Expr data,
                   double alpha) {
  auto attrs = make_node<LeakyReluAttrs>();
  attrs->alpha = alpha;
  static const Op& op = Op::Get("nn.leaky_relu");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.nn._make.leaky_relu")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 2>(MakeLeakyRelu, args, rv);
  });


RELAY_REGISTER_OP("nn.leaky_relu")
.describe(R"code(Leaky version of a Rectified Linear Unit.

`y = x > 0 ? x : alpha * x`

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "Input data.")
.set_support_level(3)
.add_type_rel("Identity", IdentityRel);


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

}  // namespace relay
}  // namespace tvm
