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

}  // namespace relay
}  // namespace tvm
