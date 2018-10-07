/*!
 *  Copyright (c) 2018 by Contributors
 * \file nn.cc
 * \brief Property def of nn operators.
 */

#include <tvm/relay/op.h>
#include <tvm/relay/attrs/nn.h>
#include "../type_relations.h"

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

}  // namespace relay
}  // namespace tvm
