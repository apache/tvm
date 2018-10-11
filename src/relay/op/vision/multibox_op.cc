/*!
 *  Copyright (c) 2018 by Contributors
 * \file multibox_op.cc
 * \brief Multibox related operators
 */
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/vision.h>
#include <vector>

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(MultiBoxPriorAttrs);

bool MultiboxPriorRel(const Array<Type>& types,
                      int num_inputs,
                      const Attrs& attrs,
                      const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  const MultiBoxPriorAttrs* param = attrs.as<MultiBoxPriorAttrs>();
  const auto& dshape = data->shape;
  CHECK_EQ(dshape.size(), 4) << "Input data should be 4D: "
      "[batch, channel, height, width]";
  IndexExpr in_height = dshape[2];
  IndexExpr in_width = dshape[3];
  int num_sizes = static_cast<int>(param->sizes.size());
  int num_ratios = static_cast<int>(param->ratios.size());

  // since input sizes are same in each batch, we could share MultiBoxPrior
  std::vector<IndexExpr> oshape(
    {1, in_height * in_width * (num_sizes + num_ratios - 1), 4});

  // assign output type
  reporter->Assign(types[1], TensorTypeNode::make(oshape, data->dtype));
  return true;
}


Expr MakeMultiBoxPrior(Expr data,
                       Array<IndexExpr> sizes,
                       Array<IndexExpr> ratios,
                       Array<IndexExpr> steps,
                       Array<IndexExpr> offsets,
                       bool clip) {
  auto attrs = make_node<MultiBoxPriorAttrs>();
  attrs->sizes = std::move(sizes);
  attrs->ratios = std::move(ratios);
  attrs->steps = std::move(steps);
  attrs->offsets = std::move(offsets);
  attrs->clip = clip;
  static const Op& op = Op::Get("vision.multibox_prior");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.vision._make.multibox_prior")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
  runtime::detail::unpack_call<Expr, 6>(MakeMultiBoxPrior, args, rv);
});


RELAY_REGISTER_OP("vision.multibox_prior")
.describe(R"doc("Generate prior(anchor) boxes from data, sizes and ratios."
)doc" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(4)
.add_type_rel("MultiBoxPrior", MultiboxPriorRel);

}  // namespace relay
}  // namespace tvm
