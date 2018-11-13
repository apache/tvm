/*!
 *  Copyright (c) 2018 by Contributors
 * \file nms.cc
 * \brief Non-maximum suppression operators
 */
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/vision.h>

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(NMSAttrs);

bool NMSRel(const Array<Type>& types,
            int num_inputs,
            const Attrs& attrs,
            const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* valid_count = types[1].as<TensorTypeNode>();
  const auto& dshape = data->shape;
  const auto& vshape = valid_count->shape;
  CHECK_EQ(dshape.size(), 3) << "Input data should be 3-D.";
  CHECK_EQ(vshape.size(), 1) << "Input valid count should be 1-D.";

  // assign output type
  reporter->Assign(types[2], TensorTypeNode::make(dshape, data->dtype));
  return true;
}


Expr MakeNMS(Expr data,
             Expr valid_count,
             double overlap_threshold,
             bool force_suppress,
             int topk) {
  auto attrs = make_node<NMSAttrs>();
  attrs->overlap_threshold = overlap_threshold;
  attrs->force_suppress = force_suppress;
  attrs->topk = topk;
  static const Op& op = Op::Get("vision.nms");
  return CallNode::make(op, {data, valid_count}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.vision._make.nms")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
  runtime::detail::unpack_call<Expr, 5>(MakeNMS, args, rv);
});


RELAY_REGISTER_OP("vision.nms")
.describe(R"doc("Non-maximum suppression."
)doc" TVM_ADD_FILELINE)
.set_num_inputs(2)
.add_argument("data", "Tensor", "Input data.")
.add_argument("valid_count", "Tensor", "Number of valid anchor boxes.")
.set_support_level(5)
.add_type_rel("NMS", NMSRel);

}  // namespace relay
}  // namespace tvm
