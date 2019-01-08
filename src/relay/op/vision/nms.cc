/*!
 *  Copyright (c) 2018 by Contributors
 * \file nms.cc
 * \brief Non-maximum suppression operators
 */
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/vision.h>

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(GetValidCountsAttrs);

bool GetValidCountRel(const Array<Type>& types,
                      int num_inputs,
                      const Attrs& attrs,
                      const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto& dshape = data->shape;
  CHECK_EQ(dshape.size(), 3) << "Input data should be 3-D.";

  std::vector<IndexExpr> oshape({data->shape[0]});
  std::vector<Type> fields;
  fields.push_back(TensorTypeNode::make(oshape, Int(32)));
  fields.push_back(TensorTypeNode::make(data->shape, data->dtype));

  // assign output type
  reporter->Assign(types[1], TupleTypeNode::make(Array<Type>(fields)));
  return true;
}

Expr MakeGetValidCounts(Expr data,
                        double score_threshold) {
  auto attrs = make_node<GetValidCountsAttrs>();
  attrs->score_threshold = score_threshold;
  static const Op& op = Op::Get("vision.get_valid_counts");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.vision._make.get_valid_counts")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
  runtime::detail::unpack_call<Expr, 2>(MakeGetValidCounts, args, rv);
});


RELAY_REGISTER_OP("vision.get_valid_counts")
.describe(R"doc(Get valid count of bounding boxes given
a score threshold. Also moves valid boxes to the top of
input data.
)doc" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "Input data.")
.set_support_level(5)
.add_type_rel("GetValidCount", GetValidCountRel);


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
             double iou_threshold,
             bool force_suppress,
             int topk,
             int id_index,
             bool do_rearrange) {
  auto attrs = make_node<NMSAttrs>();
  attrs->iou_threshold = iou_threshold;
  attrs->force_suppress = force_suppress;
  attrs->topk = topk;
  attrs->id_index = id_index;
  attrs->do_rearrange = do_rearrange;
  static const Op& op = Op::Get("vision.nms");
  return CallNode::make(op, {data, valid_count}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.vision._make.nms")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
  runtime::detail::unpack_call<Expr, 7>(MakeNMS, args, rv);
});


RELAY_REGISTER_OP("vision.nms")
.describe(R"doc(Non-maximum suppression.
)doc" TVM_ADD_FILELINE)
.set_num_inputs(2)
.add_argument("data", "Tensor", "Input data.")
.add_argument("valid_count", "Tensor", "Number of valid anchor boxes.")
.set_support_level(5)
.add_type_rel("NMS", NMSRel);

}  // namespace relay
}  // namespace tvm
