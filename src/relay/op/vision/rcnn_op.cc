/*!
 *  Copyright (c) 2019 by Contributors
 * \file rcnn_op.cc
 * \brief Faster RCNN and Mask RCNN operators
 */
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/attrs/vision.h>

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(ROIAlignAttrs);

bool ROIAlignRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  auto roi_align_attrs = attrs.as<ROIAlignAttrs>();
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* rois = types[1].as<TensorTypeNode>();
  const auto& dshape = data->shape;
  const auto& rshape = rois->shape;
  CHECK(roi_align_attrs);
  CHECK_EQ(dshape.size(), 4) << "Input data should be 4-D.";
  CHECK_EQ(rshape.size(), 2) << "Input rois should be 2-D.";
  CHECK_EQ(roi_align_attrs->layout, "NCHW") << "ROI Align only supports NCHW layout";
  // assign output type
  std::vector<IndexExpr> oshape(
      {rshape[0], dshape[1], roi_align_attrs->pooled_size[0], roi_align_attrs->pooled_size[1]});
  reporter->Assign(types[2], TensorTypeNode::make(oshape, data->dtype));
  return true;
}

Expr MakeROIAlign(Expr data, Expr rois, Array<IndexExpr> pooled_size, double spatial_scale,
                  int sample_ratio, std::string layout) {
  auto attrs = make_node<ROIAlignAttrs>();
  attrs->pooled_size = pooled_size;
  attrs->spatial_scale = spatial_scale;
  attrs->sample_ratio = sample_ratio;
  attrs->layout = layout;
  static const Op& op = Op::Get("vision.roi_align");
  return CallNode::make(op, {data, rois}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.vision._make.roi_align")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 6>(MakeROIAlign, args, rv);
  });

RELAY_REGISTER_OP("vision.roi_align")
    .describe(R"doc(ROI Align operator.

 - **data**: This depends on the `layout` parameter. Input is 4D array of shape
             (batch_size, channels, height, width) if `layout` is `NCHW`.
 - **rois**: 2D array of shape (num_roi, 5). The last dimension should be in format of
             [batch_index, w_start, h_start, w_end, h_end].
 - **out**: This depends on the `layout` parameter. Output is 4D array of shape
            (num_roi, channels, pooled_height, pooled_width) if `layout` is `NCHW`.
 )doc" TVM_ADD_FILELINE)
.set_num_inputs(2)
.add_argument("data", "Tensor", "The input tensor.")
.add_argument("rois", "Tensor", "The input rois")
.set_support_level(5)
.add_type_rel("ROIAlign", ROIAlignRel);

}  // namespace relay
}  // namespace tvm
