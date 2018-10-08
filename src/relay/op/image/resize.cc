/*!
 *  Copyright (c) 2018 by Contributors
 * \file resize.cc
 * \brief Image operators
 */
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/image.h>
#include "../nn/layout.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(ResizeAttrs);

bool ResizeRel(const Array<Type>& types,
               int num_inputs,
               const Attrs& attrs,
               const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  static const Layout kNCHW("NCHW");

  const ResizeAttrs* param = attrs.as<ResizeAttrs>();
  CHECK(param != nullptr);
  const Layout in_layout(param->layout);
  CHECK(in_layout.convertible(kNCHW))
    << "Resize only support input layouts that are convertible from NCHW."
    << " But got " << in_layout;

  auto oshape = ConvertLayout(data->shape, in_layout, kNCHW);
  oshape[2] = param->size[0];
  oshape[3] = param->size[1];

  // assign output type
  reporter->Assign(types[1],
                   TensorTypeNode::make(ConvertLayout(oshape, kNCHW, in_layout),
                                        data->dtype));
  return true;
}


// Positional relay function to create image operator
// used by frontend FFI.
Expr MakeResize(Expr data,
                Array<IndexExpr> size,
                std::string layout,
                std::string method,
                bool align_corners) {
  auto attrs = make_node<ResizeAttrs>();
  attrs->size = std::move(size);
  attrs->layout = std::move(layout);
  attrs->method = std::move(method);
  attrs->align_corners = align_corners;
  static const Op& op = Op::Get("image.resize");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.image._make.resize")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 5>(MakeResize, args, rv);
  });


RELAY_REGISTER_OP("image.resize")
.describe(R"code(Perform resize to input array with nearest neighbour or bilinear interpolation.

- **data**: data is 4D array of shape
            (batch_size, channels, in_height, in_width) for NCHW
            (batch_size, in_height, in_width, channels) for NHWC

- **out**: Output is 4D array of shape
           for layout NCHW
           (batch_size, channels, size[0], size[1])

           for layout NHWC
           (batch_size, size[0], size[1], channels)
)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(5)
.add_type_rel("Resize", ResizeRel);

}  // namespace relay
}  // namespace tvm
