/*!
 *  Copyright (c) 2018 by Contributors
 * \file upsampling.cc
 * \brief upsampling operator
 */
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/nn.h>
#include "layout.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(UpSamplingAttrs);

bool UpSamplingRel(const Array<Type>& types,
                   int num_inputs,
                   const Attrs& attrs,
                   const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  static const Layout kNCHW("NCHW");

  const UpSamplingAttrs* param = attrs.as<UpSamplingAttrs>();
  CHECK(param != nullptr);
  const Layout in_layout(param->layout);
  CHECK(in_layout.convertible(kNCHW))
    << "UpSampling only support input layouts that are convertible from NCHW."
    << " But got " << in_layout;

  auto oshape = ConvertLayout(data->shape, in_layout, kNCHW);

  oshape[2] = oshape[2] * param->scale;
  oshape[3] = oshape[3] * param->scale;

  // assign output type
  reporter->Assign(types[1],
                   TensorTypeNode::make(ConvertLayout(oshape, kNCHW, in_layout),
                                        data->dtype));
  return true;
}


// Positional relay function to create upsampling operator
// used by frontend FFI.
Expr MakeUpSampling(Expr data,
                    int scale,
                    std::string layout,
                    std::string method) {
  auto attrs = make_node<UpSamplingAttrs>();
  attrs->layout = std::move(layout);
  attrs->method = std::move(method);
  attrs->scale = scale;
  static const Op& op = Op::Get("nn.upsampling");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.nn._make.upsampling")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 4>(MakeUpSampling, args, rv);
  });


RELAY_REGISTER_OP("nn.upsampling")
.describe(R"code(Perform upsampling on input array with nearest neighbour or bilinear interpolation.

- **data**: data is 4D array of shape
            (batch_size, channels, in_height, in_width) for NCHW
            (batch_size, in_height, in_width, channels) for NHWC

- **out**: Output is 4D array of shape
           for layout NCHW
           (batch_size, channels, in_height*scale, in_width*scale)

           for layout NHWC
           (batch_size, in_height*scale, in_width*scale, channels)

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(2)
.add_type_rel("UpSampling", UpSamplingRel);

}  // namespace relay
}  // namespace tvm
