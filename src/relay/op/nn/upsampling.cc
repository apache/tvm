/*!
 *  Copyright (c) 2018 by Contributors
 * \file upsampling.cc
 * \brief upsampling operator
 */
#include <tvm/data_layout.h>
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/build_module.h>
#include <topi/elemwise.h>
#include <topi/nn/upsampling.h>
#include <vector>
#include "../op_common.h"

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

  auto layout_converter = BijectiveLayoutNode::make(in_layout, kNCHW);
  CHECK(layout_converter.defined())
    << "UpSampling only support input layouts that are convertible from NCHW."
    << " But got " << in_layout;

  auto oshape = layout_converter.ForwardShape(data->shape);

  oshape.Set(2, oshape[2] * param->scale);
  oshape.Set(3, oshape[3] * param->scale);

  // assign output type
  reporter->Assign(types[1],
                   TensorTypeNode::make(layout_converter.BackwardShape(oshape),
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
.set_attrs_type_key("relay.attrs.UpSamplingAttrs")
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(2)
.add_type_rel("UpSampling", UpSamplingRel)
.set_attr<TOpPattern>("TOpPattern", kInjective)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const Attrs& attrs,
                    const Array<Tensor>& inputs,
                    const Type& out_type,
                    const Target& target) {
    const auto* uattrs = attrs.as<UpSamplingAttrs>();
    CHECK(uattrs != nullptr);
    auto out_tt = out_type.as<TensorTypeNode>();
    CHECK(out_tt) << "expected a tensor type: " << out_type;
    CHECK(uattrs->layout == "NCHW" || uattrs->layout == "NHWC")
      << "unknown layout: " << uattrs->layout;

    Array<HalideIR::Expr> oshape;
    if (uattrs->layout == "NCHW") {
      oshape.push_back(out_tt->shape[2]);
      oshape.push_back(out_tt->shape[3]);
    } else if (uattrs->layout == "NHWC") {
      oshape.push_back(out_tt->shape[1]);
      oshape.push_back(out_tt->shape[2]);
    }

    return Array<Tensor>{
      topi::nn::upsampling(
        inputs[0],
        oshape,
        uattrs->layout,
        uattrs->method)
    };
});


}  // namespace relay
}  // namespace tvm
