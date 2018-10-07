/*!
 *  Copyright (c) 2018 by Contributors
 * \file convolution.cc
 * \brief Convolution operators
 */
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/nn.h>
#include <vector>
#include "layout.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(ConvAttrs);

bool Conv2DRel(const Array<Type>& types,
               int num_inputs,
               const Attrs& attrs,
               const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr) return false;

  static const Layout kNCHW("NCHW");
  static const Layout kOIHW("OIHW");

  const ConvAttrs* param = attrs.as<ConvAttrs>();
  CHECK(param != nullptr);
  const Layout in_layout(param->data_layout);
  const Layout kernel_layout(param->weight_layout);
  CHECK(in_layout.convertible(kNCHW))
    << "Conv only support input layouts that are convertible from NCHW."
    << " But got " << in_layout;
  CHECK(kernel_layout.convertible(kOIHW))
    << "Conv only support kernel layouts that are convertible from OIHW."
    << " But got "<< kernel_layout;

  Layout out_layout(param->out_layout);
  if (!out_layout.defined()) out_layout = in_layout;
  CHECK(out_layout.convertible(kNCHW))
      << "Conv only support output layouts that are convertible from NCHW."
      << " But got " << out_layout;

  IndexExpr channels, dilated_ksize_y, dilated_ksize_x;
  // infer weight if the kernel_size and channels are defined
  if (param->kernel_size.defined() && param->channels.defined()) {
    CHECK_EQ(param->kernel_size.size(), 2);
    CHECK_EQ(param->dilation.size(), 2);
    std::vector<IndexExpr> wshape(
        {param->channels / param->groups,
         data->shape[1] / param->groups,
         param->kernel_size[0],
         param->kernel_size[1]});
    wshape = ConvertLayout(wshape, kOIHW, kernel_layout);
    wshape[kernel_layout.indexof('O')] *= param->groups;
    channels = param->channels;
    dilated_ksize_y = 1 + (param->kernel_size[0] - 1) * param->dilation[0];
    dilated_ksize_x = 1 + (param->kernel_size[1] - 1) * param->dilation[1];
    // assign result to reporter
    reporter->Assign(types[1], TensorTypeNode::make(wshape, data->dtype));
  } else {
    // use weight to infer the conv shape.
    if (weight == nullptr) return false;
    auto wshape = ConvertLayout(weight->shape, kernel_layout, kOIHW);
    if (param->kernel_size.defined()) {
      CHECK_EQ(param->kernel_size.size(), 2);
      // check the size
      CHECK(reporter->AssertEQ(param->kernel_size[0], wshape[2]) &&
            reporter->AssertEQ(param->kernel_size[1], wshape[3]))
          << "Conv2D: shape of weight is inconsistent with kernel_size, "
          << " kernel_size=" << param->kernel_size
          << " wshape=" << Array<IndexExpr>(wshape);
    }
    if (param->channels.defined()) {
      CHECK(reporter->AssertEQ(param->channels, wshape[0]))
          << "Conv2D: shape of weight is inconsistent with channels, "
          << " channels=" << param->channels
          << " wshape=" << Array<IndexExpr>(wshape);
    }
    CHECK(reporter->AssertEQ(data->shape[1] / param->groups, wshape[1]));
    channels = wshape[0];
    dilated_ksize_y = 1 + (wshape[2] - 1) * param->dilation[0];
    dilated_ksize_x = 1 + (wshape[3] - 1) * param->dilation[1];
  }
  // dilation
  std::vector<IndexExpr> oshape({data->shape[0], channels, 0, 0});

  oshape[2] = (data->shape[2] + param->padding[0] * 2 - dilated_ksize_y) / param->strides[0] + 1;
  oshape[3] = (data->shape[3] + param->padding[1] * 2 - dilated_ksize_x) / param->strides[1] + 1;
  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }
  oshape = ConvertLayout(oshape, kNCHW, out_layout);
  // assign output type
  reporter->Assign(types[2], TensorTypeNode::make(oshape, out_dtype));
  return true;
}


// Positional relay function to create conv2d operator
// used by frontend FFI.
Expr MakeConv2D(Expr data,
                Expr weight,
                Array<IndexExpr> strides,
                Array<IndexExpr> padding,
                Array<IndexExpr> dilation,
                int groups,
                IndexExpr channels,
                Array<IndexExpr> kernel_size,
                std::string data_layout,
                std::string weight_layout,
                std::string out_layout,
                DataType out_dtype) {
  auto attrs = make_node<ConvAttrs>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->channels = channels;
  attrs->kernel_size = kernel_size;
  attrs->data_layout = std::move(data_layout);
  attrs->weight_layout = std::move(weight_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->out_dtype = std::move(out_dtype);
  static const Op& op = Op::Get("nn.conv2d");
  return CallNode::make(op, {data, weight}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.nn._make.conv2d")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 12>(MakeConv2D, args, rv);
  });


RELAY_REGISTER_OP("nn.conv2d")
.describe(R"code(2D convolution layer (e.g. spatial convolution over images).

This layer creates a convolution kernel that is convolved
with the layer input to produce a tensor of outputs.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, in_channels, height, width) if `layout` is `NCHW`.
- **weight**: (channels, in_channels, kernel_size[0], kernel_size[1])
- **out**:  This depends on the `layout` parameter. Output is 4D array of shape
            (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.

)code" TVM_ADD_FILELINE)
.set_num_inputs(2)
.add_argument("data", "Tensor", "The input tensor.")
.add_argument("weight", "Tensor", "The weight tensor.")
.set_support_level(2)
.add_type_rel("Conv2D", Conv2DRel);

}  // namespace relay
}  // namespace tvm
