/*!
 *  Copyright (c) 2018 by Contributors
 * \file convolution.cc
 * \brief Convolution operators
 */
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/nn.h>
#include <vector>

#include "../layout.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(Conv2DAttrs);

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

  const Conv2DAttrs* param = attrs.as<Conv2DAttrs>();
  CHECK(param != nullptr);
  const Layout in_layout(param->data_layout);
  const Layout kernel_layout(param->weight_layout);
  CHECK(in_layout.Convertible(kNCHW))
    << "Conv only support input layouts that are convertible from NCHW."
    << " But got " << in_layout;
  CHECK(kernel_layout.Convertible(kOIHW))
    << "Conv only support kernel layouts that are convertible from OIHW."
    << " But got "<< kernel_layout;

  Layout out_layout(param->out_layout);
  if (!out_layout.defined()) out_layout = in_layout;
  CHECK(out_layout.Convertible(kNCHW))
      << "Conv only support output layouts that are convertible from NCHW."
      << " But got " << out_layout;

  std::vector<IndexExpr> dshape_nchw = ConvertLayout(
      data->shape, in_layout, kNCHW);

  IndexExpr channels, dilated_ksize_y, dilated_ksize_x;
  // infer weight if the kernel_size and channels are defined
  if (param->kernel_size.defined() && param->channels.defined()) {
    CHECK_EQ(param->kernel_size.size(), 2);
    CHECK_EQ(param->dilation.size(), 2);
    std::vector<IndexExpr> wshape(
       {param->channels / param->groups,
         dshape_nchw[1] / param->groups,
         param->kernel_size[0],
         param->kernel_size[1]});
    wshape = ConvertLayout(wshape, kOIHW, kernel_layout);
    wshape[kernel_layout.Indexof('O')] *= param->groups;
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
    CHECK(reporter->AssertEQ(dshape_nchw[1] / param->groups, wshape[1]));
    channels = wshape[0];
    dilated_ksize_y = 1 + (wshape[2] - 1) * param->dilation[0];
    dilated_ksize_x = 1 + (wshape[3] - 1) * param->dilation[1];
  }
  // dilation
  std::vector<IndexExpr> oshape({dshape_nchw[0], channels, 0, 0});

  oshape[2] = (dshape_nchw[2] + param->padding[0] * 2 - dilated_ksize_y) / param->strides[0] + 1;
  oshape[3] = (dshape_nchw[3] + param->padding[1] * 2 - dilated_ksize_x) / param->strides[1] + 1;
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
  auto attrs = make_node<Conv2DAttrs>();
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
.set_attrs_type_key("relay.attrs.Conv2DAttrs")
.set_num_inputs(2)
.add_argument("data", "Tensor", "The input tensor.")
.add_argument("weight", "Tensor", "The weight tensor.")
.set_support_level(2)
.add_type_rel("Conv2D", Conv2DRel);


// Conv2DTranspose
TVM_REGISTER_NODE_TYPE(Conv2DTransposeAttrs);

bool Conv2DTransposeRel(const Array<Type>& types,
                        int num_inputs,
                        const Attrs& attrs,
                        const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr) return false;

  static const Layout kNCHW("NCHW");
  static const Layout kOIHW("OIHW");

  const Conv2DTransposeAttrs* param = attrs.as<Conv2DTransposeAttrs>();
  CHECK(param != nullptr);
  const Layout in_layout(param->data_layout);
  const Layout kernel_layout(param->weight_layout);
  CHECK(in_layout.Convertible(kNCHW))
    << "Conv only support input layouts that are convertible from NCHW."
    << " But got " << in_layout;
  CHECK(kernel_layout.Convertible(kOIHW))
    << "Conv only support kernel layouts that are convertible from OIHW."
    << " But got "<< kernel_layout;

  IndexExpr channels, dilated_ksize_y, dilated_ksize_x;

  auto dshape_nchw = ConvertLayout(data->shape, in_layout, kNCHW);

  // infer weight if the kernel_size and channels are defined
  if (param->kernel_size.defined() && param->channels.defined()) {
    CHECK_EQ(param->kernel_size.size(), 2);
    CHECK_EQ(param->dilation.size(), 2);

    std::vector<IndexExpr> wshape({dshape_nchw[1],
                                   param->channels / param->groups,
                                   param->kernel_size[0],
                                   param->kernel_size[1]});

    wshape = ConvertLayout(wshape, kOIHW, kernel_layout);
    dilated_ksize_y = 1 + (param->kernel_size[0] - 1) * param->dilation[0];
    dilated_ksize_x = 1 + (param->kernel_size[1] - 1) * param->dilation[1];
    channels = param->channels;

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
      CHECK(reporter->AssertEQ(param->channels, wshape[1]))
          << "Conv2D: shape of weight is inconsistent with channels, "
          << " channels=" << param->channels
          << " wshape=" << Array<IndexExpr>(wshape);
    }
    CHECK(reporter->AssertEQ(dshape_nchw[1] / param->groups, wshape[0]));
    channels = wshape[1];
    dilated_ksize_y = 1 + (wshape[2] - 1) * param->dilation[0];
    dilated_ksize_x = 1 + (wshape[3] - 1) * param->dilation[1];
  }
  // dilation
  std::vector<IndexExpr> oshape({dshape_nchw[0], channels, 0, 0});
  oshape[2] = (param->strides[0] * (dshape_nchw[2] - 1) + dilated_ksize_y -
               2 * param->padding[0] + param->output_padding[0]);
  oshape[3] = (param->strides[1] * (dshape_nchw[3] - 1) + dilated_ksize_x -
               2 * param->padding[1] + param->output_padding[1]);

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }
  oshape = ConvertLayout(oshape, kNCHW, in_layout);
  reporter->Assign(types[2], TensorTypeNode::make(oshape, out_dtype));
  return true;
}


Expr MakeConv2DTranspose(Expr data,
                         Expr weight,
                         Array<IndexExpr> strides,
                         Array<IndexExpr> padding,
                         Array<IndexExpr> dilation,
                         int groups,
                         IndexExpr channels,
                         Array<IndexExpr> kernel_size,
                         std::string data_layout,
                         std::string weight_layout,
                         Array<IndexExpr> output_padding,
                         DataType out_dtype) {
  auto attrs = make_node<Conv2DTransposeAttrs>();
  attrs->channels = channels;
  attrs->kernel_size = kernel_size;
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->output_padding = std::move(output_padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->data_layout = std::move(data_layout);
  attrs->weight_layout = std::move(weight_layout);
  attrs->out_dtype = std::move(out_dtype);
  static const Op& op = Op::Get("nn.conv2d_transpose");
  return CallNode::make(op, {data, weight}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.nn._make.conv2d_transpose")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 12>(MakeConv2DTranspose, args, rv);
  });

RELAY_REGISTER_OP("nn.conv2d_transpose")
.describe(R"code(Transposed 2D convolution layer (sometimes called Deconvolution).

The need for transposed convolutions generally arises
from the desire to use a transformation going in the opposite direction
of a normal convolution, i.e., from something that has the shape of the
output of some convolution to something that has the shape of its input
while maintaining a connectivity pattern that is compatible with
said convolution.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, in_channels, height, width) if `layout` is `NCHW`.
- **weight**: (in_channels, channels, kernel_size[0], kernel_size[1])
- **bias**: (channels,)
- **out**:  This depends on the `layout` parameter. Output is 4D array of shape
v            (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.

            out_height and out_width are calculated as::
                out_height = (height-1)*strides[0]-2*padding[0]+kernel_size[0]+output_padding[0]
                out_width = (width-1)*strides[1]-2*padding[1]+kernel_size[1]+output_padding[1]

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.Conv2DTransposeAttrs")
.set_num_inputs(2)
.add_argument("data", "Tensor", "The input tensor.")
.add_argument("weight", "Tensor", "The weight tensor.")
.set_support_level(2)
.add_type_rel("Conv2DTranspose", Conv2DTransposeRel);

}  // namespace relay
}  // namespace tvm
