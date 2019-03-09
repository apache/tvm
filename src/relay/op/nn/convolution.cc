/*!
 *  Copyright (c) 2018 by Contributors
 * \file convolution.cc
 * \brief Convolution operators
 */
#include <tvm/data_layout.h>
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/nn.h>
#include <vector>

#include "../../pass/alter_op_layout.h"

namespace tvm {
namespace relay {

// relay.nn.conv2d
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
  const Layout kernel_layout(param->kernel_layout);

  const auto trans_in_layout = BijectiveLayoutNode::make(in_layout, kNCHW);
  CHECK(trans_in_layout.defined())
    << "Conv only support input layouts that are convertible from NCHW."
    << " But got " << in_layout;

  const auto trans_kernel_layout = BijectiveLayoutNode::make(kernel_layout, kOIHW);
  CHECK(trans_kernel_layout.defined())
    << "Conv only support kernel layouts that are convertible from OIHW."
    << " But got "<< kernel_layout;

  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  const auto trans_out_layout = BijectiveLayoutNode::make(out_layout, kNCHW);
  CHECK(trans_out_layout.defined())
      << "Conv only support output layouts that are convertible from NCHW."
      << " But got " << out_layout;

  Array<IndexExpr> dshape_nchw = trans_in_layout.ForwardShape(data->shape);

  IndexExpr channels, dilated_ksize_y, dilated_ksize_x;
  // infer weight if the kernel_size and channels are defined
  if (param->kernel_size.defined() && param->channels.defined()) {
    CHECK_EQ(param->kernel_size.size(), 2);
    CHECK_EQ(param->dilation.size(), 2);
    Array<IndexExpr> wshape(
       {param->channels,
         dshape_nchw[1] / param->groups,
         param->kernel_size[0],
         param->kernel_size[1]});
    wshape = trans_kernel_layout.BackwardShape(wshape);
    channels = param->channels;
    dilated_ksize_y = 1 + (param->kernel_size[0] - 1) * param->dilation[0];
    dilated_ksize_x = 1 + (param->kernel_size[1] - 1) * param->dilation[1];
    // assign result to reporter
    reporter->Assign(types[1], TensorTypeNode::make(wshape, data->dtype));
  } else {
    // use weight to infer the conv shape.
    if (weight == nullptr) return false;
    auto wshape = trans_kernel_layout.ForwardShape(weight->shape);
    if (param->kernel_size.defined()) {
      CHECK_EQ(param->kernel_size.size(), 2);
      // check the size
      CHECK(reporter->AssertEQ(param->kernel_size[0], wshape[2]) &&
            reporter->AssertEQ(param->kernel_size[1], wshape[3]))
          << "Conv2D: shape of weight is inconsistent with kernel_size, "
          << " kernel_size=" << param->kernel_size
          << " wshape=" << wshape;
    }
    if (param->channels.defined()) {
      CHECK(reporter->AssertEQ(param->channels, wshape[0]))
          << "Conv2D: shape of weight is inconsistent with channels, "
          << " channels=" << param->channels
          << " wshape=" << wshape;
    }
    CHECK(reporter->AssertEQ(dshape_nchw[1] / param->groups, wshape[1]));
    channels = wshape[0];
    dilated_ksize_y = 1 + (wshape[2] - 1) * param->dilation[0];
    dilated_ksize_x = 1 + (wshape[3] - 1) * param->dilation[1];
  }
  // dilation
  Array<IndexExpr> oshape({dshape_nchw[0], channels, 0, 0});

  oshape.Set(2, (dshape_nchw[2] + param->padding[0] * 2 - dilated_ksize_y) / param->strides[0] + 1);
  oshape.Set(3, (dshape_nchw[3] + param->padding[1] * 2 - dilated_ksize_x) / param->strides[1] + 1);
  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }
  oshape = trans_out_layout.BackwardShape(oshape);
  // assign output type
  reporter->Assign(types[2], TensorTypeNode::make(oshape, out_dtype));
  return true;
}

template<typename T>
Array<Array<Layout> > Conv2DInferCorrectLayout(
    const Attrs& attrs,
    const Array<Layout>& new_in_layouts,
    const Array<Layout>& old_in_layouts,
    const Array<Array<IndexExpr>> &old_in_shapes) {
  const T* params = attrs.as<T>();

  // We always make other operators to fit the layouts of convolution layers
  // So this inference ignores all inputs
  return Array<Array<Layout> >{{params->data_layout, params->kernel_layout},
                               {params->out_layout == "" ?
                                   params->data_layout : params->out_layout}};
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
                std::string kernel_layout,
                std::string out_layout,
                DataType out_dtype) {
  auto attrs = make_node<Conv2DAttrs>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->channels = std::move(channels);
  attrs->kernel_size = std::move(kernel_size);
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
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
.add_type_rel("Conv2D", Conv2DRel)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout", Conv2DInferCorrectLayout<Conv2DAttrs>);


// relay.nn.conv2d_transpose
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
  const Layout kernel_layout(param->kernel_layout);

  const auto trans_in_layout = BijectiveLayoutNode::make(in_layout, kNCHW);
  CHECK(trans_in_layout.defined())
    << "Conv only support input layouts that are convertible from NCHW."
    << " But got " << in_layout;

  const auto trans_kernel_layout = BijectiveLayoutNode::make(kernel_layout, kOIHW);
  CHECK(trans_kernel_layout.defined())
    << "Conv only support kernel layouts that are convertible from OIHW."
    << " But got "<< kernel_layout;

  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  const auto trans_out_layout = BijectiveLayoutNode::make(out_layout, kNCHW);
  CHECK(trans_out_layout.defined())
    << "Conv only support output layouts that are convertible from NCHW."
    << " But got " << out_layout;

  IndexExpr channels, dilated_ksize_y, dilated_ksize_x;

  auto dshape_nchw = trans_in_layout.ForwardShape(data->shape);

  // infer weight if the kernel_size and channels are defined
  if (param->kernel_size.defined() && param->channels.defined()) {
    CHECK_EQ(param->kernel_size.size(), 2);
    CHECK_EQ(param->dilation.size(), 2);

    Array<IndexExpr> wshape({dshape_nchw[1],
                             param->channels / param->groups,
                             param->kernel_size[0],
                             param->kernel_size[1]});

    wshape = trans_kernel_layout.BackwardShape(wshape);
    dilated_ksize_y = 1 + (param->kernel_size[0] - 1) * param->dilation[0];
    dilated_ksize_x = 1 + (param->kernel_size[1] - 1) * param->dilation[1];
    channels = param->channels;

    // assign result to reporter
    reporter->Assign(types[1], TensorTypeNode::make(wshape, data->dtype));
  } else {
    // use weight to infer the conv shape.
    if (weight == nullptr) return false;
    auto wshape = trans_kernel_layout.ForwardShape(weight->shape);
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
  Array<IndexExpr> oshape({dshape_nchw[0], channels, 0, 0});
  oshape.Set(2, (param->strides[0] * (dshape_nchw[2] - 1) + dilated_ksize_y -
                 2 * param->padding[0] + param->output_padding[0]));
  oshape.Set(3, (param->strides[1] * (dshape_nchw[3] - 1) + dilated_ksize_x -
                 2 * param->padding[1] + param->output_padding[1]));

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }
  oshape = trans_out_layout.BackwardShape(oshape);
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
                         std::string kernel_layout,
                         Array<IndexExpr> output_padding,
                         DataType out_dtype) {
  auto attrs = make_node<Conv2DTransposeAttrs>();
  attrs->channels = std::move(channels);
  attrs->kernel_size = std::move(kernel_size);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->output_padding = std::move(output_padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
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
.set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                               Conv2DInferCorrectLayout<Conv2DTransposeAttrs>)
.add_type_rel("Conv2DTranspose", Conv2DTransposeRel);


// relay.nn.contrib_conv2d_winograd_without_weight_transform
TVM_REGISTER_NODE_TYPE(Conv2DWinogradAttrs);

bool Conv2DWinogradRel(const Array<Type>& types,
                       int num_inputs,
                       const Attrs& attrs,
                       const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  static const Layout kNCHW("NCHW");
  static const Layout kOIHW("OIHW");

  const Conv2DWinogradAttrs* param = attrs.as<Conv2DWinogradAttrs>();
  CHECK(param != nullptr);
  const Layout in_layout(param->data_layout);
  const Layout kernel_layout(param->kernel_layout);

  const auto trans_in_layout = BijectiveLayoutNode::make(in_layout, kNCHW);
  CHECK(trans_in_layout.defined())
    << "Conv only support input layouts that are convertible from NCHW."
    << " But got " << in_layout;

  const auto trans_kernel_layout = BijectiveLayoutNode::make(kernel_layout, kOIHW);
  CHECK(trans_kernel_layout.defined())
    << "Conv only support kernel layouts that are convertible from OIHW."
    << " But got "<< kernel_layout;

  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  const auto trans_out_layout = BijectiveLayoutNode::make(out_layout, kNCHW);
  CHECK(trans_out_layout.defined())
      << "Conv only support output layouts that are convertible from NCHW."
      << " But got " << out_layout;

  Array<IndexExpr> dshape_nchw = trans_in_layout.ForwardShape(data->shape);

  IndexExpr channels, dilated_ksize_y, dilated_ksize_x;

  CHECK(param->kernel_size.defined() && param->channels.defined())
      << "The kernel size and channels of a Conv must be set or infered by previous pass";

  CHECK_EQ(param->kernel_size.size(), 2);
  CHECK_EQ(param->dilation.size(), 2);

  channels = param->channels;
  dilated_ksize_y = 1 + (param->kernel_size[0] - 1) * param->dilation[0];
  dilated_ksize_x = 1 + (param->kernel_size[1] - 1) * param->dilation[1];

  // NOTE: Do not check weight shape here!
  // Different backend requires different layout to compute
  // the batch gemm stage in winograd efficiently, but we want to
  // make this op work for all backends.
  // So we accept all weight shapes, and assume the TOPI developers
  // can handle this correctly in alter_op_layout.

  // dilation
  Array<IndexExpr> oshape({dshape_nchw[0], channels, 0, 0});

  oshape.Set(2, (dshape_nchw[2] + param->padding[0] * 2 - dilated_ksize_y) / param->strides[0] + 1);
  oshape.Set(3, (dshape_nchw[3] + param->padding[1] * 2 - dilated_ksize_x) / param->strides[1] + 1);
  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }
  oshape = trans_out_layout.BackwardShape(oshape);
  // assign output type
  reporter->Assign(types[2], TensorTypeNode::make(oshape, out_dtype));
  return true;
}


// Positional relay function to create conv2d winograd operator
// used by frontend FFI.
Expr MakeConv2DWinograd(Expr data,
                        Expr weight,
                        int tile_size,
                        Array<IndexExpr> strides,
                        Array<IndexExpr> padding,
                        Array<IndexExpr> dilation,
                        int groups,
                        IndexExpr channels,
                        Array<IndexExpr> kernel_size,
                        std::string data_layout,
                        std::string kernel_layout,
                        std::string out_layout,
                        DataType out_dtype) {
  auto attrs = make_node<Conv2DWinogradAttrs>();
  attrs->tile_size = tile_size;
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->channels = channels;
  attrs->kernel_size = std::move(kernel_size);
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->out_dtype = std::move(out_dtype);
  static const Op& op = Op::Get("nn.contrib_conv2d_winograd_without_weight_transform");
  return CallNode::make(op, {data, weight}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.nn._make.contrib_conv2d_winograd_without_weight_transform")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 13>(MakeConv2DWinograd, args, rv);
  });


RELAY_REGISTER_OP("nn.contrib_conv2d_winograd_without_weight_transform")
.describe(R"code(Compute conv2d with winograd algorithm. Only supports NCHW layout.
                 This operator assumes the weight tensor is already pre-transformed by
                 nn.contrib_conv2d_winograd_weight_transform.

- **data**: Input is 4D array of shape  (batch_size, in_channels, height, width)
- **weight**: Any shape
            We do not check the shape for this input tensor. Since different backend
            has different layout strategy.

- **out**:  Output is 4D array of shape (batch_size, channels, out_height, out_width)
)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.Conv2DWinograd")
.set_num_inputs(2)
.add_argument("data", "Tensor", "The input tensor.")
.add_argument("weight", "Tensor", "The weight tensor.")
.set_support_level(10)
.add_type_rel("Conv2DWinograd", Conv2DWinogradRel)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout",
        Conv2DInferCorrectLayout<Conv2DWinogradAttrs>);

// relay.nn.contrib_conv2d_winograd_weight_transform
TVM_REGISTER_NODE_TYPE(Conv2DWinogradWeightTransformAttrs);

bool Conv2DWinogradWeightTransformRel(const Array<Type>& types,
                                      int num_inputs,
                                      const Attrs& attrs,
                                      const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const Conv2DWinogradWeightTransformAttrs* param = attrs.as<Conv2DWinogradWeightTransformAttrs>();
  CHECK(param != nullptr);

  CHECK_EQ(data->shape.size(), 4) << "Only support NCHW normal kernel layout";

  // each pad width element should be a pair of positive integers
  std::vector<IndexExpr> oshape {
      param->tile_size + data->shape[2] - 1,
      param->tile_size + data->shape[3] - 1,
      data->shape[0],
      data->shape[1],
  };

  reporter->Assign(types[1], TensorTypeNode::make(Array<IndexExpr>(oshape),
                                                  data->dtype));
  return true;
}

Expr MakeConv2DWinogradWeightTransform(Expr weight,
                                       int tile_size) {
  auto attrs = make_node<Conv2DWinogradWeightTransformAttrs>();
  attrs->tile_size = tile_size;
  static const Op& op = Op::Get("nn.contrib_conv2d_winograd_weight_transform");
  return CallNode::make(op, {weight}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.nn._make.contrib_conv2d_winograd_weight_transform")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 2>(MakeConv2DWinogradWeightTransform, args, rv);
  });


RELAY_REGISTER_OP("nn.contrib_conv2d_winograd_weight_transform")
.describe(R"code(Weight transformation of winograd fast convolution algorithm.

Separate this into another nnvm symbol in order to enable Precompute Pass to compute the
weight transformation in advance.

- **weight**: (channels, in_channels, kernel_size[0], kernel_size[1])
)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.Conv2DWinogradWeightTransformAttrs")
.set_num_inputs(1)
.add_argument("weight", "Tensor", "The weight tensor.")
.set_support_level(10)
.add_type_rel("Conv2DWinogradWeightTransform", Conv2DWinogradWeightTransformRel);


// Positional relay function to create conv2d NCHWc operator
// used by frontend FFI.
Expr MakeConv2DNCHWc(Expr data,
                     Expr kernel,
                     Array<IndexExpr> strides,
                     Array<IndexExpr> padding,
                     Array<IndexExpr> dilation,
                     int groups,
                     IndexExpr channels,
                     Array<IndexExpr> kernel_size,
                     std::string data_layout,
                     std::string kernel_layout,
                     std::string out_layout,
                     DataType out_dtype) {
  auto attrs = make_node<Conv2DAttrs>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->channels = channels;
  attrs->kernel_size = std::move(kernel_size);
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->out_dtype = std::move(out_dtype);
  static const Op& op = Op::Get("nn.contrib_conv2d_NCHWc");
  return CallNode::make(op, {data, kernel}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.nn._make.contrib_conv2d_NCHWc")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 12>(MakeConv2DNCHWc, args, rv);
  });


RELAY_REGISTER_OP("nn.contrib_conv2d_NCHWc")
.describe(R"code(Compute conv2d with NCHWc data layout. Only supports NCHW layout.
- **data**: Input is 5D packed tensor.
- **weight**: 6D packed tensor.

- **out**:  Output is 5D packed tensor
)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.Conv2D")
.set_num_inputs(2)
.add_argument("data", "Tensor", "The input tensor.")
.add_argument("weight", "Tensor", "The weight tensor.")
.set_support_level(10)
.add_type_rel("Conv2D", Conv2DRel)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout",
        Conv2DInferCorrectLayout<Conv2DAttrs>);


}  // namespace relay
}  // namespace tvm
