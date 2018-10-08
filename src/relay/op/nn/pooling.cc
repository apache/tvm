/*!
 *  Copyright (c) 2018 by Contributors
 * \file pooling.cc
 * \brief Pooling operators
 */
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/nn.h>
#include <vector>
#include "layout.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(MaxPool2DAttrs);

template <typename AttrTtype>
bool Pool2DRel(const Array<Type>& types,
               int num_inputs,
               const Attrs& attrs,
               const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();

  CHECK(data != nullptr);
  const auto dshape = data->shape;
  CHECK_NE(dshape.size(), 0);
  CHECK_GE(dshape.size(), 2U)
      << "Pool2D only support input >= 2-D: input must have height and width";
  const auto param = attrs.as<AttrTtype>();
  CHECK(param != nullptr);

  Layout layout(param->layout);
  CHECK(layout.contains('H') && layout.contains('W') &&
        !layout.contains('h') && !layout.contains('w'))
    << "Invalid layout " << layout
    << ". Pool2D layout must have H and W, which cannot be split";

  const auto hidx = layout.indexof('H');
  const auto widx = layout.indexof('W');

  IndexExpr pad_h, pad_w;
  if (param->padding.size() == 1) {
    pad_h = param->padding[0] * 2;
    pad_w = param->padding[0] * 2;
  } else if (param->padding.size() == 2) {
    // (top, left)
    pad_h = param->padding[0] * 2;
    pad_w = param->padding[1] * 2;
  } else if (param->padding.size() == 4) {
    // (top, left, bottom, right)
    pad_h = param->padding[0] + param->padding[2];
    pad_w = param->padding[1] + param->padding[3];
  } else {
    return false;
  }

  std::vector<IndexExpr> oshape({dshape[0], dshape[1], dshape[2], dshape[3]});
  if (param->ceil_mode) {
    oshape[hidx] = ((dshape[hidx] + pad_h - param->pool_size[0] +
                    param->strides[0] - 1) / param->strides[0]) + 1;
    oshape[widx] = ((dshape[widx] + pad_w - param->pool_size[1] +
                    param->strides[1] - 1) / param->strides[1]) + 1;
  } else {
    oshape[hidx] = ((dshape[hidx] + pad_h - param->pool_size[0]) / param->strides[0]) + 1;
    oshape[widx] = ((dshape[widx] + pad_w - param->pool_size[1]) / param->strides[1]) + 1;
  }

  // assign output type
  reporter->Assign(types[1], TensorTypeNode::make(oshape, data->dtype));
  return true;
}

// MaxPool2D
Expr MakeMaxPool2D(Expr data,
                   Array<IndexExpr> pool_size,
                   Array<IndexExpr> strides,
                   Array<IndexExpr> padding,
                   std::string layout,
                   bool ceil_mode) {
  auto attrs = make_node<MaxPool2DAttrs>();
  attrs->pool_size = std::move(pool_size);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->layout = std::move(layout);
  attrs->ceil_mode = ceil_mode;
  static const Op& op = Op::Get("nn.max_pool2d");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.nn._make.max_pool2d")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 6>(MakeMaxPool2D, args, rv);
  });


RELAY_REGISTER_OP("nn.max_pool2d")
.describe(R"code(Max pooling operation for two dimensional data.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, channels, height, width) if `layout` is `NCHW`.
- **out**: This depends on the `layout` parameter. Output is 4D array of shape
           (batch_size, channels, out_height, out_width)  if `layout` is `NCHW`.
           out_height and out_width are calculated as::

               out_height = floor((height+padding[0]+padding[2]-pool_size[0])/strides[0])+1
               out_width = floor((width+padding[1]+padding[3]-pool_size[1])/strides[1])+1

           where padding will be an expanded array based on number of values passed as::
               one int : all sides same padding used.
               two int : bottom, right use same as top and left.
               four int: padding width in the order of (top, left, bottom, right).

           When `ceil_mode` is `True`, ceil will be used instead of floor in this
           equation.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(2)
.add_type_rel("MaxPool2D", Pool2DRel<MaxPool2DAttrs>);


// AvgPool2D
Expr MakeAvgPool2D(Expr data,
                   Array<IndexExpr> pool_size,
                   Array<IndexExpr> strides,
                   Array<IndexExpr> padding,
                   std::string layout,
                   bool ceil_mode,
                   bool count_include_pad) {
  auto attrs = make_node<AvgPool2DAttrs>();
  attrs->pool_size = std::move(pool_size);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->layout = std::move(layout);
  attrs->ceil_mode = ceil_mode;
  attrs->count_include_pad = count_include_pad;
  static const Op& op = Op::Get("nn.avg_pool2d");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.nn._make.avg_pool2d")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 7>(MakeAvgPool2D, args, rv);
  });


RELAY_REGISTER_OP("nn.avg_pool2d")
.describe(R"code(
Average pooling operation for one dimensional data.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, channels, height, width) if `layout` is `NCHW`.
- **out**: This depends on the `layout` parameter. Output is 4D array of shape
           (batch_size, channels, out_height, out_width)  if `layout` is `NCHW`.
           out_height and out_width are calculated as::

               out_height = floor((height+padding[0]+padding[2]-pool_size[0])/strides[0])+1
               out_width = floor((width+padding[1]+padding[3]-pool_size[1])/strides[1])+1

           where padding will be an expanded array based on number of values passed as::
               one int : all sides same padding used.
               two int : bottom, right use same as top and left.
               four int: padding width in the order of (top, left, bottom, right).

           When `ceil_mode` is `True`, ceil will be used instead of floor in this
           equation.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(2)
.add_type_rel("AvgPool2D", Pool2DRel<AvgPool2DAttrs>);

// Global Pool
TVM_REGISTER_NODE_TYPE(GlobalPool2DAttrs);

bool GlobalPool2DRel(const Array<Type>& types,
                     int num_inputs,
                     const Attrs& attrs,
                     const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();

  CHECK(data != nullptr);
  const auto dshape = data->shape;
  CHECK_NE(dshape.size(), 0);
  CHECK_GE(dshape.size(), 2U)
      << "Pool2D only support input >= 2-D: input must have height and width";
  const auto param = attrs.as<GlobalPool2DAttrs>();
  CHECK(param != nullptr);

  Layout layout(param->layout);
  CHECK(layout.contains('H') && layout.contains('W') &&
        !layout.contains('h') && !layout.contains('w'))
    << "Invalid layout " << layout
    << ". Pool2D layout must have H and W, which cannot be split";

  const auto hidx = layout.indexof('H');
  const auto widx = layout.indexof('W');
  std::vector<IndexExpr> oshape({dshape[0], dshape[1], dshape[2], dshape[3]});
  oshape[hidx] = oshape[widx] = 1;

  // assign output type
  reporter->Assign(types[1], TensorTypeNode::make(oshape, data->dtype));
  return true;
}

Expr MakeGlobalAvgPool2D(Expr data,
                         std::string layout) {
  auto attrs = make_node<GlobalPool2DAttrs>();
  attrs->layout = std::move(layout);
  static const Op& op = Op::Get("nn.global_avg_pool2d");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.nn._make.global_avg_pool2d")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 2>(MakeGlobalAvgPool2D, args, rv);
  });

// GlobalAvgPool
RELAY_REGISTER_OP("nn.global_avg_pool2d")
.describe(R"code(Global average pooling operation for 2D data.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, channels, height, width) if `layout` is `NCHW`.
- **out**: This depends on the `layout` parameter. Output is 4D array of shape
           (batch_size, channels, 1, 1)  if `layout` is `NCHW`.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(2)
.add_type_rel("GlobalAvgPool2D", GlobalPool2DRel);

// GlobalMaxPool
Expr MakeGlobalMaxPool2D(Expr data,
                         std::string layout) {
  auto attrs = make_node<GlobalPool2DAttrs>();
  attrs->layout = std::move(layout);
  static const Op& op = Op::Get("nn.global_max_pool2d");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.nn._make.global_max_pool2d")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 2>(MakeGlobalMaxPool2D, args, rv);
  });


RELAY_REGISTER_OP("nn.global_max_pool2d")
.describe(R"code(Global max pooling operation for 2D data.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, channels, height, width) if `layout` is `NCHW`.
- **out**: This depends on the `layout` parameter. Output is 4D array of shape
           (batch_size, channels, 1, 1)  if `layout` is `NCHW`.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(2)
.add_type_rel("GlobalMaxPool2D", GlobalPool2DRel);

}  // namespace relay
}  // namespace tvm
