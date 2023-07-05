/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file bitserial.cc
 * \brief Property def of bitserial operators.
 */

#include <tvm/relay/attrs/bitserial.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>

#include "../../transforms/infer_layout_utils.h"
#include "../op_common.h"

namespace tvm {
namespace relay {

// relay.nn.bitpack
TVM_REGISTER_NODE_TYPE(BitPackAttrs);

template <typename T>
InferCorrectLayoutOutput BinaryConv2DInferCorrectLayout(
    const Attrs& attrs, const Array<Layout>& new_in_layouts, const Array<Layout>& old_in_layouts,
    const Array<tvm::relay::Type>& old_in_types) {
  const T* params = attrs.as<T>();

  // We always make other operators to fit the layouts of convolution layers
  // So this inference ignores all inputs
  return InferCorrectLayoutOutput({params->data_layout, params->kernel_layout},
                                  {params->data_layout}, attrs);
}

bool BitPackRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                const TypeReporter& reporter) {
  const BitPackAttrs* param = attrs.as<BitPackAttrs>();
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  ICHECK(data);
  int ndim = data->shape.size();
  int bits = param->bits;
  int pack_axis = param->pack_axis;
  int bit_axis = param->bit_axis;
  DataType pack_type = param->pack_type;

  int pack_bits = pack_type.bits();

  Array<IndexExpr> out_shape;
  for (int i = 0; i < ndim; ++i) {
    if (i == bit_axis) {
      out_shape.push_back(bits);
      if (i == pack_axis) {
        out_shape.push_back(indexdiv(data->shape[i], pack_bits));
      } else {
        out_shape.push_back(data->shape[i]);
      }
    } else if (i == pack_axis) {
      out_shape.push_back(indexdiv(data->shape[i], pack_bits));
    } else {
      out_shape.push_back(data->shape[i]);
    }
  }
  // Add extra check for last axis expansion.
  if (bit_axis == ndim) {
    out_shape.push_back(bits);
  }

  reporter->Assign(types[1], TensorType(out_shape, pack_type));
  return true;
}

Expr MakeBitPack(Expr data, int bits, int pack_axis, int bit_axis, DataType pack_type,
                 String name) {
  auto attrs = make_object<BitPackAttrs>();
  attrs->bits = bits;
  attrs->pack_axis = pack_axis;
  attrs->bit_axis = bit_axis;
  attrs->pack_type = pack_type;
  attrs->name = name;
  static const Op& op = Op::Get("nn.bitpack");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.bitpack").set_body_typed(MakeBitPack);

RELAY_REGISTER_OP("nn.bitpack")
    .describe(R"code(Bitpack layer that prepares data for bitserial operations.

This layer backs the bits of an input into a single datatype, allowing
efficient implementation of bitserial operations.

- **data**: Input tensor of any shape, dimension that is to be
            packed must be divisible by number of bits.
- **out**:  Packed tensor with shape appropriately compressed.
)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type<BitPackAttrs>()
    .add_argument("data", "Tensor", "Input data.")
    .set_support_level(2)
    .add_type_rel("BitPack", BitPackRel)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// relay.nn.bitserial_conv2d
TVM_REGISTER_NODE_TYPE(BinaryConv2DAttrs);

bool BinaryConv2DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const BinaryConv2DAttrs* param = attrs.as<BinaryConv2DAttrs>();
  ICHECK(param != nullptr);

  static const Layout kNCHW("NCHW");

  const Layout in_layout(param->data_layout);
  const auto trans_in_layout = tir::BijectiveLayout(in_layout, kNCHW);
  Array<IndexExpr> dshape_nchw = trans_in_layout.ForwardShape(data->shape);
  ICHECK(param->channels.defined());
  ICHECK(param->kernel_size.defined());
  Array<IndexExpr> oshape({dshape_nchw[0], param->channels, 0, 0});
  IndexExpr pad_h, pad_w;
  GetPaddingHeightWidth(param->padding, &pad_h, &pad_w);
  oshape.Set(2, (dshape_nchw[2] + pad_h - param->kernel_size[0]) / param->strides[0] + 1);
  oshape.Set(3, (dshape_nchw[3] + pad_w - param->kernel_size[1]) / param->strides[1] + 1);
  DataType out_dtype = param->out_dtype;
  oshape = trans_in_layout.BackwardShape(oshape);
  // assign output type
  reporter->Assign(types[2], TensorType(oshape, out_dtype));
  return true;
}

// Positional relay function to create binaryconv2d operator
// used by frontend FFI.
Expr MakeBinaryConv2D(Expr data, Expr weight, Array<IndexExpr> strides, Array<IndexExpr> padding,
                      IndexExpr channels, Array<IndexExpr> kernel_size, int activation_bits,
                      int weight_bits, String data_layout, String kernel_layout,
                      DataType pack_dtype, DataType out_dtype, bool unipolar) {
  auto attrs = make_object<BinaryConv2DAttrs>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->channels = std::move(channels);
  attrs->kernel_size = std::move(kernel_size);
  attrs->activation_bits = activation_bits;
  attrs->weight_bits = weight_bits;
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->pack_dtype = std::move(pack_dtype);
  attrs->out_dtype = std::move(out_dtype);
  attrs->unipolar = unipolar;
  static const Op& op = Op::Get("nn.bitserial_conv2d");
  return Call(op, {data, weight}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.bitserial_conv2d").set_body_typed(MakeBinaryConv2D);

RELAY_REGISTER_OP("nn.bitserial_conv2d")
    .describe(R"code(2D convolution using packed binary computation.

This layer creates a convolution kernel that is convolved with the
layer input using bitserial computation. This enables faster processing
on some platforms.

- **data**:   4D input tensor that can be either `NCHW` or `NHWC` layout.

- **weight**: Weight tensor that can either be prepacked (5D) or unpacked (4D).
              When data is NCHW, weight is expected to be OIHW or OIHWi.
              When data is NHWC weight is expected to be HWIO or HWIOi.

- **out**:    Output with same layout as input.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<BinaryConv2DAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_support_level(2)
    .add_type_rel("BinaryConv2D", BinaryConv2DRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                                   BinaryConv2DInferCorrectLayout<BinaryConv2DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

// relay.nn.bitserial_dense
TVM_REGISTER_NODE_TYPE(BinaryDenseAttrs);

bool BinaryDenseRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const BinaryDenseAttrs* param = attrs.as<BinaryDenseAttrs>();
  ICHECK(param != nullptr);

  ICHECK(static_cast<int>(data->shape.size()) != 0);
  ICHECK(param->units.defined());

  Array<tvm::PrimExpr> oshape = data->shape;
  oshape.Set((oshape.size() - 1), param->units);

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }

  // Assign output type.
  reporter->Assign(types[2], TensorType(oshape, out_dtype));
  return true;
}

// Positional relay function to create bitserial dense operator used by frontend FFI.
Expr MakeBinaryDense(Expr data, Expr weight, IndexExpr units, int data_bits, int weight_bits,
                     DataType pack_dtype, DataType out_dtype, bool unipolar) {
  auto attrs = make_object<BinaryDenseAttrs>();
  attrs->units = units;
  attrs->data_bits = data_bits;
  attrs->weight_bits = weight_bits;
  attrs->pack_dtype = pack_dtype;
  attrs->out_dtype = out_dtype;
  attrs->unipolar = unipolar;
  static const Op& op = Op::Get("nn.bitserial_dense");
  return Call(op, {data, weight}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.bitserial_dense").set_body_typed(MakeBinaryDense);

RELAY_REGISTER_OP("nn.bitserial_dense")
    .describe(R"code(Applies a quantized linear transformation: :math:`Y = XW^T`.

- **data**: `(x1, x2, ..., xn, input_dim)`
- **weight**: `(units, input_dim)`
- **out**: `(x1, x2, ..., xn, units)`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<BinaryDenseAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "2D Tensor", "Input data.")
    .add_argument("weight", "2D Tensor", "Weight matrix.")
    .set_support_level(1)
    .add_type_rel("BinaryDense", BinaryDenseRel)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

}  // namespace relay
}  // namespace tvm
