#include <tvm/data_layout.h>
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/bitserial.h>

namespace tvm {
namespace relay {

// relay.nn.bitpack
TVM_REGISTER_NODE_TYPE(BitPackAttrs);

bool BitPackRel(const Array<Type>& types,
                int num_inputs,
                const Attrs& attrs,
                const TypeReporter& reporter) {
  const BitPackAttrs* param = attrs.as<BitPackAttrs>();
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  CHECK(data);
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
        out_shape.push_back(data->shape[i] / pack_bits);
      } else {
        out_shape.push_back(data->shape[i]);
      }
    } else if (i == pack_axis) {
      out_shape.push_back(data->shape[i] / pack_bits);
    } else {
      out_shape.push_back(data->shape[i]);
    }
  }
  reporter->Assign(types[1], TensorTypeNode::make(out_shape, pack_type));
  return true;
}


Expr MakeBitPack(Expr data,
                 int bits,
                 int pack_axis,
                 int bit_axis,
                 DataType pack_type,
                 std::string name) {
  auto attrs = make_node<BitPackAttrs>();
  attrs->bits = bits;
  attrs->pack_axis = pack_axis;
  attrs->bit_axis = bit_axis;
  attrs->pack_type = pack_type;
  attrs->name = name;
  static const Op& op = Op::Get("nn.bitpack");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.nn._make.bitpack")
.set_body_typed(MakeBitPack);

RELAY_REGISTER_OP("nn.bitpack")
.describe(R"code(Bitpack layer that prepares data for bitserial operations.

This layer backs the bits of an input into a single datatype, allowing 
efficient implementation of bitserial operations.

- **data**: Input tensor of any shape, dimension that is to be
            packed must be divisible by number of bits.
- **out**:  Packed tensor with shape appropriately compressed. 
)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_attrs_type_key("relay.attrs.BitPackAttrs")
.add_argument("data", "Tensor", "Input data.")
.set_support_level(2)
.add_type_rel("BitPack", BitPackRel);


bool BinaryConv2DRel(const Array<Type>& types,
                     int num_inputs,
                     const Attrs& attrs,
                     const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr) return false;
  if (weight == nullptr) return false;

  const BinaryConv2DAttrs* param = attrs.as<BinaryConv2DAttrs>();
  CHECK(param != nullptr);

  static const Layout kNCHW("NCHW");

  const Layout in_layout(param->data_layout);
  const auto trans_in_layout = BijectiveLayoutNode::make(in_layout, kNCHW);
  Array<IndexExpr> dshape_nchw = trans_in_layout.ForwardShape(data->shape); 
  CHECK(param->channels.defined());
  Array<IndexExpr> oshape({dshape_nchw[0], param->channels, 0, 0});
  oshape.Set(2, (dshape_nchw[2] + param->padding[0] * 2 - param->kernel_size[0]) / param->strides[0] + 1);
  oshape.Set(3, (dshape_nchw[3] + param->padding[1] * 2 - param->kernel_size[1]) / param->strides[1] + 1);
  DataType out_dtype = param->out_dtype;
  oshape = trans_in_layout.BackwardShape(oshape);
  // assign output type
  reporter->Assign(types[2], TensorTypeNode::make(oshape, out_dtype));
  return true;
}

// Positional relay function to create binaryconv2d operator
// used by frontend FFI.
Expr MakeBinaryConv2D(Expr data,
                      Expr weight,
                      Array<IndexExpr> strides,
                      Array<IndexExpr> padding,
                      IndexExpr channels,
                      Array<IndexExpr> kernel_size,
                      int activation_bits,
                      int weight_bits,
                      std::string data_layout,
                      DataType pack_dtype,
                      DataType out_dtype,
                      bool unipolar) {
  auto attrs = make_node<BinaryConv2DAttrs>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->channels = std::move(channels);
  attrs->kernel_size = std::move(kernel_size);
  attrs->activation_bits = activation_bits;
  attrs->weight_bits = weight_bits;
  attrs->data_layout = std::move(data_layout);
  attrs->pack_dtype = std::move(pack_dtype);
  attrs->out_dtype = std::move(out_dtype);
  attrs->unipolar = unipolar;
  static const Op& op = Op::Get("nn.bitserial_conv2d");
  return CallNode::make(op, {data, weight}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.nn._make.bitserial_conv2d")
.set_body_typed(MakeBinaryConv2D);

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
.set_attrs_type_key("relay.attrs.BinaryConv2DAttrs")
.set_num_inputs(2)
.add_argument("data", "Tensor", "The input tensor.")
.add_argument("weight", "Tensor", "The weight tensor.")
.set_support_level(2)
.add_type_rel("BinaryConv2D", BinaryConv2DRel);

}  // namespace relay
}  // namespace tvm
