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

}  // namespace relay
}  // namespace tvm
