#ifndef TVM_RELAY_ATTRS_BITSERIAL_H_
#define TVM_RELAY_ATTRS_BITSERIAL_H_

#include <tvm/attrs.h>
#include <string>

namespace tvm {
namespace relay {

/*! \brief Attributes used in bitpack operators */
struct BitPackAttrs : public tvm::AttrsNode<BitPackAttrs> {
  int bits;
  int pack_axis;
  int bit_axis;
  DataType pack_type;
  std::string name;

  TVM_DECLARE_ATTRS(BitPackAttrs, "relay.attrs.BitPackAttrs") {
    TVM_ATTR_FIELD(bits).set_default(1).describe("Number of bits to quantize with.");
    TVM_ATTR_FIELD(pack_axis).set_default(1).describe(
        "Axis that should be compressed, typically channels.");
    TVM_ATTR_FIELD(bit_axis).set_default(-1).describe("New axis for packed bits.");
    TVM_ATTR_FIELD(pack_type)
        .set_default(NullValue<DataType>())
        .describe("Type of int to pack bits into.");
    TVM_ATTR_FIELD(name).set_default("BitPack").describe("Name of operation.");
  }
};

/*! \brief Attribues used in bitserial convolution operators */
struct BinaryConv2DAttrs : public tvm::AttrsNode<BinaryConv2DAttrs> {
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  IndexExpr channels;
  Array<IndexExpr> kernel_size;
  int activation_bits;
  int weight_bits;
  std::string data_layout;
  DataType pack_dtype;
  DataType out_dtype;
  bool unipolar;

  TVM_DECLARE_ATTRS(BinaryConv2DAttrs, "relay.attrs.BinaryConv2DAttrs") {
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0}))
        .describe(
            "If padding is non-zero the input is implicitly zero-padded"
            "on both sides for padding number of points.");
    TVM_ATTR_FIELD(kernel_size)
        .set_default(Array<IndexExpr>({3, 3}))
        .describe("Specifies the dimensions of the convolution window.");
    TVM_ATTR_FIELD(channels)
        .set_default(NullValue<IndexExpr>())
        .describe("Number of output channels, needed for shape inference.");
    TVM_ATTR_FIELD(activation_bits)
        .set_default(1)
        .describe("Number of bits activation should be packed with.");
    TVM_ATTR_FIELD(weight_bits)
        .set_default(1)
        .describe("Number of bits kernel should be packed with.");
    TVM_ATTR_FIELD(data_layout)
        .set_default("NCHW")
        .describe("Dimension ordering of input data, can be 'NCHW' or NHWC'.");
    TVM_ATTR_FIELD(pack_dtype)
        .set_default(NullValue<DataType>())
        .describe("Datatype to pack bits into.");
    TVM_ATTR_FIELD(out_dtype).set_default(NullValue<DataType>()).describe("Output datatype.");
    TVM_ATTR_FIELD(unipolar).set_default(true).describe(
        "Whether to use unipolar or bipolar quantization.");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_BITSERIAL_H_
