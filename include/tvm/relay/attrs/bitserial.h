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
    TVM_ATTR_FIELD(bits).set_default(1)
        .describe("Number of bits to quantize with.");
    TVM_ATTR_FIELD(pack_axis).set_default(1)
        .describe("Axis that should be compressed, typically channels.");
    TVM_ATTR_FIELD(bit_axis).set_default(-1)
        .describe("New axis for packed bits.");
    TVM_ATTR_FIELD(pack_type).set_default(NullValue<DataType>())
        .describe("Type of int to pack bits into.");
    TVM_ATTR_FIELD(name).set_default("BitPack")
        .describe("Name of operation.");
  }
};


}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_BITSERIAL_H_
