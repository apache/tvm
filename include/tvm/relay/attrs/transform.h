/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/attrs/nn.h
 * \brief Auxiliary attributes for nn operators.
 */
#ifndef TVM_RELAY_ATTRS_NN_H_
#define TVM_RELAY_ATTRS_NN_H_

#include <tvm/attrs.h>
#include <string>

namespace tvm {
namespace relay {

/*! \brief Attributes used in expand_dims operators */
struct ExpandDimsAttrs : public tvm::AttrsNode<ExpandDimsAttrs> {
  // TODO(Junru): docstring
  int axis;
  int num_newaxis;

  TVM_DECLARE_ATTRS(ExpandDimsAttrs, "relay.attrs.ExpandDimsAttrs") {
    TVM_ATTR_FIELD(axis)
        .describe("Position in the expanded axes where the new axis is placed.")
        .set_default(0);
    TVM_ATTR_FIELD(num_newaxis)
        .describe("Number of new axis to be inserted.")
        .set_default(1);
  }
};  // struct ExpandDimsAttrs

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_NN_H_
