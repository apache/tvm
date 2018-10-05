/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/attrs/transform.h
 * \brief Transform operators.
 */
#ifndef TVM_RELAY_ATTRS_TRANSFORM_H_
#define TVM_RELAY_ATTRS_TRANSFORM_H_

#include <tvm/attrs.h>
#include <string>

namespace tvm {
namespace relay {

/*! \brief Attributes used in expand_dims operators */
struct ExpandDimsAttrs : public tvm::AttrsNode<ExpandDimsAttrs> {
  int axis;
  int num_newaxis;

  TVM_DECLARE_ATTRS(ExpandDimsAttrs, "relay.attrs.ExpandDimsAttrs") {
    TVM_ATTR_FIELD(axis)
        .describe("The axis at which the input array is expanded."
                  "Should lie in range `[-data.ndim - 1, data.ndim]`."
                  "If `axis < 0`, it is the first axis inserted;"
                  "If `axis >= 0`, it is the last axis inserted in Python's negative indexing.");
    TVM_ATTR_FIELD(num_newaxis)
        .describe("Number of axises to be inserted. Should be >= 0.")
        .set_lower_bound(0)
        .set_default(1);
  }
};  // struct ExpandDimsAttrs

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_TRANSFORM_H_
