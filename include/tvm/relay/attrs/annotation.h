/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/attrs/annotation.h
 * \brief Attribute for annotation operators.
 */
#ifndef TVM_RELAY_ATTRS_ANNOTATION_H_
#define TVM_RELAY_ATTRS_ANNOTATION_H_

#include <tvm/attrs.h>
#include <string>

namespace tvm {
namespace relay {

/*!
 * \brief Options for the device annotation operators.
 */
struct OnDeviceAttrs : public tvm::AttrsNode<OnDeviceAttrs> {
  int device_type;

  TVM_DECLARE_ATTRS(OnDeviceAttrs, "relay.attrs.OnDeviceAttrs") {
    TVM_ATTR_FIELD(device_type)
      .describe(
         "The virutal device/context type that an expression is annotated with.")
      .set_default(0);
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_ANNOTATION_H_
