/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/attrs/device_copy.h
 * \brief Attribute for the device copy operator.
 */
#ifndef TVM_RELAY_ATTRS_DEVICE_COPY_H_
#define TVM_RELAY_ATTRS_DEVICE_COPY_H_

#include <tvm/attrs.h>
#include <string>

namespace tvm {
namespace relay {

/*!
 * \brief Options for the device copy operators.
 */
struct DeviceCopyAttrs : public tvm::AttrsNode<DeviceCopyAttrs> {
  int dst_dev_type;
  int src_dev_type;

  TVM_DECLARE_ATTRS(DeviceCopyAttrs, "relay.attrs.DeviceCopyAttrs") {
    TVM_ATTR_FIELD(src_dev_type)
      .describe(
         "The virutal device/context type where the op copies data from.")
      .set_default(0);
    TVM_ATTR_FIELD(dst_dev_type)
      .describe(
         "The virutal device/context type where the op copies data to.")
      .set_default(0);
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_DEVICE_COPY_H_
