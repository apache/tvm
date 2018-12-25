/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/attrs/debug.h
 * \brief Auxiliary attributes for debug operators.
 */
#ifndef TVM_RELAY_ATTRS_DEBUG_H_
#define TVM_RELAY_ATTRS_DEBUG_H_

#include <tvm/attrs.h>
#include <string>

namespace tvm {
namespace relay {

/*!
 * \brief Options for the debug operators.
 */
struct DebugAttrs : public tvm::AttrsNode<DebugAttrs> {
  EnvFunc debug_func;

  TVM_DECLARE_ATTRS(DebugAttrs, "relay.attrs.DebugAttrs") {
    TVM_ATTR_FIELD(debug_func)
        .describe("The function to use when debugging.");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_DEBUG_H_
