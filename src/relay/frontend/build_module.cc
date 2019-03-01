/*!
 *  Copyright (c) 2018 by Contributors
 * \file relay/backend/build_module.cc
 * \brief Internal compilation engine handle function cache.
 *  and interface to low level code generation.
 */
#include "build_module.h"

namespace tvm {
namespace relay {
namespace frontend {

TVM_REGISTER_GLOBAL("relay.build_module._BuildModule").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = RelayBuildCreate();
});

}  // namespace frontend
}  // namespace relay
}  // namespace tvm
