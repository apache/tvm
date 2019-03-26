/*!
 *  Copyright (c) 2019 by Contributors
 * \file micro_session.cc
 * \brief session to manage multiple micro modules
 */

#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include "micro_session.h"
#include "low_level_device.h"

namespace tvm {
namespace runtime {
// TODO: create Python frontend for this
// initializes micro session and low-level device from Python frontend
TVM_REGISTER_GLOBAL("micro.init")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  // create global micro session
  // setup either host or OpenOCD low-level device
  // setup init stub
  });
}  // namespace runtime
}  // namespace tvm
