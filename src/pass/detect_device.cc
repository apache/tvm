/*!
 *  Copyright (c) 2018 by Contributors
 * \file detect_device.cc
 */

#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include "../pass/ir_util.h"

namespace tvm {
namespace ir {
Stmt DecorateDeviceScope(Stmt stmt) {
  Stmt body = AttrStmt::make(make_zero(Int(32)),
                             ir::attr::device_scope,
                             0,
                             stmt);
  return body;
}

}  // namespace ir
}  // namespace tvm
