/*!
 *  Copyright (c) 2018 by Contributors
 * \file detect_device.cc
 */

#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include "../pass/ir_util.h"

namespace tvm {
namespace ir {

class DetectDevice : public IRMutator {
 public:
  DetectDevice() {}
  Stmt Detect(Stmt stmt) {
    Stmt body = AttrStmt::make(make_zero(Int(32)),
                               ir::attr::device_scope,
                               0,
                               stmt);
    return body;
  }
};

Stmt MarkDevice(Stmt stmt) {
  return DetectDevice().Detect(stmt);
}

}  // namespace ir
}  // namespace tvm
