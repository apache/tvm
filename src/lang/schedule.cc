/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule.cc
 */
#include <tvm/schedule.h>

namespace tvm {

Schedule::Schedule(Tensor tensor, std::string scope) {
  auto n = std::make_shared<ScheduleNode>();
  n->tensor = tensor;
  n->scope = scope;
  node_ = n;
}

TVM_REGISTER_NODE_TYPE(AttachSpecNode);
TVM_REGISTER_NODE_TYPE(ScheduleNode);

}  // namespace tvm
