/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule.cc
 */
#include <tvm/schedule.h>

namespace tvm {

Schedule::Schedule(Operation op, std::string scope) {
  auto n = std::make_shared<ScheduleNode>();
  n->op = op;
  n->scope = scope;
  node_ = n;
}

TVM_REGISTER_NODE_TYPE(AttachSpecNode);
TVM_REGISTER_NODE_TYPE(ScheduleNode);

}  // namespace tvm
