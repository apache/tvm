/*!
 *  Copyright (c) 2017 by Contributors
 *  Implementation of API functions related to schedule pass.
 * \file api_schedule.cc
 */
#include <tvm/expr.h>
#include <tvm/tensor.h>
#include <tvm/schedule.h>
#include <tvm/schedule_pass.h>
#include <tvm/api_registry.h>
#include "../schedule/graph.h"

namespace tvm {
namespace schedule {

TVM_REGISTER_API("schedule.AutoInlineElemWise")
.set_body_simple(AutoInlineElemWise);


TVM_REGISTER_API("schedule.AutoInlineInjective")
.set_body_simple(AutoInlineInjective);

TVM_REGISTER_API("schedule.ScheduleOps")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  if (args.size() == 2)
    *ret = ScheduleOps(args[0], args[1], false);
  else
    *ret = ScheduleOps(args[0], args[1], args[2]);
});

#define REGISTER_SCHEDULE_PASS(PassName)                          \
  TVM_REGISTER_API("schedule."#PassName)                          \
  .set_body_simple(PassName);                                     \


REGISTER_SCHEDULE_PASS(InferBound);
REGISTER_SCHEDULE_PASS(CreateReadGraph);
REGISTER_SCHEDULE_PASS(PostDFSOrder);
REGISTER_SCHEDULE_PASS(CreateAttachPath);
REGISTER_SCHEDULE_PASS(ScanGetBody);
REGISTER_SCHEDULE_PASS(ScanFixPointAnalysis);

}  // namespace schedule
}  // namespace tvm
