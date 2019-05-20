/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

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
.set_body_typed(AutoInlineElemWise);


TVM_REGISTER_API("schedule.AutoInlineInjective")
.set_body_typed(AutoInlineInjective);

TVM_REGISTER_API("schedule.ScheduleOps")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  if (args.size() == 2)
    *ret = ScheduleOps(args[0], args[1], false);
  else
    *ret = ScheduleOps(args[0], args[1], args[2]);
});

#define REGISTER_SCHEDULE_PASS(PassName)                          \
  TVM_REGISTER_API("schedule."#PassName)                          \
  .set_body_typed(PassName);                                     \


REGISTER_SCHEDULE_PASS(InferBound);
REGISTER_SCHEDULE_PASS(CreateReadGraph);
REGISTER_SCHEDULE_PASS(PostDFSOrder);
REGISTER_SCHEDULE_PASS(CreateAttachPath);
REGISTER_SCHEDULE_PASS(ScanGetBody);
REGISTER_SCHEDULE_PASS(ScanFixPointAnalysis);

}  // namespace schedule
}  // namespace tvm
