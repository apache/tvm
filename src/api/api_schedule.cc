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
.set_body([](TVMArgs args, TVMRetValue* ret) {
    AutoInlineElemWise(args[0]);
  });


TVM_REGISTER_API("schedule.AutoInlineInjective")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    AutoInlineInjective(args[0]);
  });

TVM_REGISTER_API("schedule.ScheduleOps")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  if (args.size() == 2)
    *ret = ScheduleOps(args[0], args[1], false);
  else
    *ret = ScheduleOps(args[0], args[1], args[2]);
});

#define REGISTER_SCHEDULE_PASS1(PassName)                         \
  TVM_REGISTER_API("schedule."#PassName)                          \
  .set_body([](TVMArgs args,  TVMRetValue *ret) {                 \
      *ret = PassName(args[0]);                                   \
    })                                                            \

#define REGISTER_SCHEDULE_PASS2(PassName)                         \
  TVM_REGISTER_API("schedule."#PassName)                          \
  .set_body([](TVMArgs args,  TVMRetValue *ret) {                 \
      *ret = PassName(args[0], args[1]);                          \
    })                                                            \


REGISTER_SCHEDULE_PASS1(InferBound);
REGISTER_SCHEDULE_PASS1(CreateReadGraph);
REGISTER_SCHEDULE_PASS2(PostDFSOrder);
REGISTER_SCHEDULE_PASS1(CreateAttachPath);
REGISTER_SCHEDULE_PASS1(ScanGetBody);
REGISTER_SCHEDULE_PASS1(ScanFixPointAnalysis);

}  // namespace schedule
}  // namespace tvm
