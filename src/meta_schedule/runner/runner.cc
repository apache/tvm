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
#include <tvm/runtime/registry.h>

#include "../utils.h"

namespace tvm {
namespace meta_schedule {

RunnerResult::RunnerResult(Optional<Array<FloatImm>> run_secs, Optional<String> error_msg) {
  ObjectPtr<RunnerResultNode> n = make_object<RunnerResultNode>();
  n->run_secs = run_secs;
  n->error_msg = error_msg;
  this->data_ = n;
}

TVM_REGISTER_NODE_TYPE(RunnerResultNode);

TVM_REGISTER_GLOBAL("meta_schedule.RunnerResult")
    .set_body_typed([](Array<FloatImm> run_secs, Optional<String> error_msg) -> RunnerResult {
      return RunnerResult(run_secs, error_msg);
    });

}  // namespace meta_schedule
}  // namespace tvm
