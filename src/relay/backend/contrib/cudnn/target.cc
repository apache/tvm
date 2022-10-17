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
 * \file src/relay/backend/contrib/cudnn/target.cc
 * \brief Registers the "cudnn" external codegen TargetKind.
 */

#include <tvm/target/target.h>

namespace tvm {
namespace relay {
namespace contrib {

/*!
 * \brief This external codegen target can use the CuDNN library linked into the TVM runtime.
 *  - Patterns and custom compiler: python/tvm/relay/op/contrib/cudnn.py
 *  - Custom schedules: python/tvm/contrib/cudnn.py
 *  - Runtime: src/runtime/contrib/cudnn/ *.cc
 */
TVM_REGISTER_TARGET_KIND("cudnn", kDLCUDA)
    .set_attr<Bool>(tvm::attr::kIsExternalCodegen, Bool(true));

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
