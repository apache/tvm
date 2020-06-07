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
 * \file tvm/relay/qnn/transform.h
 *
 * This file implements a pass manager for QNN ops using Relay Pass manager.
 */
#ifndef TVM_RELAY_QNN_TRANSFORM_H_
#define TVM_RELAY_QNN_TRANSFORM_H_

#include <tvm/relay/transform.h>
#include <tvm/runtime/c_runtime_api.h>

namespace tvm {
namespace relay {

using relay::transform::Pass;

namespace qnn {
namespace transform {

/*!
 * \brief Legalizes a QNN expr. Contains specifically two types of Legalizations. First,
 * converts/Lowers an expression containing QNN ops to an expression containing only core Relay ops.
 * Each QNN op is lowered to a sequence of exisiting Relay ops. This is a target-independent pass.
 * One can register the lowering/transformation function for this op using FTVMQnnCanonicalize
 * attr_name for FTVMLegalize op attribute. Second, as opposed to Relay Legalize, this one legalizes
 * only QNN ops. One can register a transformation/legalization function for an op by using the
 * FTVMQnnLegalize attr_name for FTVMLegalize op attribute. The isolation of QNN and Relay Legalize
 * gives us separation of concerns, leading to a better software practice. The legalization can be
 * configured to happen per target.
 *
 * \return The pass.
 */
TVM_DLL Pass Legalize();

}  // namespace transform

}  // namespace qnn
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_QNN_TRANSFORM_H_
