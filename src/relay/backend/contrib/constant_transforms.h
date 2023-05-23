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
 * \file src/relay/backend/contrib/constant_transforms.h
 * \brief Transforms applied to constant operations during codegen for BYOC backends.
 */

#ifndef TVM_RELAY_BACKEND_CONTRIB_CONSTANT_TRANSFORMS_H_
#define TVM_RELAY_BACKEND_CONTRIB_CONSTANT_TRANSFORMS_H_

#include <tvm/relay/expr.h>

#include <string>

namespace tvm {
namespace relay {
namespace contrib {

/*!
 *\brief Transpose weights from `source_layout` to `target_layout`
 *
 * \param data The constant expression to transpose.
 * \param source_layout The current layout of the constant e.g. "OHWI".
 * \param target_layout The target layout of the constant e.g. "HWIO".
 */
Constant TransposeWeights(const Constant& data, const std::string& source_layout,
                          const std::string& target_layout);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_CONSTANT_TRANSFORMS_H_
