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
 * \file src/runtime/contrib/clml/clml_utils.h
 * \brief Utils and common functions for the interface.
 */

#ifndef TVM_RUNTIME_CONTRIB_CLML_CLML_UTILS_H_
#define TVM_RUNTIME_CONTRIB_CLML_CLML_UTILS_H_

#include <memory>
#include <string>
#include <vector>

#include "../json/json_node.h"

#ifdef __ANDROID__
#include <CL/opencl.h>

#include "CL/cl.h"
#include "CL/cl_qcom_ml_ops.h"
#endif

namespace tvm {
namespace runtime {
namespace contrib {

using JSONGraphNode = tvm::runtime::json::JSONGraphNode;

#ifdef __ANDROID__
std::vector<cl_uint> GetVectorValues(const std::vector<std::string>& val);
#endif

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_CLML_CLML_UTILS_H_
