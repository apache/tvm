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
 * \file data_type.cc
 * \brief Data-type handling
 */
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {
TVM_REGISTER_GLOBAL("runtime.String2DLDataType").set_body_typed(String2DLDataType);
TVM_REGISTER_GLOBAL("runtime.DLDataType2String").set_body_typed(DLDataType2String);

}  // namespace runtime
}  // namespace tvm
