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
 * \file tvm/relay/attrs/debug.h
 * \brief Auxiliary attributes for debug operators.
 */
#ifndef TVM_RELAY_ATTRS_DEBUG_H_
#define TVM_RELAY_ATTRS_DEBUG_H_

#include <tvm/ir/attrs.h>
#include <tvm/ir/env_func.h>

#include <string>

namespace tvm {
namespace relay {

/*!
 * \brief Options for the debug operators.
 */
struct DebugAttrs : public tvm::AttrsNode<DebugAttrs> {
  EnvFunc debug_func;

  TVM_DECLARE_ATTRS(DebugAttrs, "relay.attrs.DebugAttrs") {
    TVM_ATTR_FIELD(debug_func).describe("The function to use when debugging.");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_DEBUG_H_
