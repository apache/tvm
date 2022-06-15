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
 * \file tvm/relay/attrs/call.h
 * \brief Attribute for call_lowered operator.
 */
#ifndef TVM_RELAY_ATTRS_CALL_H_
#define TVM_RELAY_ATTRS_CALL_H_

#include <tvm/ir/attrs.h>

#include <string>

namespace tvm {
namespace relay {

/*!
 * \brief Metadata for calls to TIR functions, useful for program analysis crossing Relay and TIR.
 */
struct CallLoweredAttrs : public tvm::AttrsNode<CallLoweredAttrs> {
  /*! \brief Additional metadata attached to the call node. Should be replaced by explict fields. */
  Map<String, ObjectRef> metadata;

  TVM_DECLARE_ATTRS(CallLoweredAttrs, "relay.attrs.CallLoweredAttrs") {
    TVM_ATTR_FIELD(metadata)
        .describe("Metadata attached to the lowered function call.")
        .set_default(Map<String, ObjectRef>());
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_CALL_H_
