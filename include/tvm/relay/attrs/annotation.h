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
 * \file tvm/relay/attrs/annotation.h
 * \brief Attribute for annotation operators.
 */
#ifndef TVM_RELAY_ATTRS_ANNOTATION_H_
#define TVM_RELAY_ATTRS_ANNOTATION_H_

#include <tvm/ir/attrs.h>

#include <string>

namespace tvm {
namespace relay {

/*!
 * \brief Attributes for the "on_device" operator.
 *
 * The relay call
 * \code
 *   on_device(expr, device_type=2)
 * \endcode
 * denotes that the result of \p expr should be stored on the device with \p DLDeviceType 2
 * (i.e. \p kDLCuda). Semantically the operator is the identity function.
 *
 * See also FunctionOnDeviceAttrs in include/relay/attrs/function.h for the function-level
 * companion.
 */
struct OnDeviceAttrs : public tvm::AttrsNode<OnDeviceAttrs> {
  // TODO(mbs): Replace device types with TargetDevice.
  /*! \brief Device type on which argument expression should be evaluated. */
  int device_type = kInvalidDeviceType;
  /*!
   * \brief If true, the result device must also be \p device_type and device planning should
   * not insert any "device_copy" calls to respect this annotation.
   *
   * This is used by the device planning pass itself when annotating the planned program.
   */
  bool is_fixed = false;

  TVM_DECLARE_ATTRS(OnDeviceAttrs, "relay.attrs.OnDeviceAttrs") {
    TVM_ATTR_FIELD(device_type)
        .describe("The type of the virtual device which should hold the expression result.")
        .set_default(0);
    TVM_ATTR_FIELD(is_fixed)
        .describe("If true, do not insert a \"device_copy\" call to respect this annotation.")
        .set_default(false);
  }
};

/*!
 * \brief Annotate an expression to be cast into specific data type.
 */
struct CastHintAttrs : public tvm::AttrsNode<CastHintAttrs> {
  DataType dtype;

  TVM_DECLARE_ATTRS(CastHintAttrs, "relay.attrs.CastHintAttrs") {
    TVM_ATTR_FIELD(dtype).describe("The data type denoted to be cast.");
  }
};

/*!
 * \brief Options for the operators used to annotate a compiler.
 */
struct CompilerAttrs : public tvm::AttrsNode<CompilerAttrs> {
  /*! \brief A 3rd party compiler for code generation. */
  std::string compiler;

  TVM_DECLARE_ATTRS(CompilerAttrs, "relay.attrs.CompilerAttrs") {
    TVM_ATTR_FIELD(compiler).describe("A 3rd party compiler used for code generation.");
  }
};

/*!
 * \brief Metadata for calls to TIR functions, useful for program analysis crossing Relay and TIR.
 */
struct TIRCallAttrs : public tvm::AttrsNode<TIRCallAttrs> {
  /*! \brief The metadata attached to the call node. */
  Map<String, ObjectRef> metadata;

  TVM_DECLARE_ATTRS(TIRCallAttrs, "relay.attrs.TIRCallAttrs") {
    TVM_ATTR_FIELD(metadata).describe("Metadata attached to the TIR function call.");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_ANNOTATION_H_
