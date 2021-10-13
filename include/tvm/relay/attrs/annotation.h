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
 * \brief Attributes for the "on_device" special operator.
 *
 * The Relay call (aka 'annotation'):
 * \code
 *   on_device(sub_expr, device_type=2)
 * \endcode
 * constrains \p sub_expr to execute and store its result on a device with \p DLDeviceType \p 2
 * (i.e. a \p kDLCuda device). However the annotation itself may appear in an expression to be
 * executed and stored on a different device. If so the compiler will automatically insert a
 * "device_copy" call to mediate the transition between devices.
 *
 * E.g.: Assuming %x and %y reside on the GPU and %z on the CPU then:
 * \code
 *   multiply(on_device(add(%x, %y), device_type=2), %z)
 * \endcode
 * indicates the \p add should execute on the GPU but the \p multiply should execute on the CPU.
 * The compiler will rewrite this to:
 * \code
 *   multiply(device_copy(add(%x, %y), src_dev_type=2, dst_dev_type=1), %z)
 * \endcode
 *
 * The Relay call
 * \code
 *   on_device(sub_expr, device_type=2, is_fixed=True)
 * \endcode
 * is similar to the above, however the annotation itself must appear in an expression on the
 * same device. The compiler will check the devices are consistent, and will not insert any
 * "device_copy" call. This form of annotation shouldn't be necessary in user programs. However
 * it is needed by the \p PlanDevices pass to fully specify the results of device planning so that
 * the pass is idempotent.
 *
 * E.g.: The following program is equivalent to the above:
 * \code
 *   let %a = on_device(add(%x, %y), device_type=2, is_fixed=True)
 *   multiply(device_copy(%a, src_dev_type=2, dst_dev_type=1), %z)
 * \endcode
 * The "on_device" annotation with \p is_fixed=True indicates unambiguously that \p %a is stored
 * on the GPU.
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


}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_ANNOTATION_H_
