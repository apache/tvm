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
 *  Copyright (c) 2018 by Contributors.
 *
 * \file tvm/hago/quantize.h
 * \brief Header of definitions for quantization
 */
#ifndef TVM_HAGO_QUANTIZE_H_
#define TVM_HAGO_QUANTIZE_H_

#include <tvm/relay/op.h>
#include <tvm/relay/expr.h>
#include <string>

namespace tvm {
namespace hago {

/*! \brief Attribute for simulated quantize operator */
struct SimulatedQuantizeAttrs : public tvm::AttrsNode<SimulatedQuantizeAttrs> {
  DataType in_dtype;
  DataType out_dtype;
  bool sign;
  std::string rounding;

  TVM_DECLARE_ATTRS(SimulatedQuantizeAttrs, "hago.SimulatedQuantizeAttrs") {
    TVM_ATTR_FIELD(in_dtype)
      .describe("in data type");
    TVM_ATTR_FIELD(out_dtype)
      .describe("out data type");
    TVM_ATTR_FIELD(sign).set_default(true)
        .describe("whether to use signed data type.");
    TVM_ATTR_FIELD(rounding).set_default("round")
        .describe("rounding mode. Can be 'floor', 'ceil', 'round'");
  }
};

class QConfig;
/*!
* \brief Container for build configuration options
*/
class QConfigObj : public Object {
 public:
  Array<PrimExpr> skip_conv_layers = Array<PrimExpr>(ObjectPtr<Object>(nullptr));
  std::string threshold_estimate_method = "power_of_two_range";
  double global_scale = 8.0;
  std::string log_file = ".quantize_strategy_search.log";

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("skip_conv_layers", &skip_conv_layers);
    v->Visit("threshold_estimate_method", &threshold_estimate_method);
    v->Visit("global_scale", &global_scale);
    v->Visit("log_file", &log_file);
  }

  static constexpr const char* _type_key = "hago.QConfig";
  TVM_DECLARE_FINAL_OBJECT_INFO(QConfigObj, Object);
};

/*!
* \brief Container for build configuration options
*/
class QConfig : public ObjectRef {
 public:
  QConfig() {}
  explicit QConfig(ObjectPtr<Object> n) : ObjectRef(n) {}

  const QConfigObj* operator->() const {
    return static_cast<const QConfigObj*>(get());
  }

  QConfigObj* operator->() {
    return static_cast<QConfigObj*>(get_mutable());
  }

  /*!
   * \brief Push a new BuildConfig context onto the thread local stack.
   * \param build_config The configuration to set as the current context.
   */
  static void EnterQConfigScope(const QConfig& qconfig);

  /*!
   * \brief Pop a build config off the thread local context stack, restoring the previous
   * configuration as the current context.
   */
  static void ExitQConfigScope();

  /*!
   * \brief Get the current BuildConfig context from thread local storage, or a default
   * configuration if a BuildConfig scope has not been entered.
   * \return The configuration that is the current context.
   */
  static QConfig Current();

  using ContainerType = QConfigObj;
};

/*!
 * \brief RAII container to provide a scoped BuildConfig context. Pushes a configuration onto the
 * context stack when constructed, and pops it when destructed.
 */
struct QConfigContext {
  /*!
   * \brief Enter a new BuildConfig context. The given BuildConfig becomes the new current
   * context. When the BuildConfigContext is destructed, the previous context is restored.
   * \param build_config The BuildConfig to set as the new current context.
   */
  explicit QConfigContext(const QConfig& qconfig) {
    QConfig::EnterQConfigScope(qconfig);
  }

  /*! \brief Destructor. Pops the context off the thread local stack. */
  ~QConfigContext() {
    QConfig::ExitQConfigScope();
  }
};

class OpDesc;
/*!
* \brief Container for build configuration options
*/
class OpDescObj : public Object {
 public:
  Array<relay::Type> in_types;
  Array<relay::Type> out_types;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("in_types", &in_types);
    v->Visit("out_types", &out_types);
  }

  TVM_DLL static OpDesc make(Array<relay::Type> in_types,
                             Array<relay::Type> out_types);

  static constexpr const char* _type_key = "hago.OpDesc";
  TVM_DECLARE_FINAL_OBJECT_INFO(OpDescObj, Object);
};

class OpDesc : public ObjectRef {
 public:
  OpDesc() {}
  explicit OpDesc(ObjectPtr<Object> n) : ObjectRef(n) {}

  const OpDescObj* operator->() const {
    return static_cast<const OpDescObj*>(data_.get());
  }
  OpDescObj* operator->() {
    return static_cast<OpDescObj*>(data_.get());
  }
  using ContainerType = OpDescObj;
};
}  // namespace hago
}  // namespace tvm
#endif  // TVM_HAGO_QUANTIZE_H_
