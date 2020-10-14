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
 * \file tvm/relay/quantize.h
 * \brief Header of definitions for quantization
 */
#ifndef TVM_RELAY_QUANTIZE_QUANTIZE_H_
#define TVM_RELAY_QUANTIZE_QUANTIZE_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>

#include <string>

#include "../transforms/pattern_utils.h"

namespace tvm {
namespace relay {
namespace quantize {

/*! \brief Kind of annotate field */
enum QAnnotateKind : int { kQIdentity = 0, kQInput = 1, kQWeight = 2, kQActivation = 3 };

/*! \brief Attribute for simulated quantize operator */
struct SimulatedQuantizeAttrs : public tvm::AttrsNode<SimulatedQuantizeAttrs> {
  int kind;
  bool sign;
  std::string rounding;

  TVM_DECLARE_ATTRS(SimulatedQuantizeAttrs, "relay.attrs.SimulatedQuantizeAttrs") {
    TVM_ATTR_FIELD(kind).describe("kind of field, hint for nbit/dtype configuration.");
    TVM_ATTR_FIELD(sign).set_default(true).describe("whether to use signed data type.");
    TVM_ATTR_FIELD(rounding).set_default("round").describe(
        "rounding mode. Can be 'floor', 'ceil', 'round'");
  }
};

class QConfig;
/*!
 * \brief Container for build configuration options
 */
class QConfigNode : public Object {
 public:
  int nbit_input = 8;
  int nbit_weight = 8;
  int nbit_activation = 32;
  DataType dtype_input = DataType::Int(8);
  DataType dtype_weight = DataType::Int(8);
  DataType dtype_activation = DataType::Int(32);
  std::string calibrate_mode = "global_scale";
  double global_scale = 8.0;
  std::string weight_scale = "power2";
  bool skip_dense_layer = true;
  Array<Expr> skip_conv_layers = Array<Expr>(ObjectPtr<Object>(nullptr));
  bool do_simulation = false;
  bool round_for_shift = true;
  Array<Expr> debug_enabled_ops = Array<Expr>(ObjectPtr<Object>(nullptr));
  std::string rounding = "UPWARD";
  int calibrate_chunk_by = -1;
  std::string partition_conversions = "disabled";

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("nbit_input", &nbit_input);
    v->Visit("nbit_weight", &nbit_weight);
    v->Visit("nbit_activation", &nbit_activation);
    v->Visit("dtype_input", &dtype_input);
    v->Visit("dtype_weight", &dtype_weight);
    v->Visit("dtype_activation", &dtype_activation);
    v->Visit("calibrate_mode", &calibrate_mode);
    v->Visit("global_scale", &global_scale);
    v->Visit("weight_scale", &weight_scale);
    v->Visit("skip_dense_layer", &skip_dense_layer);
    v->Visit("skip_conv_layers", &skip_conv_layers);
    v->Visit("do_simulation", &do_simulation);
    v->Visit("round_for_shift", &round_for_shift);
    v->Visit("debug_enabled_ops", &debug_enabled_ops);
    v->Visit("rounding", &rounding);
    v->Visit("calibrate_chunk_by", &calibrate_chunk_by);
    v->Visit("partition_conversions", &partition_conversions);
  }

  static constexpr const char* _type_key = "relay.quantize.QConfig";
  TVM_DECLARE_FINAL_OBJECT_INFO(QConfigNode, Object);
};

/*!
 * \brief Container for build configuration options
 */
class QConfig : public ObjectRef {
 public:
  QConfig() {}
  explicit QConfig(ObjectPtr<Object> n) : ObjectRef(n) {}

  const QConfigNode* operator->() const { return static_cast<const QConfigNode*>(get()); }

  QConfigNode* operator->() { return static_cast<QConfigNode*>(get_mutable()); }

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
  static QConfig& Current();

  using ContainerType = QConfigNode;
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
  explicit QConfigContext(const QConfig& qconfig) { QConfig::EnterQConfigScope(qconfig); }

  /*! \brief Destructor. Pops the context off the thread local stack. */
  ~QConfigContext() { QConfig::ExitQConfigScope(); }
};

}  // namespace quantize
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_QUANTIZE_QUANTIZE_H_
