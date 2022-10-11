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
enum QAnnotateKind : int { kQIdentity = 0, kQInput = 1, kQWeight = 2, kQActivation = 3, kQBias = 4 };

/*! \brief Attribute for simulated quantize operator */
struct SimulatedQuantizeAttrs : public tvm::AttrsNode<SimulatedQuantizeAttrs> {
  int kind;
  std::string rounding;
  bool per_channel;
  bool asymmetric;
  std::string name;

  TVM_DECLARE_ATTRS(SimulatedQuantizeAttrs, "relay.attrs.SimulatedQuantizeAttrs") {
    TVM_ATTR_FIELD(kind).describe("kind of field, hint for nbit/dtype configuration.");
    TVM_ATTR_FIELD(rounding).set_default("round").describe(
        "rounding mode. Can be 'floor', 'ceil', 'round'");
    TVM_ATTR_FIELD(per_channel).set_default(false).describe(
        "whether to use per channel quantize.");
    TVM_ATTR_FIELD(asymmetric).set_default(false).describe(
        "whether it is asmmetric quantization.");
    TVM_ATTR_FIELD(name).set_default("qnode").describe(
        "quantize node's name");
  }
};

class QConfig;
/*!
 * \brief Container for build configuration options
 */
class QConfigNode : public Object {
 public:
  std::string network_name = "Default";
  bool have_prequantized = false;
  int nbit_input = 8;
  int nbit_weight = 8;
  int nbit_activation = 32;
  int nbit_bias = 32;
  DataType dtype_input = DataType::Int(32);
  DataType dtype_weight = DataType::Int(32);
  DataType dtype_activation = DataType::Int(32);
  DataType dtype_bias = DataType::Int(32);
  std::string estimator_activation = "MSE";
  std::string estimator_weight = "MSE";
  std::string estimator_bias = "MSE";
  bool skip_dense_layer = false;
  Array<Expr> skip_conv_layers = Array<Expr>(ObjectPtr<Object>(nullptr));
  Array<Expr> skip_add_layers = Array<Expr>(ObjectPtr<Object>(nullptr));
  bool do_simulation = false;
  bool round_for_shift = true;
  Array<Expr> debug_enabled_ops = Array<Expr>(ObjectPtr<Object>(nullptr));
  std::string rounding = "UPWARD";
  int calibrate_chunk_by = -1;
  std::string partition_conversions = "disabled";
  std::string quantizer_weight = "Symmetric";
  std::string quantizer_activation = "Asymmetric";
  std::string quantizer_bias = "Symmetric";
  bool per_channel = true;
  std::string opt_method = "grid";
  bool debug_mode = false;
  double global_scale = 8.0;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("network_name", &network_name);
    v->Visit("have_prequantized", &have_prequantized);
    v->Visit("nbit_input", &nbit_input);
    v->Visit("nbit_weight", &nbit_weight);
    v->Visit("nbit_activation", &nbit_activation);
    v->Visit("nbit_bias", &nbit_bias);
    v->Visit("dtype_input", &dtype_input);
    v->Visit("dtype_weight", &dtype_weight);
    v->Visit("dtype_activation", &dtype_activation);
    v->Visit("dtype_bias", &dtype_bias);
    v->Visit("estimator_activation", &estimator_activation);
    v->Visit("estimator_weight", &estimator_weight);
    v->Visit("estimator_bias", &estimator_bias);
    v->Visit("skip_dense_layer", &skip_dense_layer);
    v->Visit("skip_conv_layers", &skip_conv_layers);
    v->Visit("skip_add_layers", &skip_add_layers);
    v->Visit("do_simulation", &do_simulation);
    v->Visit("round_for_shift", &round_for_shift);
    v->Visit("debug_enabled_ops", &debug_enabled_ops);
    v->Visit("rounding", &rounding);
    v->Visit("calibrate_chunk_by", &calibrate_chunk_by);
    v->Visit("partition_conversions", &partition_conversions);
    v->Visit("quantizer_weight", &quantizer_weight);
    v->Visit("quantizer_activation", &quantizer_activation);
    v->Visit("quantizer_bias", &quantizer_bias);
    v->Visit("per_channel", &per_channel);
    v->Visit("opt_method", &opt_method);
    v->Visit("debug_mode", &debug_mode);
    v->Visit("global_scale", &global_scale);
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
