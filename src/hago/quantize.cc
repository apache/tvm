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
 * Copyright (c) 2018 by Contributors
 *
 * \file quantize.cc
 *
 * \brief transform a graph to a low-bit graph
 *   for compression and acceleration.
 */
#include <dmlc/thread_local.h>
#include <tvm/dtype.h>
#include <tvm/relay/type.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/op_attr_types.h>
#include <stack>
#include "./quantize.h"

namespace tvm {
namespace hago {

using namespace ::tvm::relay;
using ::tvm::relay::Expr;
using ::tvm::relay::Type;

TVM_REGISTER_NODE_TYPE(SimulatedQuantizeAttrs);

bool SimulatedQuantizeRel(const Array<Type>& types,
                          int num_inputs,
                          const Attrs& attrs,
                          const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto param = attrs.as<SimulatedQuantizeAttrs>();
  CHECK(param != nullptr);

  const auto* data = types[0].as<TensorTypeNode>();
  CHECK(data != nullptr);
  CHECK_NE(data->shape.size(), 0) << "Input shape cannot be empty";

  reporter->Assign(types[0], types[1]);                                // output
  return true;
}


RELAY_REGISTER_OP("hago.simulated_quantize")
.describe(R"code(simulated quantize op)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input data.")
.set_attrs_type_key("hago.SimulatedQuantizeAttrs")
.set_support_level(11)
.add_type_rel("SimulatedQuantize", SimulatedQuantizeRel);

TVM_REGISTER_API("hago._quantize.simulated_quantize")
.set_body_typed<Expr(Expr, double, double, int64_t, int64_t, DataType, DataType, bool, std::string)>(
  [](Expr data, double in_scale, double out_scale, int64_t clip_min, int64_t clip_max,
     DataType in_dtype, DataType out_dtype, bool sign, std::string rounding) {
    auto attrs = make_node<SimulatedQuantizeAttrs>();
    attrs->in_scale = in_scale;
    attrs->out_scale = out_scale;
    attrs->clip_min = clip_min;
    attrs->clip_max = clip_max;
    attrs->in_dtype = in_dtype;
    attrs->out_dtype = out_dtype;
    attrs->sign = sign;
    attrs->rounding = rounding;
    static const Op& op = Op::Get("hago.simulated_quantize");
    return CallNode::make(op, {data}, Attrs(attrs), {});
  });


/*! \brief Entry to hold the BuildConfig context stack. */
struct TVMQConfigThreadLocalEntry {
  /*! \brief The default build config if the stack is empty */
  QConfig default_config;

  /*! \brief The current build config context */
  std::stack<QConfig> context_stack;

  TVMQConfigThreadLocalEntry() :
    default_config(make_node<QConfigNode>()) {
  }
};

/*! \brief Thread local store to hold the BuildConfig context stack. */
typedef dmlc::ThreadLocalStore<TVMQConfigThreadLocalEntry> TVMQConfigThreadLocalStore;

void QConfig::EnterQConfigScope(const QConfig& build_config) {
  TVMQConfigThreadLocalEntry *entry = TVMQConfigThreadLocalStore::Get();
  entry->context_stack.push(build_config);
}

void QConfig::ExitQConfigScope() {
  TVMQConfigThreadLocalEntry *entry = TVMQConfigThreadLocalStore::Get();
  entry->context_stack.pop();
}

QConfig& QConfig::Current() {
  TVMQConfigThreadLocalEntry *entry = TVMQConfigThreadLocalStore::Get();
  if (entry->context_stack.size() > 0) {
    return entry->context_stack.top();
  }

  return entry->default_config;
}

TVM_REGISTER_NODE_TYPE(QConfigNode);

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<QConfigNode>([](const QConfigNode *op, IRPrinter *p) {
  p->stream << "qconfig(";
  p->stream << "skip_conv_layers==" << op->skip_conv_layers << ", ";
  p->stream << "search_strategy=" << op->search_strategy << ", ";
  p->stream << "threshold_estimate_strategy=" << op->threshold_estimate_strategy << ", ";
  p->stream << "global_scale=" << op->global_scale << ", ";
  p->stream << "do_simulation==" << op->do_simulation << ", ";
  p->stream << "round_for_shift==" << op->round_for_shift << ", ";
  p->stream << "debug_enabled_ops==" << op->debug_enabled_ops;
  p->stream << ")";
});

TVM_REGISTER_API("hago._quantize._GetCurrentQConfig")
.set_body_typed(QConfig::Current);

TVM_REGISTER_API("hago._quantize._EnterQConfigScope")
.set_body_typed(QConfig::EnterQConfigScope);

TVM_REGISTER_API("hago._quantize._ExitQConfigScope")
.set_body_typed(QConfig::ExitQConfigScope);

OpDesc OpDescNode::make(Array<Type> in_types,
                        Array<Type> out_types) {
  NodePtr<OpDescNode> n = make_node<OpDescNode>();
  n->in_types = std::move(in_types);
  n->out_types = std::move(out_types);
  return OpDesc(n);
}

TVM_REGISTER_API("hago._make.OpDesc")
.set_body_typed(OpDescNode::make);

}  // namespace hago
}  // namespace tvm
