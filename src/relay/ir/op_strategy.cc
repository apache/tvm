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
 * \file src/relay/ir/op_strategy.cc
 * \brief The Relay operator Strategy and related data structure.
 */

#include <tvm/relay/op_strategy.h>

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(OpImplementationNode);
TVM_REGISTER_NODE_TYPE(OpSpecializationNode);
TVM_REGISTER_NODE_TYPE(OpStrategyNode);

Array<te::Tensor> OpImplementation::Compute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                            const Type& out_type) {
  return (*this)->fcompute(attrs, inputs, out_type);
}

te::Schedule OpImplementation::Schedule(const Attrs& attrs, const Array<te::Tensor>& outs,
                                        const Target& target) {
  return (*this)->fschedule(attrs, outs, target);
}

void OpSpecialization::AddImplementation(tvm::relay::FTVMCompute fcompute,
                                         tvm::relay::FTVMSchedule fschedule, String name,
                                         int plevel) {
  auto n = make_object<OpImplementationNode>();
  n->fcompute = fcompute;
  n->fschedule = fschedule;
  n->name = std::move(name);
  n->plevel = plevel;
  (*this)->implementations.push_back(OpImplementation(n));
}

void OpStrategy::AddImplementation(FTVMCompute fcompute, FTVMSchedule fschedule, String name,
                                   int plevel) {
  auto curr_cond = te::SpecializedCondition::Current();
  auto self = this->operator->();
  Array<OpSpecialization> specializations = self->specializations;
  OpSpecialization op_spec;
  for (OpSpecialization op_spec : specializations) {
    if (op_spec->condition == curr_cond) {
      op_spec.AddImplementation(fcompute, fschedule, std::move(name), plevel);
      return;
    }
  }
  ObjectPtr<OpSpecializationNode> n = make_object<OpSpecializationNode>();
  n->condition = curr_cond;
  op_spec = OpSpecialization(n);
  op_spec.AddImplementation(fcompute, fschedule, std::move(name), plevel);
  self->specializations.push_back(op_spec);
}

TVM_REGISTER_GLOBAL("relay.op._OpImplementationCompute")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      OpImplementation imp = args[0];
      Attrs attrs = args[1];
      Array<te::Tensor> inputs = args[2];
      Type out_type = args[3];
      *rv = imp.Compute(attrs, inputs, out_type);
    });

TVM_REGISTER_GLOBAL("relay.op._OpImplementationSchedule")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      OpImplementation imp = args[0];
      Attrs attrs = args[1];
      Array<te::Tensor> outs = args[2];
      Target target = args[3];
      *rv = imp.Schedule(attrs, outs, target);
    });

TVM_REGISTER_GLOBAL("relay.op._make.OpStrategy").set_body([](TVMArgs args, TVMRetValue* rv) {
  ObjectPtr<OpStrategyNode> n = make_object<OpStrategyNode>();
  *rv = OpStrategy(n);
});

TVM_REGISTER_GLOBAL("relay.op._OpStrategyAddImplementation")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      OpStrategy strategy = args[0];
      FTVMCompute compute = args[1];
      FTVMSchedule schedule = args[2];
      std::string name = args[3];
      int plevel = args[4];
      strategy.AddImplementation(compute, schedule, name, plevel);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<OpStrategyNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const OpStrategyNode*>(node.get());
      p->stream << "op_strategy(" << op->specializations << ")";
    })
    .set_dispatch<OpSpecializationNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const OpSpecializationNode*>(node.get());
      p->stream << "op_spec(" << op->condition << ", " << op->implementations << ")";
    })
    .set_dispatch<OpImplementationNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const OpImplementationNode*>(node.get());
      p->stream << "op_impl(name=" << op->name << ", level=" << op->plevel << ")";
    });

}  // namespace relay
}  // namespace tvm
