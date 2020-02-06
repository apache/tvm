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

#include <tvm/ir/expr.h>
#include <tvm/relay/op_attr_types.h>

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(OpImplementNode);
TVM_REGISTER_NODE_TYPE(OpSpecializationNode);
TVM_REGISTER_NODE_TYPE(OpStrategyNode);

Array<te::Tensor> OpImplement::Compute(const Attrs& attrs,
                                       const Array<te::Tensor>& inputs,
                                       const Type& out_type) {
  return (*this)->fcompute(attrs, inputs, out_type);
}

te::Schedule OpImplement::Schedule(const Attrs& attrs,
                                   const Array<te::Tensor> &outs,
                                   const Target& target) {
  return (*this)->fschedule(attrs, outs, target);
}

void OpSpecialization::AddImplement(tvm::relay::FTVMCompute fcompute,
                                    tvm::relay::FTVMSchedule fschedule,
                                    std::string name,
                                    int plevel) {
  auto n = make_object<OpImplementNode>();
  n->fcompute = fcompute;
  n->fschedule = fschedule;
  n->name = name;
  n->plevel = plevel;
  (*this)->implements.push_back(OpImplement(n));
}

void OpStrategy::AddImplement(FTVMCompute fcompute,
                              FTVMSchedule fschedule,
                              std::string name,
                              int plevel) {
  auto curr_cond = te::SpecializedCondition::Current();
  auto specializations = (*this)->specializations;
  OpSpecialization op_spec;
  for (auto e : specializations) {
    if (e->condition == curr_cond) {
      op_spec = e;
      break;
    }
  }
  if (op_spec.defined()) {
    op_spec.AddImplement(fcompute, fschedule, name, plevel);
  } else {
    ObjectPtr<OpSpecializationNode> n = make_object<OpSpecializationNode>();
    n->condition = curr_cond;
    op_spec = OpSpecialization(n);
    op_spec.AddImplement(fcompute, fschedule, name, plevel);
    (*this)->specializations.push_back(op_spec);
  }
}

TVM_REGISTER_GLOBAL("relay.op._OpImplementCompute")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    OpImplement imp = args[0];
    Attrs attrs = args[1];
    Array<te::Tensor> inputs = args[2];
    Type out_type = args[3];
    *rv = imp.Compute(attrs, inputs, out_type);
});

TVM_REGISTER_GLOBAL("relay.op._OpImplementSchedule")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    OpImplement imp = args[0];
    Attrs attrs = args[1];
    Array<te::Tensor> outs = args[2];
    Target target = args[3];
    *rv = imp.Schedule(attrs, outs, target);
});

TVM_REGISTER_GLOBAL("relay.op._make.OpStrategy")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    ObjectPtr<OpStrategyNode> n = make_object<OpStrategyNode>();
    *rv = OpStrategy(n);
});

TVM_REGISTER_GLOBAL("relay.op._OpStrategyAddImplement")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    OpStrategy strategy = args[0];
    FTVMCompute compute = args[1];
    FTVMSchedule schedule = args[2];
    std::string name = args[3];
    int plevel = args[4];
    strategy.AddImplement(compute, schedule, name, plevel);
});


}  // namespace relay
}  // namespace tvm
