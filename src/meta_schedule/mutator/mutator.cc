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
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

void PyMutatorNode::InitializeWithTuneContext(const TuneContext& context) {
  ICHECK(f_initialize_with_tune_context != nullptr)
      << "PyMutator's InitializeWithTuneContext method not implemented!";
  f_initialize_with_tune_context(context);
}

Optional<tir::Trace> PyMutatorNode::Apply(
    const tir::Trace& trace, support::LinearCongruentialEngine::TRandState* rand_state) {
  ICHECK(f_apply != nullptr) << "PyMutator's Apply method not implemented!";
  return f_apply(trace, *rand_state);
}

Mutator PyMutatorNode::Clone() const {
  ICHECK(f_clone != nullptr) << "PyMutator's Clone method not implemented!";
  return f_clone();
}

Mutator Mutator::PyMutator(
    PyMutatorNode::FInitializeWithTuneContext f_initialize_with_tune_context,  //
    PyMutatorNode::FApply f_apply,                                             //
    PyMutatorNode::FClone f_clone,                                             //
    PyMutatorNode::FAsString f_as_string) {
  ObjectPtr<PyMutatorNode> n = make_object<PyMutatorNode>();
  n->f_initialize_with_tune_context = std::move(f_initialize_with_tune_context);
  n->f_apply = std::move(f_apply);
  n->f_clone = std::move(f_clone);
  n->f_as_string = std::move(f_as_string);
  return Mutator(n);
}

Map<Mutator, FloatImm> Mutator::DefaultLLVM() {
  return Map<Mutator, FloatImm>{
      {Mutator::MutateTileSize(), FloatImm(DataType::Float(64), 0.9)},
      {Mutator::MutateComputeLocation(), FloatImm(DataType::Float(64), 0.05)},
      {Mutator::MutateUnroll(), FloatImm(DataType::Float(64), 0.03)},
      {Mutator::MutateParallel(/*max_jobs_per_core=*/16), FloatImm(DataType::Float(64), 0.02)}};
}

Map<Mutator, FloatImm> Mutator::DefaultVNNI() { return Mutator::DefaultLLVM(); }

Map<Mutator, FloatImm> Mutator::DefaultCUDA() {
  return Map<Mutator, FloatImm>{
      {Mutator::MutateTileSize(), FloatImm(DataType::Float(64), 0.9)},
      {Mutator::MutateUnroll(), FloatImm(DataType::Float(64), 0.08)},
      {Mutator::MutateThreadBinding(), FloatImm(DataType::Float(64), 0.02)}};
}

Map<Mutator, FloatImm> Mutator::DefaultCUDATensorCore() { return Mutator::DefaultCUDA(); }

Map<Mutator, FloatImm> Mutator::DefaultHexagon() {
  return Map<Mutator, FloatImm>{
      {Mutator::MutateTileSize(), FloatImm(DataType::Float(64), 0.9)},
      {Mutator::MutateComputeLocation(), FloatImm(DataType::Float(64), 0.05)},
      {Mutator::MutateUnroll(), FloatImm(DataType::Float(64), 0.03)},
      {Mutator::MutateParallel(/*max_jobs_per_core=*/16), FloatImm(DataType::Float(64), 0.02)}};
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PyMutatorNode>([](const ObjectRef& n, ReprPrinter* p) {
      const auto* self = n.as<PyMutatorNode>();
      ICHECK(self);
      PyMutatorNode::FAsString f_as_string = (*self).f_as_string;
      ICHECK(f_as_string != nullptr) << "PyMutator's AsString method not implemented!";
      p->stream << f_as_string();
    });

TVM_REGISTER_OBJECT_TYPE(MutatorNode);
TVM_REGISTER_NODE_TYPE(PyMutatorNode);

TVM_REGISTER_GLOBAL("meta_schedule.MutatorInitializeWithTuneContext")
    .set_body_method<Mutator>(&MutatorNode::InitializeWithTuneContext);
TVM_REGISTER_GLOBAL("meta_schedule.MutatorApply")
    .set_body_typed([](Mutator self, tir::Trace trace, TRandState seed) -> Optional<tir::Trace> {
      TRandState seed_ = (seed != -1) ? seed : support::LinearCongruentialEngine::DeviceRandom();
      return self->Apply(trace, &seed_);
    });
TVM_REGISTER_GLOBAL("meta_schedule.MutatorClone").set_body_method<Mutator>(&MutatorNode::Clone);
TVM_REGISTER_GLOBAL("meta_schedule.MutatorPyMutator").set_body_typed(Mutator::PyMutator);
TVM_REGISTER_GLOBAL("meta_schedule.MutatorDefaultLLVM").set_body_typed(Mutator::DefaultLLVM);
TVM_REGISTER_GLOBAL("meta_schedule.MutatorDefaultCUDA").set_body_typed(Mutator::DefaultCUDA);
TVM_REGISTER_GLOBAL("meta_schedule.MutatorDefaultCUDATensorCore")
    .set_body_typed(Mutator::DefaultCUDATensorCore);
TVM_REGISTER_GLOBAL("meta_schedule.MutatorDefaultHexagon").set_body_typed(Mutator::DefaultHexagon);

}  // namespace meta_schedule
}  // namespace tvm
