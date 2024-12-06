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

void PyPostprocNode::InitializeWithTuneContext(const TuneContext& context) {
  ICHECK(f_initialize_with_tune_context != nullptr)
      << "PyPostproc's InitializeWithTuneContext method not implemented!";
  f_initialize_with_tune_context(context);
}

bool PyPostprocNode::Apply(const tir::Schedule& sch) {
  ICHECK(f_apply != nullptr) << "PyPostproc's Apply method not implemented!";
  return f_apply(sch);
}

Postproc PyPostprocNode::Clone() const {
  ICHECK(f_clone != nullptr) << "PyPostproc's Clone method not implemented!";
  return f_clone();
}

Postproc Postproc::PyPostproc(
    PyPostprocNode::FInitializeWithTuneContext f_initialize_with_tune_context,  //
    PyPostprocNode::FApply f_apply,                                             //
    PyPostprocNode::FClone f_clone,                                             //
    PyPostprocNode::FAsString f_as_string) {
  ObjectPtr<PyPostprocNode> n = make_object<PyPostprocNode>();
  n->f_initialize_with_tune_context = std::move(f_initialize_with_tune_context);
  n->f_apply = std::move(f_apply);
  n->f_clone = std::move(f_clone);
  n->f_as_string = std::move(f_as_string);
  return Postproc(n);
}

Array<Postproc> Postproc::DefaultLLVM() {
  return Array<Postproc>{
      Postproc::DisallowDynamicLoop(),
      Postproc::RewriteParallelVectorizeUnroll(),
      Postproc::RewriteReductionBlock(),
      Postproc::RewriteLayout(),
  };
}

Array<Postproc> Postproc::DefaultCPUTensorization() {
  return Array<Postproc>{
      Postproc::DisallowDynamicLoop(),   Postproc::RewriteParallelVectorizeUnroll(),
      Postproc::RewriteReductionBlock(), Postproc::RewriteTensorize(/*vectorize_init_loop=*/true),
      Postproc::RewriteLayout(),
  };
}

Array<Postproc> Postproc::DefaultCUDA() {
  return Array<Postproc>{
      Postproc::DisallowDynamicLoop(),
      Postproc::RewriteCooperativeFetch(),
      Postproc::RewriteUnboundBlock(/*max_threadblocks=*/256),
      Postproc::RewriteParallelVectorizeUnroll(),
      Postproc::RewriteReductionBlock(),
      Postproc::VerifyGPUCode(),
  };
}

Array<Postproc> Postproc::DefaultCUDATensorCore() {
  return Array<Postproc>{
      Postproc::DisallowDynamicLoop(),
      Postproc::RewriteCooperativeFetch(),
      Postproc::RewriteUnboundBlock(/*max_threadblocks=*/256),
      Postproc::RewriteParallelVectorizeUnroll(),
      Postproc::RewriteReductionBlock(),
      Postproc::VerifyGPUCode(),
      // RewriteTensorize is relatively expensive and it doesn't affect the validity of a sample, so
      // run it only on samples that have passed VerifyGPUCode.
      Postproc::RewriteTensorize(/*vectorize_init_loop=*/false),
  };
}

Array<Postproc> Postproc::DefaultHexagon() {
  return Array<Postproc>{
      Postproc::DisallowDynamicLoop(),   Postproc::RewriteParallelVectorizeUnroll(),
      Postproc::RewriteReductionBlock(), Postproc::RewriteLayout(),
      Postproc::VerifyVTCMLimit(),
  };
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PyPostprocNode>([](const ObjectRef& n, ReprPrinter* p) {
      const auto* self = n.as<PyPostprocNode>();
      ICHECK(self);
      PyPostprocNode::FAsString f_as_string = (*self).f_as_string;
      ICHECK(f_as_string != nullptr) << "PyPostproc's AsString method not implemented!";
      p->stream << f_as_string();
    });

TVM_REGISTER_OBJECT_TYPE(PostprocNode);
TVM_REGISTER_NODE_TYPE(PyPostprocNode);

TVM_REGISTER_GLOBAL("meta_schedule.PostprocInitializeWithTuneContext")
    .set_body_method<Postproc>(&PostprocNode::InitializeWithTuneContext);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocApply").set_body_method<Postproc>(&PostprocNode::Apply);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocClone").set_body_method<Postproc>(&PostprocNode::Clone);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocPyPostproc").set_body_typed(Postproc::PyPostproc);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocDefaultLLVM").set_body_typed(Postproc::DefaultLLVM);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocDefaultCUDA").set_body_typed(Postproc::DefaultCUDA);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocDefaultCUDATensorCore")
    .set_body_typed(Postproc::DefaultCUDATensorCore);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocDefaultHexagon")
    .set_body_typed(Postproc::DefaultHexagon);

}  // namespace meta_schedule
}  // namespace tvm
