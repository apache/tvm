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
#include <tvm/ffi/reflection/registry.h>

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
  ObjectPtr<PyPostprocNode> n = ffi::make_object<PyPostprocNode>();
  n->f_initialize_with_tune_context = std::move(f_initialize_with_tune_context);
  n->f_apply = std::move(f_apply);
  n->f_clone = std::move(f_clone);
  n->f_as_string = std::move(f_as_string);
  return Postproc(n);
}

ffi::Array<Postproc> Postproc::DefaultLLVM() {
  return ffi::Array<Postproc>{
      Postproc::DisallowDynamicLoop(),
      Postproc::RewriteParallelVectorizeUnroll(),
      Postproc::RewriteReductionBlock(),
      Postproc::RewriteLayout(),
  };
}

ffi::Array<Postproc> Postproc::DefaultCPUTensorization() {
  return ffi::Array<Postproc>{
      Postproc::DisallowDynamicLoop(),   Postproc::RewriteParallelVectorizeUnroll(),
      Postproc::RewriteReductionBlock(), Postproc::RewriteTensorize(/*vectorize_init_loop=*/true),
      Postproc::RewriteLayout(),
  };
}

ffi::Array<Postproc> Postproc::DefaultRISCV() {
  return ffi::Array<Postproc>{
      Postproc::DisallowDynamicLoop(),   Postproc::RewriteParallelVectorizeUnroll(),
      Postproc::RewriteReductionBlock(), Postproc::RewriteTensorize(/*vectorize_init_loop=*/false),
      Postproc::RewriteLayout(),
  };
}

ffi::Array<Postproc> Postproc::DefaultCUDA() {
  return ffi::Array<Postproc>{
      Postproc::DisallowDynamicLoop(),
      Postproc::RewriteCooperativeFetch(),
      Postproc::RewriteUnboundBlock(/*max_threadblocks=*/256),
      Postproc::RewriteParallelVectorizeUnroll(),
      Postproc::RewriteReductionBlock(),
      Postproc::VerifyGPUCode(),
  };
}

ffi::Array<Postproc> Postproc::DefaultCUDATensorCore() {
  return ffi::Array<Postproc>{
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

ffi::Array<Postproc> Postproc::DefaultHexagon() {
  return ffi::Array<Postproc>{
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

TVM_FFI_STATIC_INIT_BLOCK() {
  PostprocNode::RegisterReflection();
  PyPostprocNode::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("meta_schedule.PostprocInitializeWithTuneContext",
                  &PostprocNode::InitializeWithTuneContext)
      .def_method("meta_schedule.PostprocApply", &PostprocNode::Apply)
      .def_method("meta_schedule.PostprocClone", &PostprocNode::Clone)
      .def("meta_schedule.PostprocPyPostproc", Postproc::PyPostproc)
      .def("meta_schedule.PostprocDefaultLLVM", Postproc::DefaultLLVM)
      .def("meta_schedule.PostprocDefaultCUDA", Postproc::DefaultCUDA)
      .def("meta_schedule.PostprocDefaultCUDATensorCore", Postproc::DefaultCUDATensorCore)
      .def("meta_schedule.PostprocDefaultHexagon", Postproc::DefaultHexagon);
}

}  // namespace meta_schedule
}  // namespace tvm
