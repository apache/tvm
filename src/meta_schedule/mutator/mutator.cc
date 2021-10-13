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

Mutator Mutator::PyMutator(PyMutatorNode::FInitializeWithTuneContext f_initialize_with_tune_context,
                           PyMutatorNode::FApply f_apply) {
  ObjectPtr<PyMutatorNode> n = make_object<PyMutatorNode>();
  n->f_initialize_with_tune_context = std::move(f_initialize_with_tune_context);
  n->f_apply = std::move(f_apply);
  return Mutator(n);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PyMutatorNode>([](const ObjectRef& n, ReprPrinter* p) {
      const auto* self = n.as<PyMutatorNode>();
      ICHECK(self);
      std::string func_name = "meta_schedule.mutator.py_mutator._f_as_string";
      const auto* f_as_string = tvm::runtime::Registry::Get(func_name);
      ICHECK(f_as_string) << "AttributeError: \"" << func_name
                          << "\" is not registered. "
                             "Please check if the python module is properly loaded";
      std::string ret = (*f_as_string)(n);
      p->stream << ret;
    });

TVM_REGISTER_OBJECT_TYPE(MutatorNode);
TVM_REGISTER_NODE_TYPE(PyMutatorNode);

TVM_REGISTER_GLOBAL("meta_schedule.MutatorInitializeWithTuneContext")
    .set_body_method<Mutator>(&MutatorNode::InitializeWithTuneContext);
TVM_REGISTER_GLOBAL("meta_schedule.MutatorApply").set_body_method<Mutator>(&MutatorNode::Apply);
TVM_REGISTER_GLOBAL("meta_schedule.MutatorPyMutator").set_body_typed(Mutator::PyMutator);

}  // namespace meta_schedule
}  // namespace tvm
