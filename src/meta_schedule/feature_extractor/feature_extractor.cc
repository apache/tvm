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

Array<tvm::runtime::NDArray> PyFeatureExtractorNode::ExtractFrom(
    const TuneContext& context, const Array<MeasureCandidate>& candidates) {
  ICHECK(f_extract_from != nullptr) << "PyFeatureExtractor's ExtractFrom method not implemented!";
  return f_extract_from(context, candidates);
}

FeatureExtractor FeatureExtractor::PyFeatureExtractor(
    PyFeatureExtractorNode::FExtractFrom f_extract_from,  //
    PyFeatureExtractorNode::FAsString f_as_string) {
  ObjectPtr<PyFeatureExtractorNode> n = make_object<PyFeatureExtractorNode>();
  n->f_extract_from = std::move(f_extract_from);
  n->f_as_string = std::move(f_as_string);
  return FeatureExtractor(n);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PyFeatureExtractorNode>([](const ObjectRef& n, ReprPrinter* p) {
      const auto* self = n.as<PyFeatureExtractorNode>();
      ICHECK(self);
      PyFeatureExtractorNode::FAsString f_as_string = (*self).f_as_string;
      ICHECK(f_as_string != nullptr) << "PyFeatureExtractor's AsString method not implemented!";
      p->stream << f_as_string();
    });

TVM_REGISTER_OBJECT_TYPE(FeatureExtractorNode);
TVM_REGISTER_NODE_TYPE(PyFeatureExtractorNode);

TVM_REGISTER_GLOBAL("meta_schedule.FeatureExtractorExtractFrom")
    .set_body_method<FeatureExtractor>(&FeatureExtractorNode::ExtractFrom);
TVM_REGISTER_GLOBAL("meta_schedule.FeatureExtractorPyFeatureExtractor")
    .set_body_typed(FeatureExtractor::PyFeatureExtractor);

}  // namespace meta_schedule
}  // namespace tvm
