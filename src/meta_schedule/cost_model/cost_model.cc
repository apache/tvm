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

void PyCostModelNode::Load(const String& path) {
  ICHECK(f_load != nullptr) << "PyCostModel's Load method not implemented!";
  f_load(path);
}

void PyCostModelNode::Save(const String& path) {
  ICHECK(f_save != nullptr) << "PyCostModel's Save method not implemented!";
  f_save(path);
}

void PyCostModelNode::Update(const TuneContext& context, const Array<MeasureCandidate>& candidates,
                             const Array<RunnerResult>& results) {
  ICHECK(f_update != nullptr) << "PyCostModel's Update method not implemented!";
  f_update(context, candidates, results);
}

std::vector<double> PyCostModelNode::Predict(const TuneContext& context,
                                             const Array<MeasureCandidate>& candidates) {
  ICHECK(f_predict != nullptr) << "PyCostModel's Predict method not implemented!";
  std::vector<double> result(candidates.size(), 0.0);
  f_predict(context, candidates, result.data());
  return result;
}

CostModel CostModel::PyCostModel(PyCostModelNode::FLoad f_load,        //
                                 PyCostModelNode::FSave f_save,        //
                                 PyCostModelNode::FUpdate f_update,    //
                                 PyCostModelNode::FPredict f_predict,  //
                                 PyCostModelNode::FAsString f_as_string) {
  ObjectPtr<PyCostModelNode> n = make_object<PyCostModelNode>();
  n->f_load = std::move(f_load);
  n->f_save = std::move(f_save);
  n->f_update = std::move(f_update);
  n->f_predict = std::move(f_predict);
  n->f_as_string = std::move(f_as_string);
  return CostModel(n);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PyCostModelNode>([](const ObjectRef& n, ReprPrinter* p) {
      const auto* self = n.as<PyCostModelNode>();
      ICHECK(self);
      PyCostModelNode::FAsString f_as_string = (*self).f_as_string;
      ICHECK(f_as_string != nullptr) << "PyCostModel's AsString method not implemented!";
      p->stream << f_as_string();
    });

TVM_REGISTER_OBJECT_TYPE(CostModelNode);
TVM_REGISTER_NODE_TYPE(PyCostModelNode);

TVM_REGISTER_GLOBAL("meta_schedule.CostModelLoad").set_body_method<CostModel>(&CostModelNode::Load);
TVM_REGISTER_GLOBAL("meta_schedule.CostModelSave").set_body_method<CostModel>(&CostModelNode::Save);
TVM_REGISTER_GLOBAL("meta_schedule.CostModelUpdate")
    .set_body_method<CostModel>(&CostModelNode::Update);
TVM_REGISTER_GLOBAL("meta_schedule.CostModelPredict")
    .set_body_typed([](CostModel model,                     //
                       const TuneContext& context,          //
                       Array<MeasureCandidate> candidates,  //
                       void* p_addr) -> void {
      std::vector<double> result = model->Predict(context, candidates);
      std::copy(result.begin(), result.end(), static_cast<double*>(p_addr));
    });
TVM_REGISTER_GLOBAL("meta_schedule.CostModelPyCostModel").set_body_typed(CostModel::PyCostModel);

}  // namespace meta_schedule
}  // namespace tvm
