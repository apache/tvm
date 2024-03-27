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
 * \brief Implementation of runtime part of tuning capabilities for cublas matmul primitives.
 */

#include "cublas_algo.h"
#include <tvm/runtime/registry.h>

namespace tvm {
namespace contrib {

/*******************************/
/*  Algo Desc                  */
/*******************************/

AlgoDesc::AlgoDesc() {
  ObjectPtr<AlgoDescNode> n = runtime::make_object<AlgoDescNode>();
  data_ = std::move(n);
};

AlgoDesc::AlgoDesc(const double estimated_time_us, const cublasLtMatmulAlgo_t &algo) {
  ObjectPtr<AlgoDescNode> n = runtime::make_object<AlgoDescNode>();
  n->estimated_time_us = estimated_time_us;
  n->algo = algo;
  data_ = std::move(n);
}

void AlgoDescNode::Save(dmlc::JSONWriter* writer) const {
  std::stringstream ss;
  ss << std::hex
     << algo.data[0] << " " << algo.data[1] << " " << algo.data[2] << " " << algo.data[3] << " "
     << algo.data[4] << " " << algo.data[5] << " " << algo.data[6] << " " << algo.data[7];
  writer->Write(ss.str());
}

void AlgoDescNode::Load(dmlc::JSONReader* reader) {
  std::string data;
  reader->Read(&data);
  std::stringstream ss(data);
  ss >> std::hex 
     >> algo.data[0] >> algo.data[1] >> algo.data[2] >> algo.data[3]
     >> algo.data[4] >> algo.data[5] >> algo.data[6] >> algo.data[7];
}

TVM_REGISTER_OBJECT_TYPE(AlgoDescNode);
// TVM_REGISTER_NODE_TYPE(AlgoDescNode);  // TODO: What is it? Why doesn't it works with GCC 11.4?

/*******************************/
/*  Algo Collection            */
/*******************************/

AlgoCollection::AlgoCollection() {
  auto n = runtime::make_object<AlgoCollectionNode>();
  data_ = std::move(n);
}

AlgoDesc AlgoCollectionNode::GetAlgoFor(size_t dyn_dim_val) {
  auto none_desk = AlgoDesc(tvm::runtime::ObjectPtr<AlgoDescNode>(nullptr));

  if (regions.size() == 0 || dyn_dim_val == 0)
    return none_desk;

  // Find region index which meet constrain: regions[i-1] < dyn_dim_val <= regions[i]
  // or use last element if there is no such region.
  size_t i = 0;
  while (i < regions.size() && regions[i].first < dyn_dim_val) i++;

  if (i == regions.size())
    return none_desk;

  auto algo_idx = regions[i].second;
  return algos[algo_idx];
}

void AlgoCollectionNode::Save(dmlc::JSONWriter* writer) const {
  writer->BeginObject();
  writer->WriteObjectKeyValue("regions", regions);
  writer->WriteObjectKeyValue("algos", algos);
  writer->EndObject();
}

void AlgoCollectionNode::Load(dmlc::JSONReader* reader) {
  std::string key;
  reader->BeginObject();
  while (reader->NextObjectItem(&key)) {
    if (key == "regions") {
      reader->Read(&regions);
    } else if (key == "algos") {
      reader->Read(&algos);
    } else {
      LOG(ERROR) << "Unknown key";
    }
  }  
}

AlgoCollection AlgoCollection::FromJSON(const std::string& data) {
  auto ac = tvm::contrib::AlgoCollection{};
  std::istringstream is(data);
  dmlc::JSONReader reader(&is);
  reader.Read(&ac);
  return ac;
}

TVM_REGISTER_OBJECT_TYPE(AlgoCollectionNode);
// TVM_REGISTER_NODE_TYPE(AlgoCollectionNode);

}  // namespace contrib
}  // namespace tvm
