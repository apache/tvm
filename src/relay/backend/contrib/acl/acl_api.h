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
 * \file src/relay/backend/contrib/acl/acl_api.h
 * \brief A common JSON interface between relay and the ACL runtime module.
 */

#ifndef TVM_RELAY_BACKEND_CONTRIB_ACL_ACL_API_H_
#define TVM_RELAY_BACKEND_CONTRIB_ACL_ACL_API_H_

#include <dmlc/json.h>
#include <dmlc/memory_io.h>
#include <tvm/runtime/ndarray.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {
namespace contrib {
namespace acl {

DMLC_JSON_ENABLE_ANY(std::vector<int>, IntVector);
DMLC_JSON_ENABLE_ANY(int, Int);
DMLC_JSON_ENABLE_ANY(size_t, Size_t);
DMLC_JSON_ENABLE_ANY(std::string, String);

/*!
 * JSON interface for ACL tensor.
 */
class JSONTensor {
 public:
  JSONTensor() = default;
  explicit JSONTensor(std::vector<int> shape) : type("var"), shape(std::move(shape)) {}

  JSONTensor(std::string type, std::vector<int> shape)
      : type(std::move(type)), shape(std::move(shape)) {}

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("type", type);
    writer->WriteObjectKeyValue("shape", shape);
    writer->EndObject();
  }

  void Load(dmlc::JSONReader* reader) {
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareField("type", &type);
    helper.DeclareField("shape", &shape);
    helper.ReadAllFields(reader);
  }

  /*! \brief The type of the tensor var/const. */
  std::string type;
  /*! \brief The shape of the tensor. */
  std::vector<int> shape;
};

/*!
 * JSON interface for an ACL operator.
 */
class JSONOp {
 public:
  JSONOp() = default;
  explicit JSONOp(std::string name) : name(std::move(name)) {}

  void Save(dmlc::JSONWriter* writer) const {
    auto op_attrs = attrs;
    op_attrs["num_inputs"] = dmlc::any(inputs.size());
    op_attrs["num_outputs"] = dmlc::any(outputs.size());
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("inputs", inputs);
    writer->WriteObjectKeyValue("outputs", outputs);
    writer->WriteObjectKeyValue("attrs", op_attrs);
    writer->EndObject();
  }

  void Load(dmlc::JSONReader* reader) {
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareField("name", &name);
    helper.DeclareField("inputs", &inputs);
    helper.DeclareField("outputs", &outputs);
    helper.DeclareField("attrs", &attrs);
    helper.ReadAllFields(reader);
  }

  /*! The name of the operator. */
  std::string name;
  /*! The required variable inputs to the operator. */
  std::vector<JSONTensor> inputs;
  /*! The required outputs to the operator. */
  std::vector<JSONTensor> outputs;
  /*! The attributes of the operator e.g. padding, strides, etc. */
  std::unordered_map<std::string, dmlc::any> attrs;
};

/*!
 * JSON interface for a series of ACL ops.
 */
class JSONSubGraph {
 public:
  JSONSubGraph() = default;
  explicit JSONSubGraph(JSONOp op) : op(std::move(op)) {}

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("node", op);
    writer->EndObject();
  }

  void Load(dmlc::JSONReader* reader) {
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareField("node", &op);
    helper.ReadAllFields(reader);
  }

  /*! \brief JSON op to be serialized. */
  JSONOp op;
};

/*!
 * \brief Deserialize a function (or subgraph). The function is serialized in the
 * format: Serialized JSON (using dmlc::JSONWriter), number of constants, serialized
 * NDArray constants.
 *
 * \param serialized_function Pointer to a serialized function (or subgraph).
 * \return A pair consisting of deserialized json subgraph object and deserialized
 * NDArray.
 */
std::pair<JSONSubGraph, std::vector<runtime::NDArray>> DeserializeSubgraph(
    std::string* serialized_function);

/*!
 * \brief Serialize a single subgraph which can be saved to disk.
 *
 * A subgraph is serialized so that the output is as follows:
 * - Serialized JSON.
 * - Number of constant tensors.
 * - Serialized constant tensors.
 *
 * \param subgraph JSON subgraph representation.
 * \constants Serialized JSON constants.
 */
std::string SerializeSubgraph(const JSONSubGraph& subgraph,
                              const std::vector<runtime::NDArray>& constants);

}  // namespace acl
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_ACL_ACL_API_H_
