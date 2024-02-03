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
 * \file src/contrib/msc/core/codegen/codegen_json.h
 * \brief Basic JSONSerializer for MSC runnable BYOC.
 */
#ifndef TVM_CONTRIB_MSC_CORE_CODEGEN_CODEGEN_JSON_H_
#define TVM_CONTRIB_MSC_CORE_CODEGEN_CODEGEN_JSON_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "../../../../relax/backend/contrib/codegen_json/codegen_json.h"
#include "../ir/graph.h"

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::relax;

using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;
using JSONGraphObjectPtr = backend::contrib::JSONGraphObjectPtr;
using JSONSerializer = backend::contrib::JSONSerializer;

/*!
 * \brief MSCCompileConfig defines config for all BYOC
 */
struct MSCCompileConfig {
  std::string graph_json;
  std::unordered_map<std::string, std::string> options;
  void Load(dmlc::JSONReader* reader) {
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      if (key == "graph_json") {
        reader->Read(&graph_json);
      } else {
        std::string value;
        reader->Read(&value);
        options.insert({key, value});
      }
    }
  }
};

class MSCJSONSerializer : public JSONSerializer {
 public:
  /*!
   * \brief Constructor
   * \param constant_names The names of all constants in the original module.
   */
  explicit MSCJSONSerializer(const Map<Constant, String>& constant_names,
                             const std::string& options)
      : JSONSerializer(constant_names) {
    MSCCompileConfig config;
    std::istringstream is(options);
    dmlc::JSONReader reader(&is);
    reader.Read(&config);
    ICHECK(config.graph_json.size() > 0) << "graph_json is needed to init MSCGraph";
    graph_ = MSCGraph(config.graph_json);
    for (const auto& pair : config.options) {
      options_.Set(pair.first, pair.second);
    }
    global_options_set_ = false;
  }

  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* call_node) final;

  const String GetOption(const String& key) {
    ICHECK(options_.count(key)) << "Can not find option " << key;
    return options_[key];
  }

  const Map<String, String> GetOptions() { return options_; }

 protected:
  void AddNodeAttr(JSONGraphObjectPtr node, const String& key, const String& value);

 private:
  MSCGraph graph_;
  Map<String, String> options_;
  bool global_options_set_;
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_CODEGEN_CODEGEN_JSON_H_
