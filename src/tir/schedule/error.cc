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
#include "./utils.h"

namespace tvm {
namespace tir {

String ScheduleError::RenderReport(const String& primitive) const {
  IRModule mod = this->mod();
  std::ostringstream os;
  os << "ScheduleError: An error occurred in the schedule primitive '" << primitive
     << "'.\n\nThe IR is:\n"
     << AsTVMScript(mod);
  Array<ObjectRef> locs = LocationsOfInterest();
  int n_locs = locs.size();
  std::vector<String> roi_names;
  roi_names.reserve(n_locs);
  if (n_locs > 0) {
    os << "Regions of interest:\n";
    for (const ObjectRef& obj : locs) {
      String name = obj->GetTypeKey() + '#' + std::to_string(roi_names.size());
      os << name << "\n" << obj;
      roi_names.emplace_back(std::move(name));
    }
    os << "\n";
  }
  std::string msg = DetailRenderTemplate();
  for (int i = 0; i < n_locs; ++i) {
    std::string src = "{" + std::to_string(i) + "}";
    for (size_t pos; (pos = msg.find(src)) != std::string::npos;) {
      msg.replace(pos, src.length(), roi_names[i]);
    }
  }
  os << "Error message: " << msg;
  return os.str();
}

}  // namespace tir
}  // namespace tvm
