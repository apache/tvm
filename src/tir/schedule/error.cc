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

  // get locations of interest
  Array<ObjectRef> locs = LocationsOfInterest();
  std::unordered_map<ObjectRef, String, ObjectPtrHash, ObjectPtrEqual> loc_obj_to_name;
  int n_locs = locs.size();
  std::string msg = DetailRenderTemplate();
  PrinterConfig cfg;
  cfg->syntax_sugar = false;
  if (n_locs > 0) {
    for (int i = 0; i < n_locs; ++i) {
      std::string name = locs[i]->GetTypeKey() + '#' + std::to_string(i);
      std::string src = "{" + std::to_string(i) + "}";
      for (size_t pos; (pos = msg.find(src)) != std::string::npos;) {
        msg.replace(pos, src.length(), name);
      }
      cfg->obj_to_annotate.Set(locs[i], name);
      cfg->obj_to_underline.push_back(locs[i]);
    }
  }
  os << "ScheduleError: An error occurred in the schedule primitive '" << primitive
     << "'.\n\nThe IR with diagnostic is:\n"
     << TVMScriptPrinter::Script(mod, cfg) << std::endl;

  // print error message
  os << "Error message: " << msg;
  return os.str();
}

}  // namespace tir
}  // namespace tvm
