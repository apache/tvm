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
#include <tvm/runtime/logging.h>
#include <tvm/script/printer/config.h>

#include <sstream>

namespace tvm {

std::string RedirectedReprPrinterMethod(const ObjectRef& obj) {
  try {
    return TVMScriptPrinter::Script(obj, std::nullopt);
  } catch (const tvm::ffi::Error& e) {
    LOG(WARNING) << "TVMScript printer falls back to the basic address printer with the error:\n"
                 << e.what();
    std::ostringstream os;
    os << obj->GetTypeKey() << '(' << obj.get() << ')';
    return os.str();
  }
}

}  // namespace tvm
