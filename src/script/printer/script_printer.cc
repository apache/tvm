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
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/logging.h>
#include <tvm/script/printer/printer.h>

#include <sstream>

namespace tvm {

namespace {

std::string RenderFallbackWithInvisiblePathInfo(const ffi::String& script,
                                                const PrinterConfig& config) {
  if (!config->render_invisible_path_info || config->path_to_underline.empty()) {
    return std::string(script);
  }

  std::ostringstream os;
  for (size_t i = 0; i < config->path_to_underline.size(); ++i) {
    if (i != 0) os << "\n";
    os << "Access path: " << config->path_to_underline[i]
       << "\nNote: No visible object for this path is rendered in TVMScript.";
  }
  os << "\n\n" << script;
  return os.str();
}

}  // namespace

TVMScriptPrinter::FType& TVMScriptPrinter::vtable() {
  static FType inst;
  return inst;
}

std::string Script(const ffi::ObjectRef& node, const ffi::Optional<PrinterConfig>& cfg) {
  PrinterConfig config = cfg.value_or(PrinterConfig());
  if (!TVMScriptPrinter::vtable().CanDispatch(node)) {
    // Fall back to ffi::ReprPrint for types not registered with TVMScriptPrinter.
    return RenderFallbackWithInvisiblePathInfo(ffi::ReprPrint(ffi::Any(node)), config);
  }
  return TVMScriptPrinter::vtable()(node, config);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("node.TVMScriptPrinterScript", tvm::Script);
}

std::string RedirectedReprPrinterMethod(const ffi::ObjectRef& obj) {
  try {
    return tvm::Script(obj, std::nullopt);
  } catch (const tvm::ffi::Error& e) {
    LOG(WARNING) << "TVMScript printer falls back to the basic address printer with the error:\n"
                 << e.what();
    std::ostringstream os;
    os << obj->GetTypeKey() << '(' << obj.get() << ')';
    return os.str();
  }
}

}  // namespace tvm
