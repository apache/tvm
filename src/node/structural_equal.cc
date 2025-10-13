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
 * \file src/node/structural_equal.cc
 */
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/access_path.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/node/functor.h>
#include <tvm/node/node.h>
#include <tvm/node/structural_equal.h>

#include <optional>
#include <unordered_map>

namespace tvm {

bool NodeStructuralEqualAdapter(const Any& lhs, const Any& rhs, bool assert_mode,
                                bool map_free_vars) {
  if (assert_mode) {
    auto first_mismatch = ffi::StructuralEqual::GetFirstMismatch(lhs, rhs, map_free_vars);
    if (first_mismatch.has_value()) {
      std::ostringstream oss;
      oss << "StructuralEqual check failed, caused by lhs";
      oss << " at " << (*first_mismatch).get<0>();
      {
        // print lhs
        PrinterConfig cfg;
        cfg->syntax_sugar = false;
        cfg->path_to_underline.push_back((*first_mismatch).get<0>());
        // The TVMScriptPrinter::Script will fallback to Repr printer,
        // if the root node to print is not supported yet,
        // e.g. Relax nodes, ArrayObj, MapObj, etc.
        oss << ":" << std::endl << TVMScriptPrinter::Script(lhs.cast<ObjectRef>(), cfg);
      }
      oss << std::endl << "and rhs";
      {
        // print rhs
        oss << " at " << (*first_mismatch).get<1>();
        {
          PrinterConfig cfg;
          cfg->syntax_sugar = false;
          cfg->path_to_underline.push_back((*first_mismatch).get<1>());
          // The TVMScriptPrinter::Script will fallback to Repr printer,
          // if the root node to print is not supported yet,
          // e.g. Relax nodes, ArrayObj, MapObj, etc.
          oss << ":" << std::endl << TVMScriptPrinter::Script(rhs.cast<ObjectRef>(), cfg);
        }
      }
      TVM_FFI_THROW(ValueError) << oss.str();
    }
    return true;
  } else {
    return ffi::StructuralEqual::Equal(lhs, rhs, map_free_vars);
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("node.StructuralEqual", NodeStructuralEqualAdapter)
      .def("node.GetFirstStructuralMismatch", ffi::StructuralEqual::GetFirstMismatch);
}

bool StructuralEqual::operator()(const ffi::Any& lhs, const ffi::Any& rhs,
                                 bool map_free_params) const {
  return ffi::StructuralEqual::Equal(lhs, rhs, map_free_params);
}
}  // namespace tvm
