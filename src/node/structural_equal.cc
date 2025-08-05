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
#include <tvm/node/object_path.h>
#include <tvm/node/structural_equal.h>

#include <optional>
#include <unordered_map>

namespace tvm {

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("node.ObjectPathPairLhsPath",
           [](const ObjectPathPair& object_path_pair) { return object_path_pair->lhs_path; })
      .def("node.ObjectPathPairRhsPath",
           [](const ObjectPathPair& object_path_pair) { return object_path_pair->rhs_path; });
});

ObjectPathPairNode::ObjectPathPairNode(ObjectPath lhs_path, ObjectPath rhs_path)
    : lhs_path(std::move(lhs_path)), rhs_path(std::move(rhs_path)) {}

ObjectPathPair::ObjectPathPair(ObjectPath lhs_path, ObjectPath rhs_path) {
  data_ = make_object<ObjectPathPairNode>(std::move(lhs_path), std::move(rhs_path));
}

Optional<ObjectPathPair> ObjectPathPairFromAccessPathPair(
    Optional<ffi::reflection::AccessPathPair> src) {
  if (!src.has_value()) return std::nullopt;
  auto translate_path = [](ffi::reflection::AccessPath path) {
    ObjectPath result = ObjectPath::Root();
    for (const auto& step : path) {
      switch (step->kind) {
        case ffi::reflection::AccessKind::kObjectField: {
          result = result->Attr(step->key.cast<String>());
          break;
        }
        case ffi::reflection::AccessKind::kArrayItem: {
          result = result->ArrayIndex(step->key.cast<int64_t>());
          break;
        }
        case ffi::reflection::AccessKind::kMapItem: {
          result = result->MapValue(step->key);
          break;
        }
        case ffi::reflection::AccessKind::kArrayItemMissing: {
          result = result->MissingArrayElement(step->key.cast<int64_t>());
          break;
        }
        case ffi::reflection::AccessKind::kMapItemMissing: {
          result = result->MissingMapEntry();
          break;
        }
        default: {
          LOG(FATAL) << "Invalid access path kind: " << static_cast<int>(step->kind);
          break;
        }
      }
    }
    return result;
  };

  return ObjectPathPair(translate_path((*src).get<0>()), translate_path((*src).get<1>()));
}

bool NodeStructuralEqualAdapter(const Any& lhs, const Any& rhs, bool assert_mode,
                                bool map_free_vars) {
  if (assert_mode) {
    auto first_mismatch = ObjectPathPairFromAccessPathPair(
        ffi::StructuralEqual::GetFirstMismatch(lhs, rhs, map_free_vars));
    if (first_mismatch.has_value()) {
      std::ostringstream oss;
      oss << "StructuralEqual check failed, caused by lhs";
      oss << " at " << (*first_mismatch)->lhs_path;
      {
        // print lhs
        PrinterConfig cfg;
        cfg->syntax_sugar = false;
        cfg->path_to_underline.push_back((*first_mismatch)->lhs_path);
        // The TVMScriptPrinter::Script will fallback to Repr printer,
        // if the root node to print is not supported yet,
        // e.g. Relax nodes, ArrayObj, MapObj, etc.
        oss << ":" << std::endl << TVMScriptPrinter::Script(lhs.cast<ObjectRef>(), cfg);
      }
      oss << std::endl << "and rhs";
      {
        // print rhs
        oss << " at " << (*first_mismatch)->rhs_path;
        {
          PrinterConfig cfg;
          cfg->syntax_sugar = false;
          cfg->path_to_underline.push_back((*first_mismatch)->rhs_path);
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

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("node.StructuralEqual", NodeStructuralEqualAdapter)
      .def("node.GetFirstStructuralMismatch",
           [](const Any& lhs, const Any& rhs, bool map_free_vars) {
             /*
              Optional<ObjectPathPair> first_mismatch;
              bool equal =
                  SEqualHandlerDefault(false, &first_mismatch, true).Equal(lhs, rhs, map_free_vars);
              ICHECK(equal == !first_mismatch.defined());
              return first_mismatch;
             */
             return ObjectPathPairFromAccessPathPair(
                 ffi::StructuralEqual::GetFirstMismatch(lhs, rhs, map_free_vars));
           });
});

bool StructuralEqual::operator()(const ffi::Any& lhs, const ffi::Any& rhs,
                                 bool map_free_params) const {
  return ffi::StructuralEqual::Equal(lhs, rhs, map_free_params);
}
}  // namespace tvm
