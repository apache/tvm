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
#include "module_equality.h"

#include <tvm/ir/module.h>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include <tvm/tir/analysis.h>

#include <memory>

#include "../node/ndarray_hash_equal.h"

namespace tvm {
namespace meta_schedule {

class ModuleEqualityStructural : public ModuleEquality {
 public:
  size_t Hash(IRModule mod) const { return tvm::StructuralHash()(mod); }
  bool Equal(IRModule lhs, IRModule rhs) const { return tvm::StructuralEqual()(lhs, rhs); }
  String GetName() const { return "structural"; }
};

class SEqualHandlerIgnoreNDArray : public SEqualHandlerDefault {
 public:
  SEqualHandlerIgnoreNDArray() : SEqualHandlerDefault(false, nullptr, false) {}

 protected:
  bool DispatchSEqualReduce(const ObjectRef& lhs, const ObjectRef& rhs, bool map_free_vars,
                            const Optional<ObjectPathPair>& current_paths) {
    if (auto lhs_ptr = lhs.as<runtime::NDArray::Container>(),
        rhs_ptr = rhs.as<runtime::NDArray::Container>();
        lhs_ptr && rhs_ptr) {
      SEqualReducer reducer(this, nullptr, map_free_vars);
      return NDArrayEqual(lhs_ptr, rhs_ptr, reducer, false);
    }
    return SEqualHandlerDefault::DispatchSEqualReduce(lhs, rhs, map_free_vars, current_paths);
  }
};

class ModuleEqualityIgnoreNDArray : public ModuleEquality {
 public:
  size_t Hash(IRModule mod) const { return SHashHandlerIgnoreNDArray().Hash(mod, false); }
  bool Equal(IRModule lhs, IRModule rhs) const {
    return SEqualHandlerIgnoreNDArray().Equal(lhs, rhs, false);
  }
  String GetName() const { return "ignore-ndarray"; }
};

// The NDArray-ignoring variant of structural equal / hash is used for the module equality
// on the extracted anchor blocks.
class ModuleEqualityAnchorBlock : public ModuleEquality {
  size_t Hash(IRModule mod) const {
    auto anchor_block = tir::FindAnchorBlock(mod);
    if (anchor_block) {
      return SHashHandlerIgnoreNDArray().Hash(GetRef<tir::Block>(anchor_block), false);
    }
    return ModuleEqualityIgnoreNDArray().Hash(mod);
  }
  bool Equal(IRModule lhs, IRModule rhs) const {
    auto anchor_block_lhs = tir::FindAnchorBlock(lhs);
    auto anchor_block_rhs = tir::FindAnchorBlock(rhs);
    if (anchor_block_lhs && anchor_block_rhs) {
      return SEqualHandlerIgnoreNDArray().Equal(GetRef<tir::Block>(anchor_block_lhs),
                                                GetRef<tir::Block>(anchor_block_rhs), false);
    }
    return ModuleEqualityIgnoreNDArray().Equal(lhs, rhs);
  }
  String GetName() const { return "anchor-block"; }
};

std::unique_ptr<ModuleEquality> ModuleEquality::Create(const std::string& mod_eq_name) {
  if (mod_eq_name == "structural") {
    return std::make_unique<ModuleEqualityStructural>();
  } else if (mod_eq_name == "ignore-ndarray") {
    return std::make_unique<ModuleEqualityIgnoreNDArray>();
  } else if (mod_eq_name == "anchor-block") {
    return std::make_unique<ModuleEqualityAnchorBlock>();
  }
  LOG(FATAL) << "Unknown module equality " << mod_eq_name;
}

}  // namespace meta_schedule
}  // namespace tvm
