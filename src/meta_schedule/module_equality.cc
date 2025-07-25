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

#include <tvm/ffi/reflection/structural_equal.h>
#include <tvm/ffi/reflection/structural_hash.h>
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

class ModuleEqualityIgnoreNDArray : public ModuleEquality {
 public:
  size_t Hash(IRModule mod) const {
    return tvm::ffi::reflection::StructuralHash::Hash(mod, /*map_free_vars=*/false,
                                                      /*skip_ndarray_content=*/true);
  }
  bool Equal(IRModule lhs, IRModule rhs) const {
    return tvm::ffi::reflection::StructuralEqual::Equal(lhs, rhs, /*map_free_vars=*/false,
                                                        /*skip_ndarray_content=*/true);
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
      return tvm::ffi::reflection::StructuralEqual::Equal(GetRef<tir::Block>(anchor_block_lhs),
                                                          GetRef<tir::Block>(anchor_block_rhs),
                                                          /*map_free_vars=*/false,
                                                          /*skip_ndarray_content=*/true);
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
