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
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <memory>

#include "../node/ndarray_hash_equal.h"
#include "../tir/schedule/analysis.h"

namespace tvm {
namespace meta_schedule {

class ModuleEqualityStructural : public ModuleEquality {
 public:
  size_t Hash(IRModule mod) const { return tvm::StructuralHash()(mod); }
  bool Equal(IRModule lhs, IRModule rhs) const { return tvm::StructuralEqual()(lhs, rhs); }
};

class SEqualHandlerIgnoreNDArray : public SEqualHandlerDefault {
 public:
  SEqualHandlerIgnoreNDArray() : SEqualHandlerDefault(false, nullptr) {}

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

class SHashHandlerIgnoreNDArray : public SHashHandlerDefault {
 protected:
  void DispatchSHash(const ObjectRef& object, bool map_free_vars) override {
    ICHECK(object.defined());
    if (auto ndarray = object.as<runtime::NDArray::Container>()) {
      SHashReducer hash_reduce(this, map_free_vars);
      NDArrayHash(ndarray, &hash_reduce, false);
    } else {
      SHashHandlerDefault::DispatchSHash(object, map_free_vars);
    }
  }
};

const tir::BlockNode* GetAnchorBlock(IRModule mod) {
  using namespace tir;

  struct BlockCollector : public StmtVisitor {
    void VisitStmt_(const BlockNode* block) override {
      blocks.push_back(block);
      StmtVisitor::VisitStmt(block->body);
    }
    std::vector<const BlockNode*> blocks;
  };

  auto prim_func = FindEntryFunc(mod, nullptr);
  BlockCollector collector;
  collector(prim_func->body);

  ICHECK(collector.blocks.size() > 0);

  for (auto block : collector.blocks) {
    if (!block->reads.empty() && !block->writes.empty()) {
      return block;
    }
  }

  LOG(FATAL) << "Cannot find a suitable anchor block";

  return collector.blocks[0];
}

class ModuleEqualityIgnoreNDArray : public ModuleEquality {
 public:
  size_t Hash(IRModule mod) const { return SHashHandlerIgnoreNDArray().Hash(mod, false); }
  bool Equal(IRModule lhs, IRModule rhs) const {
    return SEqualHandlerIgnoreNDArray().Equal(lhs, rhs, false);
  }
};

class ModuleEqualityAnchorBlock : public ModuleEquality {
  size_t Hash(IRModule mod) const {
    auto anchor_block = GetAnchorBlock(mod);
    return SHashHandlerIgnoreNDArray().Hash(GetRef<tir::Block>(anchor_block), false);
  }
  bool Equal(IRModule lhs, IRModule rhs) const {
    auto anchor_block_lhs = GetAnchorBlock(lhs);
    auto anchor_block_rhs = GetAnchorBlock(rhs);
    return SEqualHandlerIgnoreNDArray().Equal(GetRef<tir::Block>(anchor_block_lhs),
                                              GetRef<tir::Block>(anchor_block_rhs), false);
  }
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
  return nullptr;
}

}  // namespace meta_schedule
}  // namespace tvm
