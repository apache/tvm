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
 * \file domain_touched.cc
 * \brief Analyze buffer domains touched by a statement
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/sym/int_set.h>
#include <tvm/te/tensor.h>

#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "../../s_tir/ir/ir_visitor_with_analyzer.h"

namespace tvm {
namespace s_tir {

using namespace tirx;
using sym::IntSet;

namespace {

using BufferTouches = std::vector<std::vector<IntSet>>;

struct LoadAccess {
  BufferTouches set;
};

struct StoreAccess {
  BufferTouches set;
};

struct CombinedAccess {
  BufferTouches set;
};

using BufferDomainAccess = std::tuple<LoadAccess, StoreAccess, CombinedAccess>;

}  // namespace

// Find Read region of the tensor in the stmt.
class BufferTouchedDomain final : public s_tir::IRVisitorWithAnalyzer {
 public:
  using s_tir::IRVisitorWithAnalyzer::Visit_;

  std::unordered_map<const VarNode*, BufferDomainAccess>& GetAccessedBufferRegions() {
    return buffer_access_map_;
  }

  ffi::Array<ffi::Optional<Range>> FindUnion(const TensorVar& buffer, bool consider_loads,
                                             bool consider_stores) {
    ffi::Array<ffi::Optional<Range>> ret;
    auto kv = buffer_access_map_.find(buffer.get());
    if (kv == buffer_access_map_.end()) {
      LOG(WARNING) << "[s_tir::BufferDomainTouched] "
                   << "The requested buffer is not contained in the provided stmt body: " << buffer;
      return ret;
    }

    BufferTouches bounds;
    if (consider_loads && consider_stores) {
      bounds = std::get<CombinedAccess>(kv->second).set;
    } else if (consider_loads) {
      bounds = std::get<LoadAccess>(kv->second).set;
    } else if (consider_stores) {
      bounds = std::get<StoreAccess>(kv->second).set;
    } else {
      TVM_FFI_ICHECK(false)
          << "Must consider at least on of either loads and stores, but both are false";
    }
    for (size_t i = 0; i < bounds.size(); ++i) {
      ret.push_back(sym::Union(bounds[i]).CoverRange(std::nullopt));
    }
    return ret;
  }

 private:
  using Parent = s_tir::IRVisitorWithAnalyzer;

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadNode* op) final {
    TensorVar buffer = op->source.as_or_throw<tvm::tirx::TensorVar>();
    // Record load-exclusive buffer access
    Touch(&std::get<LoadAccess>(buffer_access_map_[buffer.get()]).set, op->indices);
    // Record load-store inclusive buffer access
    Touch(&std::get<CombinedAccess>(buffer_access_map_[buffer.get()]).set, op->indices);
    return Parent::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreNode* op) final {
    // Record store-exclusive buffer access
    Touch(&std::get<StoreAccess>(buffer_access_map_[op->dest.as_or_throw<TensorVar>().get()]).set,
          op->indices);
    // Record load-store inclusive buffer access
    Touch(
        &std::get<CombinedAccess>(buffer_access_map_[op->dest.as_or_throw<TensorVar>().get()]).set,
        op->indices);
    return Parent::Visit_(op);
  }

  void Touch(BufferTouches* bounds, const ffi::Array<PrimExpr>& args) {
    if (args.size() > bounds->size()) {
      bounds->resize(args.size());
    }
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i].as<prim::RampNode>()) {
        (*bounds)[i].emplace_back(IntSet::Vector(args[i]));
      } else {
        (*bounds)[i].emplace_back(analyzer_->int_set(args[i]));
      }
    }
  }

  std::unordered_map<const VarNode*, BufferDomainAccess> buffer_access_map_;
};

ffi::Array<ffi::Optional<Range>> DomainTouched(const Stmt& stmt, const TensorVar& buffer,
                                               bool consider_loads, bool consider_stores) {
  auto visitor = ffi::make_object<BufferTouchedDomain>();
  visitor->Visit(stmt);
  return visitor->FindUnion(buffer, consider_loads, consider_stores);
}

ffi::Map<TensorVar, ffi::Array<ffi::ObjectRef>> DomainTouchedAccessMap(const Function& func) {
  auto visitor = ffi::make_object<BufferTouchedDomain>();
  visitor->Visit(func->body);
  auto buffer_access_map = visitor->GetAccessedBufferRegions();
  ffi::Map<TensorVar, ffi::Array<ffi::ObjectRef>> ret;
  for (auto& var : func->params) {
    if (!var->ty.as<TensorTypeNode>()) {
      continue;
    }
    TensorVar buffer = var.as_or_throw<TensorVar>();
    auto& access = buffer_access_map[buffer.get()];
    ffi::Array<ffi::Array<IntSet>> loads, stores, combined;
    for (std::vector<IntSet>& touch : std::get<LoadAccess>(access).set) {
      loads.push_back(ffi::Array<IntSet>(touch));
    }
    for (std::vector<IntSet>& touch : std::get<StoreAccess>(access).set) {
      stores.push_back(ffi::Array<IntSet>(touch));
    }
    for (std::vector<IntSet>& touch : std::get<CombinedAccess>(access).set) {
      combined.push_back(ffi::Array<IntSet>(touch));
    }

    ffi::Array<ffi::ObjectRef> fields;
    fields.push_back(loads);
    fields.push_back(stores);
    fields.push_back(combined);
    ret.Set(buffer, fields);
  }
  return ret;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.DomainTouched", DomainTouched)
      .def("s_tir.DomainTouchedAccessMap", DomainTouchedAccessMap);
}

}  // namespace s_tir
}  // namespace tvm
