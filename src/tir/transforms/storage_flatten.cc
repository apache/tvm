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
 * \file storage_flatten.cc
 * \brief Flattens storage from multi-dimensional array to 1D buffer access
 */
// The pass definition originates from Halide pipeline.

#include <tvm/runtime/registry.h>
#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/te/operation.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/buffer.h>
#include <tvm/target/target_info.h>
#include <tvm/runtime/device_api.h>
#include <unordered_map>
#include "ir_util.h"
#include "arg_binder.h"
#include "../../arith/compute_expr.h"
#include "../../arith/ir_visitor_with_analyzer.h"
#include "../../runtime/thread_storage_scope.h"

namespace tvm {
namespace tir {

using runtime::StorageRank;
using runtime::StorageScope;
using runtime::ThreadScope;
using intrinsic::tvm_address_of;

class StorageFlattener : public StmtExprMutator {
 public:
  explicit StorageFlattener(const Map<Var, Buffer>& extern_buffer_map,
                            int cache_line_size,
                            bool create_bound_attributes,
                            IRVisitorWithAnalyzer* bound_analyzer)
      : bound_analyzer_(bound_analyzer),
        create_bound_attributes_(create_bound_attributes) {
    for (auto kv : extern_buffer_map) {
      BufferEntry e;
      e.buffer = kv.second;
      e.external = true;
      buf_map_[kv.second] = e;
    }
    cache_line_size_ = cache_line_size;
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<StoreNode>();
    auto it = var_remap_.find(op->buffer_var.get());
    if (it != var_remap_.end() &&
        !it->second.same_as(op->buffer_var)) {
      CHECK(it->second.as<VarNode>());
      Var buf_var = Downcast<Var>(it->second);
      return StoreNode::make(buf_var, op->value, op->index, op->predicate);
    } else {
      return stmt;
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::realize_scope) {
      storage_scope_[op->node.get()] = op->value.as<StringImmNode>()->value;
      return this->VisitStmt(op->body);
    } else if (op->attr_key == attr::double_buffer_scope &&
               op->node->IsInstance<tir::BufferNode>()) {
      auto buffer = Downcast<tir::Buffer>(op->node);
      Stmt body = this->VisitStmt(op->body);
      auto it = buf_map_.find(buffer);
      CHECK(it != buf_map_.end())
          << "Cannot find allocated buffer for " << buffer;
      body = AttrStmtNode::make(
          it->second.buffer->data, op->attr_key, op->value, std::move(body));
      return body;
    } else if (op->attr_key == attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      ThreadScope ts = ThreadScope::make(iv->thread_tag);
      curr_thread_scope_.push_back(ts);
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      curr_thread_scope_.pop_back();
      return stmt;
    } else if (op->attr_key == attr::buffer_bind_scope) {
      return HandleBufferBindScope(op);
    } else if (op->attr_key == attr::buffer_dim_align) {
      auto buffer = Downcast<tir::Buffer>(op->node);
      const CallNode* tuple = op->value.as<CallNode>();
      CHECK(tuple && tuple->is_intrinsic(intrinsic::tvm_tuple));
      auto& vinfo = dim_align_[buffer];
      int dim = tuple->args[0].as<IntImmNode>()->value;
      if (static_cast<size_t>(dim) >= vinfo.size()) {
        vinfo.resize(dim + 1);
      }
      vinfo[dim].align_factor = tuple->args[1].as<IntImmNode>()->value;
      vinfo[dim].align_offset = tuple->args[2].as<IntImmNode>()->value;
      return this->VisitStmt(op->body);
    } else if (op->attr_key == attr::opengl_stage_scope) {
      is_opengl_ = true;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    if (create_bound_attributes_) shape_collector_.clear();
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferStoreNode>();

    const auto& key = op->buffer;

    auto it = buf_map_.find(key);
    CHECK(it != buf_map_.end())
        << "Cannot find allocated buffer for " << key;

    const BufferEntry& e = it->second;
    CHECK(!e.released)
        << "Read a buffer that is already out of scope";

    if (is_opengl_) {
      return EvaluateNode::make(CallNode::make(
          DataType(),
          CallNode::glsl_texture_store,
          {e.buffer->data, op->value},
          CallNode::Intrinsic));
    } else {
      Stmt body = e.buffer.vstore(e.RelIndex(op->indices), op->value);
      if (create_bound_attributes_ && ShapeIsValid(e.buffer->shape)) {
        shape_collector_.push_back(
            std::make_pair(e.buffer->data, e.buffer->shape));
      }
      // To create bound attribute collector should has at least one item.
      if (create_bound_attributes_ && shape_collector_.size()) {
        for (size_t i = 0; i < shape_collector_.size(); ++i) {
          body = AttrStmtNode::make(
              shape_collector_[i].first, tir::attr::buffer_bound,
              MakeBound(e.buffer->dtype, shape_collector_[i].second), body);
        }
      }
      return body;
    }
  }

  Stmt VisitStmt_(const BufferRealizeNode* op) final {
    const auto& key = op->buffer;

    if (buf_map_.count(key)) {
      CHECK(buf_map_.at(key).external);
      return this->VisitStmt(op->body);
    } else {
      // create a buffer entry
      BufferEntry e;
      e.bounds = op->bounds;
      Array<PrimExpr> shape;
      for (auto r : e.bounds) {
        shape.push_back(r->extent);
      }
      // deduce current storage scope.
      auto it = storage_scope_.find(op->buffer.get());
      CHECK(it != storage_scope_.end())
          << "Cannot find storage scope of " << op->buffer;
      StorageScope skey;
      const std::string& strkey = it->second;
      if (strkey.length() == 0) {
        if (curr_thread_scope_.size() != 0) {
          skey.rank = runtime::DefaultStorageRank(
              curr_thread_scope_.back().rank);
        }
      } else {
        skey = StorageScope::make(strkey);
      }

      // use small alignment for small arrays
      auto dtype = op->buffer->dtype;
      int32_t const_size = AllocateNode::constant_allocation_size(shape);
      int align = GetTempAllocaAlignment(dtype, const_size);
      if (skey.tag.length() != 0) {
        MemoryInfo info = GetMemoryInfo(skey.to_string());
        if (info.defined()) {
          align = (info->max_simd_bits + dtype.bits() - 1) / dtype.bits();
          CHECK_LE(const_size * dtype.bits(), info->max_num_bits)
              << "Allocation exceed bound of memory tag " << skey.to_string();
        }
      }
      Array<PrimExpr> strides;
      if (dim_align_.count(key) != 0 && shape.size() != 0) {
        std::vector<PrimExpr> rstrides;
        const std::vector<DimAlignInfo>& avec = dim_align_[key];
        int first_dim = 0;
        PrimExpr stride = make_const(shape[first_dim].dtype(), 1);
        for (size_t i = shape.size(); i != 0; --i) {
          size_t dim = i - 1;
          if (dim < avec.size() && avec[dim].align_factor != 0) {
            PrimExpr factor = make_const(stride.dtype(), avec[dim].align_factor);
            PrimExpr offset = make_const(stride.dtype(), avec[dim].align_offset);
            stride = stride + indexmod(factor + offset - indexmod(stride, factor), factor);
            stride = bound_analyzer_->Simplify(stride);
          }
          rstrides.push_back(stride);
          stride = stride * shape[dim];
        }
        strides = Array<PrimExpr>(rstrides.rbegin(), rstrides.rend());
      }

      e.buffer = BufferNode::make(
          Var(op->buffer->data->name_hint, DataType::Handle()),
          op->buffer->dtype, shape, strides, PrimExpr(),
          op->buffer->name, skey.to_string(),
          align, 0, kDefault);

      buf_map_[key] = e;
      Stmt body = this->VisitStmt(op->body);
      buf_map_[key].released = true;
      Stmt ret;

      DataType storage_type = e.buffer->dtype;
      // specially handle bool, lower its storage
      // type to beDataType::Int(8)(byte)
      if (storage_type == DataType::Bool()) {
        storage_type = DataType::Int(8);
      }
      if (strides.size() != 0) {
        int first_dim = 0;
        ret = AllocateNode::make(
            e.buffer->data, storage_type,
            {e.buffer->strides[first_dim] * e.buffer->shape[first_dim]},
            make_const(DataType::Bool(e.buffer->dtype.lanes()), true), body);
      } else {
        shape = e.buffer->shape;
        if (shape.size() == 0) {
          shape.push_back(make_const(DataType::Int(32), 1));
        }
        ret = AllocateNode::make(
            e.buffer->data, storage_type, shape,
            make_const(DataType::Bool(e.buffer->dtype.lanes()), true), body);
      }
      ret = AttrStmtNode::make(
          e.buffer->data, attr::storage_scope,
          StringImmNode::make(e.buffer->scope), ret);

      if (create_bound_attributes_ && ShapeIsValid(e.buffer->shape)) {
        ret = AttrStmtNode::make(e.buffer->data, tir::attr::buffer_bound,
                             MakeBound(e.buffer->dtype, e.buffer->shape), ret);
      }
      return ret;
    }
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<LoadNode>();
    auto it = var_remap_.find(op->buffer_var.get());
    if (it != var_remap_.end() &&
        !it->second.same_as(op->buffer_var)) {
      CHECK(it->second.as<VarNode>());
      Var buf_var = Downcast<Var>(it->second);
      return LoadNode::make(op->dtype, buf_var, op->index, op->predicate);
    } else {
      return expr;
    }
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = var_remap_.find(op);
    if (it != var_remap_.end()) {
      return it->second;
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<BufferLoadNode>();

    const auto& key = op->buffer;

    auto it = buf_map_.find(key);
    CHECK(it != buf_map_.end())
        << "Cannot find allocated buffer for " << key;
    const BufferEntry& e = it->second;
    CHECK(!e.released)
        << "Read a buffer that is already out of scope";

    if (create_bound_attributes_ && ShapeIsValid(e.buffer->shape)) {
        shape_collector_.push_back(
            std::make_pair(e.buffer->data, e.buffer->shape));
    }
    return e.buffer.vload(e.RelIndex(op->indices), e.buffer->dtype);
  }


  Stmt VisitStmt_(const PrefetchNode *op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<PrefetchNode>();
    CHECK(op != nullptr);

    const auto& key = op->buffer;
    auto it = buf_map_.find(key);
    CHECK(it != buf_map_.end())
        << "Cannot find allocated buffer for " << key;
    const BufferEntry& e = it->second;

    CHECK(!e.released)
        << "Read a buffer that is already out of scope";
    CHECK_EQ(e.buffer->shape.size(), op->bounds.size())
      << "Prefetch dim should be the same as buffer dim";

    int block_size = 1,
        elem_cnt = cache_line_size_ / e.buffer->dtype.bytes();

    int starts = op->bounds.size() - 1;

    while (starts > 0) {
      auto* shape_as_int = e.buffer->shape[starts].as<IntImmNode>();
      if (shape_as_int == nullptr || block_size * shape_as_int->value > elem_cnt) break;
      block_size *= static_cast<int>(shape_as_int->value);
      starts--;
    }
    PrimExpr stride(elem_cnt / block_size);

    Array<PrimExpr> args;
    std::vector<Var> vars;

    for (int i = op->bounds.size() - 1; i > starts; --i) {
      args.push_back(op->bounds[i]->min);
    }
    auto &func_name = op->buffer->name;
    vars.push_back(Var(
        "prefetch." + func_name + "." + std::to_string(starts), DataType::Int(32)));
    args.push_back(op->bounds[starts]->min + stride * vars.back());
    for (int i = starts - 1; i >= 0; --i) {
      vars.push_back(Var(
          "prefetch." + func_name + "." + std::to_string(i), DataType::Int(32)));
      args.push_back(vars.back() + op->bounds[i]->min);
    }
    for (int i = starts; i >= 0; --i) {
      if (i < starts) {
        stmt = ForNode::make(
            vars[i], 0, op->bounds[i]->extent, ForType::Serial, DeviceAPI::None, stmt);
      } else {
        PrimExpr load = e.buffer.vload(e.RelIndex(args), e.buffer->dtype);
        PrimExpr address = CallNode::make(
            DataType::Handle(), tvm_address_of, {load}, CallNode::PureIntrinsic);
        PrimExpr prefetch = CallNode::make(
            op->buffer->dtype, CallNode::prefetch, {address, 0, 3, 1}, CallNode::Intrinsic);
        stmt = EvaluateNode::make(prefetch);
        PrimExpr extent = (op->bounds[i]->extent - 1) / stride + 1;
        stmt = ForNode::make(vars[i], 0, extent, ForType::Serial, DeviceAPI::None, stmt);
      }
    }
    return stmt;
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    CHECK(op->call_type != CallNode::Halide)
        << "Cannot handle Halide calls "
        << " please run SchedulePostProcToPrimFunc first";
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const ProvideNode* op) final {
    LOG(FATAL) << "Cannot handle Provide "
               << " please run SchedulePostProcToPrimFunc first";
    return Stmt();
  }

  Stmt VisitStmt_(const RealizeNode* op) final {
    LOG(FATAL) << "Cannot handle Realize "
               << " please run SchedulePostProcToPrimFunc first";
    return Stmt();
  }


 private:
  // The specific tensor data layout is not determined before
  // StorageFlatten pass. We use buffer_bind_scope
  // to specify before hand we want to bind a subregion
  // of tensor to a symbolic buffer, which get used in extern.
  //
  // Example:
  //
  // realize A in range [i*4, extent=10) {
  //   bind Ab to A in [i*4+1, extent=4) {
  //     call_func(Ab.ptr, Ab.shape[0])
  //   }
  // }
  //
  // After StorageFlatten
  //
  // alloc A[10]
  //   call(A + 1,  4)
  //
  // Buffer is a protocol to declare specific
  // data layout and shape we expect.
  // So this function need to check:
  // - If the bind range is within the realize range
  // - If we can match the requirement of buffer
  // - Remap variables such as Ab.ptr to the actual value.
  //
  // Here are a few possible failure cases:
  // - Buffer is declared to have constant shape,
  //   but we try to bind it to a different one.
  // - Buffer is declared to be compact(no strides)
  //   but this binded region is a subregion of
  //   a matrix(tensor), which means it requires strides.
  //
  // We do support a few relaxed case, such as bindingx
  // region with shape [1, 1, n, m] to buffer with shape [n, m]
  Stmt HandleBufferBindScope(const AttrStmtNode* op) {
    Array<ObjectRef> arr = Downcast<Array<ObjectRef> > (op->node);
    CHECK_EQ(arr.size(), 2U);
    const BufferNode* buffer = arr[0].as<BufferNode>();
    const BufferNode* target = arr[1].as<BufferNode>();
    const CallNode* tuple = op->value.as<CallNode>();
    CHECK(buffer && target);
    CHECK(tuple && tuple->is_intrinsic(intrinsic::tvm_tuple));
    auto key = GetRef<Buffer>(target);

    auto it = buf_map_.find(key);
    CHECK(it != buf_map_.end())
        << "Cannot find buffer of " << key;
    const BufferEntry& be = it->second;
    CHECK(!be.released);
    CHECK_EQ(tuple->args.size(), be.buffer->shape.size() * 2);
    Array<PrimExpr> begins, extents;
    if (be.bounds.size() != 0) {
      CHECK_EQ(tuple->args.size(), be.bounds.size() * 2);
      for (size_t i = 0; i < be.buffer->shape.size(); ++i) {
        begins.push_back(tuple->args[2 * i] - be.bounds[i]->min);
        extents.push_back(tuple->args[2 * i + 1]);
      }
    } else {
      for (size_t i = 0; i < tuple->args.size(); i += 2) {
        begins.push_back(tuple->args[i]);
        auto new_extent = bound_analyzer_->Simplify(tuple->args[i+1]);
        extents.push_back(new_extent);
      }
    }
    Buffer slice = be.buffer.MakeSlice(begins, extents);
    if (buffer->strides.size() == 0) {
      CHECK_EQ(slice->strides.size(), 0U)
          << "Trying to bind compact buffer to strided one strides="
          << slice->strides;
    } else {
      slice = slice.MakeStrideView();
    }
    // start binding
    ArgBinder binder(&var_remap_);
    binder.BindBuffer(Downcast<Buffer>(arr[0]), slice, buffer->name, true);
    // Apply the remaps
    Stmt body = MergeNest(binder.asserts(), op->body);
    body = MergeNest(binder.init_nest(), body);
    body = this->VisitStmt(body);
    // remove the binds
    for (const Var& v : binder.defs()) {
      var_remap_.erase(v.get());
    }
    return body;
  }

  // The buffer entry in the flatten map
  struct DimAlignInfo {
    int align_factor{0};
    int align_offset{0};
  };
  // The buffer entry in the flatten map
  struct BufferEntry {
    // the buffer of storage
    Buffer buffer;
    // the bounds of realization, can be null, means everything
    Region bounds;
    // Whether the buffer is external
    bool external{false};
    // Whether we are out of allocation bounds and buffer get released.
    bool released{false};
    // relative index
    inline Array<PrimExpr> RelIndex(Array<PrimExpr> args) const {
      if (bounds.size() != 0) {
        Array<PrimExpr> index;
        CHECK_EQ(bounds.size(), args.size());
        for (size_t i = 0; i < bounds.size(); ++i) {
          index.push_back(args[i] - bounds[i]->min);
        }
        return index;
      } else {
        return args;
      }
    }
  };

  bool ShapeIsValid(const Array<PrimExpr> &shape) {
    // Zero-dimensional tensor does not need boundary check.
    if (!shape.size())
      return false;

    for (size_t i = 0; i < shape.size(); ++i) {
      if (!shape[i].defined() || !shape[i].dtype().is_scalar() ||
          is_negative_const(shape[i])) {
        return false;
      }
    }
    return true;
  }

  PrimExpr MakeBound(const DataType &type, const Array<PrimExpr> &shape) {
    // We have already checked the shape size to be greater then 0.
    PrimExpr bound = MulNode::make(make_const(shape[0].dtype(), type.lanes()), shape[0]);
    for (size_t i = 1; i < shape.size(); ++i) {
      bound = MulNode::make(
          bound, MulNode::make(make_const(bound.dtype(), type.lanes()), shape[i]));
    }
    return bound;
  }

  // The buffer assignment map
  // Variable remap
  std::unordered_map<const VarNode*, PrimExpr> var_remap_;
  // Buffer map
  std::unordered_map<Buffer, BufferEntry, ObjectHash, ObjectEqual> buf_map_;
  // Dimension alignment
  std::unordered_map<Buffer, std::vector<DimAlignInfo>,
                     ObjectHash, ObjectEqual> dim_align_;
  // Storage scope
  std::unordered_map<const Object*, std::string> storage_scope_;
  // The current thread scope.
  std::vector<ThreadScope> curr_thread_scope_;
  // Collects shapes.
  std::vector<std::pair<Var, Array<PrimExpr>>> shape_collector_;
  // bounds populator. We really need the analyzer from it.
  // However
  IRVisitorWithAnalyzer* bound_analyzer_;
  // The size of cacheline
  int cache_line_size_;
  // The current stage is an OpenGL shader.
  bool is_opengl_{false};
  // Whether to mark load/store with theirs bounds.
  bool create_bound_attributes_{false};
};

PrimFunc StorageFlatten(PrimFunc func,
                        int cache_line_size,
                        bool create_bound_attributes) {
  auto fptr = func.CopyOnWrite();

  IRVisitorWithAnalyzer bound_analyzer;
  bound_analyzer(fptr->body);
  fptr->body = StorageFlattener(fptr->buffer_map,
                                cache_line_size,
                                create_bound_attributes,
                                &bound_analyzer)(std::move(fptr->body));
  return func;
}


namespace transform {

// TODO(tvm-team): consolidate configs to the PassContext
Pass StorageFlatten(int cache_line_size,
                    bool create_bound_attributes) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return StorageFlatten(
        std::move(f), cache_line_size, create_bound_attributes);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.StorageFlatten", {});
}

TVM_REGISTER_GLOBAL("tir.transform.StorageFlatten")
.set_body_typed(StorageFlatten);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
