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
 *  Copyright (c) 2016 by Contributors
 * \file storage_flatten.cc
 */
// Flattens storage from multi-dimensional array to 1D
// buffer access as in Halide pipeline.
#include <tvm/ir.h>
#include <tvm/expr.h>
#include <tvm/operation.h>
#include <tvm/ir_mutator.h>
#include <tvm/expr_operator.h>
#include <tvm/ir_pass.h>
#include <tvm/buffer.h>
#include <tvm/target_info.h>
#include <tvm/runtime/device_api.h>
#include <unordered_map>
#include "ir_util.h"
#include "arg_binder.h"
#include "../arithmetic/compute_expr.h"
#include "../runtime/thread_storage_scope.h"

namespace tvm {
namespace ir {

using runtime::StorageRank;
using runtime::StorageScope;
using runtime::ThreadScope;
using intrinsic::tvm_address_of;

class StorageFlattener : public IRMutator {
 public:
  explicit StorageFlattener(Map<Tensor, Buffer> extern_buffer,
                            int cache_line_size, bool create_bound_attributes)
      : create_bound_attributes_(create_bound_attributes) {
    for (auto kv : extern_buffer) {
      BufferEntry e;
      e.buffer = kv.second;
      e.external = true;
      buf_map_[TensorKey{kv.first->op, kv.first->value_index}] = e;
    }
    cache_line_size_ = cache_line_size;
  }

  Stmt Mutate_(const Store* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Store>();
    auto it = var_remap_.find(op->buffer_var.get());
    if (it != var_remap_.end() &&
        !it->second.same_as(op->buffer_var)) {
      CHECK(it->second.as<Variable>());
      VarExpr buf_var(it->second.node_);
      return Store::make(buf_var, op->value, op->index, op->predicate);
    } else {
      return stmt;
    }
  }

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == attr::realize_scope) {
      storage_scope_[op->node.get()] = op->value.as<StringImm>()->value;
      return this->Mutate(op->body);
    } else if (op->attr_key == attr::double_buffer_scope &&
               op->node.node_->derived_from<OperationNode>()) {
      Operation func(op->node.node_);
      Stmt body = Mutate(op->body);
      for (int i = 0; i < func->num_outputs(); ++i) {
        TensorKey key{func, i};
        auto it = buf_map_.find(key);
        CHECK(it != buf_map_.end())
            << "Cannot find allocated buffer for " << key.f;
        body = AttrStmt::make(
            it->second.buffer->data, op->attr_key, op->value, body);
      }
      return body;
    } else if (op->attr_key == attr::thread_extent) {
      IterVar iv(op->node.node_);
      ThreadScope ts = ThreadScope::make(iv->thread_tag);
      curr_thread_scope_.push_back(ts);
      Stmt stmt = IRMutator::Mutate_(op, s);
      curr_thread_scope_.pop_back();
      return stmt;
    } else if (op->attr_key == attr::buffer_bind_scope) {
      return HandleBufferBindScope(op);
    } else if (op->attr_key == attr::buffer_dim_align) {
      Tensor tensor(op->node.node_);
      const Call* tuple = op->value.as<Call>();
      CHECK(tuple && tuple->is_intrinsic(intrinsic::tvm_tuple));
      TensorKey key{tensor->op, tensor->value_index};
      auto& vinfo = dim_align_[key];
      int dim = tuple->args[0].as<IntImm>()->value;
      if (static_cast<size_t>(dim) >= vinfo.size()) {
        vinfo.resize(dim + 1);
      }
      vinfo[dim].align_factor = tuple->args[1].as<IntImm>()->value;
      vinfo[dim].align_offset = tuple->args[2].as<IntImm>()->value;
      return this->Mutate(op->body);
    } else if (op->attr_key == attr::opengl_stage_scope) {
      is_opengl_ = true;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide* op, const Stmt& s) final {
    if (create_bound_attributes_)
      shape_collector_.clear();
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Provide>();
    TensorKey key{op->func, op->value_index};
    auto it = buf_map_.find(key);
    CHECK(it != buf_map_.end())
        << "Cannot find allocated buffer for " << key.f;
    const BufferEntry& e = it->second;
    CHECK(!e.released)
        << "Read a buffer that is already out of scope";
    if (is_opengl_) {
      return Evaluate::make(Call::make(
          Type(),
          Call::glsl_texture_store,
          {e.buffer->data, op->value},
          Call::Intrinsic));
    } else {
      Stmt body = e.buffer.vstore(e.RelIndex(op->args), op->value);
      if (create_bound_attributes_ && ShapeIsValid(e.buffer->shape)) {
        shape_collector_.push_back(
            std::make_pair(e.buffer->data, e.buffer->shape));
      }
      // To create bound attribute collector should has at least one item.
      if (create_bound_attributes_ && shape_collector_.size()) {
        for (size_t i = 0; i < shape_collector_.size(); ++i) {
          body = AttrStmt::make(
              shape_collector_[i].first, ir::attr::buffer_bound,
              MakeBound(e.buffer->dtype, shape_collector_[i].second), body);
        }
      }
      return body;
    }
  }

  Stmt Mutate_(const Realize* op, const Stmt& s) final {
    TensorKey key{op->func, op->value_index};
    if (buf_map_.count(key)) {
      CHECK(buf_map_.at(key).external);
      return this->Mutate(op->body);
    } else {
      // create a buffer entry
      BufferEntry e;
      e.bounds = op->bounds;
      Array<Expr> shape;
      for (auto r : e.bounds) {
        shape.push_back(r->extent);
      }
      // deduce current storage scope.
      auto it = storage_scope_.find(op->func.get());
      CHECK(it != storage_scope_.end())
          << "Cannot find storage scope of " << op->func
          << " value_index=" << op->value_index;
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
      int32_t const_size = Allocate::constant_allocation_size(shape);
      int align = GetTempAllocaAlignment(op->type, const_size);
      if (skey.tag.length() != 0) {
        MemoryInfo info = GetMemoryInfo(skey.to_string());
        if (info.defined()) {
          align = (info->max_simd_bits + op->type.bits() - 1) / op->type.bits();
          CHECK_LE(const_size * op->type.bits(), info->max_num_bits)
              << "Allocation exceed bound of memory tag " << skey.to_string();
        }
      }
      Array<Expr> strides;
      if (dim_align_.count(key) != 0 && shape.size() != 0) {
        std::vector<Expr> rstrides;
        const std::vector<DimAlignInfo>& avec = dim_align_[key];
        int first_dim = 0;
        Expr stride = make_const(shape[first_dim].type(), 1);
        for (size_t i = shape.size(); i != 0; --i) {
          size_t dim = i - 1;
          if (dim < avec.size() && avec[dim].align_factor != 0) {
            Expr factor = make_const(stride.type(), avec[dim].align_factor);
            Expr offset = make_const(stride.type(), avec[dim].align_offset);
            stride = stride + (factor + offset - stride % factor) % factor;
            stride = ir::Simplify(stride);
          }
          rstrides.push_back(stride);
          stride = stride * shape[dim];
        }
        strides = Array<Expr>(rstrides.rbegin(), rstrides.rend());
      }

      e.buffer = BufferNode::make(
          Var(key.GetName(), Handle()),
          op->type, shape, strides, Expr(),
          key.GetName(), skey.to_string(),
          align, 0, kDefault);

      buf_map_[key] = e;
      Stmt body = this->Mutate(op->body);
      buf_map_[key].released = true;
      Stmt ret;

      Type storage_type = e.buffer->dtype;
      // specially handle bool, lower its storage
      // type to be Int(8)(byte)
      if (storage_type == Bool()) {
        storage_type = Int(8);
      }
      if (strides.size() != 0) {
        int first_dim = 0;
        ret = Allocate::make(
            e.buffer->data, storage_type,
            {e.buffer->strides[first_dim] * e.buffer->shape[first_dim]},
            make_const(Bool(e.buffer->dtype.lanes()), true), body);
      } else {
        shape = e.buffer->shape;
        if (shape.size() == 0) {
          shape.push_back(make_const(Int(32), 1));
        }
        ret = Allocate::make(
            e.buffer->data, storage_type, shape,
            make_const(Bool(e.buffer->dtype.lanes()), true), body);
      }
      ret = AttrStmt::make(
          e.buffer->data, attr::storage_scope,
          StringImm::make(e.buffer->scope), ret);

      if (create_bound_attributes_ && ShapeIsValid(e.buffer->shape)) {
        ret = AttrStmt::make(e.buffer->data, ir::attr::buffer_bound,
                             MakeBound(e.buffer->dtype, e.buffer->shape), ret);
      }
      return ret;
    }
  }

  Expr Mutate_(const Load* op, const Expr& e) final {
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<Load>();
    auto it = var_remap_.find(op->buffer_var.get());
    if (it != var_remap_.end() &&
        !it->second.same_as(op->buffer_var)) {
      CHECK(it->second.as<Variable>());
      VarExpr buf_var(it->second.node_);
      return Load::make(op->type, buf_var, op->index, op->predicate);
    } else {
      return expr;
    }
  }

  Expr Mutate_(const Variable* op, const Expr& e) final {
    auto it = var_remap_.find(op);
    if (it != var_remap_.end()) {
      return it->second;
    } else {
      return e;
    }
  }

  Expr Mutate_(const Call* op, const Expr& olde) final {
    Expr expr = IRMutator::Mutate_(op, olde);
    op = expr.as<Call>();
    if (op != nullptr && op->call_type == Call::Halide) {
      TensorKey key{op->func, op->value_index};
      auto it = buf_map_.find(key);
      CHECK(it != buf_map_.end())
          << "Cannot find allocated buffer for " << key.f;
      const BufferEntry& e = it->second;
      CHECK(!e.released)
          << "Read a buffer that is already out of scope";

      if (create_bound_attributes_ && ShapeIsValid(e.buffer->shape)) {
        shape_collector_.push_back(
            std::make_pair(e.buffer->data, e.buffer->shape));
      }
      return e.buffer.vload(e.RelIndex(op->args), e.buffer->dtype);
    } else {
      return expr;
    }
  }

  Stmt Mutate_(const Prefetch *op, const Stmt &s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Prefetch>();
    CHECK(op != nullptr);
    TensorKey key{op->func, op->value_index};
    auto it = buf_map_.find(key);
    CHECK(it != buf_map_.end())
        << "Cannot find allocated buffer for " << key.f;
    const BufferEntry& e = it->second;

    CHECK(!e.released)
        << "Read a buffer that is already out of scope";
    CHECK_EQ(e.buffer->shape.size(), op->bounds.size())
      << "Prefetch dim should be the same as buffer dim";

    int block_size = 1,
        elem_cnt = cache_line_size_ / e.buffer->dtype.bytes(),
        shape = 0;

    int starts = op->bounds.size() - 1;
    while (starts > 0 && arith::GetConstInt(e.buffer->shape[starts], &shape)
        && elem_cnt >= block_size * shape) {
      block_size *= shape;
      starts--;
    }
    Expr stride(elem_cnt / block_size);

    Array<Expr> args;
    std::vector<VarExpr> vars;

    for (int i = op->bounds.size() - 1; i > starts; --i) {
      args.push_back(op->bounds[i]->min);
    }
    auto &func_name = op->func->func_name();
    vars.push_back(VarExpr("prefetch." + func_name + "." + std::to_string(starts), Int(32)));
    args.push_back(op->bounds[starts]->min + stride * vars.back());
    for (int i = starts - 1; i >= 0; --i) {
      vars.push_back(VarExpr("prefetch." + func_name + "." + std::to_string(i), Int(32)));
      args.push_back(vars.back() + op->bounds[i]->min);
    }
    for (int i = starts; i >= 0; --i) {
      if (i < starts) {
        stmt = For::make(
            vars[i], 0, op->bounds[i]->extent, ForType::Serial, DeviceAPI::None, stmt);
      } else {
        Expr load = e.buffer.vload(e.RelIndex(args), e.buffer->dtype);
        Expr address = Call::make(Handle(), tvm_address_of, {load}, Call::PureIntrinsic);
        Expr prefetch = Call::make(op->type, Call::prefetch, {address, 0, 3, 1}, Call::Intrinsic);
        stmt = Evaluate::make(prefetch);
        Expr extent = (op->bounds[i]->extent - 1) / stride + 1;
        stmt = For::make(vars[i], 0, extent, ForType::Serial, DeviceAPI::None, stmt);
      }
    }
    return stmt;
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
  Stmt HandleBufferBindScope(const AttrStmt* op) {
    Array<NodeRef> arr(op->node.node_);
    CHECK_EQ(arr.size(), 2U);
    const BufferNode* buffer = arr[0].as<BufferNode>();
    const TensorNode* tensor = arr[1].as<TensorNode>();
    const Call* tuple = op->value.as<Call>();
    CHECK(buffer && tensor);
    CHECK(tuple && tuple->is_intrinsic(intrinsic::tvm_tuple));
    TensorKey key{tensor->op, tensor->value_index};
    CHECK(buf_map_.count(key))
        << "Cannot find buffer of " << tensor->op << " value=" << tensor->value_index;
    const BufferEntry& be = buf_map_.at(key);
    CHECK(!be.released);
    CHECK_EQ(tuple->args.size(), be.buffer->shape.size() * 2);
    Array<Expr> begins, extents;
    if (be.bounds.size() != 0) {
      CHECK_EQ(tuple->args.size(), be.bounds.size() * 2);
      for (size_t i = 0; i < be.buffer->shape.size(); ++i) {
        begins.push_back(tuple->args[2 * i] - be.bounds[i]->min);
        extents.push_back(tuple->args[2 * i + 1]);
      }
    } else {
      for (size_t i = 0; i < tuple->args.size(); i += 2) {
        begins.push_back(tuple->args[i]);
        extents.push_back(tuple->args[i + 1]);
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
    binder.BindBuffer(Buffer(arr[0].node_), slice, buffer->name, true);
    // Apply the remaps
    Stmt body = MergeNest(binder.asserts(), op->body);
    body = MergeNest(binder.init_nest(), body);
    body = this->Mutate(body);
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
    inline Array<Expr> RelIndex(Array<Expr> args) const {
      if (bounds.size() != 0) {
        Array<Expr> index;
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

  bool ShapeIsValid(const Array<Expr> &shape) {
    // Zero-dimensional tensor does not need boundary check.
    if (!shape.size())
      return false;

    for (size_t i = 0; i < shape.size(); ++i) {
      if (!shape[i].defined() || !shape[i].type().is_scalar() ||
          is_negative_const(shape[i])) {
        return false;
      }
    }
    return true;
  }

  Expr MakeBound(const Type &type, const Array<Expr> &shape) {
    // We have already checked the shape size to be greater then 0.
    Expr bound = Mul::make(make_const(shape[0].type(), type.lanes()), shape[0]);
    for (size_t i = 1; i < shape.size(); ++i) {
      bound = Mul::make(
          bound, Mul::make(make_const(bound.type(), type.lanes()), shape[i]));
    }
    return bound;
  }

  // The buffer assignment map
  // Variable remap
  std::unordered_map<const Variable*, Expr> var_remap_;
  // Buffer map
  std::unordered_map<TensorKey, BufferEntry> buf_map_;
  // Dimension alignment
  std::unordered_map<TensorKey, std::vector<DimAlignInfo> > dim_align_;
  // Storage scope
  std::unordered_map<const Node*, std::string> storage_scope_;
  // The current thread scope.
  std::vector<ThreadScope> curr_thread_scope_;
  // Collects shapes.
  std::vector<std::pair<VarExpr, Array<Expr>>> shape_collector_;
  // The size of cacheline
  int cache_line_size_;
  // The current stage is an OpenGL shader.
  bool is_opengl_{false};
  // Whether to mark load/store with theirs bounds.
  bool create_bound_attributes_{false};
};

Stmt StorageFlatten(Stmt stmt, Map<Tensor, Buffer> extern_buffer,
                    int cache_line_size, bool create_bound_attributes) {
  stmt =
      StorageFlattener(extern_buffer, cache_line_size, create_bound_attributes)
          .Mutate(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace tvm
