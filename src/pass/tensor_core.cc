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
 *  Copyright (c) 2019 by Contributors
 * \file tensor_core.cc
 */
// IR Passes for TensorCore CodeGen
#include <tvm/ir.h>
#include <tvm/expr.h>
#include <tvm/operation.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
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

struct Tile {
  int m{-1};
  int n{-1};
  int k{-1};
};

class BodyVisitor : public IRVisitor {
 public:
  BodyVisitor() {}
  using IRVisitor::Visit_;

  void Visit_(const Reduce* op) override {
    auto* comm_add = op->combiner->result[0].as<Add>();
    if (comm_add == nullptr || op->combiner->result.size() > 1) {
      return;
    }
    for (auto source : op->source) {
      auto* cast = source.as<Cast>();
      if (cast == nullptr ||
          !(cast->type == Float(32) || cast->type == Int(32))) {
        continue;
      }

      auto* mul = cast->value.as<Mul>();
      if (mul == nullptr) {
        continue;
      }

      tensorcore_candidate_ = true;
      IRVisitor::Visit(source);
    }
  }

  void Visit_(const Call* op) override {
    IRVisitor::Visit_(op);
    args_.insert(std::make_pair(op->name, op->args));
  }

  friend class ScheduleAnalyser;

 private:
  std::unordered_map<std::string, Array<Expr>> args_;
  bool tensorcore_candidate_{false};
};

class ScheduleAnalyser {
 public:
  ScheduleAnalyser() {}

  void MatrixIdentify(Schedule schedule) {
    // TODO(minmin): handle the case where MatMul is not the output stage
    for (Operation output : schedule->outputs) {
      const ComputeOpNode* compute = output.as<ComputeOpNode>();
      if (compute == nullptr) {
        // Not a ComputeOp
        continue;
      }
      auto axis = compute->axis;
      auto reduce_axis = compute->reduce_axis;
      if (axis.size() < 2 || reduce_axis.size() != 1) {
        continue;
      }
      const Variable* axis_var[2];
      const Variable* reduce_axis_var;
      axis_var[0] = axis[axis.size()-2]->var.as<Variable>();
      axis_var[1] = axis[axis.size()-1]->var.as<Variable>();
      reduce_axis_var = reduce_axis[0]->var.as<Variable>();

      BodyVisitor body_visitor;
      for (Expr expr : compute->body) {
        body_visitor.Visit(expr);
      }
      if (!body_visitor.tensorcore_candidate_) {
        continue;
      }
      for (auto iter : body_visitor.args_) {
        auto name = iter.first;
        auto args = iter.second;
        if (args.size() != 2) {
          continue;
        }
        const Variable* var0 = args[0].as<Variable>();
        const Variable* var1 = args[1].as<Variable>();
        if (var0 == nullptr || var1 == nullptr) {
          continue;
        }
        std::string matrix_abc, major;
        if (var0 == reduce_axis_var && var1 == axis_var[1]) {
          matrix_abc = "matrix_a";
          major = "col_major";
        } else if (var0 == reduce_axis_var && var1 == axis_var[0]) {
          matrix_abc = "matrix_b";
          major = "row_major";
        } else if (var0 == axis_var[1] && var1 == reduce_axis_var) {
          matrix_abc = "matrix_a";
          major = "row_major";
        } else if (var0 == axis_var[0] && var1 == reduce_axis_var) {
          matrix_abc = "matrix_b";
          major = "col_major";
        }
        matrix_abc_.insert(std::make_pair(name, matrix_abc));
        matrix_major_.insert(std::make_pair(name, major));
      }
      matrix_abc_.insert(std::make_pair(compute->name, "accumulator"));
      matrix_major_.insert(std::make_pair(compute->name, "col_major"));
    }
  }

  friend class MMAMatcher;
  friend class BufferAnalyser;
  friend class TensorCoreIRMutator;

 private:
  std::unordered_map<std::string, std::string> matrix_abc_;
  std::unordered_map<std::string, std::string> matrix_major_;
};

class IndexVisitor : public IRVisitor {
 public:
  IndexVisitor() {}
  using IRVisitor::Visit_;

  void Visit_(const Variable* op) override {
    loop_scaling_.insert(std::make_pair(op, scaling_factor_));
  }

  friend class BufferAnalyser;
  friend class TensorCoreIRMutator;

 private:
  std::unordered_map<const Variable*, unsigned> loop_scaling_;
  unsigned scaling_factor_;
};

class MMAMatcher: public IRVisitor {
 public:
  explicit MMAMatcher(Map<Tensor, Buffer> extern_buffer,
                      const ScheduleAnalyser &schedule_analyser,
                      double cuda_compute_capability, double cuda_version)
        : matrix_abc_(schedule_analyser.matrix_abc_),
          matched_(false) {
    for (auto kv : extern_buffer) {
      BufferInfo bi;
      bi.name = kv.second->name;
      bi.dtype = kv.second->dtype;
      bi.external = true;
      buf_map_[TensorKey{kv.first->op, kv.first->value_index}] = bi;
    }
    // Int wmma is supported when cuda version >= 10.0 && cuda arch >= 720
    if (cuda_compute_capability >= 7.20 && cuda_version >= 10.0) {
      support_int_wmma_ = true;
    }
  }
  using IRVisitor::Visit_;

  void Visit_(const AttrStmt* op) final {
    if (op->attr_key == attr::realize_scope) {
      storage_scope_[op->node.get()] = op->value.as<StringImm>()->value;
      Visit(op->body);
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const Provide* op) final {
    IRVisitor::Visit_(op);
    TensorKey key{op->func, op->value_index};
    auto it = buf_map_.find(key);
    if (it == buf_map_.end()) {
      return;
    }
    const BufferInfo& bi = it->second;
    if (bi.released) {
      return;
    }

    Expr a, b, c;
    if (mma_sync_match_(op, bi, &a, &b, &c)) {
      matched_ = true;
      Array<Expr> operands({a, b, c});
      mma_sync_.insert(std::make_pair(op, operands));
    }
  }

  void Visit_(const Realize* op) final {
    TensorKey key{op->func, op->value_index};
    if (buf_map_.count(key)) {
      if (!buf_map_.at(key).external) {
        return;
      }
      Visit(op->body);
    } else {
      BufferInfo bi;
      bi.name = key.GetName();
      bi.dtype = op->type;

      buf_map_[key] = bi;
      Visit(op->body);
      buf_map_[key].released = true;
    }
  }

  bool Matched() {
    return matched_;
  }

  friend class BufferAnalyser;
  friend class TensorCoreIRMutator;

 private:
  struct BufferInfo {
    std::string name;
    Type dtype;
    bool external{false};
    bool released{false};

    bool same_as(const BufferInfo &bi) {
      if (this->dtype != bi.dtype) return false;
      if (this->name != bi.name) return false;
      if (this->external != bi.external) return false;
      if (this->released != bi.released) return false;
      return true;
    }
  };

  bool check_local_buffer_(const Call* op, BufferInfo* bi) {
    if (op->call_type == Call::Halide) {
      auto it = storage_scope_.find(op->func.get());
      if (it == storage_scope_.end()) {
        return false;
      }
      const std::string& strkey = it->second;
      if (strkey != "local") {
        return false;
      }

      TensorKey key{op->func, op->value_index};
      auto it1 = buf_map_.find(key);
      if (it1 == buf_map_.end()) {
        return false;
      }
      *bi = it1->second;
      if (bi->released) {
        return false;
      }
      return true;
    }
    return false;
  }

  bool mma_sync_match_(const Provide* op, BufferInfo store_buffer,
                        Expr* a, Expr* b, Expr* c) {
    auto* add = op->value.as<Add>();
    if (add == nullptr) {
      return false;
    }

    auto* load_c = add->a.as<Call>();
    BufferInfo buffer_c;
    if (!check_local_buffer_(load_c, &buffer_c)
        || !buffer_c.same_as(store_buffer)
        || !(buffer_c.dtype == Float(32) ||
             (support_int_wmma_ && buffer_c.dtype == Int(32)))) {
      return false;
    }

    auto* cast = add->b.as<Cast>();
    if (cast == nullptr ||
        !(cast->type == Float(32) ||
          (support_int_wmma_ && cast->type == Int(32)))) {
      return false;
    }

    auto* mul = cast->value.as<Mul>();
    if (mul == nullptr) {
      return false;
    }
    auto* load_a = mul->a.as<Call>();
    BufferInfo buffer_a;
    if (!check_local_buffer_(load_a, &buffer_a)
        || !(buffer_a.dtype == Float(16) ||
             (support_int_wmma_ && buffer_a.dtype == Int(8)))) {
      return false;
    }

    auto* load_b = mul->b.as<Call>();
    BufferInfo buffer_b;
    if (!check_local_buffer_(load_b, &buffer_b)
        || !(buffer_b.dtype == Float(16) ||
             (support_int_wmma_ && buffer_b.dtype == Int(8)))) {
      return false;
    }

    frag_reg_.insert(buffer_c.name);
    frag_reg_.insert(buffer_a.name);
    frag_reg_.insert(buffer_b.name);

    std::string input0 = buffer_a.name;
    auto pos0 = input0.find(".");
    if (pos0 != std::string::npos) {
      input0 = input0.substr(0, pos0);
    }
    auto it0 = matrix_abc_.find(input0);
    std::string input1 = buffer_b.name;
    auto pos1 = input1.find(".");
    if (pos1 != std::string::npos) {
      input1 = input1.substr(0, pos1);
    }
    auto it1 = matrix_abc_.find(input1);
    if (it0 == matrix_abc_.end() || it1 == matrix_abc_.end()) {
      return false;
    }
    if (it0->second == "matrix_a" && it1->second == "matrix_b") {
      *a = mul->a;
      *b = mul->b;
    } else if (it0->second == "matrix_b" && it1->second == "matrix_a") {
      *a = mul->b;
      *b = mul->a;
    } else {
      return false;
    }
    *c = add->a;
    return true;
  }

  std::unordered_map<TensorKey, BufferInfo> buf_map_;
  std::unordered_map<const Node*, std::string> storage_scope_;

  std::unordered_map<std::string, std::string> matrix_abc_;
  std::unordered_set<std::string> frag_reg_;
  std::unordered_map<const Provide*, Array<Expr>> mma_sync_;
  bool matched_;
  bool support_int_wmma_;
};

class BufferAnalyser : public IRVisitor {
 public:
  explicit BufferAnalyser(Map<Tensor, Buffer> extern_buffer,
                          double cuda_version,
                          const ScheduleAnalyser &schedule_analyser,
                          const MMAMatcher &mma_matcher)
      : matrix_abc_(schedule_analyser.matrix_abc_),
        matrix_major_(schedule_analyser.matrix_major_),
        frag_reg_(mma_matcher.frag_reg_),
        invalid_(false),
        cuda_version_(cuda_version) {
    for (auto kv : extern_buffer) {
      BufferInfo bi;
      bi.name = kv.second->name;
      bi.dtype = kv.second->dtype;
      bi.strides = kv.second->strides;
      bi.shape = kv.second->shape;
      bi.external = true;
      buf_map_[TensorKey{kv.first->op, kv.first->value_index}] = bi;
    }
  }
  using IRVisitor::Visit_;

  void Visit_(const AttrStmt* op) final {
    if (op->attr_key == attr::thread_extent) {
      if (const IntImm* value = op->value.as<IntImm>()) {
        thread_extent_.insert(
            std::make_pair(
                op->node.as<IterVarNode>()->var->name_hint,
                value->value));
      }
      IRVisitor::Visit_(op);
    } else if (op->attr_key == attr::realize_scope) {
      storage_scope_[op->node.get()] = op->value.as<StringImm>()->value;
      Visit(op->body);
    } else if (op->attr_key == attr::buffer_dim_align) {
      Tensor tensor = Downcast<Tensor>(op->node);
      const Call* tuple = op->value.as<Call>();
      CHECK(tuple && tuple->is_intrinsic(intrinsic::tvm_tuple));
      TensorKey key{tensor->op, tensor->value_index};
      auto& vinfo = dim_align_[key];
      size_t dim = tuple->args[0].as<IntImm>()->value;
      if (dim >= vinfo.size()) {
        vinfo.resize(dim + 1);
      }
      vinfo[dim].align_factor = tuple->args[1].as<IntImm>()->value;
      vinfo[dim].align_offset = tuple->args[2].as<IntImm>()->value;
      Visit(op->body);
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const Provide* op) final {
    IRVisitor::Visit_(op);
    TensorKey key{op->func, op->value_index};
    auto it = buf_map_.find(key);
    CHECK(it != buf_map_.end())
        << "Cannot find allocated buffer for " << key.f;
    const BufferInfo& bi = it->second;
    CHECK(!bi.released)
        << "Read a buffer that is already out of scope";

    Array<Expr> strides;
    strides = bi.strides.size() > 0 ? bi.strides : bi.shape;
    strides_.insert(std::make_pair(key.GetName(), strides));

    if (matrix_abc_.count(key.GetName())) {
      if (strides.size() < 2) {
        invalid_ = true;
        return;
      }
      for (int i = int(strides.size()) - 1; i >= int(strides.size()) - 2; --i) {
        const IntImm* stride = strides[i].as<IntImm>();
        if (stride == nullptr || stride->value % 16 != 0) {
          invalid_ = true;
          return;
        }
      }
    }

    if (frag_reg_.count(bi.name)) {
      Expr dst = Call::make(bi.dtype,
                            bi.name,
                            op->args,
                            Call::Halide,
                            op->func,
                            0);
      frag_load_.insert(std::make_pair(op, dst));

      auto rel_index = bi.RelIndex(op->args);
      if (op->args.size() < 2) {
        invalid_ = true;
        return;
      }
      std::vector<int> tile_size;
      for (int i = int(op->args.size()) - 1; i >= int(op->args.size()) - 2; --i) {
        index_visitor.scaling_factor_ = 16;
        if (const IntImm* shape = bi.shape[i].as<IntImm>()) {
          tile_size.push_back(shape->value);
          index_visitor.scaling_factor_ = shape->value;
        } else {
          invalid_ = true;
          return;
        }
        auto index = rel_index[i];
        auto simplified_index = ir::Simplify(index);
        index_visitor.Visit(simplified_index);
      }

      std::string input_name = bi.name;
      auto pos = input_name.find(".");
      if (pos != std::string::npos) {
        input_name = input_name.substr(0, pos);
      }
      auto it = matrix_abc_.find(input_name);
      auto it2 = matrix_major_.find(input_name);
      bool ret = true;
      if (it != matrix_abc_.end() && it2 != matrix_major_.end()) {
        if (it->second == "matrix_a" && it2->second == "col_major") {
          ret &= assign_or_check_(&thread_tile_.m, tile_size[0]);
          ret &= assign_or_check_(&thread_tile_.k, tile_size[1]);
        }
        if (it->second == "matrix_a" && it2->second == "row_major") {
          ret &= assign_or_check_(&thread_tile_.k, tile_size[0]);
          ret &= assign_or_check_(&thread_tile_.m, tile_size[1]);
        }
        if (it->second == "matrix_b" && it2->second == "col_major") {
          ret &= assign_or_check_(&thread_tile_.k, tile_size[0]);
          ret &= assign_or_check_(&thread_tile_.n, tile_size[1]);
        }
        if (it->second == "matrix_b" && it2->second == "row_major") {
          ret &= assign_or_check_(&thread_tile_.n, tile_size[0]);
          ret &= assign_or_check_(&thread_tile_.k, tile_size[1]);
        }
        if (it->second == "accumulator") {
          ret &= assign_or_check_(&thread_tile_.m, tile_size[0]);
          ret &= assign_or_check_(&thread_tile_.n, tile_size[1]);
        }
        if (!ret) {
          invalid_ = true;
          return;
        }
      }
    }

    const Call* value = op->value.as<Call>();
    if (value != nullptr && frag_reg_.count(value->name)) {
      Expr dst = Call::make(bi.dtype,
                            bi.name,
                            op->args,
                            Call::Halide,
                            op->func,
                            0);
      frag_store_.insert(std::make_pair(op, dst));
    }
  }

  void Visit_(const Call* op) final {
    IRVisitor::Visit_(op);
    if (op->call_type == Call::Halide) {
      TensorKey key{op->func, op->value_index};
      auto it = buf_map_.find(key);
      CHECK(it != buf_map_.end())
          << "Cannot find allocated buffer for " << key.f;
      const BufferInfo& bi = it->second;
      CHECK(!bi.released)
          << "Read a buffer that is already out of scope";

      Array<Expr> strides;
      strides = bi.strides.size() > 0 ? bi.strides : bi.shape;
      strides_.insert(std::make_pair(op->name, strides));

      if (matrix_abc_.count(op->name)) {
        if (strides.size() < 2) {
          invalid_ = true;
          return;
        }
        for (int i = int(strides.size()) - 1; i >= int(strides.size()) - 2; --i) {
          const IntImm* stride = strides[i].as<IntImm>();
          if (stride == nullptr || stride->value % 16 != 0) {
            invalid_ = true;
            return;
          }
        }
      }

      if (!frag_reg_.count(bi.name)) {
        return;
      }

      auto rel_index = bi.RelIndex(op->args);
      if (op->args.size() < 2) {
        invalid_ = true;
        return;
      }
      for (int i = int(op->args.size()) - 1; i >= int(op->args.size()) - 2; --i) {
        index_visitor.scaling_factor_ = 16;
        if (const IntImm* shape = bi.shape[i].as<IntImm>()) {
          index_visitor.scaling_factor_ = shape->value;
        }
        auto index = rel_index[i];
        auto simplified_index = ir::Simplify(index);
        index_visitor.Visit(simplified_index);
      }
    }
  }

  void Visit_(const Realize* op) final {
    TensorKey key{op->func, op->value_index};
    if (buf_map_.count(key)) {
      CHECK(buf_map_.at(key).external);
      Visit(op->body);
    } else {
      // create a buffer entry
      BufferInfo bi;

      bi.bounds = op->bounds;
      Array<Expr> shape;
      for (auto r : bi.bounds) {
        shape.push_back(r->extent);
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
            stride = stride + \
              indexmod(factor + offset - indexmod(stride, factor), factor);
            stride = ir::Simplify(stride);
          }
          rstrides.push_back(stride);
          stride = stride * shape[dim];
        }
        strides = Array<Expr>(rstrides.rbegin(), rstrides.rend());
      }

      bi.name = key.GetName();
      bi.dtype = op->type;
      bi.strides = strides;
      bi.shape = shape;

      buf_map_[key] = bi;
      Visit(op->body);
      buf_map_[key].released = true;
    }
  }

  bool QualifiedForTensorCore() {
    if (invalid_) {
      return false;
    }
    auto itx = thread_extent_.find("threadIdx.x");
    if (itx == thread_extent_.end()) {
      return false;
    }
    int warp_threads_x = itx->second;
    warp_tile_.m = warp_threads_x * thread_tile_.m;
    warp_threads_y_ = 32 / warp_threads_x;
    auto ity = thread_extent_.find("threadIdx.y");
    if (ity == thread_extent_.end()) {
      return false;
    }
    if (ity->second < warp_threads_y_ || ity->second % warp_threads_y_ != 0) {
      return false;
    }
    warp_tile_.n = warp_threads_y_ * thread_tile_.n;
    warp_tile_.k = thread_tile_.k;
    return SupportedWarpTile(warp_tile_);
  }

  friend class TensorCoreIRMutator;

 private:
  struct DimAlignInfo {
    int align_factor{0};
    int align_offset{0};
  };

  struct BufferInfo {
    std::string name;
    Type dtype;
    Array<Expr> strides;
    Array<Expr> shape;
    Region bounds;
    bool external{false};
    bool released{false};

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

  bool assign_or_check_(int* dst, int src) {
    if (*dst <= 0) {
      *dst = src;
      return true;
    }
    if (*dst == src) {
      return true;
    }
    return false;
  }

  bool SupportedWarpTile(Tile warp_tile) {
    if (warp_tile.m == 16 &&
        warp_tile.n == 16 &&
        warp_tile.k == 16) {
      return true;
    }
    if (cuda_version_ >= 10.0 &&
        warp_tile.m == 8 &&
        warp_tile.n == 32 &&
        warp_tile.k == 16) {
      return true;
    }
    if (cuda_version_ >= 10.0 &&
        warp_tile.m == 32 &&
        warp_tile.n == 8 &&
        warp_tile.k == 16) {
      return true;
    }
    return false;
  }

  std::unordered_map<TensorKey, BufferInfo> buf_map_;
  std::unordered_map<TensorKey, std::vector<DimAlignInfo> > dim_align_;
  std::unordered_map<const Node*, std::string> storage_scope_;

  std::unordered_map<std::string, std::string> matrix_abc_;
  std::unordered_map<std::string, std::string> matrix_major_;
  std::unordered_set<std::string> frag_reg_;
  std::unordered_map<std::string, Array<Expr>> strides_;
  std::unordered_map<const Provide*, Expr> frag_load_;
  std::unordered_map<const Provide*, Expr> frag_store_;
  Tile warp_tile_;
  int warp_threads_y_{-1};
  IndexVisitor index_visitor;
  bool invalid_;
  std::unordered_map<std::string, int> thread_extent_;
  Tile thread_tile_;
  double cuda_version_;
};

class ThreadIdxMutator : public IRMutator {
 public:
  explicit ThreadIdxMutator(Expr warp_y): warp_y_(warp_y) {}

  Expr Mutate_(const Variable* op, const Expr& olde) override {
    // thread index unification inside a warp
    Expr expr = IRMutator::Mutate_(op, olde);
    op = expr.as<Variable>();
    if (op != nullptr) {
      if (op->name_hint == "threadIdx.x") {
        Expr zero = IntImm::make(Int(32), 0);
        return zero;
      }
      if (op->name_hint == "threadIdx.y") {
        Expr div = Div::make(expr, warp_y_);
        Expr mul = Mul::make(div, warp_y_);
        return mul;
      }
    }
    return expr;
  }

 private:
  Expr warp_y_;
};

class TensorCoreIRMutator : public IRMutator {
 public:
  explicit TensorCoreIRMutator(const ScheduleAnalyser &schedule_analyser,
    const MMAMatcher &mma_matcher, const BufferAnalyser &buffer_analyser,
    double cuda_compute_capability, double cuda_version)
      : matrix_abc_(schedule_analyser.matrix_abc_),
      matrix_major_(schedule_analyser.matrix_major_),
      mma_sync_(mma_matcher.mma_sync_),
      strides_(buffer_analyser.strides_),
      frag_reg_(buffer_analyser.frag_reg_),
      loop_scaling_(buffer_analyser.index_visitor.loop_scaling_),
      frag_load_(buffer_analyser.frag_load_),
      frag_store_(buffer_analyser.frag_store_),
      warp_tile_(buffer_analyser.warp_tile_),
      warp_threads_y_(buffer_analyser.warp_threads_y_),
      cuda_compute_capability_(cuda_compute_capability),
      cuda_version_(cuda_version) {}

  Stmt Mutate_(const Realize* op, const Stmt& s) final {
    Array<Expr> min_bound;
    for (auto i : op->bounds) {
      min_bound.push_back(i->min);
    }
    TensorKey key{op->func, op->value_index};
    min_bounds_[key] = min_bound;
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Realize>();
    if (op != nullptr) {
      if (!frag_reg_.count(key.GetName())) {
        return stmt;
      }

      // TODO(chenfan): new_extent should be same as the fragment shape
      Expr new_extent = make_const(Int(32), 16);
      Region new_bounds;
      for (size_t i = 0; i < op->bounds.size() - 2; ++i) {
        new_bounds.push_back(op->bounds[i]);
      }
      for (size_t i = op->bounds.size() - 2; i < op->bounds.size(); ++i) {
        auto min = op->bounds[i]->min;
        new_bounds.push_back(
            Range::make_by_min_extent(min, new_extent));
      }

      return Realize::make(op->func, op->value_index,
                           op->type, new_bounds,
                           op->condition, op->body);
    }
    return stmt;
  }

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    if (op->attr_key == attr::realize_scope) {
      auto node = op->node.as<OperationNode>();
      if (node != nullptr) {
        if (!frag_reg_.count(node->name)) {
          return stmt;
        }

        auto input0 = node->name;
        auto pos0 = input0.find(".");
        if (pos0 != std::string::npos) {
          input0 = input0.substr(0, pos0);
        }
        auto it0 = matrix_abc_.find(input0);
        if (it0 == matrix_abc_.end()) {
          std::cout << "Error!!!! matrix_abc_ not found" << std::endl;
        }
        auto matrix_abc = "wmma." + it0->second;
        Stmt body = Mutate(op->body);
        return AttrStmt::make(op->node,
                              op->attr_key,
                              matrix_abc,
                              body);
      }
    }
    return stmt;
  }

  Stmt Mutate_(const Provide* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    auto it = mma_sync_.find(op);
    if (it != mma_sync_.end()) {
      auto operands = it->second;
      Expr a = operands[0];
      auto ca = a.as<Call>();
      Expr b = operands[1];
      auto cb = b.as<Call>();
      Expr c = operands[2];
      auto cc = c.as<Call>();

      NodePtr<BufferNode> buffer_node_a = make_node<BufferNode>();
      NodePtr<BufferNode> buffer_node_b = make_node<BufferNode>();
      NodePtr<BufferNode> buffer_node_c = make_node<BufferNode>();

      auto mma_sync_call =
        [&buffer_node_a, &buffer_node_b, &buffer_node_c]
        (const Buffer &buffer) {
          Buffer buffer_a(buffer_node_a);
          Buffer buffer_b(buffer_node_b);
          Buffer buffer_c(buffer_node_c);

          // Notice: index needs to be set, now hardcode to 0
          return Evaluate::make(
                  Call::make(Handle(),
                        intrinsic::tvm_mma_sync,
                        {buffer_c->data, buffer_c->elem_offset,
                        buffer_a->data, buffer_a->elem_offset,
                        buffer_b->data, buffer_b->elem_offset,
                        buffer_c->data, buffer_c->elem_offset},
                        Call::Intrinsic));
        };

      auto call_add_c =
        [this, &cc, &buffer_node_c, &mma_sync_call](const Buffer &buffer) {
          return add_buffer_bind_scope(cc, buffer_node_c,
            TensorKey{cc->func, cc->value_index}, mma_sync_call, Float(32));
        };

      auto call_add_b =
        [this, &cb, &buffer_node_b, &call_add_c](const Buffer &buffer) {
          return add_buffer_bind_scope(cb, buffer_node_b,
            TensorKey{cb->func, cb->value_index}, call_add_c, Float(16));
        };

      return add_buffer_bind_scope(ca, buffer_node_a,
        TensorKey{ca->func, ca->value_index}, call_add_b, Float(16));
    }

    auto it2 = frag_load_.find(op);
    if (it2 != frag_load_.end()) {
      Expr dst = it2->second;
      bool support_int_wmma = \
        (cuda_compute_capability_ >= 7.2 && cuda_version_ >= 10.0);
      if (op->value.as<FloatImm>() != nullptr ||
          (support_int_wmma && op->value.as<IntImm>() != nullptr)) {
        auto call = dst.as<Call>();

        auto fill_fragment_call =
          [this, &op](const Buffer &buffer) {
            return Evaluate::make(
                    Call::make(Handle(),
                              intrinsic::tvm_fill_fragment,
                              {buffer->data,
                              warp_tile_.m, warp_tile_.n, warp_tile_.k,
                              buffer->elem_offset, op->value},
                              Call::Intrinsic));
          };

        NodePtr<BufferNode> buffer_node = make_node<BufferNode>();
        return add_buffer_bind_scope(call, buffer_node,
                                      TensorKey{call->func, call->value_index},
                                      fill_fragment_call, Float(32));
      }

      const Call* value = op->value.as<Call>();
      CHECK(value != nullptr)
          << "Can only load fragment from a buffer";

      auto it = strides_.find(value->name);
      CHECK(it != strides_.end())
          << "Cannot find stride for " << value->name;

      auto strides = it->second;
      Expr stride = strides[strides.size()-1];

      // thread index unification inside a warp
      Expr warp_y = IntImm::make(Int(32), warp_threads_y_);
      ThreadIdxMutator thread_idx_mutator(warp_y);
      Expr mutated_value = thread_idx_mutator.Mutate(op->value);
      Expr src = Call::make(Float(16),
                            "&",
                            {mutated_value},
                            Call::Extern);

      auto call = dst.as<Call>();
      Expr matrix_major;
      auto call_name = call->name;
      auto pos = call_name.find(".");
      if (pos != std::string::npos) {
        call_name = call_name.substr(0, pos);
      }
      auto iter2 = matrix_major_.find(call_name);
      CHECK(iter2 != matrix_major_.end())
          << "Can not determine matrix major for " << call_name;
      if (iter2->second == "col_major") {
        matrix_major = StringImm::make("col_major");
      } else if (iter2->second == "row_major") {
        matrix_major = StringImm::make("row_major");
      } else {
        LOG(FATAL) << "invalid matrix major for " << call_name;
      }

      auto load_matrix_call =
        [this, &src, &stride, &matrix_major](const Buffer &buffer) {
        return Evaluate::make(
                Call::make(Handle(),
                          intrinsic::tvm_load_matrix_sync,
                          {buffer->data,
                          warp_tile_.m, warp_tile_.n, warp_tile_.k,
                          buffer->elem_offset, src, stride, matrix_major},
                          Call::Intrinsic));
      };

      NodePtr<BufferNode> buffer_node = make_node<BufferNode>();
      return add_buffer_bind_scope(call, buffer_node,
                                    TensorKey{op->func, op->value_index},
                                    load_matrix_call, Float(16));
    }

    auto it3 = frag_store_.find(op);
    if (it3 != frag_store_.end()) {
      TensorKey key{op->func, op->value_index};
      auto it = strides_.find(key.GetName());
      CHECK(it != strides_.end())
          << "Cannot find stride for " << key.GetName();
      auto strides = it->second;
      Expr stride = strides[strides.size()-1];

      Expr dst = it3->second;
      // thread index unification inside a warp
      Expr warp_y = IntImm::make(Int(32), warp_threads_y_);
      ThreadIdxMutator thread_idx_mutator(warp_y);
      dst = thread_idx_mutator.Mutate(dst);
      dst = Call::make(Handle(),
                       "&",
                       {dst},
                       Call::Extern);

      auto call = op->value.as<Call>();

      auto store_matrix_call =
        [this, &dst, &stride](const Buffer &buffer) {
          return Evaluate::make(
                  Call::make(Handle(),
                            intrinsic::tvm_store_matrix_sync,
                            {buffer->data,
                            warp_tile_.m, warp_tile_.n, warp_tile_.k,
                            buffer->elem_offset, dst, stride,
                            StringImm::make("col_major")},
                            Call::Intrinsic));
        };

      NodePtr<BufferNode> buffer_node = make_node<BufferNode>();
      return add_buffer_bind_scope(call, buffer_node,
                                    TensorKey{call->func, call->value_index},
                                    store_matrix_call, Float(32));
    }

    return stmt;
  }

  Stmt Mutate_(const For* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<For>();
    if (op != nullptr) {
      auto it = loop_scaling_.find(op->loop_var.get());
      if (it != loop_scaling_.end()) {
        int scale_factor = it->second;
        int scaled_extent_value = 1;
        if (const IntImm *ori_extent = op->extent.as<IntImm>()) {
          int ori_extent_value = ori_extent->value;
          scaled_extent_value = ori_extent_value / scale_factor;
        }
        Expr scaled_extent = make_const(op->extent.type(), scaled_extent_value);
        stmt = For::make(op->loop_var, op->min, scaled_extent, op->for_type,
          op->device_api, op->body);
      }
    }
    return stmt;
  }

 private:
  Stmt add_buffer_bind_scope(const Call* call,
    const NodePtr<BufferNode> &buffer_node, const TensorKey &key,
    const std::function<Stmt(const Buffer &buffer)> &call_back,
    DataType datatype) {
    auto call_name = call->name;
    auto pos = call_name.find(".");
    if (pos != std::string::npos) {
      call_name = call_name.substr(0, pos);
    }
    auto it = matrix_abc_.find(call_name);
    if (it == matrix_abc_.end()) {
      std::cout << "Error!!!! matrix_abc_ not found" << std::endl;
    }

    // TODO(chenfan): This should be same as the fragment shape
    auto shape = Array<Expr>{16, 16};
    auto strides = Array<Expr>{16, 1};

    auto bound_it = min_bounds_.find(key);
    if (bound_it == min_bounds_.end()) {
      std::cout << "Error!!!! bound not found" << std::endl;
    }
    Array<Expr> min_bound = bound_it->second;
    Expr elem_offset = IntImm::make(Int(32), 0);
    CHECK_EQ(call->args.size(), min_bound.size());
    for (size_t i = 0; i < min_bound.size(); i++) {
      elem_offset = Add::make(
        elem_offset, Mul::make(
          strides[i], Sub::make(call->args[i], min_bound[i])));
    }

    buffer_node->data = Variable::make(Handle(), call->name);
    buffer_node->name = call->name;
    buffer_node->scope = "wmma." + it->second;
    buffer_node->dtype = datatype;
    buffer_node->strides = strides;
    buffer_node->shape = Array<Expr>{16, 16};
    buffer_node->data_alignment = 1;
    buffer_node->elem_offset = Simplify(elem_offset);
    buffer_node->offset_factor = 1;
    Buffer buffer(buffer_node);

    NodePtr<TensorNode> tensor_node = make_node<TensorNode>();
    tensor_node->value_index = key.value_index;
    tensor_node->op = Downcast<Operation>(key.f);
    tensor_node->shape = shape;
    tensor_node->dtype = datatype;
    Tensor tensor(tensor_node);

    Array<Expr> args;
    for (size_t i = 0; i < call->args.size(); ++i) {
      args.push_back(call->args[i]);
      args.push_back(shape[i]);
    }
    auto tuple = Call::make(Handle(),
                            intrinsic::tvm_tuple,
                            args,
                            Call::Intrinsic);

    auto call_back_call = call_back(buffer);

    Array<NodeRef> node = {buffer, tensor};
    return AttrStmt::make(node,
                          "buffer_bind_scope",
                          tuple,
                          call_back_call);
  }

  std::unordered_map<std::string, std::string> matrix_abc_;
  std::unordered_map<std::string, std::string> matrix_major_;
  std::unordered_map<const Provide*, Array<Expr>> mma_sync_;
  std::unordered_map<std::string, Array<Expr>> strides_;
  std::unordered_set<std::string> frag_reg_;
  std::unordered_map<const Variable*, unsigned> loop_scaling_;
  std::unordered_map<const Provide*, Expr> frag_load_;
  std::unordered_map<const Provide*, Expr> frag_store_;
  std::unordered_map<TensorKey, Array<Expr>> min_bounds_;
  Tile warp_tile_;
  int warp_threads_y_{-1};
  double cuda_compute_capability_;
  double cuda_version_;
};

Stmt TensorCore(Stmt stmt, Schedule schedule,
                double cuda_compute_capability,
                double cuda_version,
                Map<Tensor, Buffer> extern_buffer) {
  ScheduleAnalyser schedule_analyser;
  schedule_analyser.MatrixIdentify(schedule);

  MMAMatcher mma_matcher(extern_buffer, schedule_analyser,
                         cuda_compute_capability, cuda_version);
  mma_matcher.Visit(stmt);
  if (!mma_matcher.Matched()) {
    return stmt;
  }

  BufferAnalyser buffer_analyser(extern_buffer, cuda_version,
                                 schedule_analyser, mma_matcher);
  buffer_analyser.Visit(stmt);
  if (!buffer_analyser.QualifiedForTensorCore()) {
    return stmt;
  }

  TensorCoreIRMutator tensorcore_mutator(schedule_analyser, mma_matcher,
                      buffer_analyser, cuda_compute_capability, cuda_version);
  return tensorcore_mutator.Mutate(stmt);
}

}  // namespace ir
}  // namespace tvm
