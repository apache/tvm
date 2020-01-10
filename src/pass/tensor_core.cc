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
#include <tvm/build_module.h>
#include <tvm/runtime/device_api.h>
#include <unordered_map>
#include "ir_util.h"
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

std::string simplify_name(std::string input) {
  auto pos = input.find(".");
  if (pos != std::string::npos) {
    return input.substr(0, pos);
  } else {
    return input;
  }
}

Expr unpack_type_cast(const Expr &input, const Type &target_type) {
  auto cast = input.as<Cast>();
  if (cast == nullptr) {
    return input;
  } else if (cast->type == target_type) {
    return cast->value;
  }
  return Expr();
}

// MMAMatcher matches C = Cast(A)*Cast(B)+C,
// where A & B are fp16/int8 local buffers,
// and C is fp32/int32 local buffer.
class MMAMatcher: public IRVisitor {
 public:
  explicit MMAMatcher(Map<Tensor, Buffer> extern_buffer) {
    for (auto kv : extern_buffer) {
      BufferInfo bi;
      bi.name = kv.second->name;
      bi.dtype = kv.second->dtype;
      bi.external = true;
      buf_map_[TensorKey{kv.first->op, kv.first->value_index}] = bi;
    }
  }
  using IRVisitor::Visit_;

  void Visit_(const AttrStmt* op) final {
    if (op->attr_key == attr::pragma_tensor_core) {
      tensor_core_on_ = true;
      IRVisitor::Visit_(op);
    } else if (op->attr_key == attr::realize_scope) {
      storage_scope_[op->node.get()] = op->value.as<StringImm>()->value;
      Visit(op->body);
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const Provide* op) final {
    IRVisitor::Visit_(op);
    auto it = buf_map_.find(TensorKey{op->func, op->value_index});
    if (it == buf_map_.end()) {
      return;
    }
    const BufferInfo& bi = it->second;
    if (bi.released) {
      return;
    }
    if (tensor_core_on_ && mma_sync_match_(op, bi)) {
      matched_ = true;
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

  inline bool Matched() const {return matched_;}

  friend class ScheduleAnalyser;
  friend class BufferAnalyser;

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

  // Check whether the storage scope is local
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
      auto it1 = buf_map_.find(TensorKey{op->func, op->value_index});
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

  // Do the pattern matching
  bool mma_sync_match_(const Provide* op, BufferInfo store_buffer) {
    auto* add = op->value.as<Add>();
    if (add == nullptr) {
      return false;
    }

    auto* load_c = add->a.as<Call>();
    BufferInfo buffer_c;
    if (!check_local_buffer_(load_c, &buffer_c)
        || !buffer_c.same_as(store_buffer)
        || !(buffer_c.dtype == Float(32) ||
             buffer_c.dtype == Int(32))) {
      return false;
    }

    auto mul = unpack_type_cast(add->b, buffer_c.dtype).as<Mul>();
    if (mul == nullptr) {
      return false;
    }

    auto load_a_expr = unpack_type_cast(mul->a, buffer_c.dtype);
    auto load_a = load_a_expr.as<Call>();
    BufferInfo buffer_a;
    if (!check_local_buffer_(load_a, &buffer_a)
        || !(buffer_a.dtype == Float(16) ||
             buffer_a.dtype == Int(8))) {
      return false;
    }

    auto load_b_expr = unpack_type_cast(mul->b, buffer_c.dtype);
    auto load_b = load_b_expr.as<Call>();
    BufferInfo buffer_b;
    if (!check_local_buffer_(load_b, &buffer_b)
        || !(buffer_b.dtype == Float(16) ||
             buffer_b.dtype == Int(8))) {
      return false;
    }

    frag_reg_.insert(buffer_c.name);
    frag_reg_.insert(buffer_a.name);
    frag_reg_.insert(buffer_b.name);
    buf_name_.insert(std::make_pair(load_a, buffer_a.name));
    buf_name_.insert(std::make_pair(load_b, buffer_b.name));
    mma_sync_.insert(std::make_pair(op,
      Array<Expr>{load_a_expr, load_b_expr, add->a}));

    return true;
  }

  std::unordered_map<TensorKey, BufferInfo> buf_map_;
  std::unordered_map<const Node*, std::string> storage_scope_;
  std::unordered_map<const Provide*, Array<Expr>> mma_sync_;
  std::unordered_map<const Node*, std::string> buf_name_;
  std::unordered_set<std::string> frag_reg_;
  bool matched_{false};
  bool tensor_core_on_{false};
};

// BodyVisitor visits the body stmt of original ComputeOp
// to get the access indices of input matrices,
// if it is recognized as matrix multiply.
class BodyVisitor : public IRVisitor {
 public:
  BodyVisitor() {}
  using IRVisitor::Visit_;

  void Visit_(const Reduce* op) final {
    auto* comm_add = op->combiner->result[0].as<Add>();
    if (comm_add == nullptr || op->combiner->result.size() > 1) {
      return;
    }
    for (Expr source : op->source) {
      auto mul_0 = unpack_type_cast(source, Float(32)).as<Mul>();
      auto mul_1 = unpack_type_cast(source, Int(32)).as<Mul>();
      if (mul_0 == nullptr && mul_1 == nullptr) {
        continue;
      }

      tensorcore_candidate_ = true;
      IRVisitor::Visit(source);
    }
  }

  void Visit_(const Call* op) final {
    IRVisitor::Visit_(op);
    args_.insert(std::make_pair(op->name, op->args));
  }

  friend class ScheduleAnalyser;

 private:
  std::unordered_map<std::string, Array<Expr>> args_;
  bool tensorcore_candidate_{false};
};

// ScheduleAnalyser figures out matrix_a/matrix_b and row_major/col_major
class ScheduleAnalyser {
 public:
  explicit ScheduleAnalyser(const MMAMatcher &mma_matcher)
    : mma_sync_(mma_matcher.mma_sync_),
      buf_name_(mma_matcher.buf_name_) {}

  bool MatrixIdentify(Schedule schedule) {
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
        if (args.size() < 2) {
          continue;
        }
        const Variable* var0 = args[args.size() - 2].as<Variable>();
        const Variable* var1 = args[args.size() - 1].as<Variable>();
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

    for (auto &mma_sync : mma_sync_) {
      auto &operands = mma_sync.second;
      auto* load_a = operands[0].as<Call>();
      auto* load_b = operands[1].as<Call>();
      auto input0 = simplify_name(buf_name_.find(load_a)->second);
      auto input1 = simplify_name(buf_name_.find(load_b)->second);
      auto it0 = matrix_abc_.find(input0);
      auto it1 = matrix_abc_.find(input1);

      if (it0 == matrix_abc_.end() || it1 == matrix_abc_.end()) {
        return false;
      }
      if (it0->second == "matrix_a" && it1->second == "matrix_b") {
        return true;
      } else if (it0->second == "matrix_b" && it1->second == "matrix_a") {
        mma_sync.second = Array<Expr>{operands[1], operands[0], operands[2]};
      } else {
        return false;
      }
    }
    return true;
  }

  friend class BufferAnalyser;
  friend class TensorCoreIRMutator;

 private:
  std::unordered_map<std::string, std::string> matrix_abc_;
  std::unordered_map<std::string, std::string> matrix_major_;
  std::unordered_map<const Provide*, Array<Expr>> mma_sync_;
  std::unordered_map<const Node*, std::string> buf_name_;
};

// IndexVisitor visits access index of fragment
// to record variable for loop scaling
class IndexVisitor : public IRVisitor {
 public:
  IndexVisitor() {}
  using IRVisitor::Visit_;

  void Visit_(const Variable* op) final {
    loop_scaling_.insert(std::make_pair(op, scaling_factor_));
  }

  friend class BufferAnalyser;
  friend class TensorCoreIRMutator;

 private:
  std::unordered_map<const Variable*, unsigned> loop_scaling_;
  unsigned scaling_factor_{0};
};

// BufferAnalyser gets buffer info,
// e.g. thread tile and warp tile, for TensorCore CodeGen
class BufferAnalyser : public IRVisitor {
 public:
  explicit BufferAnalyser(Map<Tensor, Buffer> extern_buffer,
                          const ScheduleAnalyser &schedule_analyser,
                          const MMAMatcher &mma_matcher)
      : matrix_abc_(schedule_analyser.matrix_abc_),
        matrix_major_(schedule_analyser.matrix_major_),
        frag_reg_(mma_matcher.frag_reg_) {
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
      auto& vinfo = dim_align_[TensorKey{tensor->op, tensor->value_index}];
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

    if (matrix_abc_.count(key.GetName())) {
      if (bi.shape.size() < 2) {
        invalid_ = true;
        return;
      }
      for (auto i = bi.shape.size() - 1; i + 2 >= bi.shape.size(); --i) {
        const IntImm* shape = bi.shape[i].as<IntImm>();
        if (shape == nullptr || shape->value % 16 != 0) {
          invalid_ = true;
          return;
        }
      }
    }

    Array<Expr> strides;
    if (bi.strides.size() > 0) {
      strides = bi.strides;
    } else {
      for (size_t i = 1; i < bi.shape.size(); ++i) {
        Expr stride = IntImm::make(Int(32), 1);
        for (size_t j = bi.shape.size() - 1; j >= i; --j) {
          stride = Mul::make(stride, bi.shape[j]);
        }
        strides.push_back(stride);
      }
      strides.push_back(make_const(Int(32), 1));
    }
    strides_.insert(std::make_pair(key.GetName(), strides));

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
      for (auto i = op->args.size() - 1; i + 2 >= op->args.size(); --i) {
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

      std::string input_name = simplify_name(bi.name);
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

      if (matrix_abc_.count(op->name)) {
        if (bi.shape.size() < 2) {
          invalid_ = true;
          return;
        }
        for (auto i = bi.shape.size() - 1; i + 2 >= bi.shape.size(); --i) {
          const IntImm* shape = bi.shape[i].as<IntImm>();
          if (shape == nullptr || shape->value % 16 != 0) {
            invalid_ = true;
            return;
          }
        }
      }

      Array<Expr> strides;
      if (bi.strides.size() > 0) {
        strides = bi.strides;
      } else {
        for (size_t i = 1; i < bi.shape.size(); ++i) {
          Expr stride = IntImm::make(Int(32), 1);
          for (size_t j = bi.shape.size() - 1; j >= i; --j) {
            stride = Mul::make(stride, bi.shape[j]);
          }
          strides.push_back(stride);
        }
        strides.push_back(make_const(Int(32), 1));
      }
      strides_.insert(std::make_pair(key.GetName(), strides));

      if (!frag_reg_.count(bi.name)) {
        return;
      }

      auto rel_index = bi.RelIndex(op->args);
      if (op->args.size() < 2) {
        invalid_ = true;
        return;
      }
      for (auto i = op->args.size() - 1; i + 2 >= op->args.size(); --i) {
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

  // Derive warp tile from thread tile,
  // and check whether it is qualified for TensorCore.
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
    return supported_warp_tile_();
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

  bool supported_warp_tile_() {
    if (warp_tile_.m == 16 &&
        warp_tile_.n == 16 &&
        warp_tile_.k == 16) {
      return true;
    }
    if (warp_tile_.m == 8 &&
        warp_tile_.n == 32 &&
        warp_tile_.k == 16) {
      return true;
    }
    if (warp_tile_.m == 32 &&
        warp_tile_.n == 8 &&
        warp_tile_.k == 16) {
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
  std::unordered_map<std::string, int> thread_extent_;
  IndexVisitor index_visitor;
  Tile warp_tile_;
  Tile thread_tile_;
  int warp_threads_y_{-1};
  bool invalid_{false};
};

// ThreadIdxMutator does the thread index unification inside a warp
class ThreadIdxMutator : public IRMutator {
 public:
  explicit ThreadIdxMutator(Expr warp_y): warp_y_(warp_y) {}

  Expr Mutate_(const Variable* op, const Expr& olde) final {
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

// TensorCoreIRMutator mutates the AST for TensorCore CodeGen
// based on tensor core intrinsics
class TensorCoreIRMutator : public IRMutator {
 public:
  explicit TensorCoreIRMutator(const ScheduleAnalyser &schedule_analyser,
    const BufferAnalyser &buffer_analyser)
      : matrix_abc_(schedule_analyser.matrix_abc_),
      matrix_major_(schedule_analyser.matrix_major_),
      mma_sync_(schedule_analyser.mma_sync_),
      strides_(buffer_analyser.strides_),
      frag_reg_(buffer_analyser.frag_reg_),
      loop_scaling_(buffer_analyser.index_visitor.loop_scaling_),
      frag_load_(buffer_analyser.frag_load_),
      frag_store_(buffer_analyser.frag_store_),
      warp_tile_(buffer_analyser.warp_tile_),
      warp_threads_y_(buffer_analyser.warp_threads_y_) {}

  Stmt Mutate_(const Realize* op, const Stmt& s) final {
    TensorKey key{op->func, op->value_index};
    bounds_[key] = op->bounds;
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Realize>();
    if (op != nullptr) {
      if (!frag_reg_.count(key.GetName())) {
        return stmt;
      }

      auto new_extents = get_tile_size_(simplify_name(key.GetName()));

      Region new_bounds;
      for (size_t i = 0; i < op->bounds.size() - 2; ++i) {
        new_bounds.push_back(op->bounds[i]);
      }
      CHECK_GE(op->bounds.size(), 2)
          << "Less than 2 dimensions for matrix " << key.GetName();
      new_bounds.push_back(Range::make_by_min_extent(
          op->bounds[op->bounds.size() - 2]->min, new_extents[0]));
      new_bounds.push_back(Range::make_by_min_extent(
          op->bounds[op->bounds.size() - 1]->min, new_extents[1]));

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

        auto it = matrix_abc_.find(simplify_name(node->name));
        CHECK(it != matrix_abc_.end())
              << "Cannot find matrix info for " << node->name;
        auto matrix_abc = "wmma." + it->second;
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
      const auto &operands = it->second;
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
        [&buffer_node_a, &buffer_node_b]
        (const Buffer &buffer) {
          Buffer buffer_a(buffer_node_a);
          Buffer buffer_b(buffer_node_b);
          return Evaluate::make(
                  Call::make(Handle(),
                        intrinsic::tvm_mma_sync,
                        {buffer->data, buffer->elem_offset,
                        buffer_a->data, buffer_a->elem_offset,
                        buffer_b->data, buffer_b->elem_offset,
                        buffer->data, buffer->elem_offset},
                        Call::Intrinsic));
        };

      auto call_add_c =
        [this, &cc, &buffer_node_c, &mma_sync_call](const Buffer &buffer) {
          return add_buffer_bind_scope_(cc, buffer_node_c,
            TensorKey{cc->func, cc->value_index}, mma_sync_call, cc->type);
        };

      auto call_add_b =
        [this, &cb, &buffer_node_b, &call_add_c](const Buffer &buffer) {
          return add_buffer_bind_scope_(cb, buffer_node_b,
            TensorKey{cb->func, cb->value_index}, call_add_c, cb->type);
        };

      return add_buffer_bind_scope_(ca, buffer_node_a,
        TensorKey{ca->func, ca->value_index}, call_add_b, ca->type);
    }

    auto it2 = frag_load_.find(op);
    if (it2 != frag_load_.end()) {
      Expr dst = it2->second;
      if (op->value.as<FloatImm>() != nullptr ||
          op->value.as<IntImm>() != nullptr) {
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
        return add_buffer_bind_scope_(call, buffer_node,
                                      TensorKey{call->func, call->value_index},
                                      fill_fragment_call, call->type);
      }

      const Call* value = op->value.as<Call>();
      CHECK(value != nullptr)
          << "Can only load fragment from a buffer";

      auto it = strides_.find(value->name);
      CHECK(it != strides_.end())
          << "Cannot find stride for " << value->name;
      auto strides = it->second;
      CHECK_GE(strides.size(), 2);
      Expr stride = strides[strides.size()-2];

      // thread index unification inside a warp
      Expr warp_y = IntImm::make(Int(32), warp_threads_y_);
      ThreadIdxMutator thread_idx_mutator(warp_y);
      Expr mutated_value = thread_idx_mutator.Mutate(op->value);
      Expr src = Call::make(value->type,
                            "&",
                            {mutated_value},
                            Call::Extern);

      auto call = dst.as<Call>();
      Expr matrix_major;
      auto iter2 = matrix_major_.find(simplify_name(call->name));
      CHECK(iter2 != matrix_major_.end())
          << "Can not determine matrix major for " << call->name;
      if (iter2->second == "col_major") {
        matrix_major = StringImm::make("col_major");
      } else if (iter2->second == "row_major") {
        matrix_major = StringImm::make("row_major");
      } else {
        LOG(FATAL) << "invalid matrix major for " << call->name;
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
      return add_buffer_bind_scope_(call, buffer_node,
                                    TensorKey{op->func, op->value_index},
                                    load_matrix_call, call->type);
    }

    auto it3 = frag_store_.find(op);
    if (it3 != frag_store_.end()) {
      TensorKey key{op->func, op->value_index};
      auto it = strides_.find(key.GetName());
      CHECK(it != strides_.end())
          << "Cannot find stride for " << key.GetName();
      auto strides = it->second;
      CHECK_GE(strides.size(), 2);
      Expr stride = strides[strides.size()-2];

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
      return add_buffer_bind_scope_(call, buffer_node,
                                    TensorKey{call->func, call->value_index},
                                    store_matrix_call, call->type);
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
  Array<Expr> get_tile_size_(const std::string &name) {
      auto it = matrix_abc_.find(name);
      auto it2 = matrix_major_.find(name);
      CHECK(it != matrix_abc_.end() && it2 != matrix_major_.end())
          << "Cannot find matrix info for " << name;
      Expr size0 = make_const(Int(32), 16);
      Expr size1 = make_const(Int(32), 16);
      if (it->second == "matrix_a" && it2->second == "col_major") {
        size0 = make_const(Int(32), warp_tile_.k);
        size1 = make_const(Int(32), warp_tile_.m);
      }
      if (it->second == "matrix_a" && it2->second == "row_major") {
        size0 = make_const(Int(32), warp_tile_.m);
        size1 = make_const(Int(32), warp_tile_.k);
      }
      if (it->second == "matrix_b" && it2->second == "row_major") {
        size0 = make_const(Int(32), warp_tile_.k);
        size1 = make_const(Int(32), warp_tile_.n);
      }
      if (it->second == "matrix_b" && it2->second == "col_major") {
        size0 = make_const(Int(32), warp_tile_.n);
        size1 = make_const(Int(32), warp_tile_.k);
      }
      if (it->second == "matrix_c") {
        size0 = make_const(Int(32), warp_tile_.n);
        size1 = make_const(Int(32), warp_tile_.m);
      }
      Array<Expr> tile_size = {size0, size1};
      return tile_size;
  }

  Stmt add_buffer_bind_scope_(const Call* call,
      const NodePtr<BufferNode> &buffer_node, const TensorKey &key,
      const std::function<Stmt(const Buffer &buffer)> &call_back,
      DataType datatype) {
    auto it = bounds_.find(key);
    CHECK(it != bounds_.end());
    Array<Expr> min_bound;
    for (auto i : it->second) {
      min_bound.push_back(i->min);
    }

    CHECK_GE(it->second.size(), 2);
    Array<Expr> shape;
    for (size_t i = 0; i < it->second.size() - 2; ++i) {
      shape.push_back(it->second[i]->extent);
    }
    auto tile_size = get_tile_size_(simplify_name(call->name));
    shape.push_back(tile_size[0]);
    shape.push_back(tile_size[1]);

    Array<Expr> strides;
    for (size_t i = 1; i < shape.size(); ++i) {
      Expr stride = IntImm::make(Int(32), 1);
      for (size_t j = shape.size() - 1; j >= i; --j) {
        stride = Mul::make(stride, shape[j]);
      }
      strides.push_back(stride);
    }
    strides.push_back(make_const(Int(32), 1));

    Expr elem_offset = IntImm::make(Int(32), 0);
    CHECK_EQ(call->args.size(), min_bound.size());
    for (size_t i = 0; i < min_bound.size(); i++) {
      elem_offset = Add::make(
        elem_offset, Mul::make(
          strides[i], Sub::make(call->args[i], min_bound[i])));
    }

    auto it2 = matrix_abc_.find(simplify_name(call->name));
    CHECK(it2 != matrix_abc_.end())
          << "Cannot find matrix info for " << call->name;
    buffer_node->data = Variable::make(Handle(), call->name);
    buffer_node->name = call->name;
    buffer_node->scope = "wmma." + it2->second;
    buffer_node->dtype = datatype;
    buffer_node->strides = strides;
    buffer_node->shape = shape;
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
    Array<NodeRef> node = {buffer, tensor};
    return AttrStmt::make(node,
                          "buffer_bind_scope",
                          tuple,
                          call_back(buffer));
  }

  std::unordered_map<std::string, std::string> matrix_abc_;
  std::unordered_map<std::string, std::string> matrix_major_;
  std::unordered_map<const Provide*, Array<Expr>> mma_sync_;
  std::unordered_map<std::string, Array<Expr>> strides_;
  std::unordered_set<std::string> frag_reg_;
  std::unordered_map<const Variable*, unsigned> loop_scaling_;
  std::unordered_map<const Provide*, Expr> frag_load_;
  std::unordered_map<const Provide*, Expr> frag_store_;
  std::unordered_map<TensorKey, Region> bounds_;
  Tile warp_tile_;
  int warp_threads_y_{-1};
};

Stmt RewriteForTensorCore(Stmt stmt,
                          Schedule schedule,
                          Map<Tensor, Buffer> extern_buffer) {
  // Check if current lower target is CUDA
  auto target = tvm::Target::Current(true);
  if (target.defined() && target->target_name != "cuda") {
    return stmt;
  }

  // Check if current runtime support GPU CUDA
  TVMContext ctx{kDLGPU, 0};
  auto api = tvm::runtime::DeviceAPI::Get(ctx, true);
  if (api == nullptr) {
    return stmt;
  }

  MMAMatcher mma_matcher(extern_buffer);
  mma_matcher.Visit(stmt);
  if (!mma_matcher.Matched()) {
    return stmt;
  }

  ScheduleAnalyser schedule_analyser(mma_matcher);
  if (!schedule_analyser.MatrixIdentify(schedule)) {
    return stmt;
  }

  BufferAnalyser buffer_analyser(extern_buffer,
                                 schedule_analyser, mma_matcher);
  buffer_analyser.Visit(stmt);
  if (!buffer_analyser.QualifiedForTensorCore()) {
    return stmt;
  }

  return TensorCoreIRMutator(schedule_analyser, buffer_analyser).Mutate(stmt);
}

}  // namespace ir
}  // namespace tvm
