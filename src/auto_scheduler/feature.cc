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
 * \file auto_scheduler/feature.cc
 * \brief Feature extraction for the cost model
 */

#include <tvm/arith/analyzer.h>
#include <tvm/auto_scheduler/feature.h>
#include <tvm/auto_scheduler/measure.h>
#include <tvm/auto_scheduler/measure_record.h>
#include <tvm/driver/driver_api.h>
#include <tvm/ir/global_var_supply.h>
#include <tvm/runtime/registry.h>
#include <tvm/support/parallel_for.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "search_policy/utils.h"
#include "utils.h"

namespace tvm {
namespace auto_scheduler {

using namespace tvm::tir;
using arith::Analyzer;
using arith::ConstIntBound;

template <class T>
using BufferMap = std::unordered_map<Var, T, ObjectHash, ObjectEqual>;

// The number of samples to extract for arithmetic intensity curves
static const int ARITH_INTENSITY_CURVE_SAMPLE_N = 10;

// Annotation position encoding
enum class AnnotationPosType : int {
  kPosNone = 0,           // Does not have this kind of annotation
  kPosInnerSpatial = 1,   // The annotated iterator is the innermost spatial iterator
  kPosMiddleSpatial = 2,  // The annotated iterator is a middle spatial iterator
  kPosOuterSpatial = 3,   // The annotated iterator is the outermost spatial iterator
  kPosInnerReduce = 4,    // The annotated iterator is the innermost reduce iterator
  kPosMiddleReduce = 5,   // The annotated iterator is a middle reduce iterator
  kPosOuterReduce = 6,    // The annotated iterator is the outermost reduce iterator
  kPosMixed = 7           // The annotated iterator is a mixed space and reduce iterator
};

// Buffer access type
enum class BufferAccessType : int { kRead = 0, kWrite = 1, kReadWrite = 2, kUnknownRW = 3 };

// Accesses to a buffer
struct BufferAccess {
  // data reuse type
  BufferAccessType acc_type{BufferAccessType::kUnknownRW};
  // Use a two-dimensional array to store multiple multi-dimensional accesses.
  // The innermost vector stores the multi-dimensional indices of one access.
  std::vector<std::vector<PrimExpr>> indices;
};

// Data reuse type
enum class ReuseType : int { kLoopMultipleRead = 0, kSerialMultipleReadWrite = 1, kNoReuse = 2 };

// Feature for an access of a buffer
struct BufferAccessFeature {
  std::string buffer_name;        // The name of the buffer
  BufferAccessType acc_type;      // The type of the access
  float bytes;                    // The touched memory in bytes
  float unique_bytes;             // The touched unique memory in bytes
  float lines;                    // The number of touched cache lines
  float unique_lines;             // The number touched unique cache lines
  ReuseType reuse_type;           // Tye type of data reuse
  float reuse_dis_iter;           // The reuse distance in iterator number
  float reuse_dis_bytes;          // The reuse distance in total touched bytes
  float reuse_ct;                 // The reuse ratio
  float bytes_d_reuse_ct;         // bytes / reuse_ct
  float unique_bytes_d_reuse_ct;  // unique_bytes / reuse_ct
  float lines_d_reuse_ct;         // lines / reuse_ct
  float unique_lines_d_reuse_ct;  // unique_lines / reuse_ct
  float stride;                   // The stride in access
};

// Feature set of a BufferStore statement
struct FeatureSet {
  // Group 1: Computation related features
  float float_mad;                  // The number of float MAD (Multiply–add) ops
  float float_addsub;               // The number of float add and sub ops
  float float_mul;                  // The number of float multiply ops
  float float_divmod;               // The number of float div and mod ops
  float float_cmp;                  // The number of float comparison ops
  float float_math_func;            // The number of float math func calls
  float float_other_func;           // The number of other float func calls
  float int_mad;                    // The number of integer MAD (Multiply–add) ops
  float int_addsub;                 // The number of integer add and sub ops
  float int_mul;                    // The number of float multiply ops
  float int_divmod;                 // The number of float div and mod ops
  float int_cmp;                    // The number of float comparison ops
  float int_math_func;              // The number of float math func calls
  float int_other_func;             // The number of other float func calls
  float bool_op;                    // The number of bool ops
  float select_op;                  // The number of select ops
  float vec_num;                    // The number of vectorized iterators
  float vec_prod;                   // The product of the lengths of vectorized iterators
  float vec_len;                    // The length of the innermost vectorized iterator
  AnnotationPosType vec_type;       // The type of vectorization position
  float unroll_num;                 // The number of unrolled iterators
  float unroll_prod;                // The product of the lengths of vectorized iterators
  float unroll_len;                 // The length of the innermost unrolled iterator
  AnnotationPosType unroll_type;    // The type of unroll position
  float parallel_num;               // The number of paralleled iterators
  float parallel_prod;              // The product of the lengths of paralleled iterators
  float parallel_len;               // The length of the innermost paralleled iterators
  AnnotationPosType parallel_type;  // The type of parallel position
  float is_gpu;                     // Whether it is a GPU task
  float blockIdx_x_len;             // The length of blockIdx.x
  float blockIdx_y_len;             // The length of blockIdx.y
  float blockIdx_z_len;             // The length of blockIdx.z
  float threadIdx_x_len;            // The length of threadIdx.x
  float threadIdx_y_len;            // The length of threadIdx.y
  float threadIdx_z_len;            // The length of threadIdx.z
  float vthread_len;                // The length of virtual thread

  // Group 2: Buffer access related features (per buffer)
  std::vector<BufferAccessFeature> access_feas;

  // Group 3: Arithmetic intensity related features
  float arith_intensity_curve[ARITH_INTENSITY_CURVE_SAMPLE_N];  // points sampled from the
                                                                // arithmetic intensity curve

  // Group 4: Allocation related features
  float alloc_size;        // The size of allocated buffer in bytes
  float alloc_outer_prod;  // The product of lengths of loops outside the scope of the allocation
  float alloc_inner_prod;  // The product of lengths of loops inside the score of the allocation
  float alloc_prod;        // alloc_outer_prod * alloc_inner_prod

  // Group 5: Outer scope related features
  float outer_prod;            // The product of lengths of outer loops
  float num_loops;             // The number of outer loops
  float auto_unroll_max_step;  // The value of pragma "auto_unroll_max_step"
};

// Return whether a var is in an expr
bool VarInExpr(const Var& var, const PrimExpr& expr) {
  bool find = false;

  PostOrderVisit(expr, [&find, &var](const ObjectRef& node) {
    if (find) {
      return;
    }

    if (const VarNode* op = node.as<VarNode>()) {
      if (op == var.get()) {
        find = true;
      }
    }
  });

  return find;
}

// Get position encoding for annotation
AnnotationPosType GetAnnotationPosEncoding(const Var& var, const Array<PrimExpr>& spatial_args,
                                           const Array<IterVar>& axis,
                                           const Array<IterVar>& reduce_axis) {
  // Try to match spatial args first
  size_t find_i = 0;
  size_t find_ct = 0;
  for (size_t i = 0; i < spatial_args.size(); ++i) {
    if (VarInExpr(var, spatial_args[i])) {
      find_i = i;
      find_ct += 1;
    }
  }

  if (find_ct == 0) {
    // If it is not found in spacial args, then it is a reduce iterator.
    // Use name to match
    const std::string& var_name = var->name_hint;
    for (size_t i = 0; i < reduce_axis.size(); ++i) {
      if (var_name.find(reduce_axis[i]->var->name_hint) != std::string::npos) {
        find_i = i;
        find_ct++;
      }
    }
    if (find_ct >= 1) {
      if (find_i == 0) {
        return AnnotationPosType::kPosInnerReduce;
      } else if (find_i == reduce_axis.size() - 1) {
        return AnnotationPosType::kPosOuterReduce;
      } else {
        return AnnotationPosType::kPosMiddleReduce;
      }
    } else {
      // If the axis is not found in both spatial args and reduce axis,
      // then this stage must compute_at somewhere under this axis and this axis is simplified out
      // We assume it is an outer spatial
      return AnnotationPosType::kPosOuterSpatial;
    }
  } else if (find_ct == 1) {
    if (find_i == spatial_args.size() - 1) {
      return AnnotationPosType::kPosInnerSpatial;
    } else if (find_i == 0) {
      return AnnotationPosType::kPosOuterSpatial;
    } else {
      return AnnotationPosType::kPosMiddleSpatial;
    }
  } else {
    return AnnotationPosType::kPosMixed;
  }
}

// Return the maximum extent of a for loop
int64_t GetLoopExtent(const ForNode* node, const Analyzer& ana) {
  int64_t bound = ana.const_int_bound(node->extent)->max_value;
  if (bound == ConstIntBound::kPosInf) {
    return 1;  // Analyzer could not determine a valid bound, use 1 instead.
  } else {
    return bound;
  }
}

// Count math ops in an expr
class MathOpCounter : public StmtExprVisitor {
 public:
#define VisitBinary(Type, float_ct, int_ct)                        \
  void VisitExpr_(const Type* op) final {                          \
    if (op->a.dtype().is_float() || op->a.dtype().is_bfloat16()) { \
      float_ct += op->a.dtype().lanes();                           \
    } else {                                                       \
      int_ct += op->a.dtype().lanes();                             \
    }                                                              \
    StmtExprVisitor::VisitExpr_(op);                               \
  }

  VisitBinary(AddNode, float_addsub, int_addsub);
  VisitBinary(SubNode, float_addsub, int_addsub);
  VisitBinary(MulNode, float_mul, int_mul);
  VisitBinary(DivNode, float_divmod, int_divmod);
  VisitBinary(ModNode, float_divmod, int_divmod);
  VisitBinary(FloorDivNode, float_divmod, int_divmod);
  VisitBinary(FloorModNode, float_divmod, int_divmod);
  VisitBinary(MaxNode, float_cmp, int_cmp);
  VisitBinary(MinNode, float_cmp, int_cmp);
  VisitBinary(EQNode, float_cmp, int_cmp);
  VisitBinary(NENode, float_cmp, int_cmp);
  VisitBinary(LTNode, float_cmp, int_cmp);
  VisitBinary(LENode, float_cmp, int_cmp);
  VisitBinary(GTNode, float_cmp, int_cmp);
  VisitBinary(GENode, float_cmp, int_cmp);

#undef VisitBinary

  void VisitExpr_(const AndNode* op) final {
    bool_op++;
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const OrNode* op) final {
    bool_op++;
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const NotNode* op) final {
    bool_op++;
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const SelectNode* op) final {
    select_op++;
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode* op) final {
    auto* pop = op->op.as<OpNode>();
    ICHECK(pop != nullptr);
    auto effect_kind = op_call_effect_[GetRef<Op>(pop)];
    bool is_pure =
        effect_kind == CallEffectKind::kPure || effect_kind == CallEffectKind::kExprAnnotation;

    if (is_pure) {
      if (op->dtype.is_float() || op->dtype.is_bfloat16()) {
        float_math_func++;
      } else {
        int_math_func++;
      }
    } else {
      if (op->dtype.is_float() || op->dtype.is_bfloat16()) {
        float_other_func++;
      } else {
        int_other_func++;
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  // todo(merrymercy): Detect MAD (Multiply–add)
  size_t float_mad{0};         // The number of float MAD (Multiply–add) ops
  size_t float_addsub{0};      // The number of float add and sub ops
  size_t float_mul{0};         // The number of float multiply ops
  size_t float_divmod{0};      // The number of float div and mod ops
  size_t float_cmp{0};         // The number of float comparison ops
  size_t float_math_func{0};   // The number of float math func calls
  size_t float_other_func{0};  // The number of other float func calls
  size_t int_mad{0};           // The number of integer MAD (Multiply–add) ops
  size_t int_addsub{0};        // The number of integer add and sub ops
  size_t int_mul{0};           // The number of float multiply ops
  size_t int_divmod{0};        // The number of float div and mod ops
  size_t int_cmp{0};           // The number of float comparison ops
  size_t int_math_func{0};     // The number of float math func calls
  size_t int_other_func{0};    // The number of other float func calls
  size_t bool_op{0};           // The number of bool ops
  size_t select_op{0};         // The number of select ops

  OpAttrMap<TCallEffectKind> op_call_effect_ = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
};

// Extract all buffer accesses in an expr
class BufferAccessExtractor : public StmtExprVisitor {
 public:
  void ExtractReads(const PrimExpr& expr) { this->VisitExpr(expr); }

  void InsertAccess(const Var& buf, BufferAccessType acc_type, const Array<PrimExpr>& indices) {
    BufferAccess& acc = buf_accesses[buf];
    acc.acc_type = acc_type;
    acc.indices.push_back(std::vector<PrimExpr>(indices.begin(), indices.end()));
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    AddAccess(op->buffer->data, op->indices);
    StmtExprVisitor::VisitExpr_(op);
  }

  void AddAccess(const Var& buffer, const Array<PrimExpr>& indices) {
    BufferAccess& acc = buf_accesses[buffer];
    switch (acc.acc_type) {
      case BufferAccessType::kRead:
        break;
      case BufferAccessType::kWrite:
        acc.acc_type = BufferAccessType::kReadWrite;
        break;
      case BufferAccessType::kReadWrite:
        break;
      case BufferAccessType::kUnknownRW:
      default:
        acc.acc_type = BufferAccessType::kRead;
        break;
    }

    if (acc.acc_type != BufferAccessType::kReadWrite) {
      // If a buffer is both read and written, in the tvm DSL, it must be a update,
      // so the indices should be the same. Then we can skip appending indices for it.
      // Otherwise we do the following.
      buf_accesses[buffer].indices.push_back(std::vector<PrimExpr>(indices.begin(), indices.end()));
    }
  }

  BufferMap<BufferAccess> buf_accesses;
};

// Compute the coefficient for an loop iterator in an expression
// Note: we use an approximation strategy to find coefficient.
// Hopefully, it is faster than DetectLinearEquation and can handle more cases (non-linear)
class CoefficientExtractor : public StmtExprVisitor {
 public:
  void VisitExpr_(const MulNode* node) final {
    StmtExprVisitor::VisitExpr_(node);
    if (visited_var) {
      if (!visited_add) {
        if (auto a = node->a.as<IntImmNode>()) {
          visited_mul = true;
          stride = a->value;
        } else if (auto b = node->b.as<IntImmNode>()) {
          visited_mul = true;
          stride = b->value;
        }
      }
    }
  }

  void VisitExpr_(const AddNode* node) final {
    StmtExprVisitor::VisitExpr_(node);
    if (visited_var) {
      if (!visited_mul) {
        visited_add = true;
        stride = 1;
      }
    }
  }

  void VisitExpr_(const VarNode* node) final {
    if (node == var_) {
      visited_var = true;
      // This is a magic default stride in case our approximation strategy fails
      stride = 2;
    }
  }

  int ExtractCoefficient(const PrimExpr& expr, const VarNode* var) {
    visited_var = visited_mul = visited_add = false;
    var_ = var;

    this->VisitExpr(expr);

    if (visited_var && !visited_mul && !visited_add) {
      return 1;
    } else {
      return stride;
    }
  }

  bool visited_var{false};
  bool visited_mul{false};
  bool visited_add{false};
  int stride{0};

 private:
  const VarNode* var_{nullptr};
};

// Compute stride for the accesses to a buffer
int64_t ComputeStride(const std::vector<std::vector<PrimExpr>>& indices,
                      const std::vector<int>& shape, const VarNode* stride_var) {
  // Use stride of 1 for 0-dimensional buffers. 0-dim buffers has a single
  // index access, so we have to check here.
  if (shape.size() == 0) {
    return 1;
  }
  int64_t min_stride = std::numeric_limits<int64_t>::max();
  bool find = false;
  CoefficientExtractor extractor;

  for (const auto& index : indices) {
    int64_t shape_stride = 1;
    for (int i = static_cast<int>(index.size()) - 1; i >= 0; i--) {
      int coefficient = extractor.ExtractCoefficient(index[i], stride_var);
      if (extractor.visited_var) {
        find = true;
        min_stride = std::min(min_stride, std::abs(coefficient) * shape_stride);
        break;
      }
      shape_stride *= shape[i];
    }
  }

  return find ? min_stride : 0;
}

// Compute touched bytes and cache lines for accesses to a buffer
void ComputeRegion(const std::vector<std::vector<PrimExpr>>& indices, arith::Analyzer* ana,
                   std::vector<int>* region) {
  region->clear();

  if (indices.empty()) {
    return;
  }

  region->reserve(indices[0].size());

  if (indices.size() == 1) {
    for (const auto& index : indices[0]) {
      ConstIntBound bound = ana->const_int_bound(index);
      region->push_back(bound->max_value - bound->min_value + 1);
    }
  } else {
    // future(lmzheng): implement a more accurate IntSet?
    for (size_t i = 0; i < indices[0].size(); ++i) {
      int64_t minimum = ConstIntBound::kPosInf, maximum = ConstIntBound::kNegInf;
      for (size_t j = 0; j < indices.size(); ++j) {
        ConstIntBound bound = ana->const_int_bound(indices[j][i]);

        minimum = std::min(minimum, bound->min_value);
        maximum = std::max(maximum, bound->max_value);
      }
      region->push_back(maximum - minimum + 1);
    }
  }
}

// Compute reuse distance and reuse ratio for accesses to a buffer
// return values: reuse_type, reuse_dis_iter, reuse_dis_bytes, reuse_ct
std::tuple<ReuseType, float, float, float> ComputeReuse(
    const Var& buf, const std::vector<std::vector<PrimExpr>>& indices,
    const std::vector<const ForNode*>& for_loop_stack,
    const std::unordered_map<const ForNode*,
                             BufferMap<std::vector<std::tuple<BufferAccessType, int64_t, int>>>>&
        for_touch_regions,
    const Analyzer& ana) {
  float reuse_dis_iter = 1.0f;
  float reuse_dis_bytes = -1.0f;

  for (int i = static_cast<int>(for_loop_stack.size()) - 1; i >= 0; --i) {
    const ForNode* cur_for = for_loop_stack[i];
    bool find = false;

    for (size_t j = 0; j < indices.size(); j++) {
      for (size_t k = 0; k < indices[j].size(); k++) {
        if (VarInExpr(cur_for->loop_var, indices[j][k])) {
          find = true;
          break;
        }
      }
      if (find) {
        break;
      }
    }

    int64_t extent = GetLoopExtent(for_loop_stack[i], ana);
    if (find) {
      // accumulate/update reuse distance
      reuse_dis_iter *= extent;
      reuse_dis_bytes = 0.0f;
      for (const auto& iter : for_touch_regions.at(cur_for)) {
        for (const auto& access : iter.second) {
          reuse_dis_bytes += std::get<1>(access) * std::get<2>(access);
        }
      }
    } else {
      // Have LoopMultipleRead reuse
      if (reuse_dis_bytes < 0) {
        // For the reuse in the innermost axis, the above code won't be executed.
        // So we compute bytes here
        reuse_dis_bytes = 0.0f;
        for (const auto& iter : for_touch_regions.at(cur_for)) {
          for (const auto& access : iter.second) {
            reuse_dis_bytes += 1 * std::get<2>(access);
          }
        }
      }
      return std::make_tuple(ReuseType::kLoopMultipleRead, reuse_dis_iter, reuse_dis_bytes, extent);
    }

    const BufferMap<std::vector<std::tuple<BufferAccessType, int64_t, int>>>& buffer_map =
        for_touch_regions.at(cur_for);

    int serial_reuse = static_cast<int>(buffer_map.at(buf).size()) - 1;
    if (serial_reuse > 0) {
      int64_t extent = GetLoopExtent(cur_for, ana);

      // Have SerialMultipleReadWrite reuse
      reuse_dis_iter = std::numeric_limits<float>::max();
      for (const auto& acc_info : buffer_map.at(buf)) {
        reuse_dis_iter = std::min(reuse_dis_iter, static_cast<float>(std::get<1>(acc_info)));
      }

      reuse_dis_bytes = 0.0f;
      for (const auto& iter : for_touch_regions.at(cur_for)) {
        for (const auto& access : iter.second) {
          reuse_dis_bytes += std::get<1>(access) * std::get<2>(access);
        }
      }

      return std::make_tuple(ReuseType::kSerialMultipleReadWrite, reuse_dis_iter / extent,
                             reuse_dis_bytes / extent, serial_reuse);
    }
  }

  return std::make_tuple(ReuseType::kNoReuse, 0, 0, 0);
}

// Extract features for every BufferStore statement
//
// This visitor assumes that loop bounds do no depend on data or on parent loop
// bounds. For example, `for i in .. { for j in range(i, ..) }` would result in
// inaccurate features. This visitor also does not take conditionals into
// consideration when creating features. Each branch of the conditional is
// taken at the same time.
class PerStoreFeatureExtractor : public StmtExprVisitor {
 public:
  explicit PerStoreFeatureExtractor(int cache_line_size, const Map<Var, Buffer>& existing_buffers)
      : cache_line_size_(cache_line_size) {
    for (const auto& buffer : existing_buffers) {
      buffer_shapes[buffer.first] = buffer.second->shape;
      buffer_dtypes[buffer.first] = buffer.second->dtype;
      // Also need to add a reference from the buffers internal variable. This
      // is usually how buffers are referenced within the body of a PrimFunc
      buffer_shapes[buffer.second->data] = buffer.second->shape;
      buffer_dtypes[buffer.second->data] = buffer.second->dtype;
    }
  }

  void VisitStmt_(const AttrStmtNode* node) final {
    if (node->attr_key == tir::attr::thread_extent || node->attr_key == tir::attr::virtual_thread) {
      const Var& var = node->node.as<IterVarNode>()->var;
      int extent = GetIntImm(node->value);

      int* plen = nullptr;

      const std::string& name = var.get()->name_hint;
      if (node->attr_key == tir::attr::thread_extent) {
        if (name == "blockIdx.x") {
          plen = &blockIdx_x_len_;
        } else if (name == "blockIdx.y") {
          plen = &block_idx_y_len_;
        } else if (name == "blockIdx.z") {
          plen = &block_idx_z_len_;
        } else if (name == "threadIdx.x") {
          plen = &threadIdx_x_len_;
        } else if (name == "threadIdx.y") {
          plen = &thread_idx_y_len_;
        } else if (name == "threadIdx.z") {
          plen = &thread_idx_z_len_;
        } else {
          LOG(FATAL) << "invalid thread itervar " + name;
        }
      } else {
        plen = &vthread_len_;
      }

      int extent_before = *plen;
      if (node->attr_key == tir::attr::thread_extent) {
        *plen = extent;
      } else {
        *plen *= extent;
      }

      is_gpu_ = true;

      // make a fake for node for blockIdx.x or threadIdx.x
      Stmt fake_for_node = For(var, 0, extent, ForKind::kParallel, node->body);

      outer_loop_prod_ *= extent;
      for_loop_stack_.push_back(fake_for_node.as<ForNode>());
      variable_definition_stack_.push_back({});
      StmtExprVisitor::VisitStmt_(node);
      variable_definition_stack_.pop_back();
      for_loop_stack_.pop_back();
      outer_loop_prod_ /= extent;

      *plen = extent_before;
    } else if (node->attr_key == "pragma_auto_unroll_max_step") {
      int value = GetIntImm(node->value);

      int16_t old_value = cur_auto_unroll_max_step_;
      cur_auto_unroll_max_step_ = value;
      StmtExprVisitor::VisitStmt_(node);
      cur_auto_unroll_max_step_ = old_value;
    } else {
      StmtExprVisitor::VisitStmt_(node);
    }
  }

  void VisitStmt_(const ForNode* node) final {
    ana_.Bind(node->loop_var, Range::FromMinExtent(node->min, node->extent));
    int64_t loop_extent = GetLoopExtent(node, ana_);

    if (node->kind == ForKind::kVectorized) {
      vec_for_stack_.push_back(node);
    } else if (node->kind == ForKind::kUnrolled) {
      unroll_for_stack_.push_back(node);
    } else if (node->kind == ForKind::kParallel) {
      parallel_for_stack_.push_back(node);
    }

    outer_loop_prod_ *= loop_extent;
    for_loop_stack_.push_back(node);
    variable_definition_stack_.push_back({});
    StmtExprVisitor::VisitStmt_(node);
    variable_definition_stack_.pop_back();
    for_loop_stack_.pop_back();
    outer_loop_prod_ /= loop_extent;

    if (node->kind == ForKind::kVectorized) {
      vec_for_stack_.pop_back();
    } else if (node->kind == ForKind::kUnrolled) {
      unroll_for_stack_.pop_back();
    } else if (node->kind == ForKind::kParallel) {
      parallel_for_stack_.pop_back();
    }
  }

  void VisitExpr_(const BufferLoadNode* node) final {
    // Store buffer shape/dtype. It may already be stored.
    buffer_shapes[node->buffer->data] = node->buffer->shape;
    buffer_dtypes[node->buffer->data] = node->buffer->dtype;
    StmtExprVisitor::VisitExpr_(node);
  }

  void VisitStmt_(const BufferStoreNode* node) final {
    // Store buffer shape/dtype. It may already be stored.
    buffer_shapes[node->buffer->data] = node->buffer->shape;
    buffer_dtypes[node->buffer->data] = node->buffer->dtype;

    MathOpCounter math_op_counter;
    math_op_counter(node->value);
    std::vector<float> mem_bytes_list;
    std::vector<float> compute_ops_list;
    double cur_compute_ops;

    // Group 1: Computation related features
    ExtractComputationFeature(node->buffer->data, node->indices, math_op_counter);

    // Group 2: Buffer access related features (per buffer)
    ExtractBufferAccessFeature(node->buffer->data, node->indices, node->value, math_op_counter,
                               &cur_compute_ops, &compute_ops_list, &mem_bytes_list);

    // Group 3: Arithmetic intensity related features
    ExtractArithmeticIntensityFeature(node->buffer->data, cur_compute_ops, compute_ops_list,
                                      mem_bytes_list);

    // Group 4: Allocation related features
    ExtractOuterScopeFeature(node->buffer->data);
  }

  void VisitStmt_(const BufferRealizeNode* node) final {
    // Store buffer shape/dtype. It may already be stored.
    buffer_shapes[node->buffer->data] = node->buffer->shape;
    buffer_dtypes[node->buffer->data] = node->buffer->dtype;
    StmtExprVisitor::VisitStmt_(node);

    // Group 5: Outer scope related features
    ExtractAllocationFeature(node);
  }

  void VisitStmt_(const AllocateNode* node) final {
    buffer_dtypes[node->buffer_var] = node->dtype;
    buffer_shapes[node->buffer_var] = node->extents;
    StmtExprVisitor::VisitStmt_(node);

    // Group 5: Outer scope related features
    ExtractAllocationFeature(node);
  }

  void VisitStmt_(const LetStmtNode* node) final {
    // TODO(tkonolige): add arithmetic counts from this statement to counts of inner stores.
    ana_.Bind(node->var, node->value);
    ICHECK(variable_definition_stack_.size() > 0)
        << "Variable definition outside of a for loop is not handled by feature extraction";
    variable_definition_stack_.back().push_back(std::make_tuple(node->var, node->value));
    StmtExprVisitor::VisitStmt_(node);
  }

  // Extract computation related features (group 1)
  void ExtractComputationFeature(const Var& buffer, const Array<PrimExpr>& indices,
                                 const MathOpCounter& math_op_counter) {
    FeatureSet& fea = buffer_features[buffer];

    // Computation related features
    fea.float_mad += outer_loop_prod_ * math_op_counter.float_mad;
    fea.float_addsub += outer_loop_prod_ * math_op_counter.float_addsub;
    fea.float_mul += outer_loop_prod_ * math_op_counter.float_mul;
    fea.float_divmod += outer_loop_prod_ * math_op_counter.float_divmod;
    fea.float_cmp += outer_loop_prod_ * math_op_counter.float_cmp;
    fea.float_math_func += outer_loop_prod_ * math_op_counter.float_math_func;
    fea.float_other_func += outer_loop_prod_ * math_op_counter.float_other_func;
    fea.int_mad += outer_loop_prod_ * math_op_counter.int_mad;
    fea.int_addsub += outer_loop_prod_ * math_op_counter.int_addsub;
    fea.int_mul += outer_loop_prod_ * math_op_counter.int_mul;
    fea.int_divmod += outer_loop_prod_ * math_op_counter.int_divmod;
    fea.int_math_func += outer_loop_prod_ * math_op_counter.int_math_func;
    fea.int_cmp += outer_loop_prod_ * math_op_counter.int_cmp;
    fea.int_other_func += outer_loop_prod_ * math_op_counter.int_other_func;
    fea.bool_op += outer_loop_prod_ * math_op_counter.bool_op;
    fea.select_op += outer_loop_prod_ * math_op_counter.select_op;

    fea.vec_len = fea.unroll_len = fea.parallel_len = 0.0f;
    fea.vec_type = fea.unroll_type = fea.parallel_type = AnnotationPosType::kPosNone;

    fea.vec_num = vec_for_stack_.size();
    if (!vec_for_stack_.empty()) {
      fea.vec_len = GetLoopExtent(vec_for_stack_.back(), ana_);
      fea.vec_prod = 1.0;
      for (const ForNode* pfor : vec_for_stack_) {
        fea.vec_prod *= GetLoopExtent(pfor, ana_);
      }
      fea.vec_type = AnnotationPosType::kPosMixed;
      // todo(merrymercy): this feature requires operation (tvm.compute) information
      // GetAnnotationPosEncoding(vec_for_stack_.back()->loop_var,
      // node->args, pcompute->axis, pcompute->reduce_axis);
    }

    fea.unroll_num = unroll_for_stack_.size();
    if (!unroll_for_stack_.empty()) {
      fea.unroll_len = GetLoopExtent(unroll_for_stack_.back(), ana_);
      fea.unroll_prod = 1.0;
      for (const ForNode* pfor : unroll_for_stack_) {
        fea.unroll_prod *= GetLoopExtent(pfor, ana_);
      }
      fea.unroll_type = AnnotationPosType::kPosMixed;
      // GetAnnotationPosEncoding(unroll_for_stack_.back()->loop_var,
      // node->args, pcompute->axis, pcompute->reduce_axis);
    }

    fea.parallel_num = parallel_for_stack_.size();
    if (!parallel_for_stack_.empty()) {
      fea.parallel_len = GetLoopExtent(parallel_for_stack_.back(), ana_);
      fea.parallel_prod = 1.0;
      for (const ForNode* pfor : parallel_for_stack_) {
        fea.parallel_prod *= GetLoopExtent(pfor, ana_);
      }
      fea.parallel_type = AnnotationPosType::kPosMixed;
      // GetAnnotationPosEncoding(parallel_for_stack_.back()->loop_var,
      // node->args, pcompute->axis, pcompute->reduce_axis);
    }

    // GPU threads
    fea.is_gpu = is_gpu_;
    fea.blockIdx_x_len = blockIdx_x_len_;
    fea.blockIdx_y_len = block_idx_y_len_;
    fea.blockIdx_z_len = block_idx_z_len_;
    fea.threadIdx_x_len = threadIdx_x_len_;
    fea.threadIdx_y_len = thread_idx_y_len_;
    fea.threadIdx_z_len = thread_idx_z_len_;
    fea.vthread_len = vthread_len_;
  }

  // Extract buffer access related features (group 2)
  void ExtractBufferAccessFeature(const Var& buffer, const Array<PrimExpr>& indices,
                                  const PrimExpr& value, const MathOpCounter& math_op_counter,
                                  double* cur_compute_ops, std::vector<float>* compute_ops_list,
                                  std::vector<float>* mem_bytes_list) {
    FeatureSet& fea = buffer_features[buffer];

    // Extract all buffer accesses
    std::vector<BufferAccessFeature> acc_feas;
    BufferAccessExtractor buf_extractor;
    buf_extractor.InsertAccess(buffer, BufferAccessType::kWrite, indices);
    buf_extractor.ExtractReads(value);

    mem_bytes_list->reserve(for_loop_stack_.size());
    compute_ops_list->reserve(for_loop_stack_.size());

    *cur_compute_ops = math_op_counter.float_mad + math_op_counter.float_addsub +
                       math_op_counter.float_mul + math_op_counter.float_divmod +
                       math_op_counter.float_cmp + math_op_counter.float_math_func +
                       math_op_counter.float_other_func;

    ICHECK_EQ(for_loop_stack_.size(), variable_definition_stack_.size())
        << "variable_definition_stack_ should mirror for_loop_stack_ in size";
    std::vector<int> tmp_region;
    for (int i = static_cast<int>(for_loop_stack_.size()) - 1; i >= 0; i--) {
      const ForNode* p_for = for_loop_stack_[i];

      // Construct a local analyzer context which contains definitions (for and
      // let) from innermost loops up to and including `i`. For loop variable
      // definitions in loops more outer than `i` are set to 1 so that we can
      // get per-loop-iteration features. Note that we add these definitions
      // from outermost to innermost because inner definitions may depend on
      // outer ones.
      Analyzer local_analyzer;
      for (int j = 0; j < i; j++) {
        local_analyzer.Bind(for_loop_stack_.at(j)->loop_var,
                            Range::FromMinExtent(for_loop_stack_.at(j)->min, 1));
      }
      for (int j = i; j < static_cast<int>(for_loop_stack_.size()); j++) {
        local_analyzer.Bind(
            for_loop_stack_.at(j)->loop_var,
            Range::FromMinExtent(for_loop_stack_.at(j)->min, for_loop_stack_.at(j)->extent));
        for (auto definition : variable_definition_stack_.at(j)) {
          local_analyzer.Bind(std::get<0>(definition), std::get<1>(definition));
        }
      }

      // Note, here we do overwrite.
      // So if there are multiple BufferStoreNode, the last one will overwrite the first few.
      // e.g. The update part in gemm will overwrite the init part.
      BufferMap<std::vector<std::tuple<BufferAccessType, int64_t, int>>>& buffer_regions_map =
          for_touch_regions_[p_for];

      int64_t mem_bytes = 0;
      for (const auto& x : buf_extractor.buf_accesses) {
        const Var& t = x.first;
        const BufferAccess& acc = x.second;

        ComputeRegion(acc.indices, &local_analyzer, &tmp_region);
        int64_t touched_size = ElementProduct(tmp_region);
        touched_size = std::max<int64_t>(0, touched_size);
        buffer_regions_map[t].push_back(
            std::make_tuple(acc.acc_type, touched_size, buffer_dtypes.at(t).bytes()));
        mem_bytes += touched_size * buffer_dtypes.at(t).bytes();
      }

      mem_bytes_list->push_back(mem_bytes);
      *cur_compute_ops *= GetLoopExtent(for_loop_stack_[i], local_analyzer);
      compute_ops_list->push_back(*cur_compute_ops);
    }

    //  Buffer access related features (per buffer)
    for (const auto& x : buf_extractor.buf_accesses) {
      const Var& t = x.first;
      const BufferAccess& acc = x.second;

      std::vector<int> int_shape;
      for (const auto& dim : buffer_shapes.at(t)) {
        int_shape.push_back(GetIntImm(dim));
      }

      size_t ele_bytes = buffer_dtypes.at(t).bytes();

      // calculate bytes
      float bytes = outer_loop_prod_ * ele_bytes;
      float unique_bytes;

      // calculate cache lines
      int64_t stride;
      float lines;
      float unique_lines;

      if (for_loop_stack_.empty()) {
        unique_bytes = ele_bytes;
        stride = 0;
        lines = 1.0f;
        unique_lines = 1.0f;
      } else {
        unique_bytes = static_cast<float>(
                           std::get<1>(for_touch_regions_[for_loop_stack_.front()][t].front())) *
                       ele_bytes;

        stride = 0;
        int64_t reduce_ratio = 1;

        int i;
        for (i = static_cast<int>(for_loop_stack_.size()) - 1; i >= 0; i--) {
          stride = ComputeStride(acc.indices, int_shape, for_loop_stack_[i]->loop_var.get());
          if (stride != 0) {
            break;
          }
          reduce_ratio *= GetLoopExtent(for_loop_stack_.back(), ana_);
        }

        lines = outer_loop_prod_ / reduce_ratio *
                std::min(1.0f, 1.0f * stride * ele_bytes / cache_line_size_);
        lines = std::max(lines, 1.0f);

        // convert `stride` back to the stride of the innermost iterator
        stride = (i == static_cast<int>(for_loop_stack_.size()) - 1 ? stride : 0);

        float n_continuous = ele_bytes;
        for (int i = std::min(static_cast<int>(tmp_region.size()) - 1,
                              static_cast<int>(int_shape.size()) - 1);
             i >= 0; i--) {
          if (tmp_region[i] == int_shape[i]) {
            n_continuous *= tmp_region[i];
            break;
          }
        }
        unique_lines = unique_bytes / std::min(n_continuous, static_cast<float>(cache_line_size_));
        unique_lines = std::max(unique_lines, 1.0f);
      }

      auto [reuse_type, reuse_dis_iter, reuse_dis_bytes, reuse_ct] =
          ComputeReuse(t, acc.indices, for_loop_stack_, for_touch_regions_, ana_);

      acc_feas.emplace_back();
      BufferAccessFeature& acc_fea = acc_feas.back();

      // TODO(tkonolige): save buffer names and use those instead?
      acc_fea.buffer_name = t->name_hint;
      acc_fea.acc_type = acc.acc_type;
      acc_fea.stride = stride;
      acc_fea.bytes = bytes;
      acc_fea.unique_bytes = unique_bytes;
      acc_fea.lines = lines;
      acc_fea.unique_lines = unique_lines;
      acc_fea.reuse_type = reuse_type;
      acc_fea.reuse_dis_iter = reuse_dis_iter;
      acc_fea.reuse_dis_bytes = reuse_dis_bytes;
      acc_fea.reuse_ct = reuse_ct;
      if (acc_fea.reuse_ct > 0.5) {
        acc_fea.bytes_d_reuse_ct = bytes / reuse_ct;
        acc_fea.unique_bytes_d_reuse_ct = unique_bytes / reuse_ct;
        acc_fea.lines_d_reuse_ct = lines / reuse_ct;
        acc_fea.unique_lines_d_reuse_ct = unique_lines / reuse_ct;
      } else {
        // no reuse, multiply by a magic number '2'
        acc_fea.bytes_d_reuse_ct = bytes * 2;
        acc_fea.unique_bytes_d_reuse_ct = unique_bytes * 2;
        acc_fea.lines_d_reuse_ct = lines * 2;
        acc_fea.unique_lines_d_reuse_ct = unique_lines * 2;
      }
    }

    fea.access_feas = acc_feas;
  }

  // Extract arithmetic intensity related feature (group 3)
  void ExtractArithmeticIntensityFeature(const Var& buffer, double cur_compute_ops,
                                         const std::vector<float>& compute_ops_list,
                                         const std::vector<float>& mem_bytes_list) {
    FeatureSet& fea = buffer_features[buffer];

    // Compute arithmetic intensity curve (y axis : arithmetic intensity, x axis : flops).
    // We use piecewise linear interpolation to fit this curve.
    int pt = 0;
    if (cur_compute_ops <= 0 || compute_ops_list.empty()) {
      std::fill(fea.arith_intensity_curve,
                fea.arith_intensity_curve + ARITH_INTENSITY_CURVE_SAMPLE_N, 0.0);
    } else {
      for (size_t i = 0; i < ARITH_INTENSITY_CURVE_SAMPLE_N; ++i) {
        float cur_compute_ops = compute_ops_list.back() * (i + 1) / ARITH_INTENSITY_CURVE_SAMPLE_N;
        while (compute_ops_list[pt] < cur_compute_ops - 1e-4) {
          pt++;
        }
        ICHECK_LT(pt, compute_ops_list.size());

        float value;
        if (pt == 0) {
          value = compute_ops_list[pt] / mem_bytes_list[pt];
        } else {
          float base = compute_ops_list[pt - 1] / mem_bytes_list[pt - 1];
          float slope = (compute_ops_list[pt] / mem_bytes_list[pt] -
                         compute_ops_list[pt - 1] / mem_bytes_list[pt - 1]) /
                        (compute_ops_list[pt] - compute_ops_list[pt - 1]);
          value = base + slope * (cur_compute_ops - compute_ops_list[pt - 1]);
        }
        fea.arith_intensity_curve[i] = value;
      }
    }
  }

  // Extract allocation related features (group 4)
  void ExtractAllocationFeature(const BufferRealizeNode* node) {
    FeatureSet& fea = buffer_features[node->buffer->data];

    float allocation_size = 1.0f;
    for (const auto& x : node->bounds) {
      allocation_size *= GetIntImm(x->extent);
    }
    // allocation feature
    fea.alloc_size = allocation_size * node->buffer->dtype.bytes();
    fea.alloc_prod = allocation_size * outer_loop_prod_;
    fea.alloc_outer_prod = outer_loop_prod_;
    fea.alloc_inner_prod = fea.outer_prod / outer_loop_prod_;
  }

  void ExtractAllocationFeature(const AllocateNode* node) {
    FeatureSet& fea = buffer_features[node->buffer_var];

    float allocation_size = 1.0f;
    for (const auto& x : node->extents) {
      // TODO(tkonolige): will not handle dynamic shape
      allocation_size *= GetIntImm(x);
    }
    // allocation feature
    fea.alloc_size = allocation_size * node->dtype.bytes();
    fea.alloc_prod = allocation_size * outer_loop_prod_;
    fea.alloc_outer_prod = outer_loop_prod_;
    fea.alloc_inner_prod = fea.outer_prod / outer_loop_prod_;
  }

  // Extract outer scope related features (group 5)
  void ExtractOuterScopeFeature(const Var& buffer) {
    FeatureSet& fea = buffer_features[buffer];

    fea.outer_prod = outer_loop_prod_;
    fea.num_loops = for_loop_stack_.size();
    fea.auto_unroll_max_step = cur_auto_unroll_max_step_;
  }

  // Stores FeatureSet for every buffer
  BufferMap<FeatureSet> buffer_features;

 private:
  // The shared arithmetic analyzer
  Analyzer ana_;

  // The product of outer loop
  float outer_loop_prod_ = 1.0f;

  // The stacks to store parent loops during DFS
  std::vector<const ForNode*> for_loop_stack_;
  std::vector<const ForNode*> parallel_for_stack_;
  std::vector<const ForNode*> vec_for_stack_;
  std::vector<const ForNode*> unroll_for_stack_;
  std::vector<std::vector<std::tuple<Var, PrimExpr>>> variable_definition_stack_;

  // GPU-related features
  bool is_gpu_{false};
  int blockIdx_x_len_{1};
  int block_idx_y_len_{1};
  int block_idx_z_len_{1};
  int threadIdx_x_len_{1};
  int thread_idx_y_len_{1};
  int thread_idx_z_len_{1};
  int vthread_len_{1};
  int16_t cur_auto_unroll_max_step_{0};

  // Store touch region information for all for loops. The format of this nested map:
  // For a loop, for all its touched buffers, for all different accesses to the buffers,
  // its (access type, number of touched elements, number of bytes of single element)
  std::unordered_map<const ForNode*,
                     BufferMap<std::vector<std::tuple<BufferAccessType, int64_t, int>>>>
      for_touch_regions_;

  // The default cache line size in bytes
  const int cache_line_size_ = 64;

  // Storage of buffer shape and dtype information. Needed because Load/Store
  // nodes only do not contain this information.
  BufferMap<Array<PrimExpr>> buffer_shapes;
  BufferMap<DataType> buffer_dtypes;
};

// shifted log to incorporate the property that log2p(0) = 0
inline float log2p(float x) { return x < 0 ? -std::log2(-x + 1) : std::log2(x + 1); }

void GetPerStoreFeature(const PrimFunc& func, int cache_line_size, int max_n_bufs,
                        std::vector<float>* ret, bool log_scale) {
  PerStoreFeatureExtractor extractor(cache_line_size, func->buffer_map);
  extractor(func->body);

  auto slog = log_scale ? log2p : [](float x) { return x; };

  ret->push_back(extractor.buffer_features.size());

  for (const auto& x : extractor.buffer_features) {
    const FeatureSet& fea_set = x.second;

    /***** Group 1: Computation related features *****/
    ret->push_back(slog(fea_set.float_mad));
    ret->push_back(slog(fea_set.float_addsub));
    ret->push_back(slog(fea_set.float_mul));
    ret->push_back(slog(fea_set.float_divmod));
    ret->push_back(slog(fea_set.float_cmp));
    ret->push_back(slog(fea_set.float_math_func));
    ret->push_back(slog(fea_set.float_other_func));
    ret->push_back(slog(fea_set.int_mad));
    ret->push_back(slog(fea_set.int_addsub));
    ret->push_back(slog(fea_set.int_mul));
    ret->push_back(slog(fea_set.int_divmod));
    ret->push_back(slog(fea_set.int_cmp));
    ret->push_back(slog(fea_set.int_math_func));
    ret->push_back(slog(fea_set.int_other_func));
    ret->push_back(slog(fea_set.bool_op));
    ret->push_back(slog(fea_set.select_op));

    ret->push_back(slog(fea_set.vec_num));
    ret->push_back(slog(fea_set.vec_prod));
    ret->push_back(slog(fea_set.vec_len));
    for (int i = 0; i <= static_cast<int>(AnnotationPosType::kPosMixed); i++) {
      ret->push_back(i == static_cast<int>(fea_set.vec_type));
    }

    ret->push_back(slog(fea_set.unroll_num));
    ret->push_back(slog(fea_set.unroll_prod));
    ret->push_back(slog(fea_set.unroll_len));
    for (int i = 0; i <= static_cast<int>(AnnotationPosType::kPosMixed); i++) {
      ret->push_back(i == static_cast<int>(fea_set.unroll_type));
    }

    ret->push_back(slog(fea_set.parallel_num));
    ret->push_back(slog(fea_set.parallel_prod));
    ret->push_back(slog(fea_set.parallel_len));
    for (int i = 0; i <= static_cast<int>(AnnotationPosType::kPosMixed); i++) {
      ret->push_back(i == static_cast<int>(fea_set.parallel_type));
    }

    ret->push_back(fea_set.is_gpu);
    ret->push_back(slog(fea_set.blockIdx_x_len));
    ret->push_back(slog(fea_set.blockIdx_y_len));
    ret->push_back(slog(fea_set.blockIdx_z_len));
    ret->push_back(slog(fea_set.threadIdx_x_len));
    ret->push_back(slog(fea_set.threadIdx_y_len));
    ret->push_back(slog(fea_set.threadIdx_z_len));
    ret->push_back(slog(fea_set.vthread_len));

    /***** Group 2: Buffer access related features *****/
    // sort according to pair (lines, bytes)
    std::vector<std::pair<float, float>> buf_order_key;
    for (const auto& acc_fea : fea_set.access_feas) {
      buf_order_key.emplace_back(acc_fea.lines, acc_fea.bytes);
    }
    std::vector<int> buf_order(buf_order_key.size());
    std::iota(buf_order.begin(), buf_order.end(), 0);

    auto cmp = [&buf_order_key](int l, int r) {
      return buf_order_key[l].first > buf_order_key[r].first ||
             (buf_order_key[l].first == buf_order_key[r].first &&
              buf_order_key[l].second > buf_order_key[r].second);
    };
    std::sort(buf_order.begin(), buf_order.end(), cmp);
    int n_bufs = std::min(max_n_bufs, static_cast<int>(buf_order.size()));
    buf_order.resize(n_bufs);

    for (int idx : buf_order) {
      const auto& acc_fea = fea_set.access_feas[idx];
      for (int j = 0; j <= static_cast<int>(BufferAccessType::kReadWrite); ++j) {
        ret->push_back(j == static_cast<int>(acc_fea.acc_type));
      }
      ret->push_back(slog(acc_fea.bytes));
      ret->push_back(slog(acc_fea.unique_bytes));
      ret->push_back(slog(acc_fea.lines));
      ret->push_back(slog(acc_fea.unique_lines));
      for (int j = 0; j <= static_cast<int>(ReuseType::kNoReuse); ++j) {
        ret->push_back(j == static_cast<int>(acc_fea.reuse_type));
      }
      ret->push_back(slog(acc_fea.reuse_dis_iter));
      ret->push_back(slog(acc_fea.reuse_dis_bytes));
      ret->push_back(slog(acc_fea.reuse_ct));
      ret->push_back(slog(acc_fea.bytes_d_reuse_ct));
      ret->push_back(slog(acc_fea.unique_bytes_d_reuse_ct));
      ret->push_back(slog(acc_fea.lines_d_reuse_ct));
      ret->push_back(slog(acc_fea.unique_lines_d_reuse_ct));
      ret->push_back(slog(acc_fea.stride));
    }
    // - fill padding
    for (int i = 0; i < max_n_bufs - n_bufs; ++i) {
      for (int j = 0; j <= static_cast<int>(BufferAccessType::kReadWrite); ++j) {  // 3
        ret->push_back(0.0f);
      }
      ret->push_back(0.0f);
      ret->push_back(0.0f);
      ret->push_back(0.0f);
      ret->push_back(0.0f);
      for (int j = 0; j <= static_cast<int>(ReuseType::kNoReuse); ++j) {  // 3
        ret->push_back(0.0f);
      }
      ret->push_back(0.0f);
      ret->push_back(0.0f);
      ret->push_back(0.0f);
      ret->push_back(0.0f);
      ret->push_back(0.0f);
      ret->push_back(0.0f);
      ret->push_back(0.0f);
      ret->push_back(0.0f);
    }

    /***** Group 3: Arithmetic intensity related features *****/
    for (size_t i = 0; i < ARITH_INTENSITY_CURVE_SAMPLE_N; ++i) {
      ret->push_back(slog(fea_set.arith_intensity_curve[i]));
    }

    /***** Group 4: Allocation related features *****/
    ret->push_back(slog(fea_set.alloc_size));
    ret->push_back(slog(fea_set.alloc_prod));
    ret->push_back(slog(fea_set.alloc_outer_prod));
    ret->push_back(slog(fea_set.alloc_inner_prod));

    /***** Group 5: Outer scope related features *****/
    ret->push_back(slog(fea_set.outer_prod));
    ret->push_back(slog(fea_set.num_loops));
    ret->push_back(slog(fea_set.auto_unroll_max_step));
  }
}

void GetPerStoreFeatureName(int max_n_bufs, std::vector<std::string>* ret) {
  /***** Group 1: Computation related features *****/
  ret->push_back(("float_mad"));
  ret->push_back(("float_addsub"));
  ret->push_back(("float_mul"));
  ret->push_back(("float_divmod"));
  ret->push_back(("float_cmp"));
  ret->push_back(("float_mathfunc"));
  ret->push_back(("float_otherfunc"));
  ret->push_back(("int_mad"));
  ret->push_back(("int_addsub"));
  ret->push_back(("int_mul"));
  ret->push_back(("int_divmod"));
  ret->push_back(("int_cmp"));
  ret->push_back(("int_mathfunc"));
  ret->push_back(("int_otherfunc"));
  ret->push_back(("bool_op"));
  ret->push_back(("select_op"));
  ret->push_back(("vec_num"));
  ret->push_back(("vec_prod"));
  ret->push_back(("vec_len"));
  ret->push_back(("vec_type.kPosNone"));
  ret->push_back(("vec_type.kPosInnerSpatial"));
  ret->push_back(("vec_type.kPosMiddleSpatial"));
  ret->push_back(("vec_type.kPosOuterSpatial"));
  ret->push_back(("vec_type.kPosInnerReduce"));
  ret->push_back(("vec_type.kPosMiddleReduce"));
  ret->push_back(("vec_type.kPosOuterReduce"));
  ret->push_back(("vec_type.kPosMixed"));
  ret->push_back(("unroll_num"));
  ret->push_back(("unroll_prod"));
  ret->push_back(("unroll_len"));
  ret->push_back(("unroll_type.kPosNone"));
  ret->push_back(("unroll_type.kPosInnerSpatial"));
  ret->push_back(("unroll_type.kPosMiddleSpatial"));
  ret->push_back(("unroll_type.kPosOuterSpatial"));
  ret->push_back(("unroll_type.kPosInnerReduce"));
  ret->push_back(("unroll_type.kPosMiddleReduce"));
  ret->push_back(("unroll_type.kPosOuterReduce"));
  ret->push_back(("unroll_type.kPosMixed"));
  ret->push_back(("parallel_num"));
  ret->push_back(("parallel_prod"));
  ret->push_back(("parallel_len"));
  ret->push_back(("parallel_type.kPosNone"));
  ret->push_back(("parallel_type.kPosInnerSpatial"));
  ret->push_back(("parallel_type.kPosMiddleSpatial"));
  ret->push_back(("parallel_type.kPosOuterSpatial"));
  ret->push_back(("parallel_type.kPosInnerReduce"));
  ret->push_back(("parallel_type.kPosMiddleReduce"));
  ret->push_back(("parallel_type.kPosOuterReduce"));
  ret->push_back(("parallel_type.kPosMixed"));
  ret->push_back(("is_gpu"));
  ret->push_back(("blockIdx_x_len"));
  ret->push_back(("blockIdx_y_len"));
  ret->push_back(("blockIdx_z_len"));
  ret->push_back(("threadIdx_x_len"));
  ret->push_back(("threadIdx_y_len"));
  ret->push_back(("threadIdx_z_len"));
  ret->push_back(("vthread_len"));
  // section total: 57

  /***** Group 2: Buffer access related features *****/
  for (size_t i = 0; i < static_cast<size_t>(max_n_bufs); ++i) {
    std::string prefix = "B" + std::to_string(i) + ".";
    ret->push_back((prefix + "acc_type.kRead"));
    ret->push_back((prefix + "acc_type.kWrite"));
    ret->push_back((prefix + "acc_type.kReadWrite"));
    ret->push_back((prefix + "bytes"));
    ret->push_back((prefix + "unique_bytes"));
    ret->push_back((prefix + "lines"));
    ret->push_back((prefix + "unique_lines"));
    ret->push_back((prefix + "reuse_type.kLoopMultipleRead"));
    ret->push_back((prefix + "reuse_type.kSerialMultipleReadWrite"));
    ret->push_back((prefix + "reuse_type.kNoReuse"));
    ret->push_back((prefix + "reuse_dis_iter"));
    ret->push_back((prefix + "reuse_dis_bytes"));
    ret->push_back((prefix + "reuse_ct"));
    ret->push_back((prefix + "bytes_d_reuse_ct"));
    ret->push_back((prefix + "unique_bytes_d_reuse_ct"));
    ret->push_back((prefix + "lines_d_reuse_ct"));
    ret->push_back((prefix + "unique_lines_d_reuse_ct"));
    ret->push_back((prefix + "stride"));
  }
  // section total : max_n_bufs * 18

  /***** Group 3: Arithmetic intensity related features *****/
  for (size_t i = 0; i < ARITH_INTENSITY_CURVE_SAMPLE_N; ++i) {
    ret->push_back(("arith_intensity_curve_" + std::to_string(i)));
  }
  // section total: ARITH_INTENSITY_CURVE_SAMPLE_N = 10

  /***** Group 4: Allocation related features *****/
  ret->push_back(("alloc_size"));
  ret->push_back(("alloc_prod"));
  ret->push_back(("alloc_outer_prod"));
  ret->push_back(("alloc_inner_prod"));
  // section total : 4

  /***** Group 5: Outer scope related features *****/
  ret->push_back(("outer_prod"));
  ret->push_back(("num_loops"));
  ret->push_back(("auto_unroll_max_step"));
  // section total : 3
}

void GetPerStoreFeaturesWorkerFunc(const SearchTask& task, const State& state, int max_n_bufs,
                                   std::vector<float>* feature, std::atomic<int>* error_ct) {
  auto [sch, tensors] = task->compute_dag.ApplySteps(state->transform_steps);

  // When inlining, replace const matrices with const values.
  // Produces wrong IR, but good enough for feature extraction, and
  // can improve the speed of feature extraction/search.  Must be
  // called before ScheduleToModule to have an effect.
  sch = sch.normalize_for_feature_extraction();

  try {
    const std::string& name = "main";
    auto pass_ctx = tvm::transform::PassContext::Current();

    auto mod = ScheduleToModule(sch, Array<ObjectRef>{tensors.begin(), tensors.end()}, name,
                                std::unordered_map<te::Tensor, te::Buffer>(),
                                GlobalVarSupply(NameSupply("")));

    bool disable_vectorize =
        pass_ctx->GetConfig<Bool>("tir.disable_vectorize", Bool(false)).value();
    bool instrument_bound_checkers =
        pass_ctx->GetConfig<Bool>("tir.instrument_bound_checkers", Bool(false)).value();

    if (IsGPUTask(task)) {
      auto pass_list = Array<tvm::transform::Pass>();
      // Phase 0
      pass_list.push_back(tir::transform::InjectPrefetch());
      pass_list.push_back(tir::transform::StorageFlatten(64, instrument_bound_checkers));
      // Phase 1
      pass_list.push_back(tir::transform::NarrowDataType(32));
      pass_list.push_back(tir::transform::Simplify());
      pass_list.push_back(tir::transform::VectorizeLoop(!disable_vectorize));
      pass_list.push_back(tir::transform::InjectVirtualThread());
      pass_list.push_back(tir::transform::StorageRewrite());
      pass_list.push_back(tir::transform::Simplify());
      tvm::Map<String, tvm::PrimExpr> gpu_params{
          {"max_shared_memory_per_block", task->hardware_params->max_shared_memory_per_block},
          {"max_local_memory_per_block", task->hardware_params->max_local_memory_per_block},
          {"max_threads_per_block", task->hardware_params->max_threads_per_block},
          {"max_vector_bytes", task->hardware_params->vector_unit_bytes},
          {"max_vthread", task->hardware_params->max_vthread_extent},
      };
      pass_list.push_back(tir::transform::VerifyGPUCode(gpu_params));
      const auto& optimize = tir::transform::Sequential(pass_list);
      optimize(mod);
    }
    if (IsHexagonTask(task)) {
      Target target = task->target;
      const auto& optimize = tir::transform::Sequential({tir::transform::VerifyVTCMLimit(target)});
      optimize(mod);
    }
    const auto& optimize =
        tir::transform::Sequential(Array<tvm::transform::Pass>{tir::transform::Simplify()});
    mod = optimize(std::move(mod));
    PrimFunc prim_func = Downcast<PrimFunc>(mod->Lookup(name));
    GetPerStoreFeature(prim_func, task->hardware_params->cache_line_bytes, max_n_bufs, feature);
  } catch (Error& e) {
    (*error_ct)++;
  }
}

void GetPerStoreFeaturesFromStates(const Array<State>& states, const SearchTask& task,
                                   int skip_first_n_feature_extraction, int max_n_bufs,
                                   std::vector<std::vector<float>>* features) {
  // extract features
  features->assign(states.size(), std::vector<float>());

  std::atomic<int> error_ct(0);

  support::parallel_for(skip_first_n_feature_extraction, states.size(),
                        [&task, &states, &max_n_bufs, &features, &error_ct](int i) {
                          GetPerStoreFeaturesWorkerFunc(task, states[i], max_n_bufs,
                                                        &(*features)[i], &error_ct);
                        });
}

void GetPerStoreFeaturesFromStates(const Array<State>& states, const std::vector<SearchTask>& tasks,
                                   int skip_first_n_feature_extraction, int max_n_bufs,
                                   std::vector<std::vector<float>>* features) {
  // extract features
  features->assign(states.size(), std::vector<float>());

  std::atomic<int> error_ct(0);

  support::parallel_for(skip_first_n_feature_extraction, states.size(),
                        [&tasks, &states, &max_n_bufs, &features, &error_ct](int i) {
                          GetPerStoreFeaturesWorkerFunc(tasks[i], states[i], max_n_bufs,
                                                        &(*features)[i], &error_ct);
                        });
}

void GetPerStoreFeaturesFromFile(const std::string& filename, int max_lines, int max_n_bufs,
                                 std::vector<std::vector<float>>* features,
                                 std::vector<float>* normalized_throughputs,
                                 std::vector<int>* task_ids) {
  Array<State> states;
  std::vector<SearchTask> tasks;

  normalized_throughputs->clear();
  task_ids->clear();

  // (workload_key, target) -> (search_task, task_id)
  std::unordered_map<std::pair<std::string, std::string>, std::pair<SearchTask, size_t>> task_cache;
  // task_id -> min_cost
  std::vector<float> min_costs;

  const auto* workload_key_to_tensors =
      tvm::runtime::Registry::Get("auto_scheduler.workload_key_to_tensors");
  ICHECK(workload_key_to_tensors != nullptr);

  // read from file
  RecordReader reader(filename);
  auto cur_inp = make_object<MeasureInputNode>();
  auto cur_res = make_object<MeasureResultNode>();
  while (reader->ReadNext(cur_inp.get(), cur_res.get())) {
    float cost = static_cast<float>(FloatArrayMean(cur_res->costs));
    const std::string& workload_key = cur_inp->task->workload_key;

    SearchTask task;
    size_t task_id;
    std::pair<std::string, std::string> key(workload_key, cur_inp->task->target->str());
    auto find_res = task_cache.find(key);
    if (find_res == task_cache.end()) {
      // rebuild task
      Array<te::Tensor> tensors = (*workload_key_to_tensors)(workload_key);
      Target target = cur_inp->task->target;
      Target target_host = cur_inp->task->target_host;
      CheckAndUpdateHostConsistency(&target, &target_host);
      task = SearchTask(ComputeDAG(tensors), workload_key, target, target_host,
                        cur_inp->task->hardware_params, cur_inp->task->layout_rewrite_option,
                        cur_inp->task->task_input_names);
      task_id = task_cache.size();

      // compute min cost for each task
      task_cache.insert(std::make_pair(key, std::make_pair(task, task_id)));
      min_costs.push_back(cost);
    } else {
      std::tie(task, task_id) = find_res->second;
      min_costs[task_id] = std::min(min_costs[task_id], cost);
    }

    tasks.push_back(std::move(task));
    task_ids->push_back(task_id);
    states.push_back(cur_inp->state);
    normalized_throughputs->push_back(cost);

    if (max_lines > 0 && static_cast<int>(states.size()) >= max_lines) {
      break;
    }
  }

  for (size_t i = 0; i < normalized_throughputs->size(); ++i) {
    (*normalized_throughputs)[i] = min_costs[(*task_ids)[i]] / (*normalized_throughputs)[i];
  }

  GetPerStoreFeaturesFromStates(states, tasks, 0, max_n_bufs, features);
}

void GetPerStoreFeaturesFromMeasurePairs(const Array<MeasureInput>& inputs,
                                         const Array<MeasureResult>& results,
                                         int skip_first_n_feature_extraction, int max_n_bufs,
                                         std::vector<std::vector<float>>* features,
                                         std::vector<float>* normalized_throughputs,
                                         std::vector<int>* task_ids) {
  Array<State> states;
  std::vector<SearchTask> tasks;

  normalized_throughputs->clear();
  task_ids->clear();

  // (workload_key, target) -> (search_task, task_id)
  std::unordered_map<std::pair<std::string, std::string>, std::pair<SearchTask, size_t>> task_cache;
  // task_id -> min_cost
  std::vector<float> min_costs;

  const auto* workload_key_to_tensors =
      tvm::runtime::Registry::Get("auto_scheduler.workload_key_to_tensors");
  ICHECK(workload_key_to_tensors != nullptr);

  tasks.reserve(inputs.size());
  normalized_throughputs->reserve(inputs.size());
  task_ids->reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    float cost = static_cast<float>(FloatArrayMean(results[i]->costs));
    const std::string& workload_key = inputs[i]->task->workload_key;
    SearchTask task;

    size_t task_id;
    std::pair<std::string, std::string> key(workload_key, inputs[i]->task->target->str());
    auto find_res = task_cache.find(key);
    if (find_res == task_cache.end()) {
      if (inputs[i]->task->compute_dag.defined()) {  // the measure input is complete
        task = inputs[i]->task;
      } else {
        // The measure input is incomplete, rebuild task for incomplete measure pairs read from file
        try {
          Array<te::Tensor> tensors = (*workload_key_to_tensors)(workload_key);
          Target target = inputs[i]->task->target;
          Target target_host = inputs[i]->task->target_host;
          CheckAndUpdateHostConsistency(&target, &target_host);
          task =
              SearchTask(ComputeDAG(tensors), workload_key, target, target_host,
                         inputs[i]->task->hardware_params, inputs[i]->task->layout_rewrite_option,
                         inputs[i]->task->task_input_names);
        } catch (std::exception& e) {
          // Cannot build ComputeDAG from workload key, the task may have not been registered in
          // this search round
          continue;
        }
      }
      task_id = task_cache.size();

      // compute min cost for each task
      task_cache.insert(std::make_pair(key, std::make_pair(task, task_id)));
      min_costs.push_back(cost);
    } else {
      std::tie(task, task_id) = find_res->second;
      min_costs[task_id] = std::min(min_costs[task_id], cost);
    }

    tasks.push_back(std::move(task));
    task_ids->push_back(task_id);
    states.push_back(inputs[i]->state);
    normalized_throughputs->push_back(cost);
  }

  for (size_t i = 0; i < normalized_throughputs->size(); ++i) {
    (*normalized_throughputs)[i] = min_costs[(*task_ids)[i]] / (*normalized_throughputs)[i];
  }

  GetPerStoreFeaturesFromStates(states, tasks, skip_first_n_feature_extraction, max_n_bufs,
                                features);
}

/*
 * \brief Serialize a two-dimensional variable-size feature vector with normalized throughputs
 * and task ids to a one-dimensional flatten byte array.
 *
 * For faster data copy between c++ and python, the c++ part returns features in a single
 * flatten array using a packed format. The python part then unpacks the flatten array.
 *
 * The packed format for n records is:
 * {
 *   int   n;
 *   int   sizes[n+2];           // The sizes for the following arrays
 *
 *   float features_0[size[0]];  // The features for record 0
 *   float features_1[size[1]];  // The features for record 1
 *   ...
 *   float features_i[size[i]];  // The features for record i
 *   ... // until i == n - 1
 *
 *   float throughputs[sizes[n]];  // The normalized throughputs for n records
 *   int   task_ids[size[n+1]];   // The task ids for n records
 *
 * }
 * To implement this format, we also store int as float, so we can store all numbers
 * into a single float array.
 */
TVMByteArray SerializeFeatures(std::vector<std::vector<float>>&& features,
                               std::vector<float>&& normalized_throughputs,
                               std::vector<int>&& task_ids, std::vector<char>* out_data) {
  size_t total_bytes = 0;
  std::vector<int> size_vector;

  int n = features.size();

  // serialize sizes
  size_t size_vector_size = 1 + n + 2;
  total_bytes += size_vector_size * sizeof(int);

  size_vector.reserve(size_vector_size);
  size_vector.push_back(features.size());
  for (const auto& x : features) {
    size_vector.push_back(static_cast<int>(x.size()));
    total_bytes += sizeof(float) * x.size();
  }
  size_vector.push_back(static_cast<int>(normalized_throughputs.size()));
  total_bytes += sizeof(float) * normalized_throughputs.size();
  size_vector.push_back(static_cast<int>(task_ids.size()));
  total_bytes += sizeof(int) * task_ids.size();

  ICHECK_EQ(size_vector.size(), size_vector_size);

  // allocate memory
  out_data->reserve(total_bytes);
  char* ptr = out_data->data();

  // serialize size_vector
  memmove(ptr, reinterpret_cast<char*>(size_vector.data()), size_vector.size() * sizeof(int));
  ptr += size_vector.size() * sizeof(int);

  // serialize features
  for (auto& x : features) {
    memmove(ptr, x.data(), sizeof(float) * x.size());
    ptr += sizeof(float) * x.size();
    x.clear();
  }

  // serialize normalized_throughputs
  memmove(ptr, reinterpret_cast<char*>(normalized_throughputs.data()),
          normalized_throughputs.size() * sizeof(int));
  ptr += normalized_throughputs.size() * sizeof(int);

  // serialize task_ids
  memmove(ptr, reinterpret_cast<char*>(task_ids.data()), task_ids.size() * sizeof(int));
  ptr += task_ids.size() * sizeof(int);

  ICHECK_EQ(ptr - out_data->data(), total_bytes);

  return TVMByteArray{out_data->data(), total_bytes};
}

TVM_REGISTER_GLOBAL("auto_scheduler.GetPerStoreFeaturesFromFile")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      std::string filename = args[0];
      int max_lines = args[1];
      int max_n_bufs = args[2];

      std::vector<std::vector<float>> features;
      std::vector<float> normalized_throughputs;
      std::vector<int> task_ids;

      GetPerStoreFeaturesFromFile(filename, max_lines, max_n_bufs, &features,
                                  &normalized_throughputs, &task_ids);

      std::vector<char> byte_data;
      *ret = SerializeFeatures(std::move(features), std::move(normalized_throughputs),
                               std::move(task_ids), &byte_data);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.GetPerStoreFeaturesFromMeasurePairs")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      Array<MeasureInput> inputs = args[0];
      Array<MeasureResult> results = args[1];
      int skip_first_n_feature_extraction = args[2];
      int max_n_bufs = args[3];

      std::vector<std::vector<float>> features;
      std::vector<float> normalized_throughputs;
      std::vector<int> task_ids;

      GetPerStoreFeaturesFromMeasurePairs(inputs, results, skip_first_n_feature_extraction,
                                          max_n_bufs, &features, &normalized_throughputs,
                                          &task_ids);

      std::vector<char> byte_data;
      *ret = SerializeFeatures(std::move(features), std::move(normalized_throughputs),
                               std::move(task_ids), &byte_data);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.GetPerStoreFeaturesFromStates")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      Array<State> states = args[0];
      SearchTask task = args[1];
      int max_n_bufs = args[2];

      std::vector<std::vector<float>> features;
      std::vector<float> normalized_throughputs;
      std::vector<int> task_ids;

      GetPerStoreFeaturesFromStates(states, task, 0, max_n_bufs, &features);

      std::vector<char> byte_data;
      *ret = SerializeFeatures(std::move(features), std::move(normalized_throughputs),
                               std::move(task_ids), &byte_data);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.GetPerStoreFeatureNames")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      int max_n_bufs = args[0];
      std::vector<std::string> names;

      GetPerStoreFeatureName(max_n_bufs, &names);

      Array<String> arr;
      for (const auto& x : names) {
        arr.push_back(x);
      }
      *ret = arr;
    });

TVM_REGISTER_GLOBAL("auto_scheduler.FeaturesFromPrimFunc")
    .set_body_typed([](const PrimFunc& func, int cache_line_size, int max_n_bufs, bool log_scale) {
      std::vector<float> vec;
      GetPerStoreFeature(func, cache_line_size, max_n_bufs, &vec, log_scale);
      int64_t num_feature_rows = vec[0];  // first element is number of rows
      int64_t row_length = 0;
      if (num_feature_rows != 0) {
        row_length = (vec.size() - 1) / num_feature_rows;
      }
      auto ary =
          runtime::NDArray::Empty({num_feature_rows, row_length}, {kDLFloat, 32, 1}, {kDLCPU, 0});
      // NDArray is row major by default
      ary.CopyFromBytes(vec.data() + 1, sizeof(float) * num_feature_rows * row_length);
      return ary;
    });

}  // namespace auto_scheduler
}  // namespace tvm
