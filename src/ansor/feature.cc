/*!
 *  Copyright (c) 2020 by Contributors
 */

#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/analysis.h>
#include <tvm/arith/analyzer.h>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>
#include "measure.h"
#include "serialization.h"
#include "utils.h"
// #include "../arithmetic/compute_expr.h"

namespace tvm {
/* Import the function from build_module.cc */
extern void GetBinds(const Array<te::Tensor>& args,
                     bool compact,
                     const std::unordered_map<te::Tensor, te::Buffer>& binds,
                     Map<te::Tensor, te::Buffer>* out_binds,
                     Array<ObjectRef>* out_arg_list,
                     const BuildConfig& config);
}  // namespace tvm


namespace tvm {
namespace ansor {

using namespace tvm::tir;
using arith::ConstIntBound;
using arith::Analyzer;

static const int ARITH_INTENSITY_CURVE_SAMPLE_N = 10;

// Annotation position encoding
enum AnnotationPosType {
  kPosNone, kPosInnerSpatial, kPosMiddleSpatial, kPosOuterSpatial,
  kPosInnerReduce, kPosMiddleReduce, kPosOuterReduce, kPosMixed
};

// Buffer access type
enum BufferAccessType {
  kRead, kWrite, kReadWrite, kUnknownRW
};

// Accesses to a buffer
struct BufferAccess {
  BufferAccessType acc_type{kUnknownRW};
  std::vector<std::vector<PrimExpr> > indices;
};

// Data reuse type
enum ReuseType {
  kLoopMultipleRead, kSerialMultipleReadWrite, kNoReuse
};

// Feature for an access of a buffer
struct BufferAccessFeature {
  std::string tensor_name;
  BufferAccessType acc_type;
  float bytes;
  float unique_bytes;
  float lines;
  float unique_lines;
  ReuseType reuse_type;
  float reuse_dis_iter;    // reuse distance in iterator number
  float reuse_dis_bytes;   // reuse distance in total touched bytes
  float reuse_ct;          // reuse times
  float bytes_d_reuse_ct;
  float unique_bytes_d_reuse_ct;
  float lines_d_reuse_ct;
  float unique_lines_d_reuse_ct;
  float stride;
};

// Feature set of a statement
struct FeatureSet {
  // compute feature
  float float_mad;
  float float_addsub;
  float float_mul;
  float float_divmod;
  float float_cmp;
  float float_math_func;
  float float_other_func;
  float int_mad;
  float int_addsub;
  float int_mul;
  float int_divmod;
  float int_cmp;
  float int_math_func;
  float int_other_func;
  float bool_op;
  float select_op;
  float vec_num;   // The number of vectorized iterators
  float vec_prod;  // The product of the lengths of vectorized iterators
  float vec_len;   // The length of the innermost vectorized iterator
  AnnotationPosType vec_type;
  float unroll_num;   // The number of unrolled iterators
  float unroll_prod;  // The product of the lengths of vectorized iterators
  float unroll_len;   // The length of the innermost unrolled iterator
  AnnotationPosType unroll_type;
  float parallel_num;   // The number of paralleled iterators
  float parallel_prod;  // The product of the lengths of paralleled iterators
  float parallel_len;   // The length of the innermost paralleled iterators
  AnnotationPosType parallel_type;
  float is_gpu;
  float blockIdx_x_len;
  float blockIdx_y_len;
  float blockIdx_z_len;
  float threadIdx_x_len;
  float threadIdx_y_len;
  float threadIdx_z_len;
  float vthread_len;

  float arith_intensity_curve[ARITH_INTENSITY_CURVE_SAMPLE_N];

  // buffer access feature (per buffer)
  std::vector<BufferAccessFeature> access_feas;

  // allocation feature
  float alloc_size;
  float alloc_prod;
  float alloc_outer_prod;
  float alloc_inner_prod;

  // overall feature
  float outer_prod;
  float num_loops;
  float auto_unroll_max_step;
};

// Return whether a var is in an expr
bool VarInExpr(const Var& var, const PrimExpr& expr) {
  bool find = false;

  PostOrderVisit(expr, [&find, &var](const ObjectRef &node) {
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
AnnotationPosType GetAnnotationPosEncoding(
    const Var& var, const Array<PrimExpr>& spatial_args,
    const Array<IterVar>& axis, const Array<IterVar>& reduce_axis) {
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
    // If not find in spatial args, then it is a reduce iteartor.
    // Use name to match
    for (size_t i = 0; i < reduce_axis.size(); ++i) {
      if (var->name_hint.find(reduce_axis[i]->var->name_hint) != std::string::npos) {
        find_i = i;
        find_ct++;
      }
    }
    if (find_ct >= 1) {
      if (find_i == 0) {
        return kPosInnerReduce;
      } else if (find_i == reduce_axis.size() - 1) {
        return kPosOuterReduce;
      } else {
        return kPosMiddleReduce;
      }
    } else {
      // If the axis is not found in both spatial args and reduce axis,
      // then this stage must compute_at somewhere under this aixs and this axis is simplified out
      // We assume it is an outer spatial
      return kPosOuterSpatial;
    }
  } else if (find_ct == 1) {
    if (find_i == spatial_args.size() - 1) {
      return kPosInnerSpatial;
    } else if (find_i == 0) {
      return kPosOuterSpatial;
    } else {
      return kPosMiddleSpatial;
    }
  } else {
    return kPosMixed;
  }
}

// Count math ops in an expr
class MathOpCounter : public StmtExprVisitor {
 public:
#define VisitBinary(Type, float_ct, int_ct) \
  void VisitExpr_(const Type* op) final {   \
    if (op->a.dtype().is_float()) {          \
      float_ct++;                           \
    } else {                                \
      int_ct++;                             \
    }                                       \
    StmtExprVisitor::VisitExpr_(op);        \
  }                                         \

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

  void VisitExpr_(const AndNode* op) final { bool_op++; StmtExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const OrNode* op)  final { bool_op++; StmtExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const NotNode* op) final { bool_op++; StmtExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const SelectNode* op) final { select_op++; StmtExprVisitor::VisitExpr_(op); }

  // TODO(...): CallNode with type CallNode::Halide has been modified to BufferLoadNode
  void VisitExpr_(const CallNode* op) final {
    if (op->call_type == CallNode::CallType::PureIntrinsic) {
      if (op->dtype.is_float()) {
        float_math_func++;
      } else {
        int_math_func++;
      }
    } else if (op->call_type != CallNode::CallType::Halide) {
       if (op->dtype.is_float()) {
        float_other_func++;
      } else {
        int_other_func++;
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  // todo(lmzheng): detect mad
  size_t float_mad{0}, float_addsub{0}, float_mul{0}, float_divmod{0},
         float_cmp{0}, float_math_func{0}, float_other_func{0};
  size_t int_mad{0}, int_addsub{0}, int_mul{0}, int_divmod{0},
         int_cmp{0}, int_math_func{0}, int_other_func{0};
  size_t bool_op{0}, select_op{0};
};


// Extract all buffer accesses in an expr
class BufferAccessExtractor : public StmtExprVisitor {
 public:
  void ExtractReads(const PrimExpr& expr) {
    this->VisitExpr(expr);
  }

  void InsertAccess(const te::Tensor& ten, BufferAccessType acc_type,
      const Array<PrimExpr>& indices) {
    BufferAccess& acc = buf_accesses[ten];
    acc.acc_type = acc_type;
    acc.indices.push_back(std::vector<PrimExpr>(indices.begin(), indices.end()));
  }

  // TODO(...): CallNode with type CallNode::Halide has been modified to BufferLoadNode
  void VisitExpr_(const CallNode *op) final {
    if (op->call_type == CallNode::CallType::Halide) {
      te::Tensor ten = Downcast<te::Operation>(op->func).output(op->value_index);
      BufferAccess& acc = buf_accesses[ten];
      switch (acc.acc_type) {
        case kRead:
          break;
        case kWrite:
          acc.acc_type = kReadWrite; break;
        case kReadWrite:
          break;
        case kUnknownRW:
        default:
          acc.acc_type = kRead; break;
      }

      if (acc.acc_type != kReadWrite) {
        // If a buffer is both read and written, in the tvm DSL, it must be a update,
        // so the indices should be the same. Then we can skip appending indices for it.
        // Otherwise we do the following.
        buf_accesses[ten].indices.push_back(
            std::vector<PrimExpr>(op->args.begin(), op->args.end()));
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  std::unordered_map<te::Tensor, BufferAccess> buf_accesses;
};

// Compute coefficient for an loop iterator in an expression
// Note: we use a approximation strategy to find coefficient.
// Hopefully, it is faster than DetectLinearEquation and can handle more cases (non-linear)
class CoefficientExtractor : public StmtExprVisitor {
 public:
  void VisitExpr_(const MulNode *node) final {
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

  void VisitExpr_(const AddNode *node) final {
    StmtExprVisitor::VisitExpr_(node);
    if (visited_var) {
      if (!visited_mul) {
        visited_add = true;
        stride = 1;
      }
    }
  }

  void VisitExpr_(const VarNode *node) final {
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
int64_t ComputeStride(const std::vector<std::vector<PrimExpr> >& indices,
                      const std::vector<int>& shape,
                      const VarNode* stride_var) {
  int64_t min_stride = std::numeric_limits<int64_t>::max();
  bool find = false;
  CoefficientExtractor extractor;

  for (const auto &index : indices) {
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
void ComputeRegion(
    const std::vector<std::vector<PrimExpr> > &indices,
    arith::Analyzer* ana,
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
    const te::Tensor& t,
    const std::vector<std::vector<PrimExpr> >& indices,
    const std::vector<const ForNode*>& for_loop_stack,
    const std::unordered_map<const ForNode*, std::unordered_map<te::Tensor, \
       std::vector<std::tuple<BufferAccessType, int64_t, int> > > >& for_touch_regions) {
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

    int64_t extent = GetIntImm(for_loop_stack[i]->extent);
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
      return std::make_tuple(kLoopMultipleRead, reuse_dis_iter, reuse_dis_bytes, extent);
    }

    const std::unordered_map<te::Tensor, std::vector<std::tuple<BufferAccessType, int64_t, int> > >&
        tensor_map = for_touch_regions.at(cur_for);

    int serial_reuse = static_cast<int>(tensor_map.at(t).size()) - 1;
    if (serial_reuse > 0) {
      int64_t extent = GetIntImm(cur_for->extent);

      // Have SerialMultipleReadWrite reuse
      reuse_dis_iter = std::numeric_limits<float>::max();
      for (const auto& acc_info : tensor_map.at(t)) {
        reuse_dis_iter = std::min(reuse_dis_iter, static_cast<float>(std::get<1>(acc_info)));
      }

      reuse_dis_bytes = 0.0f;
      for (const auto& iter : for_touch_regions.at(cur_for)) {
        for (const auto& access : iter.second) {
          reuse_dis_bytes += std::get<1>(access) * std::get<2>(access);
        }
      }

      return std::make_tuple(kSerialMultipleReadWrite,
          reuse_dis_iter / extent, reuse_dis_bytes / extent, serial_reuse);
    }
  }

  return std::make_tuple(kNoReuse, 0, 0, 0);
}

// Extract features for every Provide statement
class PerStmtFeatureExtractor : public StmtExprVisitor {
 public:
  explicit PerStmtFeatureExtractor(int cache_line_size) :
      cache_line_size_(cache_line_size) {}

  void VisitStmt_(const AttrStmtNode* node) final {
    if (node->attr_key == tir::attr::thread_extent ||
        node->attr_key == tir::attr::virtual_thread) {
      const Var& var = node->node.as<IterVarNode>()->var;
      int extent = GetIntImm(node->value);

      int* plen = nullptr;

      const std::string& name = var.get()->name_hint;
      if (node->attr_key == tir::attr::thread_extent) {
        if (name == "blockIdx.x") {
          plen = &blockIdx_x_len;
        } else if (name == "blockIdx.y") {
          plen = &blockIdx_y_len;
        } else if (name == "blockIdx.z") {
          plen = &blockIdx_z_len;
        } else if (name == "threadIdx.x") {
          plen = &threadIdx_x_len;
        } else if (name == "threadIdx.y") {
          plen = &threadIdx_y_len;
        } else if (name == "threadIdx.z") {
          plen = &threadIdx_z_len;
        } else {
          LOG(FATAL) << "invalid thread itervar " + name;
        }
      } else {
        plen = &vthread_len;
      }

      int extent_before = *plen;
      if (node->attr_key == tir::attr::thread_extent) {
        *plen = extent;
      } else {
        *plen *= extent;
      }

      is_gpu = true;

      // make a fake for node for blockIdx.x or threadIdx.x
      Stmt fake_for_node = ForNode::make(var, 0, extent, ForType::Parallel,
              DeviceAPI::None, node->body);

      outer_loop_prod *= extent;
      for_loop_stack.push_back(fake_for_node.as<ForNode>());
      StmtExprVisitor::VisitStmt_(node);
      for_loop_stack.pop_back();
      outer_loop_prod /= extent;

      *plen = extent_before;
    } else if (node->attr_key == "pragma_auto_unroll_max_step") {
      int value = GetIntImm(node->value);

      int16_t old_value = cur_auto_unroll_max_step;
      cur_auto_unroll_max_step = value;
      StmtExprVisitor::VisitStmt_(node);
      cur_auto_unroll_max_step = old_value;
    } else {
      StmtExprVisitor::VisitStmt_(node);
    }
  }

  void VisitStmt_(const ForNode* node) final {
    int64_t loop_extent = GetIntImm(node->extent);

    if (node->for_type == ForType::Vectorized) {
      vec_for_stack.push_back(node);
    } else if (node->for_type == ForType::Unrolled) {
      unroll_for_stack.push_back(node);
    } else if (node->for_type == ForType::Parallel) {
      parallel_for_stack.push_back(node);
    }

    outer_loop_prod *= loop_extent;
    for_loop_stack.push_back(node);
    StmtExprVisitor::VisitStmt_(node);
    for_loop_stack.pop_back();
    outer_loop_prod /= loop_extent;

    if (node->for_type == ForType::Vectorized) {
      vec_for_stack.pop_back();
    } else if (node->for_type == ForType::Unrolled) {
      unroll_for_stack.pop_back();
    } else if (node->for_type == ForType::Parallel) {
      parallel_for_stack.pop_back();
    }
  }

  // TODO(...): ProvideNode is deprecated, move to BufferStoreNode
  void VisitStmt_(const ProvideNode* node) final {
    te::Operation op = Downcast<te::Operation>(node->func);
    te::Tensor ten = op.output(node->value_index);
    const te::ComputeOpNode* pcompute = op.as<te::ComputeOpNode>();

    FeatureSet &fea = op_features[ten];

    // compute feature
    MathOpCounter mathops;
    mathops(node->value);
    fea.float_mad        = outer_loop_prod * mathops.float_mad;
    fea.float_addsub     = outer_loop_prod * mathops.float_addsub;
    fea.float_mul        = outer_loop_prod * mathops.float_mul;
    fea.float_divmod     = outer_loop_prod * mathops.float_divmod;
    fea.float_cmp        = outer_loop_prod * mathops.float_cmp;
    fea.float_math_func  = outer_loop_prod * mathops.float_math_func;
    fea.float_other_func = outer_loop_prod * mathops.float_other_func;
    fea.int_mad          = outer_loop_prod * mathops.int_mad;
    fea.int_addsub       = outer_loop_prod * mathops.int_addsub;
    fea.int_mul          = outer_loop_prod * mathops.int_mul;
    fea.int_divmod       = outer_loop_prod * mathops.int_divmod;
    fea.int_math_func    = outer_loop_prod * mathops.int_math_func;
    fea.int_cmp          = outer_loop_prod * mathops.int_cmp;
    fea.int_other_func   = outer_loop_prod * mathops.int_other_func;
    fea.bool_op          = outer_loop_prod * mathops.bool_op;
    fea.select_op        = outer_loop_prod * mathops.select_op;

    fea.outer_prod = outer_loop_prod;
    fea.num_loops = for_loop_stack.size();
    fea.auto_unroll_max_step = cur_auto_unroll_max_step;
    fea.vec_len = fea.unroll_len = fea.parallel_len = 0.0f;
    fea.vec_type = fea.unroll_type = fea.parallel_type = kPosNone;

    fea.vec_num = vec_for_stack.size();
    if (!vec_for_stack.empty()) {
      fea.vec_len = GetIntImm(vec_for_stack.back()->extent);
      fea.vec_prod = 1.0;
      for (const ForNode* pfor : vec_for_stack) {
        fea.vec_prod *= GetIntImm(pfor->extent);
      }
      fea.vec_type = GetAnnotationPosEncoding(vec_for_stack.back()->loop_var,
          node->args, pcompute->axis, pcompute->reduce_axis);
    }

    fea.unroll_num = unroll_for_stack.size();
    if (!unroll_for_stack.empty()) {
      fea.unroll_len = GetIntImm(unroll_for_stack.back()->extent);
      fea.unroll_prod = 1.0;
      for (const ForNode* pfor : unroll_for_stack) {
        fea.unroll_prod *= GetIntImm(pfor->extent);
      }
      fea.unroll_type = GetAnnotationPosEncoding(unroll_for_stack.back()->loop_var,
          node->args, pcompute->axis, pcompute->reduce_axis);
    }

    fea.parallel_num = parallel_for_stack.size();
    if (!parallel_for_stack.empty()) {
      fea.parallel_len = GetIntImm(parallel_for_stack.back()->extent);
      fea.parallel_prod = 1.0;
      for (const ForNode* pfor : parallel_for_stack) {
        fea.parallel_prod *= GetIntImm(pfor->extent);
      }
      fea.parallel_type = GetAnnotationPosEncoding(parallel_for_stack.back()->loop_var,
          node->args, pcompute->axis, pcompute->reduce_axis);
    }

    // GPU threads
    fea.is_gpu = is_gpu;
    fea.blockIdx_x_len = blockIdx_x_len;
    fea.blockIdx_y_len = blockIdx_y_len;
    fea.blockIdx_z_len = blockIdx_z_len;
    fea.threadIdx_x_len = threadIdx_x_len;
    fea.threadIdx_y_len = threadIdx_y_len;
    fea.threadIdx_z_len = threadIdx_z_len;
    fea.vthread_len = vthread_len;

    // Extract all buffer access
    std::vector<BufferAccessFeature> acc_feas;
    BufferAccessExtractor buf_extractor;
    buf_extractor.InsertAccess(ten, kWrite, node->args);
    buf_extractor.ExtractReads(node->value);

    // Compute touched region for all outer loops
    Analyzer ana;
    for (auto x : for_loop_stack) {
      ana.Bind(x->loop_var, Range::make_by_min_extent(x->min, 1));
    }

    std::vector<float> mem_bytes_list;
    std::vector<float> compute_ops_list;

    mem_bytes_list.reserve(for_loop_stack.size());
    compute_ops_list.reserve(for_loop_stack.size());

    int cur_compute_ops = mathops.float_mad + mathops.float_addsub + mathops.float_mul +
                          mathops.float_divmod + mathops.float_cmp +
                          mathops.float_math_func + mathops.float_other_func;

    std::vector<int> tmp_region;
    for (int i = static_cast<int>(for_loop_stack.size()) - 1; i >= 0; i--) {
      const ForNode* p_for = for_loop_stack[i];

      ana.Bind(p_for->loop_var,
               Range::make_by_min_extent(for_loop_stack[i]->min, for_loop_stack[i]->extent));

      // Note, here we do overwrite.
      // So if there are multiple Provides, the last one will overwrite the first few.
      // e.g. The update part in gemm will overwrite the init part.
      std::unordered_map<te::Tensor, std::vector<std::tuple<BufferAccessType, int64_t, int> > >&
          tensor_regions_map = for_touch_regions[p_for];

      int64_t mem_bytes = 0;
      for (const auto &x : buf_extractor.buf_accesses) {
        const te::Tensor& t = x.first;
        const BufferAccess& acc = x.second;

        ComputeRegion(acc.indices, &ana, &tmp_region);
        int64_t touched_size = ElementProduct(tmp_region);
        tensor_regions_map[t].push_back(std::make_tuple(acc.acc_type,
                    touched_size, t->dtype.bytes()));
        mem_bytes += touched_size * t->dtype.bytes();
      }

      mem_bytes_list.push_back(std::log2(mem_bytes));
      cur_compute_ops *= GetIntImm(for_loop_stack[i]->extent);
      compute_ops_list.push_back(std::log2(cur_compute_ops));
    }

    // Compute arithmetic intensity curve (y axis : arithmetic intensity, x axis : flops).
    // We use piecewise linear interpolation to fit this curve.
    int pt = 0;
    if (cur_compute_ops <= 0 || compute_ops_list.empty()) {
      std::fill(fea.arith_intensity_curve,
                fea.arith_intensity_curve + ARITH_INTENSITY_CURVE_SAMPLE_N, 0.0);
    } else {
      for (size_t i = 0; i < ARITH_INTENSITY_CURVE_SAMPLE_N; ++i) {
        float cur_compute_ops = compute_ops_list.back() * (i+1) / ARITH_INTENSITY_CURVE_SAMPLE_N;
        while (compute_ops_list[pt] < cur_compute_ops - 1e-4) {
          pt++;
        }
        CHECK_LT(pt, compute_ops_list.size());

        float value;
        if (pt == 0) {
          value = compute_ops_list[pt] / mem_bytes_list[pt];
        } else {
          float base = compute_ops_list[pt-1] / mem_bytes_list[pt-1];
          float slope = (compute_ops_list[pt] / mem_bytes_list[pt] -
              compute_ops_list[pt-1] / mem_bytes_list[pt-1]) /
              (compute_ops_list[pt] - compute_ops_list[pt-1]);
          value = base + slope * (cur_compute_ops - compute_ops_list[pt-1]);
        }
        fea.arith_intensity_curve[i] = value;
      }
    }

    // Compute buffer access feature
    for (const auto &x : buf_extractor.buf_accesses) {
      const te::Tensor& t = x.first;
      const BufferAccess& acc = x.second;

      std::vector<int> int_shape;
      for (const auto& dim : t->shape) {
        int_shape.push_back(GetIntImm(dim));
      }

      size_t ele_bytes = t->dtype.bytes();

      // calculate bytes
      float bytes = outer_loop_prod * ele_bytes;
      float unique_bytes;

      // calculate cache lines
      int64_t stride;
      float lines;
      float unique_lines;

      if (for_loop_stack.empty()) {
        unique_bytes = ele_bytes;
        stride = 0;
        lines = 1.0f;
        unique_lines = 1.0f;
      } else {
        unique_bytes = std::get<1>(for_touch_regions[for_loop_stack.front()][t].front())
            * ele_bytes;

        stride = 0;
        int64_t reduce_ratio = 1;

        int i;
        for (i = static_cast<int>(for_loop_stack.size()) - 1; i >= 0; i--) {
          stride = ComputeStride(acc.indices, int_shape, for_loop_stack[i]->loop_var.get());
          if (stride != 0) {
            break;
          }
          reduce_ratio *= GetIntImm(for_loop_stack.back()->extent);
        }

        lines = outer_loop_prod / reduce_ratio *
            std::min(1.0f, 1.0f * stride * ele_bytes / cache_line_size_);
        lines = std::max(lines, 1.0f);

        // convert `stride` back to the stride of the innermost iterator
        stride = (i == static_cast<int>(for_loop_stack.size()) - 1 ? stride : 0);

        float n_continuous = ele_bytes;
        for (int i = static_cast<int>(tmp_region.size()) - 1; i >= 0; i--) {
          if (tmp_region[i] == int_shape[i]) {
            n_continuous *= tmp_region[i];
            break;
          }
        }
        unique_lines = unique_bytes / std::min(n_continuous,
                                               static_cast<float>(cache_line_size_));
        unique_lines = std::max(unique_lines, 1.0f);
      }

      ReuseType reuse_type;
      float reuse_dis_iter, reuse_dis_bytes, reuse_ct;
      std::tie(reuse_type, reuse_dis_iter, reuse_dis_bytes, reuse_ct) =
          ComputeReuse(t, acc.indices, for_loop_stack, for_touch_regions);

      acc_feas.emplace_back();
      BufferAccessFeature& acc_fea = acc_feas.back();

      acc_fea.tensor_name = t->op->func_name();
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
        acc_fea.unique_lines_d_reuse_ct = unique_lines* 2;
      }
    }

    fea.access_feas = acc_feas;
  }

  // TODO(...): RealizeNode is deprecated, move to BufferRealizeNode
  void VisitStmt_(const RealizeNode *node) final {
    StmtExprVisitor::VisitStmt_(node);

    te::Operation op = Downcast<te::Operation>(node->func);
    te::Tensor ten = op.output(node->value_index);

    FeatureSet& fea = op_features[ten];

    float allocation_size = 1.0f;
    for (const auto& x : node->bounds) {
      allocation_size *= GetIntImm(x->extent);
    }
    // allocation feature
    fea.alloc_size = allocation_size * ten->dtype.bytes();
    fea.alloc_prod = allocation_size * outer_loop_prod;
    fea.alloc_outer_prod = outer_loop_prod;
    fea.alloc_inner_prod = fea.outer_prod / outer_loop_prod;
  }

  float outer_loop_prod = 1.0f;

  std::vector<const ForNode*> for_loop_stack;
  std::vector<const ForNode*> parallel_for_stack;
  std::vector<const ForNode*> vec_for_stack;
  std::vector<const ForNode*> unroll_for_stack;

  bool is_gpu;
  int blockIdx_x_len{1};
  int blockIdx_y_len{1};
  int blockIdx_z_len{1};
  int threadIdx_x_len{1};
  int threadIdx_y_len{1};
  int threadIdx_z_len{1};
  int vthread_len{1};
  int16_t cur_auto_unroll_max_step{0};

  std::unordered_map<te::Tensor, FeatureSet> op_features;

  // for a loop, for all its touched tensors, for all different accesses to the tensors,
  // its (access type, number of touched elements, number of bytes of single element)
  std::unordered_map<const ForNode*, std::unordered_map<te::Tensor, \
       std::vector<std::tuple<BufferAccessType, int64_t, int> > > > for_touch_regions;

 private:
  const int cache_line_size_ = 64;
};

// shifted log to incorporate the property that slog(0) = 0
inline float slog(float x) {
  return x < 0 ? -std::log2(-x+1) : std::log2(x+1);
}

// Get features for all ir::Provide statements in a TVM program.
// So we call it `PerStmt` feature
void GetPerStmtFeature(const Stmt& stmt,
                       int cache_line_size,
                       int max_n_bufs,
                       std::vector<float>* ret) {
  LOG(WARNING) << "RealizeNode & ProvideNode deprecated, "
               << "need to fix the implementation of PerStmtFeatureExtractor.";
  PerStmtFeatureExtractor extractor(cache_line_size);
  extractor(stmt);

  ret->push_back(extractor.op_features.size());

  for (const auto& x : extractor.op_features) {
    const FeatureSet& fea_set = x.second;

    /***** compute feature *****/
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
    for (int i = 0; i <= kPosMixed; i++) {
      ret->push_back(i == fea_set.vec_type);
    }

    ret->push_back(slog(fea_set.unroll_num));
    ret->push_back(slog(fea_set.unroll_prod));
    ret->push_back(slog(fea_set.unroll_len));
    for (int i = 0; i <= kPosMixed; i++) {
      ret->push_back(i == fea_set.unroll_type);
    }

    ret->push_back(slog(fea_set.parallel_num));
    ret->push_back(slog(fea_set.parallel_prod));
    ret->push_back(slog(fea_set.parallel_len));
    for (int i = 0; i <= kPosMixed; i++) {
      ret->push_back(i == fea_set.parallel_type);
    }

    ret->push_back(fea_set.is_gpu);
    ret->push_back(slog(fea_set.blockIdx_x_len));
    ret->push_back(slog(fea_set.blockIdx_y_len));
    ret->push_back(slog(fea_set.blockIdx_z_len));
    ret->push_back(slog(fea_set.threadIdx_x_len));
    ret->push_back(slog(fea_set.threadIdx_y_len));
    ret->push_back(slog(fea_set.threadIdx_z_len));
    ret->push_back(slog(fea_set.vthread_len));

    for (size_t i = 0; i < ARITH_INTENSITY_CURVE_SAMPLE_N; ++i) {
      ret->push_back(fea_set.arith_intensity_curve[i]);
    }

    /***** access feature *****/
    // sort according to pair (lines, bytes)
    std::vector<std::pair<float, float> > buf_order_key;
    for (const auto& acc_fea : fea_set.access_feas) {
      buf_order_key.emplace_back(acc_fea.lines, acc_fea.bytes);
    }
    std::vector<int> buf_order(buf_order_key.size());
    std::iota(buf_order.begin(), buf_order.end(), 0);

    auto cmp = [&buf_order_key](int l, int r) {
      return buf_order_key[l].first > buf_order_key[r].first
          || (buf_order_key[l].first == buf_order_key[r].first
              && buf_order_key[l].second > buf_order_key[r].second);
    };
    std::sort(buf_order.begin(), buf_order.end(), cmp);
    int n_bufs = std::min(max_n_bufs, static_cast<int>(buf_order.size()));
    buf_order.resize(n_bufs);

    for (int idx : buf_order) {
      const auto& acc_fea = fea_set.access_feas[idx];
      for (int j = 0; j <= kReadWrite; ++j) {
        ret->push_back(j == acc_fea.acc_type);
      }
      ret->push_back(slog(acc_fea.bytes));
      ret->push_back(slog(acc_fea.unique_bytes));
      ret->push_back(slog(acc_fea.lines));
      ret->push_back(slog(acc_fea.unique_lines));
      for (int j = 0; j <= kNoReuse; ++j) {
        ret->push_back(acc_fea.reuse_type == j);
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
      for (int j = 0; j <= kReadWrite; ++j) {  // 3
        ret->push_back(0.0f);
      }
      ret->push_back(0.0f); ret->push_back(0.0f); ret->push_back(0.0f); ret->push_back(0.0f);
      for (int j = 0; j <= kNoReuse; ++j) {   // 3
        ret->push_back(0.0f);
      }
      ret->push_back(0.0f); ret->push_back(0.0f); ret->push_back(0.0f); ret->push_back(0.0f);
      ret->push_back(0.0f); ret->push_back(0.0f); ret->push_back(0.0f); ret->push_back(0.0f);
    }

    /***** allocation feature *****/
    ret->push_back(slog(fea_set.alloc_size));
    ret->push_back(slog(fea_set.alloc_prod));
    ret->push_back(slog(fea_set.alloc_outer_prod));
    ret->push_back(slog(fea_set.alloc_inner_prod));

    /***** overall feature *****/
    ret->push_back(slog(fea_set.outer_prod));
    ret->push_back(slog(fea_set.num_loops));
    ret->push_back(slog(fea_set.auto_unroll_max_step));
  }
}


/* \brief Get the name of every element in the feature vector. Use this for debug and inspection */
void GetPerStmtFeatureName(int max_n_bufs, std::vector<std::string> *ret) {
  /***** compute feature *****/
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
  for (size_t i = 0; i < ARITH_INTENSITY_CURVE_SAMPLE_N; ++i) {
    ret->push_back(("arith_intensity_curve_" + std::to_string(i)));
  }
  // section total: 55 + ARITH_INTENSITY_CURVE_SAMPLE_N = 65

  /***** access feature *****/
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

  /***** allocation feature *****/
  ret->push_back(("alloc_size"));
  ret->push_back(("alloc_prod"));
  ret->push_back(("alloc_outer_prod"));
  ret->push_back(("alloc_inner_prod"));
  // section total : 4

  /***** overall feature *****/
  ret->push_back(("outer_prod"));
  ret->push_back(("num_loops"));
  ret->push_back(("auto_unroll_max_step"));
  // section total : 2
}

void GetPerStmtFeaturesWorkerFunc(const SearchTask& task, const State& state,
        int max_n_bufs, std::vector<float>* feature, std::atomic<int>* error_ct) {
  te::Schedule sch;
  Array<te::Tensor> tensors;
  Map<IterVar, Range> bounds;
  GlobalVar g("main");

  std::tie(sch, tensors) = task->compute_dag.ApplySteps(state->transform_steps);
  sch = sch.normalize();
  bounds = te::InferBound(sch);

  try {
    auto stmt = te::ScheduleOps(sch, bounds, false);
    Map<te::Tensor, te::Buffer> out_binds; Array<ObjectRef> out_arg_list;
    bool compact = te::VerifyCompactBuffer(stmt);
    GetBinds(tensors, compact, std::unordered_map<te::Tensor, te::Buffer>(),
              &out_binds, &out_arg_list, BuildConfig::Create());
    tir::PrimFunc f = te::SchedulePostProcToPrimFunc(out_arg_list,
        std::move(stmt), out_binds);
    f = WithAttr(std::move(f), "global_symbol", runtime::String("main"));
    auto mod = IRModule(Map<GlobalVar, BaseFunc>({{g, f}}));
    auto pass_list = Array<tvm::transform::Pass>();
    if (task->target->device_type == kDLGPU) {
      pass_list.push_back(tir::transform::InjectPrefetch());
      pass_list.push_back(tir::transform::StorageFlatten(64));
      pass_list.push_back(tir::transform::Simplify());
      pass_list.push_back(tir::transform::VectorizeLoop());
      pass_list.push_back(tir::transform::InjectVirtualThread());
      pass_list.push_back(tir::transform::StorageRewrite());
      pass_list.push_back(tir::transform::Simplify());
      tvm::Map<std::string, tvm::PrimExpr> gpu_params {
        {"max_shared_memory_per_block",
            task->hardware_params->max_shared_memory_per_block},
        {"max_local_memory_per_block",
            task->hardware_params->max_registers_per_block},
        {"max_threads_per_block",
            task->hardware_params->max_threads_per_block},
        {"max_vector_bytes",
            task->hardware_params->vector_unit_bytes}
      };
      pass_list.push_back(tir::transform::VerifyGPUCode(gpu_params));
      const auto& optimize = tir::transform::Sequential(pass_list);
      optimize(mod);
    }
    pass_list.clear();
    pass_list.push_back(tir::transform::Simplify());
    const auto& optimize = tir::transform::Sequential(pass_list);
    mod = optimize(std::move(mod));
    const auto& it = mod->functions.find(g);
    CHECK(it != mod->functions.end());
    const auto& prim_func = (*it).second.as<PrimFuncNode>();
    GetPerStmtFeature(prim_func->body,
                      task->hardware_params->cache_line_bytes,
                      max_n_bufs, feature);
  } catch (dmlc::Error &e) {
    (*error_ct)++;
  }
}

void GetPerStmtFeaturesFromStates(const Array<State>& states,
                                  const SearchTask& task,
                                  int max_n_bufs,
                                  int skip_first_n_feature_extraction,
                                  std::vector<std::vector<float> >* features) {
  // extract features
  features->assign(states.size(), std::vector<float>());

  std::atomic<int> error_ct(0);

  ThreadPool& pool = ThreadPool::Global();
  pool.BeginBatch(static_cast<int>(states.size()) - skip_first_n_feature_extraction);
  for (size_t i = skip_first_n_feature_extraction; i < states.size(); ++i) {
    pool.Enqueue(GetPerStmtFeaturesWorkerFunc, task, states[i],
        max_n_bufs, &(*features)[i], &error_ct);
  }
  pool.WaitBatch();

  if (error_ct > 0) {
    std::cerr << "Encountered " << error_ct
              << " errors during feature extraction. Ignored." << std::endl;
  }
}


void GetPerStmtFeaturesFromStates(const Array<State>& states,
                                  const std::vector<SearchTask>& tasks,
                                  int max_n_bufs,
                                  int skip_first_n_feature_extraction,
                                  std::vector<std::vector<float> >* features) {
  // extract features
  features->assign(states.size(), std::vector<float>());

  std::atomic<int> error_ct(0);

  ThreadPool& pool = ThreadPool::Global();
  pool.BeginBatch(static_cast<int>(states.size()) - skip_first_n_feature_extraction);
  for (size_t i = skip_first_n_feature_extraction; i < states.size(); ++i) {
    pool.Enqueue(GetPerStmtFeaturesWorkerFunc, tasks[i], states[i],
        max_n_bufs, &(*features)[i], &error_ct);
  }
  pool.WaitBatch();

  if (error_ct > 0) {
    std::cerr << "Encountered " << error_ct
    << " errors during feature extraction. Ignored." << std::endl;
  }
}

void GetPerStmtFeaturesFromFile(const std::string& filename,
                                int n_lines,
                                int max_n_bufs,
                                std::vector<std::vector<float> >* features,
                                std::vector<float>* normalized_throughputs,
                                std::vector<int>* task_ids) {
  Array<State> states;
  // ArrayNode* pstates = states.CopyOnWrite();
  std::vector<SearchTask> tasks;

  normalized_throughputs->clear();
  task_ids->clear();

  // (workload_key, target) -> (search_task, task_id)
  std::unordered_map<std::pair<std::string, std::string>, std::pair<SearchTask, size_t>> task_cache;
  // task_id -> min_cost
  std::vector<float> min_costs;

  // read from file
  LogReader reader = LogReaderNode::make(filename);
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
      task = SearchTaskNode::make(ComputeDAGNode::make_by_workload_key(workload_key),
                                  workload_key,
                                  cur_inp->task->target,
                                  cur_inp->task->target_host,
                                  cur_inp->task->hardware_params);
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
    // pstates->data.push_back(cur_inp->state);
    states.push_back(cur_inp->state);
    normalized_throughputs->push_back(cost);

    if (n_lines > 0 && static_cast<int>(states.size()) >= n_lines) {
      break;
    }
  }

  for (size_t i = 0; i < normalized_throughputs->size(); ++i) {
    (*normalized_throughputs)[i] = min_costs[(*task_ids)[i]] / (*normalized_throughputs)[i];
  }

  GetPerStmtFeaturesFromStates(states, tasks, max_n_bufs, 0, features);
}

void GetPerStmtFeaturesFromMeasurePairs(const Array<MeasureInput>& inputs,
                                        const Array<MeasureResult>& results,
                                        int max_n_bufs,
                                        int skip_first_n_feature_extraction,
                                        std::vector<std::vector<float> >* features,
                                        std::vector<float>* normalized_throughputs,
                                        std::vector<int>* task_ids) {
  Array<State> states;
  // ArrayNode* pstates = states.CopyOnWrite();
  std::vector<SearchTask> tasks;

  normalized_throughputs->clear();
  task_ids->clear();

  // (workload_key, target) -> (search_task, task_id)
  std::unordered_map<std::pair<std::string, std::string>, std::pair<SearchTask, size_t>> task_cache;
  // task_id -> min_cost
  std::vector<float> min_costs;

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
      if (inputs[i]->task->compute_dag.defined()) {   // the measure input is complete
          task = inputs[i]->task;
      } else {  // the measure input is incomplete
          // rebuild task for incomplete measure pairs read from file
          task = SearchTaskNode::make(ComputeDAGNode::make_by_workload_key(workload_key),
                                      workload_key,
                                      inputs[i]->task->target,
                                      inputs[i]->task->target_host,
                                      inputs[i]->task->hardware_params);
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
    // pstates->data.push_back(inputs[i]->state);
    states.push_back(inputs[i]->state);
    normalized_throughputs->push_back(cost);
  }

  for (size_t i = 0; i < normalized_throughputs->size(); ++i) {
    (*normalized_throughputs)[i] = min_costs[(*task_ids)[i]] / (*normalized_throughputs)[i];
  }

  GetPerStmtFeaturesFromStates(states, tasks, max_n_bufs,
          skip_first_n_feature_extraction, features);
}

}   // namespace ansor
}   // namespace tvm
