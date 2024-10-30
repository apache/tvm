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
 * \file tir/analysis/identify_memcpy.cc
 * \brief Check if a loop nest is equivalent to memcpy
 */

#include <tvm/arith/bound.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/stmt.h>

#include <optional>
#include <sstream>
#include <string>
#include <variant>

#include "../../arith/ir_visitor_with_analyzer.h"

namespace tvm {
namespace tir {

std::variant<MemCpyDetails, std::string> IdentifyMemCpyImpl(const For& loop,
                                                            arith::Analyzer* analyzer) {
  Map<Var, arith::IntSet> loop_intervals;
  Map<Var, Range> loop_ranges;
  PrimExpr total_loop_iterations = 1;

  // Walk through the loop nest, stopping at the first loop whose body
  // is not a loop.
  Stmt stmt = loop;
  while (auto* for_node = stmt.as<ForNode>()) {
    loop_ranges.Set(for_node->loop_var, Range::FromMinExtent(for_node->min, for_node->extent));
    loop_intervals.Set(for_node->loop_var,
                       arith::IntSet::FromMinExtent(for_node->min, for_node->extent));
    total_loop_iterations = total_loop_iterations * for_node->extent;

    stmt = for_node->body;
  }

  BufferStore store;
  if (auto opt = stmt.as<BufferStore>()) {
    store = opt.value();
  } else {
    return static_cast<const std::stringstream&>(
               std::stringstream()
               << "Expected innermost loop to have BufferStore body, but instead found " << stmt)
        .str();
  }

  BufferLoad load;
  if (auto opt = store->value.as<BufferLoad>()) {
    load = opt.value();
  } else {
    return static_cast<const std::stringstream&>(
               std::stringstream()
               << "Expected BufferStore's value to be BufferLoad, but instead found "
               << store->value)
        .str();
  }

  // Now, we have a BufferStore whose value is a BufferLoad.  Because
  // non-flat physical indices are target-dependent, only handle cases
  // where the buffer will be flattened to a 1-d physical buffer.
  Array<PrimExpr> flattened_dst = store->buffer.OffsetOf(store->indices);
  Array<PrimExpr> flattened_src = load->buffer.OffsetOf(load->indices);

  if (flattened_dst.size() != 1 || flattened_src.size() != 1) {
    return static_cast<const std::stringstream&>(
               std::stringstream()
               << "Expected flattened dimension of src/dest to be 1, but found"
               << flattened_src.size() << "-d src and " << flattened_dst.size() << "-d dst")
        .str();
  }
  PrimExpr src_index = flattened_src[0];
  PrimExpr dst_index = flattened_dst[0];

  // First check, do the input/output form affine subsets of their
  // respective buffers?
  //
  // For example, should exclude the following, indices are not affine
  //
  // for i in T.serial(16):
  //     B[i] = A[T.abs(i-8)]

  auto src_iter_map = arith::DetectIterMap({src_index}, loop_ranges, Bool(true),
                                           arith::IterMapLevel::Bijective, analyzer);
  if (src_iter_map->errors.size()) {
    return static_cast<const std::stringstream&>(std::stringstream()
                                                 << "arith::DetectIterMap(src) returned "
                                                 << src_iter_map->errors.size() << " errors: ["
                                                 << src_iter_map->errors << "]"
                                                 << " for src_index = " << src_index)
        .str();
  }
  auto dst_iter_map = arith::DetectIterMap({dst_index}, loop_ranges, Bool(true),
                                           arith::IterMapLevel::Bijective, analyzer);
  if (dst_iter_map->errors.size()) {
    return static_cast<const std::stringstream&>(std::stringstream()
                                                 << "arith::DetectIterMap(dst) returned "
                                                 << dst_iter_map->errors.size() << " errors: ["
                                                 << dst_iter_map->errors << "]"
                                                 << " for dst_index = " << dst_index)
        .str();
  }

  // Second check, are those affine subsets contiguous?  If so, then
  // the index expressions will visit every location between the min
  // and the max.  This checks surjectivity over a linear region,
  // which may not be the same as DetectIterMap's check of
  // surjectivity over the affine subset.
  //
  // For example, should exclude the following, doesn't touch all
  // output locations within the output region touched.
  //
  // for i in T.serial(16):
  //     B[2*i] = A[i]
  //
  // Similarly, should exclude the following, doesn't touch all
  // input locations within the input region touched.
  //
  // for i in T.serial(16):
  //     B[i] = A[2*i]
  total_loop_iterations = analyzer->Simplify(total_loop_iterations);
  auto src_interval = analyzer->int_set(src_index, loop_intervals);
  auto dst_interval = analyzer->int_set(dst_index, loop_intervals);

  if (!src_interval.HasLowerBound() || !src_interval.HasUpperBound()) {
    return static_cast<const std::stringstream&>(std::stringstream()
                                                 << "Expected known bounds for src, but found "
                                                 << src_interval << " for expression " << src_index)
        .str();
  }
  if (!dst_interval.HasLowerBound() || !dst_interval.HasUpperBound()) {
    return static_cast<const std::stringstream&>(std::stringstream()
                                                 << "Expected known bounds for dst, but found "
                                                 << dst_interval << " for expression " << dst_index)
        .str();
  }

  {
    PrimExpr must_prove = total_loop_iterations == src_interval.max() - src_interval.min() + 1;
    PrimExpr simplified = analyzer->Simplify(must_prove);
    if (!analyzer->CanProve(simplified)) {
      return static_cast<const std::stringstream&>(
                 std::stringstream()
                 << "Mismatch between loop iterations (" << total_loop_iterations
                 << ") and number of src indices touched (" << src_interval
                 << ".  Equality to prove simplified to " << simplified)
          .str();
    }
  }
  {
    PrimExpr must_prove = total_loop_iterations == dst_interval.max() - dst_interval.min() + 1;
    PrimExpr simplified = analyzer->Simplify(must_prove);
    if (!analyzer->CanProve(simplified)) {
      return static_cast<const std::stringstream&>(
                 std::stringstream()
                 << "Mismatch between loop iterations (" << total_loop_iterations
                 << ") and number of dst indices touched (" << dst_interval
                 << ".  Equality to prove simplified to " << simplified)
          .str();
    }
  }

  // Third check, is there a transformation applied between the input
  // and output iterators?
  //
  // For example, the following would pass all checks so far, but
  // converts between row-major and column-major layouts, and could
  // not be specified as a memcpy.
  //
  // for i,j in T.grid(4,4):
  //     B[i,j] = A[j,i]

  auto src_iter_sum = src_iter_map->indices[0];
  auto dst_iter_sum = dst_iter_map->indices[0];

  if (src_iter_sum->args.size() != dst_iter_sum->args.size()) {
    return static_cast<const std::stringstream&>(
               std::stringstream()
               << "IterMap for src/dst unpacked to different number of IterSplitExpr: "
               << src_iter_sum->args.size() << " for src, " << dst_iter_sum->args.size()
               << " for dst.  "
               << "IterMaps were detected as src = " << src_iter_sum << ", dst = " << dst_iter_sum)
        .str();
  }
  std::vector<arith::IterSplitExpr> src_iter_terms(src_iter_sum->args.begin(),
                                                   src_iter_sum->args.end());
  std::vector<arith::IterSplitExpr> dst_iter_terms(dst_iter_sum->args.begin(),
                                                   dst_iter_sum->args.end());

  auto make_comparison_tuple = [](const arith::IterSplitExpr& expr) {
    auto as_int_or_zero = [](auto& val) -> int64_t {
      if (auto* as_int = val.template as<IntImmNode>()) {
        return as_int->value;
      } else {
        return 0;
      }
    };
    return std::tuple{
        static_cast<bool>(expr->scale.as<IntImmNode>()),        as_int_or_zero(expr->scale),
        static_cast<bool>(expr->extent.as<IntImmNode>()),       as_int_or_zero(expr->lower_factor),
        static_cast<bool>(expr->lower_factor.as<IntImmNode>()), as_int_or_zero(expr->lower_factor),
    };
  };
  auto sorting_function = [&make_comparison_tuple](const arith::IterSplitExpr& lhs,
                                                   const arith::IterSplitExpr& rhs) -> bool {
    return make_comparison_tuple(lhs) < make_comparison_tuple(rhs);
  };
  std::sort(src_iter_terms.begin(), src_iter_terms.end(), sorting_function);
  std::sort(dst_iter_terms.begin(), dst_iter_terms.end(), sorting_function);

  for (size_t i = 0; i < src_iter_terms.size(); i++) {
    const arith::IterSplitExpr& src_term = src_iter_terms[i];
    const arith::IterSplitExpr& dst_term = dst_iter_terms[i];

    if (!analyzer->CanProve(
            arith::NormalizeIterMapToExpr(src_term->source->source == dst_term->source->source))) {
      return static_cast<const std::stringstream&>(
                 std::stringstream()
                 << "Term " << i << " had different source, src_term->source = " << src_term->source
                 << ", dst_term->source = " << dst_term->source)
          .str();
    }
    if (!analyzer->CanProve(src_term->lower_factor == dst_term->lower_factor)) {
      return static_cast<const std::stringstream&>(
                 std::stringstream()
                 << "Term " << i << " had different lower_factor, src_term->lower_factor = "
                 << src_term->lower_factor
                 << ", dst_term->lower_factor = " << dst_term->lower_factor)
          .str();
    }
    if (!analyzer->CanProve(src_term->extent == dst_term->extent)) {
      return static_cast<const std::stringstream&>(
                 std::stringstream()
                 << "Term " << i << " had different extent, src_term->extent = " << src_term->extent
                 << ", dst_term->extent = " << dst_term->extent)
          .str();
    }
    if (!analyzer->CanProve(src_term->scale == dst_term->scale)) {
      return static_cast<const std::stringstream&>(
                 std::stringstream()
                 << "Term " << i << " had different scale, src_term->scale = " << src_term->scale
                 << ", dst_term->scale = " << dst_term->scale)
          .str();
    }
  }

  BufferRegion src_region(load->buffer, arith::DomainTouched(loop, load->buffer, true, true));
  BufferRegion dst_region(store->buffer, arith::DomainTouched(loop, store->buffer, true, true));

  return MemCpyDetails{src_region, dst_region};
}

std::optional<MemCpyDetails> IdentifyMemCpy(const For& loop, arith::Analyzer* analyzer) {
  auto result = IdentifyMemCpyImpl(loop, analyzer);
  if (auto* ptr = std::get_if<MemCpyDetails>(&result)) {
    return *ptr;
  } else {
    return std::nullopt;
  }
}

// Expose the IdentifyMemCpy functionality to Python API for purpose of unit testing.
TVM_REGISTER_GLOBAL("tir.analysis._identify_memcpy").set_body_typed([](const Stmt& stmt) {
  Array<ObjectRef> output;

  struct Visitor : arith::IRVisitorWithAnalyzer {
    explicit Visitor(Array<ObjectRef>* output) : output(output) {}
    Array<ObjectRef>* output;

   private:
    using IRVisitorWithAnalyzer::VisitStmt_;
    void VisitStmt_(const ForNode* op) override {
      For loop = GetRef<For>(op);
      auto result = IdentifyMemCpyImpl(loop, &(Visitor::analyzer_));
      if (auto* ptr = std::get_if<MemCpyDetails>(&result)) {
        output->push_back(Array{ptr->source, ptr->dest});
      } else if (auto* ptr = std::get_if<std::string>(&result)) {
        output->push_back(StringImm(*ptr));
      } else {
        LOG(FATAL) << "Internal error, unhandled std::variant type";
      }

      IRVisitorWithAnalyzer::VisitStmt_(op);
    }
  };

  Visitor visitor(&output);
  visitor(stmt);

  return output;
});

}  // namespace tir
}  // namespace tvm
