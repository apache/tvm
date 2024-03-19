/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file layout/swizzle.cc
 * \brief Define swizzled layout
 *
 */

#include "swizzle.h"

#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <cmath>

namespace tvm {
namespace tl {

SwizzlePattern::SwizzlePattern(int bits, int base, int shift)
    : bits_(bits), base_(base), shift_(shift) {
  ICHECK(bits >= 0);
  ICHECK(base >= 0);
  ICHECK(shift >= 0);
  ICHECK(shift >= bits);
}

PrimExpr SwizzlePattern::swizzle(PrimExpr expr) const {
  int base = (1 << base_);
  int mask = ((1 << bits_) - 1) << shift_;
  PrimExpr high = FloorDiv(expr, base);
  PrimExpr low = FloorMod(expr, base);
  high = bitwise_xor(high, right_shift(bitwise_and(high, mask), shift_));
  return low + high * base;
}

bool SwizzlePattern::operator==(const SwizzlePattern& other) const {
  return std::tie(base_, bits_, shift_) == std::tie(other.base_, other.bits_, other.shift_);
}

SwizzledLayoutNode::SwizzledLayoutNode(Array<PrimExpr> input_size, Array<PrimExpr> forward_index,
                                       SwizzlePattern pattern)
    : pattern_(pattern) {
  input_size_ = input_size;
  arith::Analyzer analyzer;
  UpdateAnalyzer(&analyzer);
  forward_index_ = forward_index.Map([&](const PrimExpr& e) { return analyzer.Simplify(e); });
}

Array<PrimExpr> SwizzledLayoutNode::Forward(const Array<PrimExpr>& vars) const {
  auto expr_list = LayoutNode::Forward(vars);
  auto expr = expr_list.back();
  expr_list.pop_back();
  expr_list.push_back(pattern_.swizzle(expr));
  return expr_list;
}

void SwizzledLayoutNode::DebugOutput() const {
  LayoutNode::DebugOutput();
  std::cout << "Layout Swizzle: " << pattern_.Base() << " " << pattern_.Bits() << " "
            << pattern_.Shift();
}

Layout SwizzledLayoutNode::Inverse() const {
  ICHECK(0) << "Not Implemented.";
  return {};
}

SwizzledLayout::SwizzledLayout(Array<IterVar> forward_var, Array<PrimExpr> forward_index,
                               SwizzlePattern pattern) {
  Map<Var, PrimExpr> vmap;
  Array<PrimExpr> input_size;
  for (size_t i = 0; i < forward_var.size(); i++) {
    vmap.Set(forward_var[i]->var, InputPlaceholder(i));
    CHECK(is_zero(forward_var[i]->dom->min));
    input_size.push_back(forward_var[i]->dom->extent);
  }
  forward_index = forward_index.Map([&](const PrimExpr& e) { return Substitute(e, vmap); });

  auto n = make_object<SwizzledLayoutNode>(input_size, forward_index, pattern);
  data_ = std::move(n);
}

SwizzledLayout::SwizzledLayout(Array<PrimExpr> input_size, Array<PrimExpr> forward_index,
                               SwizzlePattern pattern) {
  auto n = make_object<SwizzledLayoutNode>(input_size, forward_index, pattern);
  data_ = std::move(n);
}

void SwizzledLayoutNode::VisitAttrs(tvm::AttrVisitor* v) { LayoutNode::VisitAttrs(v); }

bool SwizzledLayoutNode::SEqualReduce(const SwizzledLayoutNode* other, SEqualReducer equal) const {
  return equal(this->InputShape(), other->InputShape()) &&
         equal(this->forward_index_, other->forward_index_) && pattern_ == other->pattern_;
}

TVM_REGISTER_NODE_TYPE(SwizzledLayoutNode);

}  // namespace tl
}  // namespace tvm