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
 * \file swizzle.h
 * \brief Define swizzled layout
 *
 */

#ifndef TVM_TL_LAYOUT_SWIZZLE_H_
#define TVM_TL_LAYOUT_SWIZZLE_H_

#include "layout.h"

namespace tvm {
namespace tl {

/*!
 * \brief Swizzle pattern
 */
class SwizzlePattern {
 public:
  SwizzlePattern() = default;
  SwizzlePattern(int bits, int base, int shift);
  PrimExpr swizzle(PrimExpr expr) const;
  int Bits() const { return bits_; }
  int Base() const { return base_; }
  int Shift() const { return shift_; }
  bool operator==(const SwizzlePattern& other) const;

 private:
  int bits_;
  int base_;
  int shift_;
};

/*!
 * \brief Layout with swizzle
 */
class SwizzledLayoutNode : public LayoutNode {
 public:
  SwizzledLayoutNode() = default;
  SwizzledLayoutNode(Array<PrimExpr> input_size, Array<PrimExpr> forward_index,
                     SwizzlePattern pattern);

  Array<PrimExpr> Forward(const Array<PrimExpr>& vars) const final;
  Layout Inverse() const final;
  void DebugOutput() const final;

  static constexpr const char* _type_key = "tl.SwizzledLayout";
  bool SEqualReduce(const SwizzledLayoutNode* other, SEqualReducer equal) const;
  void VisitAttrs(tvm::AttrVisitor* v);
  TVM_DECLARE_FINAL_OBJECT_INFO(SwizzledLayoutNode, LayoutNode);

 private:
  SwizzlePattern pattern_;
};

/*!
 * \brief SwizzledLayout reference class.
 */
class SwizzledLayout : public Layout {
 public:
  TVM_DLL SwizzledLayout(Array<IterVar> forward_var, Array<PrimExpr> forward_index,
                         SwizzlePattern pattern);
  TVM_DLL SwizzledLayout(Array<PrimExpr> input_size, Array<PrimExpr> forward_index,
                         SwizzlePattern pattern);

  TVM_DEFINE_OBJECT_REF_METHODS(SwizzledLayout, Layout, SwizzledLayoutNode);
};

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_LAYOUT_SWIZZLE_H_