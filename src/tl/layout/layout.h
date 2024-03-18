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
 * \file Layout.h
 * \brief Define Layout used in MMA and other operations.
 *
 */

#ifndef TVM_TL_LAYOUT_LAYOUT_H_
#define TVM_TL_LAYOUT_LAYOUT_H_

#include <tvm/arith/analyzer.h>

namespace tvm {
namespace tl {

using namespace tir;

class Layout;
class Fragment;

class LayoutNode : public Object {
 public:
  bool Valid() const { return forward_index_.size() != 0; }

  size_t InputDim() const { return forward_var_.size(); }

  size_t OutputDim() const { return forward_index_.size(); }

  Array<PrimExpr> InputShape() const;

  Array<PrimExpr> OutputShape() const;

  virtual Layout Inverse() const;

  Array<PrimExpr> Forward(const Array<PrimExpr>& vars) const;

  PrimExpr GetFlattenedIndice() const;

  int VectorSize() const;

  virtual void DebugOutput() const;

  virtual void UpdateAnalyzer(arith::Analyzer* analyzer) const;

  void VisitAttrs(tvm::AttrVisitor* v);
  static constexpr bool _type_has_method_sequal_reduce = true;
  bool SEqualReduce(const LayoutNode* other, SEqualReducer equal) const;
  static constexpr const char* _type_key = "tl.Layout";
  TVM_DECLARE_BASE_OBJECT_INFO(LayoutNode, Object);

  Array<PrimExpr> forward_index_;
  Array<IterVar> forward_var_;
};

/*!
 * \brief Layout reference class.
 */
class Layout : public ObjectRef {
 public:
  TVM_DLL Layout(Array<IterVar> forward_var, Array<PrimExpr> forward_index);

  TVM_DEFINE_OBJECT_REF_METHODS(Layout, ObjectRef, LayoutNode);
};

class FragmentNode : public LayoutNode {
 public:
  FragmentNode() = default;

  Layout Inverse() const final;

  void UpdateAnalyzer(arith::Analyzer* analyzer) const final;

  PrimExpr ThreadExtent() const;

  PrimExpr ReplicateExtent() const { return thread_replicate_->dom->extent; };

  PrimExpr ForwardThread(const Array<PrimExpr>& vars, const Optional<PrimExpr>& rep_var) const;

  Fragment Repeat(const Array<PrimExpr>& repeats, bool repeat_on_thread,
                  bool lower_dim_first = true) const;

  Fragment Replicate(int repeats) const;

  Fragment DeReplicate() const;

  Fragment CondenseReplicateVar() const;

  void DebugOutput() const final;

  void VisitAttrs(tvm::AttrVisitor* v);
  bool SEqualReduce(const FragmentNode* other, SEqualReducer equal) const;
  static constexpr const char* _type_key = "tl.Fragment";
  TVM_DECLARE_FINAL_OBJECT_INFO(FragmentNode, LayoutNode);

  PrimExpr forward_thread_;
  IterVar thread_replicate_;
};

/*!
 * \brief Fragment reference class.
 */
class Fragment : public Layout {
 public:
  TVM_DLL Fragment(Array<IterVar> forward_var, Array<PrimExpr> forward_index,
                   PrimExpr forward_thread, IterVar thread_replicate);

  TVM_DEFINE_OBJECT_REF_METHODS(Fragment, Layout, FragmentNode);
};

bool FragmentThreadEqual(const Fragment& a, const Fragment& b);

Fragment makeGemmFragmentC(const int block_m, const int block_n, const int warp_m, const int warp_n,
                           const int element_size);
Fragment makeGemmFragmentA(const int block_m, const int block_n, const int block_k,
                           const int warp_m, const int warp_n);
Fragment makeGemmFragmentB(const int block_m, const int block_n, const int block_k,
                           const int warp_m, const int warp_n);
Layout makeGemmABLayout(int stride, int continuous, int element_size, int kfactor);

Fragment makeGemmVoltaFragmentC(const int block_m, const int block_n, const int warp_m,
                                const int warp_n, const int element_size);
Fragment makeGemmVoltaFragmentA(const int block_m, const int block_n, const int block_k,
                                const int warp_m, const int warp_n);
Layout makeGemmVoltaABLayout(int stride, int continuous, bool is_a, int kfactor);

namespace attr {

// LoopAttr, Disable a for loop to infer under InferLevel kFree
constexpr const char* kSkipLayoutInfer = "skip_layout_infer";

// BlockAttr, Containing the layout for all the buffers in the block
constexpr const char* kLayoutMap = "layout_map";
}  // namespace attr

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_LAYOUT_LAYOUT_H_
