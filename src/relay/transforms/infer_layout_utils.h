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
 * \file infer_layout_utils.h
 * \brief Utility functions to alter the layouts of operators or replace primitive operators with
          other expressions. This pass can be used for computing convolution in
          custom layouts or other general weight pre-transformation.
 */

#ifndef TVM_RELAY_TRANSFORMS_INFER_LAYOUT_UTILS_H_
#define TVM_RELAY_TRANSFORMS_INFER_LAYOUT_UTILS_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/tir/data_layout.h>

#include <string>
#include <tuple>
#include <utility>

#include "pattern_utils.h"

namespace tvm {
namespace relay {

/*!
 * \brief Returns a new layout where the subordinate factors are adjusted based on the tensor
 *        shape.
 * \param old_layout The old layout before any transformation.
 * \param old_shape The shape of the original tensor.
 * \return The adjusted Layout.
 */
Layout AdjustSubordinateFactors(const Layout& src_layout, const Layout& old_layout,
                                const Array<tvm::PrimExpr>& old_shape);

bool Isomorphic(const Layout& lhs, const Layout& rhs);

/*!
 * \brief Try transforming `old` in as the smae way as how`ref_old` is transformed to `ref_new`.
 * `old` and `ref_old` are expected to describe two broadcastable tensors. Layout with fewer rank
 * will be expanded. For example,
 * if old = 'NW', ref_old = 'NC', ref_new = 'NC1c', then the result is 'NW1w';
 * if old = 'W', ref_old = 'NC', ref_new = 'NC1c', then the result is 'NW1w'.
 * When `old` and `ref_old` are isomorphic (same structure, only differ in naming), the transform
 * is guaranteed to succeed, in which case the function is simply renaming the axes of `ref_new`
 * to conform to `old`'s naming.
 * \param old The layout to be transformed.
 * \param ref_old The reference layout before transform.
 * \param ref_new The reference layout after transform.
 * \return The transformed layout.
 */
Layout TryTransformLike(const Layout& old, const Layout& ref_old, const Layout& ref_new);

/*
 * \brief An output structure to hold results from FInferCorrectLayout calls.
 * \tparam input_layouts Inferred input layouts.
 * \tparam output_layouts Inferred output layouts.
 * \tparam new_attrs Updated attributes consistent with inferred layouts.
 */
class InferCorrectLayoutOutputNode : public Object {
 public:
  Array<Layout> input_layouts;
  Array<Layout> output_layouts;
  Attrs new_attrs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("input_layouts", &input_layouts);
    v->Visit("output_layouts", &output_layouts);
    v->Visit("new_attrs", &new_attrs);
  }

  TVM_DECLARE_BASE_OBJECT_INFO(InferCorrectLayoutOutputNode, Object);

  static constexpr const char* _type_key = "relay._transform.InferCorrectLayoutOutput";
};

class InferCorrectLayoutOutput : public ObjectRef {
 public:
  InferCorrectLayoutOutput(Array<Layout> input_layouts, Array<Layout> output_layouts,
                           Attrs new_attrs) {
    auto n = make_object<InferCorrectLayoutOutputNode>();
    n->input_layouts = std::move(input_layouts);
    n->output_layouts = std::move(output_layouts);
    n->new_attrs = std::move(new_attrs);
    data_ = n;
  }
  TVM_DEFINE_OBJECT_REF_METHODS(InferCorrectLayoutOutput, ObjectRef, InferCorrectLayoutOutputNode);
};

/*!
 * \brief Infer & correct function of node layout. See \p Layout for layout convention
 * \param attrs The attribute of the node.
 * \param new_in_layouts The layouts of input arguments after alter_op_layout.
 *                       This can be undefined, which means we call this function before alternating
 *                       any operators.
 * \param old_in_layouts The layouts of input arguments before alter_op_layout.
 * \param old_in_types The types of old input arguments.
 * \return infer_layout_output Inferred layouts and updated attributes stored in
 *                             InferCorrectLayoutOutput above.
 */
using FInferCorrectLayout = runtime::TypedPackedFunc<InferCorrectLayoutOutput(
    const Attrs& attrs, const Array<Layout>& new_in_layouts, const Array<Layout>& old_in_layouts,
    const Array<tvm::relay::Type>& old_in_types)>;

inline InferCorrectLayoutOutput ElemwiseArbitraryLayout(
    const Attrs& attrs, const Array<Layout>& new_in_layouts, const Array<Layout>& old_in_layouts,
    const Array<tvm::relay::Type>& old_in_types) {
  Layout ret;

  if (new_in_layouts.defined()) {
    ICHECK_GE(new_in_layouts.size(), 1);
    ret = new_in_layouts[0];
  } else {
    for (size_t i = 0; i < old_in_layouts.size(); ++i) {
      if (old_in_layouts[i].defined()) {
        ret = old_in_layouts[i];
        break;
      }
    }
  }

  return InferCorrectLayoutOutput(Array<Layout>(old_in_layouts.size(), ret), {ret}, attrs);
}

std::pair<Array<Layout>, Array<Layout>> BinaryBroadcastLayoutHelper(
    const Attrs& attrs, const Array<Layout>& new_in_layouts, const Array<Layout>& old_in_layouts,
    const Array<tvm::relay::Type>& old_in_types);

/*! \brief Infer layout for binary broadcast operators */
inline InferCorrectLayoutOutput BinaryBroadcastLayout(const Attrs& attrs,
                                                      const Array<Layout>& new_in_layouts,
                                                      const Array<Layout>& old_in_layouts,
                                                      const Array<tvm::relay::Type>& old_in_types) {
  auto inferred_layout =
      BinaryBroadcastLayoutHelper(attrs, new_in_layouts, old_in_layouts, old_in_types);
  return InferCorrectLayoutOutput(inferred_layout.first, inferred_layout.second, attrs);
}

}  //  namespace relay
}  //  namespace tvm

#endif  // TVM_RELAY_TRANSFORMS_INFER_LAYOUT_UTILS_H_
