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
 * \file tvm/tir/data_layout.h
 * \brief Layout expression to describe the data organization of a tensor.
 *  And BijectiveLayout to mapping two data layouts between each other.
 */
#ifndef TVM_TIR_DATA_LAYOUT_H_
#define TVM_TIR_DATA_LAYOUT_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tir {

class Layout;

class LayoutAxis {
 public:
  static const LayoutAxis& Get(const char name);

  // Get the singleton LayoutAxis using itvar->var->name_hint
  static const LayoutAxis& Get(const tir::IterVar& itvar);

  // Get the singleton LayoutAxis using name[0] (size of name must be 1).
  static const LayoutAxis& Get(const std::string& name);

  inline bool IsPrimal() const { return name_ >= 'A' && name_ <= 'Z'; }
  inline std::string name() const { return std::string(1, name_); }

  // if current axis is primal, switch the axis to its subordinate one,
  // else switch to the primal.
  inline const LayoutAxis& ToDual() const {
    if (name_ >= 'A' && name_ <= 'Z') {
      return LayoutAxis::Get(name_ - 'A' + 'a');
    } else {
      return LayoutAxis::Get(name_ - 'a' + 'A');
    }
  }

  // return the primal axis. If it is already primal, return itself.
  const LayoutAxis& ToPrimal() const { return IsPrimal() ? *this : ToDual(); }

  // return the subordinate axis. If it is already subordinate, return itself.
  const LayoutAxis& ToSubordinate() const { return IsPrimal() ? ToDual() : *this; }

  inline bool operator==(const LayoutAxis& rhs) const { return name_ == rhs.name_; }

  friend std::ostream& operator<<(std::ostream& os, const LayoutAxis& l) {
    os << l.name();
    return os;
  }

 private:
  static const LayoutAxis UPPER_CASE[];
  static const LayoutAxis LOWER_CASE[];
  LayoutAxis(const LayoutAxis&);
  LayoutAxis& operator=(const LayoutAxis&);
  explicit LayoutAxis(const char name) : name_(name) {}

  const char name_;
};

/*!
 * \brief Layout is to describe how data is organized within an N-dimention tensor.
 *  It is composed of upper cases, lower cases and numbers,
 *  where upper case indicates a primal axis and
 *  the corresponding lower case with factor size indicates the subordinate axis.
 *  For example, NCHW16c can describe a 5-D tensor of
 *  [batch_size, channel, height, width, channel_block].
 *  Here subordinate axis channel_block=16 is the factor size of the primal axis C (channel).
 *  Layout for scalar is defined, while both its name and axes have size 0.
 */
class LayoutNode : public Object {
 public:
  /*! \brief string representation of layout, "" for scalar. */
  String name;
  /*! \brief specify each axis of the layout,
   *   in which the variable name is the name of the axis.
   *   The IterVar's extent indicates the size of the axis,
   *   it is a variable for a primal axis, but a constant for a subordinate axis.
   *   Empty for scalar's layout.
   */
  Array<tir::IterVar> axes;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("axes", &axes);
  }

  static constexpr const char* _type_key = "tir.Layout";
  TVM_DECLARE_FINAL_OBJECT_INFO(LayoutNode, Object);
};

/*!
 * \brief Managed reference to LayoutNode
 * \sa LayoutNode
 */
class Layout : public ObjectRef {
 public:
  explicit Layout(const Array<tir::IterVar>& axes);

  /*! \brief construct from a string */
  Layout(const tvm::String& name) : Layout(name.operator std::string()) {}  // NOLINT(*)

  /*! \brief construct from a string */
  Layout(const char* name) : Layout(std::string(name)) {}  // NOLINT(*)

  /*!
   * \brief construct from a string.
   * \param name input in layout convention:
   *        upper case indicates a dimension and
   *        the corresponding lower case with factor size
   *        indicates the split dimension.
   *        return undefined layout if "__undef__" is passed.
   * \param dtype The dtype of generated axes vars in the returned layout.
   *        It is required to be integer type.
   */
  TVM_DLL Layout(const std::string& name, DataType dtype = DataType::Int(32));  // NOLINT(*)

  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  LayoutNode* operator->() { return static_cast<LayoutNode*>(get_mutable()); }

  /*!
   * \brief Return an undefined layout.
   * \return a (global) undefined layout.
   */
  static const Layout& Undef() {
    static Layout undef;
    return undef;
  }

  /*!
   * \brief Returns a sub-layout which is the portion of the object
   *        that starts at dimension \p pos and spans \p len dimensions
   *        (or until the end of the layout, whichever comes first).
   * \param pos The start position.
   * \param len The length of the sub-layout. if 0, return layout of scalar
   * \return A newly constructed Layout object.
   */
  Layout SubLayout(size_t pos, size_t len) const;

  /*!
   * \brief Split \p axis by \p size and put the sub-axis to position \p target_pos.
   * \param axis The source axis to be split. It must be a primal-axis;
   * \param target_pos The target position of the newly split subordinate-axis.
   * \param factor size of the sub-dimension.
   * \return A newly constructed Layout object.
   */
  Layout Split(const LayoutAxis& axis, size_t target_pos, int32_t factor) const;

  /*! \return number of dimensions */
  inline size_t ndim() const {
    if (!defined()) return 0;
    return operator->()->axes.size();
  }

  /*! \return number of super dimensions */
  inline size_t ndim_primal() const {
    if (!defined()) return 0;
    size_t ct = 0;
    for (auto x : operator->()->axes) {
      if (LayoutAxis::Get(x).IsPrimal()) {
        ct++;
      }
    }
    return ct;
  }

  /*!
   * \brief Returns a new layout where the dims have been expanded to match the primal dimensions.
   * \param dst_layout The dst layout to which current layout has to be expanded.
   * \return The expanded Layout.
   */
  inline Layout ExpandPrimal(const Layout& dst_layout) {
    Layout new_src_layout;
    // 1) Find the axis which are missing in the current layout. Make them the prefix.
    std::string new_src_layout_str = "";
    for (auto dst_axis : dst_layout->axes) {
      if (LayoutAxis::Get(dst_axis).IsPrimal()) {
        if (!this->Contains(LayoutAxis::Get(dst_axis))) {
          new_src_layout_str += dst_axis->var->name_hint;
        }
      }
    }
    // 2) Now, add the primal axis of the current layout.
    new_src_layout_str += this->name();
    new_src_layout = Layout(new_src_layout_str);
    return new_src_layout;
  }

  /*!
   * \brief return the index of the input axis.
   *        If it is not found in the layout or the layout is undefined,
   *        return -1.
   * \param axis the input axis.
   * \return the index or -1 if not found.
   */
  inline int32_t IndexOf(const LayoutAxis& axis) const {
    if (!this->defined()) return -1;
    const auto axes = operator->()->axes;
    for (size_t i = 0; i < axes.size(); ++i) {
      if (axes[i]->var->name_hint == axis.name()) return static_cast<int32_t>(i);
    }
    return -1;
  }

  /*!
   * \brief Get the factor size of the subordinate axis.
   * \param axis the input primal-axis or subordinate-axis.
   * \return the size of the subordinate-axis of \p axis (if \p axis is a primal-axis),
   *         or the size of \p axis itself (if \p axis is a subordinate-axis).
   *         Return -1 if \p axis is not in the layout the layout is undefined.
   */
  int32_t FactorOf(const LayoutAxis& axis) const;

  /*!
   * \brief Whether the layout contains an axis.
   * \param axis axis to be checked.
   * \return Whether the layout contains the axis.
   */
  bool Contains(const LayoutAxis& axis) const {
    if (!defined()) return false;
    for (const tir::IterVar var : operator->()->axes) {
      if (var->var->name_hint == axis.name()) {
        return true;
      }
    }
    return false;
  }

  const LayoutAxis& operator[](int32_t i) const {
    ICHECK(defined()) << "Try to access axis from an undefined layout.";
    int32_t index = i < 0 ? static_cast<int32_t>(ndim() + i) : i;
    ICHECK(index >= 0 && static_cast<size_t>(index) < ndim()) << "Invalid index " << i;
    const tir::IterVar axis = operator->()->axes[index];
    return LayoutAxis::Get(axis);
  }

  /*! \return the string description of the layout */
  inline std::string name() const {
    if (!defined()) return "__undef__";
    return operator->()->name;
  }

  /*!
   * \brief Whether the two layouts are equal.
   * \param rhs Another layout.
   * \return whether the two layouts are equal.
   */
  inline bool Equals(const Layout& rhs) const { return name() == rhs.name(); }

  /*!
   * \brief allow output string of layout to ostream
   * \param os the output stream
   * \param l the layout
   * \return the ostream
   */
  friend std::ostream& operator<<(std::ostream& os, const Layout& l) {
    os << l.name();
    return os;
  }

  TVM_DEFINE_OBJECT_REF_METHODS(Layout, ObjectRef, LayoutNode);
};

// Internal node container BijectiveLayout
class BijectiveLayoutNode : public Object {
 public:
  /*! \brief Describes how source axes can be mapped to the destination axes,
   *   e.g., [i0 / 16, i1, i0 % 16] can describe NC -> NC16n
   */
  Array<PrimExpr> index_forward_rule;
  /*! \brief Describes how destination axes can be mapped to the source axes */
  Array<PrimExpr> index_backward_rule;
  /*! \brief Describes how source shapes can be mapped to the destination shapes */
  Array<PrimExpr> shape_forward_rule;
  /*! \brief Describes how destination shapes can be mapped to the source shapes */
  Array<PrimExpr> shape_backward_rule;

  /*! \brief The source layout */
  Layout src_layout;
  /*! \brief The destination layout */
  Layout dst_layout;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("src_layout", &src_layout);
    v->Visit("dst_layout", &dst_layout);
    v->Visit("index_forward_rule", &index_forward_rule);
    v->Visit("index_backward_rule", &index_backward_rule);
    v->Visit("shape_forward_rule", &shape_forward_rule);
    v->Visit("shape_backward_rule", &shape_backward_rule);
  }

  static constexpr const char* _type_key = "tir.BijectiveLayout";
  TVM_DECLARE_FINAL_OBJECT_INFO(BijectiveLayoutNode, Object);
};

/*!
 * \brief Bijective function mapping for data layout transformation.
 *   Given two Layout, BijectiveLayout build and store the mapping rules,
 *   provides API to transform N-dimention tensor from the source indices (i0, i1, .., im)
 *   to the destination indices (j0, j1, .., jm).
 */
class BijectiveLayout : public ObjectRef {
 public:
  /*!
   * \brief The constructor
   * \param src_layout The source layout
   * \param dst_layout The destination layout
   */
  TVM_DLL BijectiveLayout(Layout src_layout, Layout dst_layout);

  // Given the source shape, infer the destination shape.
  TVM_DLL Array<PrimExpr> ForwardShape(const Array<PrimExpr>& shape) const;
  // Given the destination shape, recover the source shape.
  TVM_DLL Array<PrimExpr> BackwardShape(const Array<PrimExpr>& dst_shape) const;
  // Given the destination indices, infer the destination indices.
  TVM_DLL Array<PrimExpr> ForwardIndex(const Array<PrimExpr>& index) const;
  // Given the destination indices, recover the source indices.
  TVM_DLL Array<PrimExpr> BackwardIndex(const Array<PrimExpr>& dst_index) const;

  TVM_DEFINE_OBJECT_REF_METHODS(BijectiveLayout, ObjectRef, BijectiveLayoutNode);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_DATA_LAYOUT_H_
