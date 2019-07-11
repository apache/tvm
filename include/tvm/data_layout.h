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
 * \file tvm/data_layout.h
 * \brief Layout expression to describe the data organization of a tensor.
 *  And BijectiveLayout to mapping two data layouts between each other.
 */
#ifndef TVM_DATA_LAYOUT_H_
#define TVM_DATA_LAYOUT_H_

#include <tvm/base.h>
#include <tvm/expr.h>

#include <string>
#include <sstream>
#include <vector>
#include <utility>
#include <algorithm>

#include "expr_operator.h"

namespace tvm {

class LayoutAxis {
 public:
  static const LayoutAxis& Get(const char name);

  // Get the singleton LayoutAxis using itvar->var->name_hint
  static const LayoutAxis& Get(const IterVar& itvar);

  // Get the singleton LayoutAxis using name[0] (size of name must be 1).
  static const LayoutAxis& make(const std::string& name);

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
  const LayoutAxis& ToPrimal() const {
    return IsPrimal() ? *this : ToDual();
  }

  // return the subordinate axis. If it is already subordinate, return itself.
  const LayoutAxis& ToSubordinate() const {
    return IsPrimal() ? ToDual() : *this;
  }

  inline bool operator==(const LayoutAxis& rhs) const {
    return name_ == rhs.name_;
  }

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

class Layout;
// Internal node container Buffer
class LayoutNode : public Node {
 public:
  /*! \brief string representation of layout, "" for scalar. */
  std::string name;
  /*! \brief specify each axis of the layout,
   *   in which the variable name is the name of the axis.
   *   The IterVar's extent indicates the size of the axis,
   *   it is a variable for a primal axis, but a constant for a subordinate axis.
   *   Empty for scalar's layout.
   */
  Array<IterVar> axes;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("axes", &axes);
  }

  TVM_DLL static Layout make(const std::string& layout);

  static constexpr const char* _type_key = "Layout";
  TVM_DECLARE_NODE_TYPE_INFO(LayoutNode, Node);
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
class Layout : public NodeRef {
 public:
  explicit Layout(NodePtr<Node> n) : NodeRef(n) {}

  /*! \brief default constructor */
  Layout() = default;

  explicit Layout(const Array<IterVar>& axes);

  /*! \brief construct from a string */
  Layout(const char* name) : Layout(std::string(name)) {} // NOLINT(*)

  /*!
   * \brief construct from a string.
   * \param name input in layout convention:
   *        upper case indicates a dimension and
   *        the corresponding lower case with factor size
   *        indicates the split dimension.
   *        return undefined layout if "__undef__" is passed.
   */
  Layout(const std::string& name); // NOLINT(*)

  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  const LayoutNode* operator->() const {
    return static_cast<const LayoutNode*>(node_.get());
  }

  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  LayoutNode* operator->() {
    return static_cast<LayoutNode*>(node_.get());
  }

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
  Layout Split(const LayoutAxis &axis, size_t target_pos, int32_t factor) const;


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
    for (const IterVar var : operator->()->axes) {
      if (var->var->name_hint == axis.name()) {
        return true;
      }
    }
    return false;
  }

  const LayoutAxis& operator[](int32_t i) const {
    CHECK(defined()) << "Try to access axis from an undefined layout.";
    int32_t index = i < 0 ? static_cast<int32_t>(ndim() + i) : i;
    CHECK(index >= 0 && static_cast<size_t>(index) < ndim()) << "Invalid index " << i;
    const IterVar axis = operator->()->axes[index];
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
  inline bool Equals(const Layout &rhs) const {
    return name() == rhs.name();
  }

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

  using ContainerType = LayoutNode;
};

class BijectiveLayout;
// Internal node container BijectiveLayout
class BijectiveLayoutNode : public Node {
 public:
  /*! \brief Describes how source axes can be mapped to the destination axes,
   *   e.g., [i0 / 16, i1, i0 % 16] can describe NC -> NC16n
   */
  Array<Expr> forward_rule;
  /*! \brief Describes how destination axes can be mapped to the source axes */
  Array<Expr> backward_rule;

  /*! \brief The source layout */
  Layout src_layout;
  /*! \brief The destination layout */
  Layout dst_layout;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("src_layout", &src_layout);
    v->Visit("dst_layout", &dst_layout);
    v->Visit("forward_rule", &forward_rule);
    v->Visit("backward_rule", &backward_rule);
  }

  static constexpr const char* _type_key = "BijectiveLayout";
  TVM_DECLARE_NODE_TYPE_INFO(BijectiveLayoutNode, Node);

  TVM_DLL static BijectiveLayout make(const Layout& src_layout,
                                      const Layout& dst_layout);
};

/*! \brief Bijective function mapping for data layout transformation.
 *   Given two Layout, BijectiveLayout build and store the mapping rules,
 *   provides API to transform N-dimention tensor from the source indices (i0, i1, …, im)
 *   to the destination indices (j0, j1, … jm).
 */
class BijectiveLayout : public NodeRef {
 public:
  BijectiveLayout() = default;
  explicit BijectiveLayout(NodePtr<Node> n) : NodeRef(n) {}

  // Given the source shape, infer the destination shape.
  TVM_DLL Array<Expr> ForwardShape(const Array<Expr>& shape) const;
  // Given the destination shape, recover the source shape.
  TVM_DLL Array<Expr> BackwardShape(const Array<Expr>& dst_shape) const;
  // Given the destination indices, infer the destination indices.
  TVM_DLL Array<Expr> ForwardIndex(const Array<Expr>& index) const;
  // Given the destination indices, recover the source indices.
  TVM_DLL Array<Expr> BackwardIndex(const Array<Expr>& dst_index) const;

  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const BijectiveLayoutNode* operator->() const;

  /*! \brief specify container node */
  using ContainerType = BijectiveLayoutNode;
};

inline const BijectiveLayoutNode* BijectiveLayout::operator->() const {
  return static_cast<const BijectiveLayoutNode*>(node_.get());
}

}  // namespace tvm

#endif  // TVM_DATA_LAYOUT_H_
