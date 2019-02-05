/*!
 *  Copyright (c) 2019 by Contributors
 * \file tvm/layout.h
 * \brief Layout expression.
 *
 *  This file is adapted from its nnvm counterpart and will keep involving
 *  to the new layout system
 *
 *  The layout is composed of upper cases, lower cases and numbers,
 *  where upper case indicates a primal axis and
 *  the corresponding lower case with factor size indicates the subordinate axis.
 *  For example, NCHW16c can describe a 5-D tensor of
 *  [batch_size, channel, height, width, channel_block].
 *  Here subordinate axis channel_block=16 is the factor size of the primal axis C (channel).
 */
#ifndef TVM_LAYOUT_H_
#define TVM_LAYOUT_H_

#include <tvm/base.h>
#include <tvm/expr.h>

#include <string>
#include <sstream>
#include <vector>
#include <utility>
#include <algorithm>

#include "ir_operator.h"

namespace tvm {

class LayoutAxis {
 public:
  // single axis definitions
  static const LayoutAxis A;
  static const LayoutAxis B;
  static const LayoutAxis C;
  static const LayoutAxis D;
  static const LayoutAxis E;
  static const LayoutAxis F;
  static const LayoutAxis G;
  static const LayoutAxis H;
  static const LayoutAxis I;
  static const LayoutAxis J;
  static const LayoutAxis K;
  static const LayoutAxis L;
  static const LayoutAxis M;
  static const LayoutAxis N;
  static const LayoutAxis O;
  static const LayoutAxis P;
  static const LayoutAxis Q;
  static const LayoutAxis R;
  static const LayoutAxis S;
  static const LayoutAxis T;
  static const LayoutAxis U;
  static const LayoutAxis V;
  static const LayoutAxis W;
  static const LayoutAxis X;
  static const LayoutAxis Y;
  static const LayoutAxis Z;
  static const LayoutAxis a;
  static const LayoutAxis b;
  static const LayoutAxis c;
  static const LayoutAxis d;
  static const LayoutAxis e;
  static const LayoutAxis f;
  static const LayoutAxis g;
  static const LayoutAxis h;
  static const LayoutAxis i;
  static const LayoutAxis j;
  static const LayoutAxis k;
  static const LayoutAxis l;
  static const LayoutAxis m;
  static const LayoutAxis n;
  static const LayoutAxis o;
  static const LayoutAxis p;
  static const LayoutAxis q;
  static const LayoutAxis r;
  static const LayoutAxis s;
  static const LayoutAxis t;
  static const LayoutAxis u;
  static const LayoutAxis v;
  static const LayoutAxis w;
  static const LayoutAxis x;
  static const LayoutAxis y;
  static const LayoutAxis z;

  static const LayoutAxis& Get(const char name) {
    switch (name) {
      case 'A': return A;
      case 'B': return B;
      case 'C': return C;
      case 'D': return D;
      case 'E': return E;
      case 'F': return F;
      case 'G': return G;
      case 'H': return H;
      case 'I': return I;
      case 'J': return J;
      case 'K': return K;
      case 'L': return L;
      case 'M': return M;
      case 'N': return N;
      case 'O': return O;
      case 'P': return P;
      case 'Q': return Q;
      case 'R': return R;
      case 'S': return S;
      case 'T': return T;
      case 'U': return U;
      case 'V': return V;
      case 'W': return W;
      case 'X': return X;
      case 'Y': return Y;
      case 'Z': return Z;
      case 'a': return a;
      case 'b': return b;
      case 'c': return c;
      case 'd': return d;
      case 'e': return e;
      case 'f': return f;
      case 'g': return g;
      case 'h': return h;
      case 'i': return i;
      case 'j': return j;
      case 'k': return k;
      case 'l': return l;
      case 'm': return m;
      case 'n': return n;
      case 'o': return o;
      case 'p': return p;
      case 'q': return q;
      case 'r': return r;
      case 's': return s;
      case 't': return t;
      case 'u': return u;
      case 'v': return v;
      case 'w': return w;
      case 'x': return x;
      case 'y': return y;
      case 'z': return z;
      default: LOG(FATAL) << "Invalid layout axis name " << name;
    }
    // suppress return-type warning.
    return A;
  }

  // Get the singleton LayoutAxis using itvar->var->name_hint
  inline static const LayoutAxis& Get(const IterVar& itvar) {
    const std::string axis = itvar->var.get()->name_hint;
    CHECK_EQ(axis.size(), 1) << "Invalid layout axis " << axis;
    return LayoutAxis::Get(axis[0]);
  }

  // Get the singleton LayoutAxis using name[0] (size of name must be 1).
  inline static const LayoutAxis& make(const std::string& name) {
    CHECK_EQ(name.length(), 1) << "Invalid axis " << name;
    return LayoutAxis::Get(name[0]);
  }

  inline bool IsPrimal() const { return name_ >= 'A' && name_ <= 'Z'; }
  inline std::string name() const { return name_str_; }

  // if current axis is primal, switch the axis to its subordinate one,
  // else switch to the primal.
  inline const LayoutAxis& to_dual() const {
    if (name_ >= 'A' && name_ <= 'Z') {
      return LayoutAxis::Get(name_ - 'A' + 'a');
    } else {
      return LayoutAxis::Get(name_ - 'a' + 'A');
    }
  }

  // return the primal axis. If it is already primal, return itself.
  inline const LayoutAxis& to_primal() const {
    return IsPrimal() ? *this : to_dual();
  }

  // return the subordinate axis. If it is already subordinate, return itself.
  inline const LayoutAxis& to_subordinate() const {
    return IsPrimal() ? to_dual() : *this;
  }

  inline bool operator==(const LayoutAxis& rhs) const {
    return name_ == rhs.name_;
  }

  friend std::ostream& operator<<(std::ostream& os, const LayoutAxis& l) {
    os << l.name();
    return os;
  }

 private:
  LayoutAxis(const LayoutAxis&);
  LayoutAxis& operator=(const LayoutAxis&);
  explicit LayoutAxis(const char name) : name_(name), name_str_(std::string(1, name)) {}

  const char name_;
  const std::string name_str_;
};

class Layout;
class LayoutNode : public Node {
 public:
  std::string name;
  Array<IterVar> axes;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("axes", &axes);
  }

  TVM_DLL static Layout make(const std::string& layout);

  static constexpr const char* _type_key = "Layout";
  TVM_DECLARE_NODE_TYPE_INFO(LayoutNode, Node);
};

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
   * \param len The length of the sub-layout.
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
  Layout Split(const LayoutAxis &axis, size_t target_pos, int64_t factor) const;


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
      if (axes[i]->var.get()->name_hint == axis.name()) return static_cast<int32_t>(i);
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
  int64_t FactorOf(const LayoutAxis& axis) const;

  /*!
   * \brief Whether the layout contains an axis.
   * \param axis axis to be checked.
   * \return Whether the layout contains the axis.
   */
  bool Contains(const LayoutAxis& axis) const {
    if (!defined()) return false;
    for (const IterVar var : operator->()->axes) {
      if (var->var.get()->name_hint == axis.name()) {
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
class BijectiveLayoutNode : public Node {
 public:
  // expression of each location, on how original location can be mapped
  // to the store location, example
  // [i0 / 16, i1, i0 % 16]
  Array<Expr> forward_rule;
  Array<Expr> backward_rule;

  Layout src_layout;
  Layout dst_layout;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("src_layout", &src_layout);
    v->Visit("dst_layout", &dst_layout);
  }

  static constexpr const char* _type_key = "BijectiveLayout";
  TVM_DECLARE_NODE_TYPE_INFO(BijectiveLayoutNode, Node);

  TVM_DLL static BijectiveLayout make(const Layout& src_layout,
                                      const Layout& dst_layout);
};

class BijectiveLayout : public NodeRef {
 public:
  BijectiveLayout() = default;
  explicit BijectiveLayout(NodePtr<Node> n) : NodeRef(n) {}

  // Given the shape of the source layout, infer the target shape.
  TVM_DLL Array<Expr> ForwardShape(const Array<Expr>& shape) const;
  // Given the target shape, recover the source shape.
  TVM_DLL Array<Expr> BackwardShape(const Array<Expr>& shape) const;
  // Given the indices of the source layout, infer the target index.
  TVM_DLL Array<Expr> ForwardIndex(const Array<Expr>& index) const;
  // Given the target indices, recover the source index.
  TVM_DLL Array<Expr> BackwardIndex(const Array<Expr>& store_index) const;

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

#endif  // TVM_LAYOUT_H_
