/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/layout.h
 * \brief Layout expression.
 *
 *  This file is adapted from its nnvm counterpart and will keep involving
 *  to the new layout system
 *
 *  The layout is composed of upper cases, lower cases and numbers,
 *  where upper case indicates a (primal) axis and
 *  the corresponding lower case with factor size indicates the split (subordinate) axis.
 *  For example, NCHW16c can describe a 5-D tensor of
 *  [batch_size, channel, height, width, channel_block].
 *  Here subordinate axis channel_block=16 is the split of the primal axis C (channel).
 */
#ifndef TVM_RELAY_OP_LAYOUT_H_
#define TVM_RELAY_OP_LAYOUT_H_

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
      default: CHECK(false) << "Invalid layout axis name " << name;
    }
    // suppress return-type warning.
    return A;
  }

  inline static const LayoutAxis& Get(const char* name) {
    return LayoutAxis::Get(*name);
  }

  inline static const LayoutAxis& Get(const IterVar& itvar) {
    const std::string axis = itvar->var.get()->name_hint;
    CHECK_EQ(axis.size(), 1) << "Invalid layout axis " << axis;
    return LayoutAxis::Get(axis[0]);
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
  Array<IterVar> axis;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("axis", &axis);
  }

  TVM_DLL static Layout make(const std::string& layout);

  static constexpr const char* _type_key = "Layout";
  TVM_DECLARE_NODE_TYPE_INFO(LayoutNode, Node);
};

class Layout : public NodeRef {
 public:
  explicit Layout(NodePtr<Node> n) : NodeRef(n) {}

  /*! \brief default constructor */
  Layout() : Layout("__undef__") {} // NOLINT(*)

  explicit Layout(const Array<IterVar>& axes) {
    node_ = make_node<LayoutNode>();
    LayoutNode *node = operator->();
    node->axis = axes;
    std::ostringstream repr;
    for (const IterVar& axis : axes) {
      if (const auto* factor = axis->dom->extent.as<IntImm>()) {
        CHECK_GT(factor->value, 0);
        repr << factor;
      }
      CHECK_EQ(axis->var.get()->name_hint.size(), 1) << "Invalid layout axis "
                                                     << axis->var.get()->name_hint;
      char c = axis->var.get()->name_hint[0];
      CHECK((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) << "Invalid layout axis " << c;
      repr << axis->var.get()->name_hint;
    }
    node->name = repr.str();
  }

  /*! \brief construct from a string */
  Layout(const char* name) : Layout(std::string(name)) {} // NOLINT(*)

  /*!
   * \brief construct from a string.
   * \param layout input in layout convention:
   *        upper case indicates a dimension and
   *        the corresponding lower case with factor size
   *        indicates the split dimension.
   *        return undefined layout if "__undef__" is passed.
   */
  Layout(const std::string& name) { // NOLINT(*)
    node_ = make_node<LayoutNode>();
    LayoutNode *node = operator->();
    node->name = name;

    if (name != "__undef__") {  // parse layout string
      int32_t factor = 0;
      for (char c : name) {
        if (c >= 'A' && c <= 'Z') {
          CHECK_EQ(factor, 0) << "Invalid layout " << name
                              << ": invalid factor size " << factor
                              << " before dimension " << c;
          std::string shape_name("_shape");
          shape_name.insert(0, 1, c);
          IterVar axis = IterVarNode::make(Range(Expr(0), Var(shape_name)),
                                           Var(std::string(1, c)), kDataPar);
          node->axis.push_back(axis);
        } else if (c >= 'a' && c <= 'z') {
          CHECK_GT(factor, 0) << "Invalid layout " << name << ": invalid factor size "
                              << factor << " for dimension " << c;
          IterVar axis = IterVarNode::make(Range(Expr(0), Expr(factor)),
                                           Var(std::string(1, c)), kDataPar);
          node->axis.push_back(axis);
          factor = 0;
        } else if (c >= '0' && c <= '9') {
          CHECK(factor >= 0) << "Invalid layout " << name << ": _ is adjacent to a number.";
          factor = factor * 10 + c - '0';
        } else {
          LOG(FATAL) << "Invalid layout " << name;
        }
      }
    }

    // validate layout
    std::vector<bool> exist_axis(256, false);
    for (const IterVar& v : node->axis) {
      auto axis_str = v->var.get()->name_hint;
      CHECK_EQ(axis_str.size(), 1);
      char axis = axis_str[0];
      CHECK((axis >= 'a' && axis <= 'z') || (axis >= 'A' && axis <= 'Z'));
      CHECK(!exist_axis[axis]) << "Invalid layout " << name << ": duplicate axis " << axis;
      exist_axis[axis] = true;
    }
    for (const IterVar& v : node->axis) {
      char axis = v->var.get()->name_hint[0];
      if (axis >= 'a' && axis <= 'z') {
        CHECK(exist_axis[axis-'a'+'A']) << "Invalid layout " << name << ": missing axis "
                                        << axis - 'a' + 'A';
      }
    }
  }

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
   * \brief Returns a sublayout which is the portion of the object
   *        that starts at dimension \p pos and spans \p len dimensions
   *        (or until the end of the layout, whichever comes first).
   * \param pos The start position.
   * \param len The length of the sub-layout.
   * \return A newly constructed Layout object.
   */
  Layout Sublayout(size_t pos, size_t len) const {
    if (pos > ndim()) return Layout::Undef();
    if (pos + len > ndim()) len = ndim() - pos;
    Array<IterVar> new_layout;
    const auto axes = operator->()->axis;
    for (size_t i = pos; i < pos + len; ++i) {
      new_layout.push_back(axes[i]);
    }
    return Layout(new_layout);
  }

  /*!
   * \brief Split \p axis by \p size and put the sub-axis to position \p target_pos.
   * \param axis The source axis to be split. It must be a primal-axis;
   * \param target_pos The target position of the newly split subordinate-axis.
   * \param size size of the sub-dimension.
   * \return A newly constructed Layout object.
   */
  Layout Split(const LayoutAxis& axis, size_t target_pos, int32_t size) const {
    const std::string& name = operator->()->name;
    const auto axes = operator->()->axis;
    CHECK(target_pos <= this->ndim()) << "Invalid split position "
                                      << target_pos << " for layout " << name;
    CHECK(axis.IsPrimal()) << "Cannot split a subordinate axis " << axis;
    CHECK(this->Contains(axis)) << "Axis " << axis << " does not exist in " << name;
    CHECK(!this->Contains(axis.to_subordinate())) << "Axis " << axis
                                                  << " has already been split in " << name;
    CHECK(size > 0) << "Invalid split size " << size;
    Array<IterVar> new_layout;
    for (size_t i = 0; i <= this->ndim(); ++i) {
      if (i == target_pos) {
        new_layout.push_back(IterVarNode::make(Range(Expr(0), Expr(size)),
                                               Var(axis.to_subordinate().name()), kDataPar));
      }
      if (i == this->ndim()) break;
      new_layout.push_back(axes[i]);
    }
    return Layout(new_layout);
  }


  /*! \return number of dimensions */
  inline size_t ndim() const {
    return operator->()->axis.size();
  }

  /*! \return number of super dimensions */
  inline size_t ndim_primal() const {
    size_t ct = 0;
    for (auto x : operator->()->axis) {
      if (LayoutAxis::Get(x->var.get()->name_hint[0]).IsPrimal()) {
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
  int32_t Indexof(const LayoutAxis& axis) const {
    if (!this->defined()) return -1;
    const auto axes = operator->()->axis;
    for (size_t i = 0; i < axes.size(); ++i) {
      if (axes[i]->var.get()->name_hint == axis.name()) return static_cast<int32_t>(i);
    }
    return -1;
  }

  /*!
   * \param axis the input primal-axis or subordinate-axis.
   * \return the size of the subordinate-axis of \p axis (if \p axis is a primal-axis),
   *         or the size of \p axis itself (if \p axis is a primal-axis).
   *         Return -1 if \p axis is not in the layout or the layout is undefined.
   */
  int64_t Subsizeof(const LayoutAxis& axis) const {
    const LayoutAxis& sub = axis.to_subordinate();
    if (!this->defined() || !this->Contains(sub)) {
      return -1;
    }

    for (const IterVar& itvar : operator->()->axis) {
      if (sub == LayoutAxis::Get(itvar)) {
        const auto* factor = itvar->dom->extent.as<IntImm>();
        CHECK(factor);
      }
    }
  }

  /*!
   * \brief Whether the layout contains an axis.
   * \param axis axis to be checked.
   * \return Whether the layout contains the axis.
   */
  bool Contains(const LayoutAxis& axis) const {
    for (const IterVar var : operator->()->axis) {
      if (var->var.get()->name_hint == axis.name()) {
        return true;
      }
    }
    return false;
  }

  const LayoutAxis& operator[](size_t i) const {
    const IterVar axis = operator->()->axis[i];
    return LayoutAxis::Get(axis);
  }

  /*! \return whether the layout is defined */
  bool defined() const {
    return operator->()->name != "__undef__";
  }

  /*! \return the string description of the layout */
  const std::string& name() const {
    return operator->()->name;
  }

  /*!
   * \brief Whether the two layouts are equal.
   * \param rhs Another layout.
   * \return whether the two layouts are equal.
   */
  bool Equals(const Layout &rhs) const {
    return operator->()->name == rhs->name;
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

class BijectiveLayoutNode;

class BijectiveLayout : public NodeRef {
 public:
  BijectiveLayout() = default;
  explicit BijectiveLayout(NodePtr<Node> n) : NodeRef(n) {}

  // Final shape of the underlying array, given the shape of the normal layout
  TVM_DLL Array <Expr> ForwardShape(const Array<Expr>& shape) const;
  // Given final shape, recover the original shape.
  TVM_DLL Array<Expr> BackwardShape(const Array<Expr>& shape) const;
  // Final index of the underlying array, given the normal layout.
  TVM_DLL Array<Expr> ForwardIndex(const Array<Expr>& index) const;
  // Given store index, recover the original representation space index.
  TVM_DLL Array<Expr> BackwardIndex(const Array<Expr>& store_index) const;

  /*!
  * \brief access the internal node container
  * \return the pointer to the internal node container
  */
  inline const BijectiveLayoutNode* operator->() const;

  /*! \brief specify container node */
  using ContainerType = BijectiveLayoutNode;
};

class BijectiveLayoutNode : public Node {
 public:
  // The original axis, with symbolic shape
  Array<IterVar> orig_axis;
  Array<IterVar> store_axis;
  // expression of each location, on how original location can be mapped
  // to the store location, example
  // [i0 / 16, i1, i0 % 16]
  Array<Expr> forward_rule;
  Array<Expr> backward_rule;

  Layout orig_layout;
  Layout store_layout;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("orig_axis", &orig_axis);
    v->Visit("store_axis", &store_axis);
    v->Visit("orig_layout", &orig_layout);
    v->Visit("store_layout", &store_layout);
  }

  static constexpr const char* _type_key = "BijectiveLayout";
  TVM_DECLARE_NODE_TYPE_INFO(BijectiveLayoutNode, Node);

  TVM_DLL static BijectiveLayout make(const Layout& orig_layout,
                                      const Layout& store_layout);

  inline static char GetAxisName(const IterVar& axis) {
    return axis->var.get()->name_hint.at(0);
  }
  inline static bool IsMajorAxis(const IterVar& axis) {
    return GetAxisName(axis) >= 'A' && GetAxisName(axis) <= 'Z';
  }
  inline static bool Match(const IterVar& x, const IterVar& y) {
    const char x_name = IsMajorAxis(x) ? GetAxisName(x) : GetAxisName(x) - 'a' + 'A';
    const char y_name = IsMajorAxis(y) ? GetAxisName(y) : GetAxisName(y) - 'a' + 'A';
    return x_name == y_name;
  }

 private:
  inline static bool GetStoreRule(Array<Expr>& rule,
                                  const Array<IterVar>& orig_axes,
                                  const Array<IterVar>& store_axes) {
    for (const IterVar& axis : store_axes) {
      Expr store(0);
      for (const IterVar& orig_axis : orig_axes) {
        if (Match(axis, orig_axis)) {
          if (IsMajorAxis(orig_axis)) {
            Expr orig_var = orig_axis->var;
            // TODO: avoid for loop
            for (const IterVar& temp_axis : orig_axes) {
              if (!IsMajorAxis(temp_axis) && Match(temp_axis, orig_axis)) {
                orig_var = orig_var * temp_axis->dom->extent;
              }
            }
            store = store + orig_var;
          } else {
            store = store + orig_axis->var;
          }
        }
      }
      if (is_zero(store)) {
        // Not convertible
        return false;
      }
      if (IsMajorAxis(axis)) {
        // TODO: avoid for loop
        for (const IterVar& temp_axis : store_axes) {
          if (!IsMajorAxis(temp_axis) && Match(temp_axis, axis)) {
            store = store / temp_axis->dom->extent;
          }
        }
      } else {
        store = store % axis->dom->extent;
      }
      rule.push_back(store);
    }
    return true;
  }
};

inline const BijectiveLayoutNode* BijectiveLayout::operator->() const {
  return static_cast<const BijectiveLayoutNode*>(node_.get());
}

}  // namespace tvm

#endif  // TVM_RELAY_OP_LAYOUT_H_
