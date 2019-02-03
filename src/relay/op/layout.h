/*!
 *  Copyright (c) 2018 by Contributors
 * \file relay/op/layout.h
 * \brief Layout expression.
 *
 *  This file is adapted from its nnvm counterpart and will keep involving
 *  to the new layout system
 *
 *  The layout is composed of upper cases, lower cases and numbers,
 *  where upper case indicates a (super-)dimension and
 *  the corresponding lower case with factor size indicates the split (sub-)dimension.
 *  For example, NCHW16c can describe a 5-D tensor of
 *  [batch_size, channel, height, width, channel_block].
 *  Here sub-dimension channel_block=16 is the split of super-dimension C (channel).
 */
#ifndef TVM_RELAY_OP_LAYOUT_H_
#define TVM_RELAY_OP_LAYOUT_H_

#include <tvm/base.h>
#include <tvm/expr.h>
#include <tvm/relay/base.h>

#include <string>
#include <sstream>
#include <vector>
#include <utility>
#include <algorithm>

namespace tvm {
namespace relay {

class LayoutNode : public Node {
 public:
  std::string name;
  Array<Integer> superdim_pos;
  Array<Integer> subdim_pos;
  Array<Integer> subdim_size;
  Array<Integer> layout_simplified;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("superdim_pos", &superdim_pos);
    v->Visit("subdim_pos", &subdim_pos);
    v->Visit("subdim_size", &subdim_size);
    v->Visit("layout_simplified", &layout_simplified);
  }

  static constexpr const char* _type_key = "Layout";
  TVM_DECLARE_NODE_TYPE_INFO(LayoutNode, Node);
};

class Layout : public NodeRef {
 public:
  using LayoutDim = char;
  static constexpr uint32_t kUniqueDim = 26;

  explicit Layout(NodePtr<Node> n) : NodeRef(n) {}

  /*! \brief default constructor */
  Layout() : Layout("__undef__") {} // NOLINT(*)

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

    std::vector<int32_t> superdim_pos(kUniqueDim, -1);
    std::vector<int32_t> subdim_pos(kUniqueDim, -1);
    std::vector<int32_t> subdim_size(kUniqueDim, -1);
    std::vector<char> layout_simplified;

    if (name != "__undef__") {  // parse layout string
      int32_t factor = 0;
      uint32_t curr = 0;
      for (size_t i = 0; i < name.size(); ++i) {
        const LayoutDim c = name.at(i);
        if (IsSuperdim(c)) {
          int pos = c - 'A';
          CHECK_EQ(factor, 0) << "Invalid layout " << name
                              << ": invalid factor size " << factor
                              << " before dimension " << c;
          CHECK_EQ(superdim_pos[pos], -1) << "Invalid layout " << name
                                          << ": duplicate dimension " << c;
          superdim_pos[pos] = curr++;
          layout_simplified.push_back(c);
        } else if (IsSubdim(c)) {
          int pos = c - 'a';
          CHECK_GT(factor, 0) << "Invalid layout " << name << ": invalid factor size "
                              << factor << " for dimension " << c;
          CHECK_EQ(subdim_pos[pos], -1) << "Invalid layout " << name
                                        << ": duplicate dimension " << c;
          CHECK_EQ(subdim_size[pos], -1) << "Invalid layout " << name
                                         << ": duplicate dimension " << c;
          subdim_pos[pos] = curr++;
          subdim_size[pos] = factor;
          layout_simplified.push_back(c);
          factor = 0;
        } else if (c >= '0' && c <= '9') {
          CHECK(factor >= 0) << "Invalid layout " << name << ": _ is adjacent to a number.";
          factor = factor * 10 + c - '0';
        } else {
          LOG(FATAL) << "Invalid layout " << name;
        }
      }
      for (LayoutDim dim : layout_simplified) {
        CHECK(IsSuperdim(dim) || superdim_pos[dim-'a'] >= 0)
          << "Invalid layout " << name << ": missing axis "
          << static_cast<char>(dim - 'a' + 'A');
      }
    }

    LayoutNode *node = operator->();
    node->name = name;

    for (uint32_t i = 0; i < kUniqueDim; ++i) {
      node->superdim_pos.push_back(superdim_pos[i]);
      node->subdim_pos.push_back(subdim_pos[i]);
      node->subdim_size.push_back(subdim_size[i]);
    }
    for (LayoutDim dim : layout_simplified) {
      node->layout_simplified.push_back(dim);
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
   * \brief Check whether a given dimension is a super-dimension.
   * \param dim input dimension
   * \return Whether a given dimension is a super-dimension.
   */
  static bool IsSuperdim(LayoutDim dim) {
    return dim >= 'A' && dim <= 'Z';
  }

  /*!
   * \brief Check whether a given dimension is a sub-dimension.
   * \param dim input dimension
   * \return Whether a given dimension is a sub-dimension.
   */
  static bool IsSubdim(LayoutDim dim) {
    return dim >= 'a' && dim <= 'z';
  }

  /*!
   * \brief Convert a given dimension to super-dimension.
   * \param dim input dimension
   * \return The converted description.
   */
  static LayoutDim ToSuperdim(LayoutDim dim) {
    if (IsSubdim(dim)) {
      return dim - 'a' + 'A';
    }
    return dim;
  }

  /*!
   * \brief Convert a given dimension to sub-dimension.
   * \param dim input dimension
   * \return The converted description.
   */
  static LayoutDim ToSubdim(LayoutDim dim) {
    if (IsSuperdim(dim)) {
      return dim - 'A' + 'a';
    }
    return dim;
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
   * \brief Two layouts are convertible only if
   *        they have same set of super-dimensions.
   *        e.g., NCHW, NCHW16c, NHWC are convertible between each other,
   *        but NCHW, CHW, OIHW are not.
   * \param dst the target layout
   * \return Whether can be converted to dst layout.
   */
  bool Convertible(const Layout &dst) const {
    const LayoutNode *n = operator->();
    if (!this->defined() || !dst.defined()) return false;
    for (size_t i = 0; i < kUniqueDim; ++i) {
      if ((n->superdim_pos[i]->value >= 0 && dst->superdim_pos[i]->value < 0) ||
          (n->superdim_pos[i]->value < 0 && dst->superdim_pos[i]->value >= 0)) {
        return false;
      }
    }
    return true;
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
    const Array<Integer>& layout_simplified = operator->()->layout_simplified;
    if (pos > ndim()) return Layout::Undef();
    if (pos + len > ndim()) len = ndim() - pos;
    std::ostringstream new_layout;
    for (size_t i = pos; i < pos + len; ++i) {
      if (IsSubdim(layout_simplified[i]->value)) {
        auto block_size = this->Subsizeof(layout_simplified[i]->value);
        CHECK_GT(block_size, 0);
        new_layout << block_size;
      }
      new_layout << static_cast<char>(layout_simplified[i]->value);
    }
    return Layout(new_layout.str());
  }

  /*! \return A newly constructed reversed Layout object. */
  Layout Reverse() const {
    const Array<Integer>& layout_simplified = operator->()->layout_simplified;
    if (!this->defined()) return Layout::Undef();
    std::ostringstream new_layout;
    for (int64_t i = this->ndim() - 1; i >= 0; --i) {
      if (IsSubdim(layout_simplified[i]->value)) {
        auto block_size = this->Subsizeof(layout_simplified[i]->value);
        CHECK_GT(block_size, 0);
        new_layout << block_size;
      }
      new_layout << layout_simplified[i]->value;
    }
    return Layout(new_layout.str());
  }

  /*!
   * \brief Split \p dim by \p size and put the sub-dimension to position \p target_pos.
   * \param dim The source dimension to be split. It must be a super-dimension.
   * \param target_pos The target position of the newly split sub-dimension.
   * \param size size of the sub-dimension.
   * \return A newly constructed Layout object.
   */
  Layout Split(LayoutDim dim, size_t target_pos, uint32_t size) const {
    const std::string &name = operator->()->name;
    CHECK(target_pos <= this->ndim()) << "Invalid split position "
                                      << target_pos << " for layout " << name;
    CHECK(IsSuperdim(dim)) << "Cannot split a sub-dimension " << dim;
    CHECK(this->Contains(dim)) << "Axis " << dim << " does not exist in " << name;
    CHECK(!this->Contains(ToSubdim(dim))) << "Dimension " << dim
                                           << " has already been split in "
                                           << name;
    CHECK(size > 0) << "Invalid split size " << size;
    std::ostringstream new_layout;
    for (size_t i = 0; i <= this->ndim(); ++i) {
      if (i == target_pos) {
        new_layout << size << Layout::ToSubdim(dim);
      }
      if (i == this->ndim()) break;
      new_layout << this->at(i);
    }
    Layout x(new_layout.str());
    return x;
  }


  /*! \return number of dimensions */
  size_t ndim() const {
    return operator->()->layout_simplified.size();
  }

  /*! \return number of super dimensions */
  size_t ndim_super() const {
    size_t ct = 0;
    for (auto x : operator->()->layout_simplified) {
      if (IsSuperdim(x))
        ct++;
    }
    return ct;
  }

  /*!
   * \brief The description of the \p i-th dimension.
   *        If it is a sub-dimension, the size will be returned as well,
   *        e.g., 16c. Otherwise a single character is returned, e.g., C.
   * \param i The position
   * \return the description of the dimension.
   */
  std::string at(size_t i) const {
    const Array<Integer>& layout_simplified = operator->()->layout_simplified;
    CHECK_LT(i, this->ndim()) << "position " << i
                              << " exceeds ndim=" << this->ndim();
    std::ostringstream repr;
    if (IsSubdim(layout_simplified[i]->value)) {
      auto factor = Subsizeof(layout_simplified[i]->value);
      CHECK_GT(factor, 0);
      repr << factor;
    }
    repr << static_cast<char>(layout_simplified[i]->value);
    return repr.str();
  }

  /*!
   * \brief return the index of the input dimension.
   *        If it is not found in the layout or the layout is undefined,
   *        return -1.
   * \param dim the input dimension.
   * \return the index or -1 if not found.
   */
  int32_t Indexof(LayoutDim dim) const {
    if (!this->defined()) return -1;
    else if (IsSuperdim(dim)) return operator->()->superdim_pos[dim - 'A']->value;
    else if (IsSubdim(dim)) return operator->()->subdim_pos[dim - 'a']->value;
    return -1;
  }

  /*!
   * \param dim the input super-dimension or sub-dimension.
   * \return the size of the sub-dimension of \p dim (if \p dim is a super-dimension),
   *         or the size of \p dim itself (if \p dim is a sub-dimension).
   *         Return -1 if \p dim is not in the layout or the layout is undefined.
   */
  int64_t Subsizeof(LayoutDim dim) const {
    CHECK(IsSuperdim(dim) || IsSubdim(dim)) << "Invalid dim " << dim;
    if (!this->defined() || !this->Contains(ToSubdim(dim))) {
      return -1;
    }
    int idx = ToSubdim(dim) - 'a';
    return operator->()->subdim_size[idx]->value;
  }

  /*!
   * \brief Whether the layout contains a dimension.
   * \param dim dimension to be checked.
   * \return Whether the layout contains the dimension.
   */
  bool Contains(LayoutDim dim) const {
    if (IsSuperdim(dim)) {
      return operator->()->superdim_pos[dim-'A']->value >= 0;
    } else if (IsSubdim(dim)) {
      return operator->()->subdim_pos[dim-'a']->value >= 0;
    }
    return false;
  }

  LayoutDim operator[](size_t i) const {
    return operator->()->layout_simplified[i];
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

/*!
 * \brief Convert shape in src_layout to shape in dst_layout
 * \param src original shape
 * \param src_layout layout of original shape
 * \param dst_layout target layout
 * \return shape in target layout
 */
std::vector<IndexExpr> ConvertLayout(
    std::vector<IndexExpr> src,
    const Layout& src_layout,
    const Layout& dst_layout);

/*!
 * \brief Convert shape in src_layout to shape in dst_layout
 * \param src original shape
 * \param src_layout layout of original shape
 * \param dst_layout target layout
 * \return shape in target layout
 */
std::vector<IndexExpr> ConvertLayout(
    const Array<IndexExpr>& src,
    const Layout& src_layout,
    const Layout& dst_layout);
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_LAYOUT_H_
