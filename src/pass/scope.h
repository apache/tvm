/*!
 *  Copyright (c) 2016 by Contributors
 * \file scope.h
 * \brief attribute scope data structure,
 *  defines attributes on current domain
 */
#ifndef TVM_PASS_SCOPE_H_
#define TVM_PASS_SCOPE_H_

#include <tvm/ir.h>
#include <unordered_map>
#include <vector>
#include <string>

namespace tvm {
namespace ir {

/*!
 * \brief Attribute scope of Nodes in the IR.
 * \tparam ValueType The value of of the scope.
 */
template<typename K, typename V>
class Scope {
 public:
  /*!
   * \brief Push value to scope
   * \param key the key to be pushed.
   * \param v The value to be pushed.
   */
  inline void Push(const K& key, V v) {
    data_[key].emplace_back(v);
  }
  /*!
   * \brief Pop value from scope.
   * \param key the key to be poped
   */
  inline void Pop(const K& key) {
    auto& v = data_[key];
    CHECK_NE(v.size(), 0);
    v.pop_back();
  }

  /*!
   * \brief Get value from the scope
   * \param key the key to fetch.
   * \return The value to be fetched.
   */
  inline V operator[](const K& key) const {
    const auto it = data_.find(key);
    CHECK(it != data_.end() && it->second.size() != 0)
        << "cannot find value in scope";
    return it->second.back();
  }

 private:
  std::unordered_map<K, std::vector<V> > data_;
};

/*! \brief Attribute key for specific attribute */
struct AttrKey {
  /*! \brief The node of the attribute */
  NodeRef node;
  /*! \brief The type key of the attribute. */
  std::string type_key;
  // overload operator ==
  inline bool operator==(const AttrKey& other) const {
    return node == other.node && type_key == other.type_key;
  }
};
}  // namespace ir
}  // namespace tvm

namespace std {
template <>
struct hash<::tvm::ir::AttrKey> {
  std::size_t operator()(const ::tvm::ir::AttrKey& k) const {
    size_t lhs = k.node.hash();
    size_t rhs = std::hash<std::string>()(k.type_key);
    lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
    return lhs;
  }
};
}  // namespace std
#endif  // TVM_PASS_SCOPE_H_
