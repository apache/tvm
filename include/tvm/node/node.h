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
 * \file tvm/node/node.h
 * \brief Node system data structure.
 */
#ifndef TVM_NODE_NODE_H_
#define TVM_NODE_NODE_H_

#include <dmlc/logging.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/node_base.h>
#include <string>
#include <vector>
#include <utility>
#include <type_traits>


namespace tvm {
// forward declaration
class DataType;
class Node;
class NodeRef;

namespace runtime {
// forward declaration
class NDArray;
// forward declaration
class Object;
}  // namespace runtime

/*!
 * \brief Visitor class to each node content.
 *  The content is going to be called for each field.
 */
class TVM_DLL AttrVisitor {
 public:
//! \cond Doxygen_Suppress
  virtual ~AttrVisitor() = default;
  virtual void Visit(const char* key, double* value) = 0;
  virtual void Visit(const char* key, int64_t* value) = 0;
  virtual void Visit(const char* key, uint64_t* value) = 0;
  virtual void Visit(const char* key, int* value) = 0;
  virtual void Visit(const char* key, bool* value) = 0;
  virtual void Visit(const char* key, std::string* value) = 0;
  virtual void Visit(const char* key, void** value) = 0;
  virtual void Visit(const char* key, DataType* value) = 0;
  virtual void Visit(const char* key, NodeRef* value) = 0;
  virtual void Visit(const char* key, runtime::NDArray* value) = 0;
  virtual void Visit(const char* key, runtime::Object* value) = 0;
  template<typename ENum,
           typename = typename std::enable_if<std::is_enum<ENum>::value>::type>
  void Visit(const char* key, ENum* ptr) {
    static_assert(std::is_same<int, typename std::underlying_type<ENum>::type>::value,
                  "declare enum to be enum int to use visitor");
    this->Visit(key, reinterpret_cast<int*>(ptr));
  }
//! \endcond
};

/*!
 * \brief base class of node container in DSL AST.
 */
class TVM_DLL Node : public NodeBase {
 public:
  /*! \brief virtual destructor */
  virtual ~Node() {}
  /*! \return The unique type key of the node */
  virtual const char* type_key() const = 0;
  /*!
   * \brief Apply visitor to each field of the Node
   *  Visitor could mutate the content of the node.
   *  override if Node contains attribute fields.
   * \param visitor The visitor
   */
  virtual void VisitAttrs(AttrVisitor* visitor) {}
  /*! \return the type index of the node */
  virtual uint32_t type_index() const = 0;
  /*!
   * \brief Whether this node derives from node with type_index=tid.
   *  Implemented by TVM_DECLARE_NODE_TYPE_INFO
   *
   * \param tid The type index.
   * \return the check result.
   */
  virtual bool _DerivedFrom(uint32_t tid) const;
  /*!
   * \brief get a runtime unique type index given a type key
   * \param type_key Type key of a type.
   * \return the corresponding type index.
   */
  static uint32_t TypeKey2Index(const char* type_key);
  /*!
   * \brief get type key from type index.
   * \param index The type index
   * \return the corresponding type key.
   */
  static const char* TypeIndex2Key(uint32_t index);
  /*!
   * \return whether the type is derived from
   */
  template<typename T>
  inline bool derived_from() const;
  /*!
   * \return whether the node is of type T
   * \tparam The type to be checked.
   */
  template<typename T>
  inline bool is_type() const;
  /*!
   * \brief Get a NodePtr that holds reference to this Node.
   * \return the NodePtr
   */
  inline NodePtr<Node> GetNodePtr() const;
  // node ref can see this
  friend class NodeRef;
  static constexpr const char* _type_key = "Node";
};

/*! \brief Base class of all node reference object */
class NodeRef {
 public:
  /*! \brief type indicate the container type */
  using ContainerType = Node;
  /*!
   * \brief Comparator
   * \param other Another node ref.
   * \return the compare result.
   */
  inline bool operator==(const NodeRef& other) const;
  /*!
   * \brief Comparator
   * \param other Another node ref.
   * \return the compare result.
   */
  inline bool same_as(const NodeRef& other) const;
  /*!
   * \brief Comparator
   * \param other Another node ref.
   * \return the compare result.
   */
  inline bool operator<(const NodeRef& other) const;
  /*!
   * \brief Comparator
   * \param other Another node ref.
   * \return the compare result.
   */
  inline bool operator!=(const NodeRef& other) const;
  /*! \return the hash function for NodeRef */
  inline size_t hash() const;
  /*! \return whether the expression is null */
  inline bool defined() const;
  /*! \return the internal type index of IRNode */
  inline uint32_t type_index() const;
  /*! \return the internal node pointer */
  inline const Node* get() const;
  /*! \return the internal node pointer */
  inline const Node* operator->() const;
  /*!
   * \brief Downcast this ir node to its actual type (e.g. Add, or
   * Select). This returns nullptr if the node is not of the requested
   * type. Example usage:
   *
   * if (const Add *add = node->as<Add>()) {
   *   // This is an add node
   * }
   * \tparam T the target type, must be subtype of IRNode
   */
  template<typename T>
  inline const T *as() const;
  /*!
   * \brief A more powerful version of as that also works with
   *  intermediate base types.
   * \tparam T the target type, must be subtype of IRNode
   */
  template<typename T>
  inline const T *as_derived() const;
  /*! \brief default constructor */
  NodeRef() = default;
  explicit NodeRef(NodePtr<Node> node) : node_(node) {}
  /*! \brief the internal node object, do not touch  */
  NodePtr<Node> node_;
};

/*!
 * \brief Get a reference type from a Node ptr type
 *
 *  It is always important to get a reference type
 *  if we want to return a value as reference or keep
 *  the node alive beyond the scope of the function.
 *
 * \param ptr The node pointer
 * \tparam RefType The reference type
 * \tparam NodeType The node type
 * \return The corresponding RefType
 */
template <typename RefType, typename NodeType>
inline RefType GetRef(const NodeType* ptr);

/*!
 * \brief Downcast a base reference type to a more specific type.
 *
 * \param ref The inptut reference
 * \return The corresponding SubRef.
 * \tparam SubRef The target specific reference type.
 * \tparam BaseRef the current reference type.
 */
template <typename SubRef, typename BaseRef>
inline SubRef Downcast(BaseRef ref);

/*!
 * \brief helper macro to declare type information in a base node.
 */
#define TVM_DECLARE_BASE_NODE_INFO(TypeName, Parent)                    \
  bool _DerivedFrom(uint32_t tid) const override {                      \
    static uint32_t tidx = TypeKey2Index(TypeName::_type_key);          \
    if (tidx == tid) return true;                                       \
    return Parent::_DerivedFrom(tid);                                   \
  }

/*!
 * \brief helper macro to declare type information in a terminal node
 */
#define TVM_DECLARE_NODE_TYPE_INFO(TypeName, Parent)                    \
  const char* type_key() const final {                                  \
    return TypeName::_type_key;                                         \
  }                                                                     \
  uint32_t type_index() const final {                                   \
    static uint32_t tidx = TypeKey2Index(TypeName::_type_key);          \
    return tidx;                                                        \
  }                                                                     \
  bool _DerivedFrom(uint32_t tid) const final {                         \
    static uint32_t tidx = TypeKey2Index(TypeName::_type_key);          \
    if (tidx == tid) return true;                                       \
    return Parent::_DerivedFrom(tid);                                   \
  }

// implementations of inline functions after this
template<typename T>
inline bool Node::derived_from() const {
  // use static field so query only happens once.
  static uint32_t type_id = Node::TypeKey2Index(T::_type_key);
  return this->_DerivedFrom(type_id);
}


template<typename T>
inline bool Node::is_type() const {
  // use static field so query only happens once.
  static uint32_t type_id = Node::TypeKey2Index(T::_type_key);
  return type_id == this->type_index();
}


inline NodePtr<Node> Node::GetNodePtr() const {
  return NodePtr<Node>(const_cast<Node*>(this));
}

template <typename RefType, typename NodeType>
inline RefType GetRef(const NodeType* ptr) {
  static_assert(std::is_base_of<typename RefType::ContainerType, NodeType>::value,
                "Can only cast to the ref of same container type");
  return RefType(ptr->GetNodePtr());
}

template <typename SubRef, typename BaseRef>
inline SubRef Downcast(BaseRef ref) {
  CHECK(ref->template is_type<typename SubRef::ContainerType>() ||
        ref->template derived_from<typename SubRef::ContainerType>())
      << "Downcast from " << ref->type_key() << " to "
      << SubRef::ContainerType::_type_key << " failed.";
  return SubRef(std::move(ref.node_));
}

inline const Node* NodeRef::get() const {
  return node_.get();
}

inline const Node* NodeRef::operator->() const {
  return node_.get();
}

inline bool NodeRef::defined() const {
  return node_.get() != nullptr;
}

inline bool NodeRef::operator==(const NodeRef& other) const {
  return node_.get() == other.node_.get();
}

inline bool NodeRef::same_as(const NodeRef& other) const {
  return node_.get() == other.node_.get();
}

inline bool NodeRef::operator<(const NodeRef& other) const {
  return node_.get() < other.node_.get();
}

inline bool NodeRef::operator!=(const NodeRef& other) const {
  return node_.get() != other.node_.get();
}

inline size_t NodeRef::hash() const {
  return std::hash<Node*>()(node_.get());
}

inline uint32_t NodeRef::type_index() const {
  CHECK(node_.get() != nullptr)
      << "null type";
  return get()->type_index();
}

template<typename T>
inline const T* NodeRef::as() const {
  const Node* ptr = static_cast<const Node*>(get());
  if (ptr && ptr->is_type<T>()) {
    return static_cast<const T*>(ptr);
  }
  return nullptr;
}

template<typename T>
inline const T* NodeRef::as_derived() const {
  const Node* ptr = static_cast<const Node*>(get());
  if (ptr && (ptr->is_type<T>() || ptr->derived_from<T>())) {
    return static_cast<const T*>(ptr);
  }
  return nullptr;
}

/*! \brief The hash function for nodes */
struct NodeHash {
  size_t operator()(const NodeRef& a) const {
    return a.hash();
  }
};

/*! \brief The equal comparator for nodes */
struct NodeEqual {
  bool operator()(const NodeRef& a, const NodeRef& b) const {
    return a.get() == b.get();
  }
};
}  // namespace tvm
#endif  // TVM_NODE_NODE_H_
