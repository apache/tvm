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
#include <tvm/runtime/object.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/ndarray.h>
#include <string>
#include <vector>
#include <utility>
#include <type_traits>


namespace tvm {
// forward declaration
class DataType;
class Node;
class NodeRef;

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
  virtual void Visit(const char* key, runtime::ObjectRef* value) = 0;
  template<typename ENum,
           typename = typename std::enable_if<std::is_enum<ENum>::value>::type>
  void Visit(const char* key, ENum* ptr) {
    static_assert(std::is_same<int, typename std::underlying_type<ENum>::type>::value,
                  "declare enum to be enum int to use visitor");
    this->Visit(key, reinterpret_cast<int*>(ptr));
  }
//! \endcond
};

/*! \brief Reuse the type index in he runtime. */
using TypeIndex = runtime::TypeIndex;

/*!
 * \brief base class of node container in DSL AST.
 */
class Node : public runtime::Object {
 public:
  /*! \brief virtual destructor */
  virtual ~Node() {}

  /*!
   * \brief Apply visitor to each field of the Node
   *  Visitor could mutate the content of the node.
   *  override if Node contains attribute fields.
   * \param visitor The visitor
   */
  virtual void VisitAttrs(AttrVisitor* visitor) {}

  static constexpr const char* _type_key = "Node";
  static constexpr uint32_t _type_index = TypeIndex::kDynamic;

  TVM_DECLARE_BASE_OBJECT_INFO(Node, runtime::Object);
};


/*!
 * \brief Base class of all node reference object
 *  NodeRef is just a alias of ObjectRef.
 */
class NodeRef : public runtime::ObjectRef {
 public:
  /*! \brief type indicate the container type */
  using ContainerType = Node;

  /*! \return the internal node pointer */
  const Node* get() const {
    return static_cast<const Node*>(ObjectRef::get());
  }
  /*! \return the internal node pointer */
  const Node* operator->() const {
    return get();
  }
  /*!
   * \brief A more powerful version of as that also works with
   *  intermediate base types.
   * \tparam T the target type, must be subtype of IRNode
   */
  template<typename T>
  const T *as_derived() const {
    return as<T>();
  }
  /*! \brief default constructor */
  NodeRef() = default;
  explicit NodeRef(runtime::ObjectPtr<runtime::Object> ptr) : ObjectRef(ptr) {}
};

/*!
 * \brief helper macro to declare type information in a base node.
 */
#define TVM_DECLARE_BASE_NODE_INFO(TypeName, Parent)  \
  TVM_DECLARE_BASE_OBJECT_INFO(TypeName, Parent)

/*!
 * \brief helper macro to declare type information in a terminal node
 */
#define TVM_DECLARE_NODE_TYPE_INFO(TypeName, Parent)  \
  TVM_DECLARE_FINAL_OBJECT_INFO(TypeName, Parent);


using runtime::Object;
using runtime::ObjectPtr;
using runtime::ObjectRef;
using runtime::GetRef;
using runtime::Downcast;
using runtime::make_object;
using runtime::ObjectHash;
using runtime::ObjectEqual;

using NodeHash = ObjectHash;
using NodeEqual = ObjectEqual;

/*!
 * \brief Allocate a node object.
 * \param args arguments to the constructor.
 * \tparam T the node type.
 * \return The NodePtr to the allocated object.
 */
template<typename T, typename... Args>
inline NodePtr<T> make_node(Args&&... args) {
  return runtime::make_object<T>(std::forward<Args>(args)...);
}
}  // namespace tvm
#endif  // TVM_NODE_NODE_H_
