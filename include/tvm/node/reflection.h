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
 * \file tvm/node/reflection.h
 * \brief Reflection and serialization of compiler IR/AST nodes.
 */
#ifndef TVM_NODE_REFLECTION_H_
#define TVM_NODE_REFLECTION_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/data_type.h>

#include <vector>
#include <string>

namespace tvm {

// forward declaration
using runtime::Object;
using runtime::ObjectPtr;
using runtime::ObjectRef;

/*!
 * \brief Visitor class for to get the attributesof a AST/IR node.
 *  The content is going to be called for each field.
 *
 *  Each objects that wants reflection will need to implement
 *  a VisitAttrs function and call visitor->Visit on each of its field.
 */
class AttrVisitor {
 public:
//! \cond Doxygen_Suppress
  TVM_DLL virtual ~AttrVisitor() = default;
  TVM_DLL virtual void Visit(const char* key, double* value) = 0;
  TVM_DLL virtual void Visit(const char* key, int64_t* value) = 0;
  TVM_DLL virtual void Visit(const char* key, uint64_t* value) = 0;
  TVM_DLL virtual void Visit(const char* key, int* value) = 0;
  TVM_DLL virtual void Visit(const char* key, bool* value) = 0;
  TVM_DLL virtual void Visit(const char* key, std::string* value) = 0;
  TVM_DLL virtual void Visit(const char* key, void** value) = 0;
  TVM_DLL virtual void Visit(const char* key, DataType* value) = 0;
  TVM_DLL virtual void Visit(const char* key, runtime::NDArray* value) = 0;
  TVM_DLL virtual void Visit(const char* key, runtime::ObjectRef* value) = 0;
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
 * \brief Virtual function table to support IR/AST node reflection.
 *
 * Functions are stored  in columar manner.
 * Each column is a vector indexed by Object's type_index.
 */
class ReflectionVTable {
 public:
  /*!
   * \brief Visitor function.
   * \note We use function pointer, instead of std::function
   *       to reduce the dispatch overhead as field visit
   *       does not need as much customization.
   */
  typedef void (*FVisitAttrs)(Object* self, AttrVisitor* visitor);
  /*!
   * \brief creator function.
   * \param global_key Key that identifies a global single object.
   *        If this is not empty then FGlobalKey must be defined for the object.
   * \return The created function.
   */
  typedef ObjectPtr<Object> (*FCreate)(const std::string& global_key);
  /*!
   * \brief Global key function, only needed by global objects.
   * \param node The node pointer.
   * \return node The global key to the node.
   */
  typedef std::string (*FGlobalKey)(const Object* self);
  /*!
   * \brief Dispatch the VisitAttrs function.
   * \param self The pointer to the object.
   * \param visitor The attribute visitor.
   */
  inline void VisitAttrs(Object* self, AttrVisitor* visitor) const;
  /*!
   * \brief Get global key of the object, if any.
   * \param self The pointer to the object.
   * \return the global key if object has one, otherwise return empty string.
   */
  inline std::string GetGlobalKey(Object* self) const;
  /*!
   * \brief Create an initial object using default constructor
   *        by type_key and global key.
   *
   * \param type_key The type key of the object.
   * \param global_key A global key that can be used to uniquely identify the object if any.
   */
  TVM_DLL ObjectPtr<Object> CreateInitObject(const std::string& type_key,
                                             const std::string& global_key = "") const;
  /*!
   * \brief Get an field object by the attr name.
   * \param self The pointer to the object.
   * \param attr_name The name of the field.
   * \return The corresponding attribute value.
   * \note This function will throw an exception if the object does not contain the field.
   */
  TVM_DLL runtime::TVMRetValue GetAttr(Object* self, const std::string& attr_name) const;

  /*!
   * \brief List all the fields in the object.
   * \return All the fields.
   */
  TVM_DLL std::vector<std::string> ListAttrNames(Object* self) const;

  /*! \return The global singleton. */
  TVM_DLL static ReflectionVTable* Global();

  class Registry;
  template<typename T>
  inline Registry Register();

 private:
  /*! \brief Attribute visitor. */
  std::vector<FVisitAttrs> fvisit_attrs_;
  /*! \brief Creation function. */
  std::vector<FCreate> fcreate_;
  /*! \brief Global key function. */
  std::vector<FGlobalKey> fglobal_key_;
};

/*! \brief Registry of a reflection table. */
class ReflectionVTable::Registry {
 public:
  Registry(ReflectionVTable* parent, uint32_t type_index)
      : parent_(parent), type_index_(type_index) { }
  /*!
   * \brief Set fcreate function.
   * \param f The creator function.
   * \return rference to self.
   */
  Registry& set_creator(FCreate f) {  // NOLINT(*)
    CHECK_LT(type_index_, parent_->fcreate_.size());
    parent_->fcreate_[type_index_] = f;
    return *this;
  }
  /*!
   * \brief Set global_key function.
   * \param f The creator function.
   * \return rference to self.
   */
  Registry& set_global_key(FGlobalKey f) {  // NOLINT(*)
    CHECK_LT(type_index_, parent_->fglobal_key_.size());
    parent_->fglobal_key_[type_index_] = f;
    return *this;
  }

 private:
  ReflectionVTable* parent_;
  uint32_t type_index_;
};

/*!
 * \brief Register a node type to object registry and reflection registry.
 * \param TypeName The name of the type.
 * \note This macro will call TVM_REGISTER_OBJECT_TYPE for the type as well.
 */
#define TVM_REGISTER_NODE_TYPE(TypeName)                                \
  TVM_REGISTER_OBJECT_TYPE(TypeName);                                   \
  static DMLC_ATTRIBUTE_UNUSED ::tvm::ReflectionVTable::Registry &      \
  __make_Node ## _ ## TypeName ## __ =                                  \
      ::tvm::ReflectionVTable::Global()->Register<TypeName>()           \
      .set_creator([](const std::string&) -> ObjectPtr<Object> {        \
          return ::tvm::runtime::make_object<TypeName>();               \
        })

// Implementation details
template<typename T>
inline ReflectionVTable::Registry
ReflectionVTable::Register() {
  uint32_t tindex = T::RuntimeTypeIndex();
  if (tindex >= fvisit_attrs_.size()) {
    fvisit_attrs_.resize(tindex + 1, nullptr);
    fcreate_.resize(tindex + 1, nullptr);
    fglobal_key_.resize(tindex + 1, nullptr);
  }
  // functor that implemnts the redirection.
  struct Functor {
    static void VisitAttrs(Object* self, AttrVisitor* v) {
      static_cast<T*>(self)->VisitAttrs(v);
     }
  };

  fvisit_attrs_[tindex] = Functor::VisitAttrs;
  return Registry(this, tindex);
}

inline void ReflectionVTable::
VisitAttrs(Object* self, AttrVisitor* visitor) const {
  uint32_t tindex = self->type_index();
  if (tindex >= fvisit_attrs_.size() || fvisit_attrs_[tindex] == nullptr) {
    LOG(FATAL) << "TypeError: " << self->GetTypeKey()
               << " is not registered via TVM_REGISTER_NODE_TYPE";
  }
  fvisit_attrs_[tindex](self, visitor);
}

inline std::string ReflectionVTable::GetGlobalKey(Object* self) const {
  uint32_t tindex = self->type_index();
  if (tindex < fglobal_key_.size() && fglobal_key_[tindex] != nullptr) {
    return fglobal_key_[tindex](self);
  } else {
    return std::string();
  }
}

}  // namespace tvm
#endif  // TVM_NODE_REFLECTION_H_
