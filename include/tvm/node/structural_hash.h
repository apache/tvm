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
 * \file tvm/node/structural_equal.h
 * \brief Structural hash class.
 */
#ifndef TVM_NODE_STRUCTURAL_HASH_H_
#define TVM_NODE_STRUCTURAL_HASH_H_

#include <tvm/node/functor.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/ndarray.h>

#include <functional>
#include <string>

namespace tvm {

/*!
 * \brief Hash definition of base value classes.
 */
class BaseValueHash {
 public:
  size_t operator()(const double& key) const { return std::hash<double>()(key); }

  size_t operator()(const int64_t& key) const { return std::hash<int64_t>()(key); }

  size_t operator()(const uint64_t& key) const { return std::hash<uint64_t>()(key); }

  size_t operator()(const int& key) const { return std::hash<int>()(key); }

  size_t operator()(const bool& key) const { return std::hash<bool>()(key); }

  size_t operator()(const std::string& key) const { return std::hash<std::string>()(key); }

  size_t operator()(const runtime::DataType& key) const {
    return std::hash<int32_t>()(static_cast<int32_t>(key.code()) |
                                (static_cast<int32_t>(key.bits()) << 8) |
                                (static_cast<int32_t>(key.lanes()) << 16));
  }

  template <typename ENum, typename = typename std::enable_if<std::is_enum<ENum>::value>::type>
  bool operator()(const ENum& key) const {
    return std::hash<size_t>()(static_cast<size_t>(key));
  }
};

/*!
 * \brief Content-aware structural hasing.
 *
 *  The structural hash value is recursively defined in the DAG of IRNodes.
 *  There are two kinds of nodes:
 *
 *  - Normal node: the hash value is defined by its content and type only.
 *  - Graph node: each graph node will be assigned a unique index ordered by the
 *    first occurence during the visit. The hash value of a graph node is
 *    combined from the hash values of its contents and the index.
 */
class StructuralHash : public BaseValueHash {
 public:
  // inheritate operator()
  using BaseValueHash::operator();
  /*!
   * \brief Compute structural hashing value for an object.
   * \param key The left operand.
   * \return The hash value.
   */
  TVM_DLL size_t operator()(const ObjectRef& key) const;
};

/*!
 * \brief A Reducer class to reduce the structural hash value.
 *
 *  The reducer will call the SEqualHash function of each objects recursively.
 *
 *  A SEqualHash function will make a sequence of calls to the reducer to
 *  indicate a sequence of child hash values that the reducer need to combine
 *  inorder to obtain the hash value of the hash value of the parent object.
 *
 *  Importantly, the reducer may not directly use recursive calls
 *  to compute the hash values of child objects directly.
 *
 *  Instead, it can store the necessary hash computing task into a stack
 *  and reduce the result later.
 */
class SHashReducer {
 public:
  /*! \brief Internal handler that defines custom behaviors. */
  class Handler {
   public:
    /*!
     * \brief Append hashed_value to the current sequence of hashes.
     *
     * \param hashed_value The hashed value
     */
    virtual void SHashReduceHashedValue(size_t hashed_value) = 0;
    /*!
     * \brief Append hash value of key to the current sequence of hashes.
     *
     * \param key The object to compute hash from.
     * \param map_free_vars Whether to map free variables by their occurence number.
     */
    virtual void SHashReduce(const ObjectRef& key, bool map_free_vars) = 0;
    /*!
     * \brief Apppend a hash value of free variable to the current sequence of hashes.
     *
     * \param var The var of interest.
     * \param map_free_vars Whether to map free variables by their occurence number.
     *
     * \note If map_free_vars is set to be true,
     *       internally the handler can maintain a counter to encode free variables
     *       by their order of occurence. This helps to resolve variable
     *       mapping of function parameters and let binding variables.
     *
     *       If map_free_vars is set to be false, the address of the variable will be used.
     */
    virtual void SHashReduceFreeVar(const runtime::Object* var, bool map_free_vars) = 0;
    /*!
     * \brief Lookup a hash value for key
     *
     * \param key The hash key.
     * \param hashed_value the result hash value
     *
     * \return Whether there is already a pre-computed hash value.
     */
    virtual bool LookupHashedValue(const ObjectRef& key, size_t* hashed_value) = 0;
    /*!
     * \brief Mark current comparison as graph node in hashing.
     *        Graph node hash will depends on the graph structure.
     */
    virtual void MarkGraphNode() = 0;
  };

  /*! \brief default constructor */
  SHashReducer() = default;
  /*!
   * \brief Constructor with a specific handler.
   * \param handler The equal handler for objects.
   * \param map_free_vars Whether to map free variables.
   */
  explicit SHashReducer(Handler* handler, bool map_free_vars)
      : handler_(handler), map_free_vars_(map_free_vars) {}
  /*!
   * \brief Push hash of key to the current sequence of hash values.
   * \param key The key to be hashed.
   */
  template <typename T,
            typename = typename std::enable_if<!std::is_base_of<ObjectRef, T>::value>::type>
  void operator()(const T& key) const {
    // handle normal values.
    handler_->SHashReduceHashedValue(BaseValueHash()(key));
  }
  /*!
   * \brief Push hash of key to the current sequence of hash values.
   * \param key The key to be hashed.
   */
  void operator()(const ObjectRef& key) const { return handler_->SHashReduce(key, map_free_vars_); }
  /*!
   * \brief Push hash of key to the current sequence of hash values.
   * \param key The key to be hashed.
   * \note This function indicate key could contain var defintions.
   */
  void DefHash(const ObjectRef& key) const { return handler_->SHashReduce(key, true); }
  /*!
   * \brief Implementation for hash for a free var.
   * \param var The variable.
   * \return the result.
   */
  void FreeVarHashImpl(const runtime::Object* var) const {
    handler_->SHashReduceFreeVar(var, map_free_vars_);
  }

  /*! \return Get the internal handler. */
  Handler* operator->() const { return handler_; }

 private:
  /*! \brief Internal class pointer. */
  Handler* handler_;
  /*!
   * \brief Whether or not to map free variables by their occurence
   *        If the flag is false, then free variables will be mapped
   *        by their in-memory address.
   */
  bool map_free_vars_;
};

/*! \brief The default handler for hash key computation
 *
 * Users can derive from this class and override the DispatchSHash method,
 * to customize hashing.
 */
class SHashHandlerDefault : public SHashReducer::Handler {
 public:
  SHashHandlerDefault();
  virtual ~SHashHandlerDefault();

  void SHashReduceHashedValue(size_t hashed_value) override;
  void SHashReduce(const ObjectRef& key, bool map_free_vars) override;
  void SHashReduceFreeVar(const runtime::Object* var, bool map_free_vars) override;
  bool LookupHashedValue(const ObjectRef& key, size_t* hashed_value) override;
  void MarkGraphNode() override;

  /*!
   * \brief The entry point for hashing
   * \param object The object to be hashed.
   * \param map_free_vars Whether or not to remap variables if possible.
   * \return The hash result.
   */
  virtual size_t Hash(const ObjectRef& object, bool map_free_vars);

 protected:
  /*!
   * \brief The dispatcher for hashing of intermediate objects
   * \param object An intermediate object to be hashed.
   * \param map_free_vars Whether or not to remap variables if possible.
   * \return The hash result.
   */
  virtual void DispatchSHash(const ObjectRef& object, bool map_free_vars);

 private:
  class Impl;
  Impl* impl;
};

class SEqualReducer;
struct NDArrayContainerTrait {
  static constexpr const std::nullptr_t VisitAttrs = nullptr;
  static void SHashReduce(const runtime::NDArray::Container* key, SHashReducer hash_reduce);
  static bool SEqualReduce(const runtime::NDArray::Container* lhs,
                           const runtime::NDArray::Container* rhs, SEqualReducer equal);
};

}  // namespace tvm
#endif  // TVM_NODE_STRUCTURAL_HASH_H_
