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
 *  Copyright (c) 2018 by Contributors
 * \file relay/backend/compile_engine.h
 * \brief Internal compialtion engine handle function cache.
 *  and interface to low level code generation.
 */
#ifndef TVM_RELAY_BACKEND_COMPILE_ENGINE_H_
#define TVM_RELAY_BACKEND_COMPILE_ENGINE_H_

#include <tvm/lowered_func.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/transform.h>
#include <string>
#include <functional>

namespace tvm {
namespace relay {

/*! \brief Indicate whether the data or shape or both of a parameter is used in the shape func. */
enum ShapeFuncParamState {
  kNoNeed = 0,
  kNeedInputData = 1,
  kNeedInputShape = 2,
  kNeedBoth = 3,
  kNeedUpperBound = 4,
};

/*! \brief Node container to represent a cached function. */
struct CachedFuncNode : public Node {
  /* \brief compiled target */
  tvm::Target target;
  /*! \brief Function name */
  std::string func_name;
  /* \brief The inputs to the function */
  tvm::Array<Tensor> inputs;
  /* \brief The outputs to the function */
  tvm::Array<Tensor> outputs;
  /*! \brief The lowered functions to support the function. */
  tvm::Array<tvm::LoweredFunc> funcs;
  /*! \brief Parameter usage states in the shape function. */
  tvm::Array<Integer> shape_func_param_states;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("target", &target);
    v->Visit("func_name", &func_name);
    v->Visit("inputs", &inputs);
    v->Visit("outputs", &outputs);
    v->Visit("funcs", &funcs);
    v->Visit("shape_func_param_states", &shape_func_param_states);
  }

  static constexpr const char* _type_key = "relay.CachedFunc";
  TVM_DECLARE_NODE_TYPE_INFO(CachedFuncNode, Node);
};

TVM_DEFINE_NODE_REF(CachedFunc, CachedFuncNode);


class CCacheKey;
/*! \brief Compile cache key */
class CCacheKeyNode : public Node {
 public:
  /*! \brief The source function to be lowered. */
  Function source_func;
  /*! \brief The hardware target.*/
  Target target;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("source_func", &source_func);
    v->Visit("target", &target);
  }
  /*! \return The hash value of CCacheKey. */
  inline size_t Hash() const;
  /*!
   * \brief check content equality
   * \param other The other value.
   * \return The result of equality check.
   */
  inline bool Equal(const CCacheKeyNode* other) const;
  /*!
   * \brief create a cache key.
   * \param source_func The source function.
   * \param target The target device.
   * \return the created key.
   */
  TVM_DLL static CCacheKey make(Function source_func,
                                Target target);

  static constexpr const char* _type_key = "relay.CCacheKey";
  TVM_DECLARE_NODE_TYPE_INFO(CCacheKeyNode, tvm::Node);

 private:
  /*!
   * \brief internal cached hash value.
   */
  mutable size_t hash_{0};
};

/*! \brief cache entry used in compile engine */
class CCacheKey : public NodeRef {
 public:
  CCacheKey() {}
  explicit CCacheKey(ObjectPtr<Object> n) : NodeRef(n) {}
  const CCacheKeyNode* operator->() const {
    return static_cast<const CCacheKeyNode*>(get());
  }
  // comparator
  inline bool operator==(const CCacheKey& other) const {
    CHECK(defined() && other.defined());
    return (*this)->Equal(other.operator->());
  }
  using ContainerType = CCacheKeyNode;
};

/*! \brief Node container for compile cache. */
class CCacheValueNode : public Node {
 public:
  /*! \brief The corresponding function */
  CachedFunc cached_func;
  /*! \brief Result of Packed function generated by JIT */
  PackedFunc packed_func;
  /*! \brief usage statistics */
  int use_count{0};

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("cached_func", &cached_func);
    v->Visit("use_count", &use_count);
  }
  static constexpr const char* _type_key = "relay.CCacheValue";
  TVM_DECLARE_NODE_TYPE_INFO(CCacheValueNode, tvm::Node);
};

/*! \brief cache entry used in compile engine */
class CCacheValue : public NodeRef {
 public:
  CCacheValue() {}
  explicit CCacheValue(ObjectPtr<Object> n) : NodeRef(n) {}
  CCacheValueNode* operator->() {
    return static_cast<CCacheValueNode*>(get_mutable());
  }
  const CCacheValueNode* operator->() const {
    return static_cast<const CCacheValueNode*>(get());
  }
  using ContainerType = CCacheValueNode;
};

/*!
 * \brief Backend compilation engine for
 *        low level code generation.
 */
class CompileEngineNode : public Node {
 public:
  /*!
   * \brief Get lowered result.
   * \param key The key to the cached function.
   * \return The result.
   */
  virtual CachedFunc Lower(const CCacheKey& key) = 0;
  /*!
   * \brief Just in time compile to get a PackedFunc.
   * \param key The key to the cached function.
   * \return The result.
   */
  virtual PackedFunc JIT(const CCacheKey& key) = 0;
  /*!
   * \brief Lower the shape function.
   * \param key The key to the cached function.
   * \return The result.
   */
  virtual CachedFunc LowerShapeFunc(const CCacheKey& key) = 0;
  /*! \brief clear the cache. */
  virtual void Clear() = 0;

  // VisitAttrs
  void VisitAttrs(AttrVisitor*) final {}

  static constexpr const char* _type_key = "relay.CompileEngine";
  TVM_DECLARE_NODE_TYPE_INFO(CompileEngineNode, Node);
};

/*! \brief cache entry used in compile engine */
class CompileEngine : public NodeRef {
 public:
  CompileEngine() {}
  explicit CompileEngine(ObjectPtr<Object> n) : NodeRef(n) {}
  CompileEngineNode* operator->() {
    return static_cast<CompileEngineNode*>(get_mutable());
  }
  using ContainerType = CompileEngineNode;
  /*! \brief The global compile engine. */
  TVM_DLL static const CompileEngine& Global();
};

/*!
 * \brief Check if the type is dynamic.
 * \param ty The type to be checked.
 * \return The result.
 */
bool IsDynamic(const Type& ty);

// implementations
inline size_t CCacheKeyNode::Hash() const {
  if (hash_ != 0) return hash_;
  // do structral hash, avoid 0.
  hash_ = StructuralHash()(this->source_func);
  hash_ = dmlc::HashCombine(
      hash_, std::hash<std::string>()(target->str()));
  if (hash_ == 0) hash_ = 1;
  return hash_;
}

inline bool CCacheKeyNode::Equal(
    const CCacheKeyNode* other) const {
  if (Hash() != other->Hash()) return false;
  return this->target->str() == other->target->str() &&
      AlphaEqual(this->source_func, other->source_func);
}

}  // namespace relay
}  // namespace tvm

namespace std {
// overload hash
template<>
struct hash<::tvm::relay::CCacheKey> {
  size_t operator()(const ::tvm::relay::CCacheKey& key) const {
    CHECK(key.defined());
    return key->Hash();
  }
};
}  // namespace std
#endif  // TVM_RELAY_BACKEND_COMPILE_ENGINE_H_
