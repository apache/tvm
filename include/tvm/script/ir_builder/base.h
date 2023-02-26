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
#ifndef TVM_SCRIPT_IR_BUILDER_BASE_H_
#define TVM_SCRIPT_IR_BUILDER_BASE_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/function.h>
#include <tvm/node/node.h>

#include <vector>

namespace tvm {
namespace script {
namespace ir_builder {

////////////////////////////// IRBuilderFrame //////////////////////////////

/*!
 * \brief A stack frame of the IRBuilder used to keep track of the current scope.
 * Furthermore, the information stored in each stack frame can be useful for context-dependent
 * IR construction.
 *
 * \example
 *
 * The `T::MatchBuffer` below adds an element in `PrimFuncNode::buffer_map`:
 *
 * \code {.cpp}
 *
 * using T = tvm::script::ir_builder::tir;
 * With <PrimFuncFrame> _(...);
 * Buffer buffer = T::MatchBuffer(...);
 *
 * \endcode
 *
 * The `T::MatchBuffer` below instead generates `MatchBufferRegion` in a TIR block:
 *
 * \code {.cpp}
 *
 * using T = tvm::script::ir_builder::tir;
 * With <PrimFuncFrame> _(...);
 * {
 *   With<BlockFrame> _2(...);
 *   Buffer buffer = T::MatchBuffer(...);
 * }
 *
 * \endcode
 */
class IRBuilderFrameNode : public runtime::Object {
 public:
  /*! \brief A list of callbacks used when exiting the frame. */
  std::vector<runtime::TypedPackedFunc<void()>> callbacks;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `callbacks` is not visited.
  }

  static constexpr const char* _type_key = "script.ir_builder.IRBuilderFrame";
  TVM_DECLARE_BASE_OBJECT_INFO(IRBuilderFrameNode, runtime::Object);

 public:
  /*! \brief Default destructor. */
  virtual ~IRBuilderFrameNode() = default;
  /*!
   * \brief The method called when entering RAII scope.
   * \sa tvm::support::With
   */
  virtual void EnterWithScope();
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  virtual void ExitWithScope();
  /*!
   * \brief Add a callback method invoked when exiting the RAII scope.
   * \param callback The callback to be added.
   */
  void AddCallback(runtime::TypedPackedFunc<void()> callback);
};

/*!
 * \brief Managed reference to an IRBuilderFrameNode.
 * \sa IRBuilderFrameNode
 */
class IRBuilderFrame : public runtime::ObjectRef {
 public:
  /*! \brief Default destructor. */
  virtual ~IRBuilderFrame() = default;
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(IRBuilderFrame, ObjectRef, IRBuilderFrameNode);

 protected:
  /*! \brief Disallow direct construction of this object. */
  IRBuilderFrame() = default;

 public:
  /*!
   * \brief Redirected to `IRBuilderFrameNode::EnterWithScope`.
   * \sa IRBuilderFrameNode::EnterWithScope
   */
  inline void EnterWithScope() {
    ICHECK(data_ != nullptr);
    static_cast<IRBuilderFrameNode*>(data_.get())->EnterWithScope();
  }
  /*!
   * \brief Redirected to `IRBuilderFrameNode::ExitWithScope`.
   * \sa IRBuilderFrameNode::ExitWithScope
   */
  inline void ExitWithScope() {
    ICHECK(data_ != nullptr);
    static_cast<IRBuilderFrameNode*>(data_.get())->ExitWithScope();
    data_.reset();
  }
};

////////////////////////////// IRBuilder //////////////////////////////

/*!
 * \brief A dialect-agnostic IRBuilder that constructs any IR of TVM.
 * An idiomatic use of this class is to put this inside the RAII with-scope,
 * call dialect-specific methods accordingly. Upon exiting the scope.
 *
 * \code
 *
 * PrimFunc ConstructPrimFunc() {
 *   using tvm::script::ir_builder::IRBuilder;
 *   using T = tvm::script::ir_builder::tir;
 *   IRBuilder builder;
 *   // Step 1. Place IRBuilder inside the with-scope.
 *   {
 *     With<IRBuilder> _(builder);
 *     // Step 2. Call dialect-specific methods.
 *     With<T::PrimFuncFrame> _2(...);
 *     T::MatchBuffer(...);
 *   }
 *   // Step 3. Return the constructed PrimFunc.
 *   return builder->Get<PrimFunc>();
 * }
 *
 * \endcode
 */
class IRBuilderNode : public runtime::Object {
 public:
  /*! \brief A stack of context frames in the IRBuilder */
  runtime::Array<IRBuilderFrame> frames;
  /*! \brief The outcome of IR construction */
  Optional<ObjectRef> result;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("frames", &frames);
    v->Visit("result", &result);
  }

  static constexpr const char* _type_key = "script.ir_builder.IRBuilder";
  TVM_DECLARE_FINAL_OBJECT_INFO(IRBuilderNode, runtime::Object);

 public:
  /*!
   * \brief Find a frame of the given type in the stack `this->frames` from top to bottom.
   * \tparam T The type of the frame to find.
   * \return The frame if found, otherwise NullOpt.
   */
  template <typename TFrame>
  inline Optional<TFrame> FindFrame() const;
  /*!
   * \brief Get the frame on top of the stack `this->frames` if its type is `TFrame`.
   * \tparam TFrame The assumed type of the last frame on stack.
   * \return The frame if the stack is non-empty and the top of the stack is of type `TFrame`.
   * Otherwise NullOpt.
   */
  template <typename TFrame>
  inline Optional<TFrame> GetLastFrame() const;
  /*!
   * \brief Get the IR being constructed.
   * \tparam TObjectRef The type of the IR being constructed.
   * \return The resulting IR. Throw an exception if the IR is not constructed yet.
   */
  template <typename TObjectRef>
  inline TObjectRef Get() const;
};

/*!
 * \brief Managed reference to an IRBuilderNode.
 * \sa IRBuilderNode
 */
class IRBuilder : public runtime::ObjectRef {
 public:
  /*! \brief Creates an IRBuilder. */
  IRBuilder();
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(IRBuilder, ObjectRef, IRBuilderNode);

 public:
  /*!
   * \brief Puts the current IRBuilder into a thread-local scope, which can be retrieved using
   * `IRBuilder::Current()`.
   *
   * \code {.cpp}
   * IRBuilder builder;
   * {
   *   With<IRBuilder> _(builder);
   *   // IRBuilder::Current() == builder
   * }
   * // IRBuilder::Current() == nullptr
   * \endcode
   *
   * \sa IRBuilder::Current
   * \sa IRBuilder::ExitWithScope
   * \sa tvm::support::With
   */
  void EnterWithScope();
  /*!
   * \brief Exit the RAII scope.
   * \sa IRBuilder::EnterWithScope
   * \sa IRBuilder::Current
   * \sa tvm::support::With
   */
  void ExitWithScope();
  /*!
   * \brief Get the current IRBuilder in the current thread-local scope.
   * \return The current IRBuilder.
   * \sa IRBuilder::EnterWithScope
   * \sa IRBuilder::ExitWithScope
   * \sa tvm::support::With
   */
  static IRBuilder Current();
  /*! \brief See if the current thread-local scope has an IRBuilder. */
  static bool IsInScope();
  /*!
   * \brief Give a string name to the `obj`
   * \tparam TObjectRef The type of the object to name.
   * \param name The name to give to the object.
   * \param obj The object to name.
   */
  template <class TObjectRef>
  inline static TObjectRef Name(String name, TObjectRef obj);
};

////////////////////////////// Details //////////////////////////////

namespace details {

class Namer {
 public:
  using FType = NodeFunctor<void(const ObjectRef&, String)>;
  static FType& vtable();
  static void Name(ObjectRef node, String name);
};

}  // namespace details

template <class TObjectRef>
inline TObjectRef IRBuilder::Name(String name, TObjectRef obj) {
  details::Namer::Name(obj, name);
  return Downcast<TObjectRef>(obj);
}

template <typename TFrame>
inline Optional<TFrame> IRBuilderNode::FindFrame() const {
  using TFrameNode = typename TFrame::ContainerType;
  for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
    if (const TFrameNode* p = (*it).template as<TFrameNode>()) {
      return GetRef<TFrame>(p);
    }
  }
  return NullOpt;
}

template <typename TFrame>
inline Optional<TFrame> IRBuilderNode::GetLastFrame() const {
  using TFrameNode = typename TFrame::ContainerType;
  if (!frames.empty() && frames.back()->IsInstance<TFrameNode>()) {
    return Downcast<TFrame>(frames.back());
  }
  return NullOpt;
}

template <typename TObjectRef>
inline TObjectRef IRBuilderNode::Get() const {
  using TObject = typename TObjectRef::ContainerType;
  CHECK(result.defined()) << "IndexError: No result exists in IRBuilder yet";
  const auto* n = result.as<TObject>();
  CHECK(n != nullptr) << "TypeError: IRBuilder result is not of type: " << TObject::_type_key;
  return GetRef<TObjectRef>(n);
}

}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_IR_BUILDER_BASE_H_
