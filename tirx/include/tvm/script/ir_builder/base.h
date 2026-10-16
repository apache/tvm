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

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/function.h>
#include <tvm/ir/object_functor.h>

#include <cstdint>
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
 * Frames capture construction context and nested scopes. Language variant builders define
 * concrete frame subclasses and decide how their contents contribute to the result.
 */
class TVM_TIRX_DLL IRBuilderFrameObj : public ffi::Object {
 public:
  /*! \brief Capture the source-call context when this frame is constructed. */
  IRBuilderFrameObj();
  /*! \brief A list of callbacks used when exiting the frame. */
  std::vector<ffi::TypedFunction<void()>> callbacks;
  /*! \brief Source context retained until this frame constructs its result. */
  mutable Location loc = UnknownLoc();

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IRBuilderFrameObj>().def_ro("loc", &IRBuilderFrameObj::loc);
    // `callbacks` is not registered as it's not visited.
  }

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("script.ir_builder.IRBuilderFrame", IRBuilderFrameObj, ffi::Object);

 public:
  /*! \brief Default destructor. */
  virtual ~IRBuilderFrameObj() = default;
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
  void AddCallback(ffi::TypedFunction<void()> callback);
};

/*!
 * \brief Managed reference to an IRBuilderFrameObj.
 * \sa IRBuilderFrameObj
 */
class IRBuilderFrame : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IRBuilderFrame, ffi::ObjectRef, IRBuilderFrameObj);

 protected:
  /*! \brief Disallow direct construction of this object. */
  IRBuilderFrame() = default;
  explicit IRBuilderFrame(ffi::ObjectPtr<IRBuilderFrameObj> data) : ffi::ObjectRef(data) {}

 public:
  /*!
   * \brief Redirected to `IRBuilderFrameObj::EnterWithScope`.
   * \sa IRBuilderFrameObj::EnterWithScope
   */
  inline void EnterWithScope() {
    TVM_FFI_ICHECK(data_ != nullptr);
    static_cast<IRBuilderFrameObj*>(data_.get())->EnterWithScope();
  }
  /*!
   * \brief Redirected to `IRBuilderFrameObj::ExitWithScope`.
   * \sa IRBuilderFrameObj::ExitWithScope
   */
  inline void ExitWithScope() {
    TVM_FFI_ICHECK(data_ != nullptr);
    static_cast<IRBuilderFrameObj*>(data_.get())->ExitWithScope();
    data_.reset();
  }
};

////////////////////////////// IRBuilder //////////////////////////////

/*!
 * \brief A language variant agnostic IRBuilder that constructs any IR of TVM.
 * An idiomatic use of this class is to put this inside the RAII with-scope,
 * call language variant specific methods accordingly. Upon exiting the scope.
 *
 * \code
 *
 * IRModule ConstructModule() {
 *   using namespace tvm::script::ir_builder;
 *   IRBuilder builder;
 *   {
 *     With<IRBuilder> scope(builder);
 *     With<ir::IRModuleFrame> module(ir::IRModule());
 *     // Add function declarations and definitions with the chosen language variant builder.
 *   }
 *   return builder->Get<IRModule>();
 * }
 *
 * \endcode
 */
class IRBuilderObj : public ffi::Object {
 public:
  /*! \brief A stack of context frames in the IRBuilder */
  ffi::Array<IRBuilderFrame> frames;
  /*! \brief The outcome of IR construction */
  ffi::Optional<ffi::ObjectRef> result;
  /*! \brief Active frontend source locations, from outermost to innermost. */
  std::vector<Location> locs;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IRBuilderObj>()
        .def_ro("frames", &IRBuilderObj::frames)
        .def_ro("result", &IRBuilderObj::result);
  }

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.IRBuilder", IRBuilderObj, ffi::Object);

 public:
  /*!
   * \brief Find a frame of the given type in the stack `this->frames` from top to bottom.
   * \tparam T The type of the frame to find.
   * \return The frame if found, otherwise std::nullopt.
   */
  template <typename TFrame>
  inline ffi::Optional<TFrame> FindFrame() const;
  /*!
   * \brief Get the top frame if its type is `TFrame`.
   * \tparam TFrame The assumed type of the last frame on stack.
   * \return The frame if the stack is non-empty and the top of the stack is of type `TFrame`.
   * Otherwise std::nullopt.
   */
  template <typename TFrame>
  inline ffi::Optional<TFrame> GetLastFrame() const;
  /*!
   * \brief Get the IR being constructed.
   * \tparam TObjectRef The type of the IR being constructed.
   * \return The resulting IR. Throw an exception if the IR is not constructed yet.
   */
  template <typename TObjectRef>
  inline TObjectRef Get() const;
  /*!
   * \brief Push a frontend source location for IR constructed in the nested scope.
   * \return The index of the pushed scope, valid until it is popped.
   */
  TVM_TIRX_DLL int64_t PushLoc(Location loc);
  /*!
   * \brief Update a source location without changing its nested scopes.
   * \param index The index returned by PushLoc for a scope that has not been popped.
   * \param loc The replacement location for that scope.
   */
  TVM_TIRX_DLL void UpdateLoc(int64_t index, Location loc);
  /*! \brief Pop the innermost frontend source location. */
  TVM_TIRX_DLL void PopLoc();
  /*! \brief Normalize the full active source stack with an optional explicit location. */
  TVM_TIRX_DLL Location ResolveLoc(Location location = UnknownLoc()) const;
  /*! \brief Compose active and explicit source context onto the same supported node or frame. */
  TVM_TIRX_DLL ffi::ObjectRef AttachLoc(ffi::ObjectRef obj, Location loc = UnknownLoc()) const;
};

/*!
 * \brief Managed reference to an IRBuilderObj.
 * \sa IRBuilderObj
 */
class IRBuilder : public ffi::ObjectRef {
 public:
  /*! \brief Creates an IRBuilder. */
  TVM_TIRX_DLL IRBuilder();
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IRBuilder, ffi::ObjectRef, IRBuilderObj);

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
  TVM_TIRX_DLL void EnterWithScope();
  /*!
   * \brief Exit the RAII scope.
   * \sa IRBuilder::EnterWithScope
   * \sa IRBuilder::Current
   * \sa tvm::support::With
   */
  TVM_TIRX_DLL void ExitWithScope();
  /*!
   * \brief Get the current IRBuilder in the current thread-local scope.
   * \return The current IRBuilder.
   * \sa IRBuilder::EnterWithScope
   * \sa IRBuilder::ExitWithScope
   * \sa tvm::support::With
   */
  TVM_TIRX_DLL static IRBuilder Current();
  /*! \brief See if the current thread-local scope has an IRBuilder. */
  TVM_TIRX_DLL static bool IsInScope();
  /*!
   * \brief Give a string name to the `obj`
   * \tparam TObjectRef The type of the object to name.
   * \param name The name to give to the object.
   * \param obj The object to name.
   */
  template <class TObjectRef>
  inline static TObjectRef Name(ffi::String name, TObjectRef obj);
};

////////////////////////////// Details //////////////////////////////

namespace details {

/*! \brief Language variant extensible access to an object's mutable source location. */
class LocationAccessor {
 public:
  using FType = ObjectFunctor<Location*(const ffi::ObjectRef&)>;
  TVM_TIRX_DLL static FType& vtable();
};

class Namer {
 public:
  using FType = ObjectFunctor<void(const ffi::ObjectRef&, ffi::String)>;
  TVM_TIRX_DLL static FType& vtable();
  TVM_TIRX_DLL static void Name(ffi::ObjectRef node, ffi::String name);
};

}  // namespace details

template <class TObjectRef>
inline TObjectRef IRBuilder::Name(ffi::String name, TObjectRef obj) {
  details::Namer::Name(obj, name);
  return obj.template as_or_throw<TObjectRef>();
}

template <typename TFrame>
inline ffi::Optional<TFrame> IRBuilderObj::FindFrame() const {
  using TFrameNode = typename TFrame::ContainerType;
  for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
    if (const TFrameNode* p = (*it).template as<TFrameNode>()) {
      return ffi::GetRef<TFrame>(p);
    }
  }
  return std::nullopt;
}

template <typename TFrame>
inline ffi::Optional<TFrame> IRBuilderObj::GetLastFrame() const {
  using TFrameNode = typename TFrame::ContainerType;
  if (!frames.empty() && frames.back()->IsInstance<TFrameNode>()) {
    return frames.back().template as_or_throw<TFrame>();
  }
  return std::nullopt;
}

template <typename TObjectRef>
inline TObjectRef IRBuilderObj::Get() const {
  using TObject = typename TObjectRef::ContainerType;
  TVM_FFI_CHECK(result.has_value(), IndexError) << "No result exists in IRBuilder yet";
  const auto* n = result.as<TObject>();
  TVM_FFI_CHECK(n != nullptr, TypeError)
      << "IRBuilder result is not of type: " << TObject::_type_key;
  return ffi::GetRef<TObjectRef>(n);
}

}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_IR_BUILDER_BASE_H_
