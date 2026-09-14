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
 * \file tvm/ir/object_functor.h
 * \brief Native object dispatch, visiting and mutation with structural fallback.
 */
#ifndef TVM_IR_OBJECT_FUNCTOR_H_
#define TVM_IR_OBJECT_FUNCTOR_H_

#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/runtime/base.h>

#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

namespace tvm {

/*!
 * \brief A dynamically dispatched functor on the type of the first argument.
 *
 * Dispatch is based on the runtime type of the first object argument.
 *
 * \code
 *   ObjectFunctor<std::string(const ffi::ObjectRef& n, std::string prefix)> tostr;
 *   tostr.SetDispatch<prim::AddNode>([](const ffi::ObjectRef& op, std::string prefix) {
 *     return prefix + "Add";
 *   });
 *   tostr.SetDispatch<IntImmNode>([](const ffi::ObjectRef& op, std::string prefix) {
 *     return prefix + "IntImm";
 *   });
 *
 *   tirx::PrimVar x("x");
 *   PrimExpr y = x + 1;
 *   // dispatch to IntImm, outputs "MyIntImm"
 *   LOG(INFO) << tostr(IntImm::Int32(1), "My");
 *   // dispatch to Add, outputs "MyAdd"
 *   LOG(INFO) << tostr(y, "My");
 * \endcode
 *
 * A shared table can register dispatch functions during static initialization:
 *
 * \code
 *   class Printer {
 *    public:
 *     using FType = ObjectFunctor<std::string(const ffi::ObjectRef&)>;
 *     static FType& vtable();
 *   };
 *
 *   Printer::FType& Printer::vtable() {
 *     static FType inst;
 *     return inst;
 *   }
 *
 *   TVM_FFI_STATIC_INIT_BLOCK() {
 *     Printer::vtable()
 *         .SetDispatch<prim::AddNode>([](const ffi::ObjectRef&) { return std::string("Add"); })
 *         .SetDispatch<IntImmNode>([](const ffi::ObjectRef&) { return std::string("IntImm"); });
 *   }
 * \endcode
 *
 * \tparam FType Function signature, with const ffi::ObjectRef& or const ffi::Object* as its first
 * argument.
 */
template <typename FType>
class ObjectFunctor;

template <typename R, typename NodeArg, typename... Args>
class ObjectFunctor<R(NodeArg, Args...)> {
  static_assert(std::is_same_v<NodeArg, const ffi::ObjectRef&> ||
                    std::is_same_v<NodeArg, const ffi::Object*>,
                "ObjectFunctor requires an ObjectRef reference or borrowed Object pointer");

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*!
   * \brief Whether a dispatch function is registered for the exact runtime type.
   * \param n The borrowed NodeArg or AnyView value to be dispatched.
   * \tparam T The input representation, convertible to NodeArg or AnyView.
   * \return Whether a dispatch function is registered for n's type, excluding ancestors.
   */
  template <typename T>
  TVM_FFI_INLINE bool CanDispatch(const T& n) const {
    uint32_t type_index = GetTypeIndex(n);
    if (type_index < begin_type_index_) return false;
    type_index -= begin_type_index_;
    return type_index < func_.size() && func_[type_index] != nullptr;
  }
  /*!
   * \brief invoke the functor, dispatch on type of n
   * \param n The object argument
   * \param args The additional arguments
   * \return The result.
   */
  TVM_FFI_INLINE R operator()(NodeArg n, Args... args) const {
    uint32_t type_index = n->type_index();
    if (type_index >= begin_type_index_) {
      uint32_t index = type_index - begin_type_index_;
      if (index < func_.size() && func_[index] != nullptr) {
        return (*func_[index])(n, std::forward<Args>(args)...);
      }
    }

    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
    for (int32_t i = type_info->type_depth - 1; i >= 0; --i) {
      type_index = type_info->type_ancestors[i]->type_index;
      if (type_index >= begin_type_index_) {
        uint32_t index = type_index - begin_type_index_;
        if (index < func_.size() && func_[index] != nullptr) {
          return (*func_[index])(n, std::forward<Args>(args)...);
        }
      }
    }
    ThrowUnregistered(n);
  }
  /*!
   * \brief set the dispatcher for type TNode
   * \param f The function to be set.
   * \tparam TNode the type of Node to be dispatched.
   * \return reference to self.
   */
  template <typename TNode>
  ObjectFunctor& SetDispatch(R (*f)(NodeArg n, Args...)) {
    static_assert(std::is_base_of_v<ffi::Object, TNode>,
                  "ObjectFunctor dispatch requires an object node type");
    uint32_t tindex = TNode::RuntimeTypeIndex();
    if (func_.size() <= tindex) {
      func_.resize(tindex + 1, nullptr);
    }
    TVM_FFI_ICHECK(func_[tindex] == nullptr)
        << "Dispatch for " << TNode::_type_key << " is already set";
    TVM_FFI_ICHECK_EQ(begin_type_index_, 0) << " Cannot call SetDispatch after calling Finalize";
    func_[tindex] = f;
    return *this;
  }
  /*!
   * \brief unset the dispatcher for type TNode
   *
   * \tparam TNode the type of Node to be dispatched.
   * \return reference to self.
   */
  template <typename TNode>
  ObjectFunctor& ClearDispatch() {
    uint32_t tindex = TNode::RuntimeTypeIndex();
    TVM_FFI_ICHECK_LT(tindex, func_.size()) << "ClearDispatch: index out of range";
    TVM_FFI_ICHECK_EQ(begin_type_index_, 0) << " Cannot call ClearDispatch after calling Finalize";
    func_[tindex] = nullptr;
    return *this;
  }
  /*!
   * \brief Finalize the functor after calling sequence of SetDispatch
   * This function will attempt to find the min type index that is not null
   * and optimize the space of the func table so it is more compact
   */
  void Finalize() {
    TVM_FFI_ICHECK_EQ(begin_type_index_, 0) << "Can only call Finalize once";
    while (begin_type_index_ < func_.size() && func_[begin_type_index_] == nullptr) {
      ++begin_type_index_;
    }
    // shift up the function value
    size_t new_ftable_size = func_.size() - begin_type_index_;
    if (begin_type_index_ != 0) {
      std::memmove(func_.data(), func_.data() + begin_type_index_,
                   new_ftable_size * sizeof(FPointer));
    }
    func_.resize(new_ftable_size);
    func_.shrink_to_fit();
  }

 private:
  TVM_FFI_INLINE static uint32_t GetTypeIndex(NodeArg n) {
    if constexpr (std::is_pointer_v<NodeArg>) {
      return n != nullptr ? n->type_index() : ffi::TypeIndex::kTVMFFINone;
    } else {
      return n.defined() ? n->type_index() : ffi::TypeIndex::kTVMFFINone;
    }
  }
  TVM_FFI_INLINE static uint32_t GetTypeIndex(ffi::AnyView n) { return n.type_index(); }

  [[noreturn]] TVM_FFI_COLD_CODE static void ThrowUnregistered(NodeArg n) {
    TVM_FFI_THROW(InternalError) << "ObjectFunctor calls un-registered function on type "
                                 << n->GetTypeKey();
    throw;
  }

  /*! \brief internal function pointer type */
  using FPointer = R (*)(NodeArg n, Args...);
  /*! \brief internal function table */
  std::vector<FPointer> func_;
  /*! \brief start range of func index */
  uint32_t begin_type_index_{0};
};

using ffi::Expected;
using ffi::UnchangedOr;
using ffi::VisitInterrupt;

/*!
 * \brief Native visiting with exact dispatch and structural fallback.
 *
 * Allocate visitors with ffi::make_object<Derived>(). Inputs are borrowed;
 * interrupts and errors own their values. A hook controls descent into its node.
 * One visitor must not be shared by overlapping traversals.
 */
class TVM_DLL ObjectVisitor : public ffi::StructuralVisitorObj {
 public:
  /*! \brief Construct a visitor using structural fallback for every value. */
  ObjectVisitor() : ObjectVisitor(GlobalVTable()) {}
  /*! \brief Release the managed visitor state. */
  ~ObjectVisitor() = default;
  ObjectVisitor(const ObjectVisitor& other) = delete;
  ObjectVisitor& operator=(const ObjectVisitor& other) = delete;

  /*!
   * \brief Visit a borrowed object or inline value.
   * \param value The borrowed value to visit.
   * \return None on completion, an owning VisitInterrupt, or an Error.
   * \note Overrides must convert thrown errors and propagate errors and interrupts from their
   *       own work. A qualified Parent::VisitExpected(value) bypasses the current entry override;
   *       descendant calls still use virtual dispatch.
   */
  virtual Expected<ffi::Optional<VisitInterrupt>> VisitExpected(ffi::AnyView value) noexcept {
    if (native_vtable_->CanDispatch(value)) {
      // Exact registrations contain only object node types, so this extraction is borrowed
      // and needs no additional object-type check or owning reference.
      const auto* object =
          ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ffi::Object>(value);
      try {
        return DispatchNative(object);
      } catch (ffi::Error& error) {
        return AttachVisitErrorContext(error, object);
      }
    }
    return StructuralVisitDefault(value);
  }

 protected:
  /*! \brief Exact native dispatch table with owning interrupt and error results. */
  using VTable =
      ObjectFunctor<Expected<ffi::Optional<VisitInterrupt>>(const ffi::Object*, ObjectVisitor*)>;

  /*!
   * \brief Construct a visitor with a finalized native dispatch table.
   * \param vtable The immutable table, which must outlive this visitor.
   */
  explicit ObjectVisitor(const VTable* vtable)
      : ffi::StructuralVisitorObj(StructuralVTable()), native_vtable_(vtable) {}
  /*!
   * \brief Initialize the base registrations in a fresh mutable table.
   * \param vtable The table to initialize before adding derived registrations.
   */
  static void InitVTable(VTable* vtable) {}

  /*!
   * \brief Register an exact node type in a fresh table initialized by its parent.
   * Derived node types require separate registrations.
   * \tparam Self The visitor implementing the hook.
   * \tparam Node The exact node type handled by the hook.
   * \param vtable The mutable table to receive the registration.
   */
  template <typename Self, typename Node>
  static void SetDispatch(VTable* vtable) {
    vtable->template SetDispatch<Node>(DispatchNode<Self, Node>);
  }

 private:
  // Native table callback; core instantiations may be shared by the library.
  template <typename Self, typename Node>
  static Expected<ffi::Optional<VisitInterrupt>> DispatchNode(const ffi::Object* node,
                                                              ObjectVisitor* self) {
    return static_cast<Self*>(self)->Visit_(static_cast<const Node*>(node));
  }

  // Keep one named return value in this scope so native results can be constructed in place.
  TVM_FFI_INLINE Expected<ffi::Optional<VisitInterrupt>> DispatchNative(const ffi::Object* value) {
    Expected<ffi::Optional<VisitInterrupt>> result = (*native_vtable_)(value, this);
    if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
      UpdateVisitErrorContext(result, value);
    }
    return result;
  }
  Expected<ffi::Optional<VisitInterrupt>> StructuralVisitDefault(ffi::AnyView value) noexcept {
    TVMFFIAny result = ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(
        ffi::StructuralVisitorObj::DefaultVisitExpected(value));
    if (TVM_FFI_PREDICT_FALSE(result.type_index == ffi::TypeIndex::kTVMFFIError)) {
      result = ffi::details::AttachStructuralVisitErrorContextRaw(result, value);
    }
    return ffi::details::ExpectedUnsafe::MoveFromTVMFFIAny<ffi::Optional<VisitInterrupt>>(result);
  }
  TVM_FFI_COLD_CODE static Expected<ffi::Optional<VisitInterrupt>> AttachVisitErrorContext(
      ffi::Error& error, const ffi::Object* value) {
    if (value) ffi::details::UpdateVisitErrorContext(error, ffi::GetRef<ffi::ObjectRef>(value));
    return ffi::Unexpected(std::move(error));
  }
  TVM_FFI_COLD_CODE static void UpdateVisitErrorContext(
      const Expected<ffi::Optional<VisitInterrupt>>& result, const ffi::Object* value) {
    ffi::Error error = result.error();
    if (value) ffi::details::UpdateVisitErrorContext(error, ffi::GetRef<ffi::ObjectRef>(value));
  }
  static const VTable* GlobalVTable() {
    static const VTable table = [] {
      VTable table;
      InitVTable(&table);
      table.Finalize();
      return table;
    }();
    return &table;
  }
  static const ffi::StructuralVisitorVTable* StructuralVTable() {
    static const ffi::StructuralVisitorVTable table{StructuralVTableVisitImpl};
    return &table;
  }
  static TVMFFIAny StructuralVTableVisitImpl(ffi::StructuralVisitorObj* self,
                                             ffi::AnyView value) noexcept {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(
        static_cast<ObjectVisitor*>(self)->VisitExpected(value));
  }

  const VTable* const native_vtable_;
};

/*!
 * \brief Native mutation with exact dispatch and structural fallback.
 *
 * Allocate mutators with ffi::make_object<Derived>(). Inputs are borrowed;
 * replacements and errors own their values. Overrides forward inplace_mode
 * to child calls, following the structural mutation ownership contract.
 */
class TVM_DLL ObjectMutator : public ffi::StructuralMapEngineBase {
 public:
  /*! \brief Construct a mutator using structural fallback for every object. */
  ObjectMutator() : ObjectMutator(GlobalVTable()) {}
  /*! \brief Release the managed mutator state. */
  ~ObjectMutator() = default;
  ObjectMutator(const ObjectMutator& other) = delete;
  ObjectMutator& operator=(const ObjectMutator& other) = delete;

  /*!
   * \brief Mutate a borrowed value, checking current uniqueness before in-place dispatch.
   * \param value The borrowed object or inline value to mutate.
   * \param inplace_mode Inherited permission along the path to this value.
   * \return A replacement, Unchanged, or an Error if mutation fails.
   * \note Overrides must establish current uniqueness before writing or forwarding permission,
   *       never promote kDisallow, and convert thrown errors from their own work into Expected.
   *       Expr inputs require Expr replacements. A qualified Parent::MutateExpected(value, mode)
   *       bypasses the current entry override while descendants remain virtual.
   *       DefaultMutateExpected trusts its caller's established mode and does not recheck it.
   */
  virtual Expected<UnchangedOr<ffi::Any>> MutateExpected(
      ffi::AnyView value, ffi::InplaceMode inplace_mode = ffi::InplaceMode::kDisallow) noexcept {
    if (native_vtable_->CanDispatch(value)) {
      // Only object node types can be registered. Exact dispatch proves this borrowed extraction
      // is valid without an additional object-type check or owning reference.
      const auto* object =
          ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ffi::Object>(value);
      if (inplace_mode == ffi::InplaceMode::kAllow && !object->unique()) {
        inplace_mode = ffi::InplaceMode::kDisallow;
      }
      try {
        return DispatchNative(object, inplace_mode);
      } catch (ffi::Error& error) {
        return AttachVisitErrorContext(error, object);
      }
    }
    if (inplace_mode == ffi::InplaceMode::kAllow) {
      const auto* object = value.as<ffi::Object>();
      if (object == nullptr || !object->unique()) inplace_mode = ffi::InplaceMode::kDisallow;
    }
    return ffi::StructuralMutatorObj::DefaultMutateExpected(value, inplace_mode);
  }

  /*!
   * \brief Look up a replacement in the variable-remap environment.
   * \param var The borrowed variable identity to look up.
   * \return The owning replacement, None on a miss, or an Error.
   */
  TVM_FFI_INLINE Expected<ffi::Any> VarRemapGetExpected(ffi::AnyView var) noexcept {
    return VarRemapGetImpl(var);
  }
  /*!
   * \brief Record a replacement in the variable-remap environment.
   * \param var The borrowed variable identity to bind.
   * \param mapped_value The borrowed replacement value.
   * \return Successful completion or an Error.
   */
  TVM_FFI_INLINE Expected<void> VarRemapSetExpected(ffi::AnyView var,
                                                    ffi::AnyView mapped_value) noexcept {
    return VarRemapSetImpl(var, mapped_value);
  }

 protected:
  /*! \brief Exact native dispatch table with owning replacement and error results. */
  using VTable = ObjectFunctor<Expected<UnchangedOr<ffi::Any>>(const ffi::Object*, ObjectMutator*,
                                                               ffi::InplaceMode)>;

  /*!
   * \brief Construct a mutator with a finalized native dispatch table.
   * \param vtable The immutable table, which must outlive this mutator.
   */
  explicit ObjectMutator(const VTable* vtable)
      : ffi::StructuralMapEngineBase(StructuralVTable()), native_vtable_(vtable) {}
  /*!
   * \brief Initialize the base registrations in a fresh mutable table.
   * \param vtable The table to initialize before adding derived registrations.
   */
  static void InitVTable(VTable* vtable) {}

  /*!
   * \brief Register an exact node type in a fresh table initialized by its parent.
   * Derived node types require separate registrations.
   * \tparam Self The mutator implementing the hook.
   * \tparam Node The exact node type handled by the hook.
   * \param vtable The mutable table to receive the registration.
   */
  template <typename Self, typename Node>
  static void SetDispatch(VTable* vtable) {
    vtable->template SetDispatch<Node>(DispatchNode<Self, Node>);
  }

 private:
  // Native table callback; core instantiations may be shared by the library.
  template <typename Self, typename Node>
  static Expected<UnchangedOr<ffi::Any>> DispatchNode(const ffi::Object* node, ObjectMutator* self,
                                                      ffi::InplaceMode inplace_mode) {
    return static_cast<Self*>(self)->Mutate_(static_cast<const Node*>(node), inplace_mode);
  }

  using ffi::StructuralMutatorObj::DefaultMutateExpected;
  using ffi::StructuralMutatorObj::Mutate;

  // Keep one named return value in this scope so native results can be constructed in place.
  TVM_FFI_INLINE Expected<UnchangedOr<ffi::Any>> DispatchNative(const ffi::Object* value,
                                                                ffi::InplaceMode inplace_mode) {
    Expected<UnchangedOr<ffi::Any>> result = (*native_vtable_)(value, this, inplace_mode);
    if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
      UpdateVisitErrorContext(result, value);
    }
    return result;
  }
  TVM_FFI_COLD_CODE static Expected<UnchangedOr<ffi::Any>> AttachVisitErrorContext(
      ffi::Error& error, const ffi::Object* value) {
    if (value) ffi::details::UpdateVisitErrorContext(error, ffi::GetRef<ffi::ObjectRef>(value));
    return ffi::Unexpected(std::move(error));
  }
  TVM_FFI_COLD_CODE static void UpdateVisitErrorContext(
      const Expected<UnchangedOr<ffi::Any>>& result, const ffi::Object* value) {
    ffi::Error error = result.error();
    if (value) ffi::details::UpdateVisitErrorContext(error, ffi::GetRef<ffi::ObjectRef>(value));
  }
  static const VTable* GlobalVTable() {
    static const VTable table = [] {
      VTable table;
      InitVTable(&table);
      table.Finalize();
      return table;
    }();
    return &table;
  }
  static const ffi::StructuralMutatorVTable* StructuralVTable() {
    static const ffi::StructuralMutatorVTable table{
        StructuralVTableMutateImpl, StructuralVTableMaybeInplaceMutateImpl,
        StructuralVTableVarRemapGetImpl, StructuralVTableVarRemapSetImpl};
    return &table;
  }
  static TVMFFIAny StructuralVTableMutateImpl(ffi::StructuralMutatorObj* self,
                                              ffi::AnyView value) noexcept {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(
        static_cast<ObjectMutator*>(self)->MutateExpected(value, ffi::InplaceMode::kDisallow));
  }
  static TVMFFIAny StructuralVTableMaybeInplaceMutateImpl(ffi::StructuralMutatorObj* self,
                                                          ffi::AnyView value) noexcept {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(
        static_cast<ObjectMutator*>(self)->MutateExpected(value, ffi::InplaceMode::kAllow));
  }
  static TVMFFIAny StructuralVTableVarRemapGetImpl(ffi::StructuralMutatorObj* self,
                                                   ffi::AnyView key) noexcept {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(
        static_cast<ObjectMutator*>(self)->VarRemapGetImpl(key));
  }
  static TVMFFIAny StructuralVTableVarRemapSetImpl(ffi::StructuralMutatorObj* self,
                                                   ffi::AnyView key, ffi::AnyView value) noexcept {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(
        static_cast<ObjectMutator*>(self)->VarRemapSetImpl(key, value));
  }

  const VTable* const native_vtable_;
};

/*!
 * \brief Define a default constructor backed by one initialized native dispatch table.
 * \param Class The class whose default constructor is defined.
 * \param Parent The parent accepting a const VTable pointer in its constructor.
 * \note Class must provide InitVTable(VTable*) including inherited registrations.
 * The table is initialized and finalized once; explicit table-taking constructors are separate.
 */
#define TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(Class, Parent) \
  Class()                                                            \
      : Parent([] {                                                  \
          static const VTable table = [] {                           \
            VTable table;                                            \
            Class::InitVTable(&table);                               \
            table.Finalize();                                        \
            return table;                                            \
          }();                                                       \
          return &table;                                             \
        }()) {}

}  // namespace tvm
#endif  // TVM_IR_OBJECT_FUNCTOR_H_
