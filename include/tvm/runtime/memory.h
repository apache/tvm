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
 * \file tvm/runtime/memory.h
 * \brief Runtime memory management.
 */
#ifndef TVM_RUNTIME_MEMORY_H_
#define TVM_RUNTIME_MEMORY_H_

#include <utility>
#include <type_traits>
#include "object.h"

namespace tvm {
namespace runtime {
/*!
 * \brief Allocate an object using default allocator.
 * \param args arguments to the constructor.
 * \tparam T the node type.
 * \return The NodePtr to the allocated object.
 */
template<typename T, typename... Args>
inline ObjectPtr<T> make_object(Args&&... args);

// Detail implementations after this
//
// The current design allows swapping the
// allocator pattern when necessary.
//
// Possible future allocator optimizations:
// - Arena allocator that gives ownership of memory to arena (deleter_= nullptr)
// - Thread-local object pools: one pool per size and alignment requirement.
// - Can specialize by type of object to give the specific allocator to each object.

/*!
 * \brief Base class of object allocators that implements make.
 *  Use curiously recurring template pattern.
 *
 * \tparam Derived The derived class.
 */
template<typename Derived>
class ObjAllocatorBase {
 public:
  /*!
   * \tparam T The type to be allocated.
   * \tparam Args The constructor signature.
   * \param args The arguments.
   */
  template<typename T, typename... Args>
  inline ObjectPtr<T> make(Args&&... args) {
    using Handler = typename Derived::template Handler<T>;
    static_assert(std::is_base_of<Object, T>::value,
                  "make_node can only be used to create NodeBase");
    T* ptr = Handler::New(static_cast<Derived*>(this),
                         std::forward<Args>(args)...);
    ptr->type_index_ = T::RuntimeTypeIndex();
    ptr->deleter_ = Handler::Deleter();
    return ObjectPtr<T>(ptr);
  }
};

// Simple allocator that uses new/delete.
class SimpleObjAllocator :
      public ObjAllocatorBase<SimpleObjAllocator> {
 public:
  template<typename T>
  class Handler {
   public:
    template<typename... Args>
    static T* New(SimpleObjAllocator*, Args&&... args) {
      // NOTE: the first argument is not needed for SimpleObjAllocator
      // It is reserved for special allocators that needs to recycle
      // the object to itself (e.g. in the case of object pool).
      //
      // In the case of an object pool, an allocator needs to create
      // a special chunk memory that hides reference to the allocator
      // and call allocator's release function in the deleter.
      return new T(std::forward<Args>(args)...);
    }

    static Object::FDeleter Deleter() {
      return Deleter_;
    }

   private:
    static void Deleter_(Object* ptr) {
      delete static_cast<T*>(ptr);
    }
  };
};

template<typename T, typename... Args>
inline ObjectPtr<T> make_object(Args&&... args) {
  return SimpleObjAllocator().make<T>(std::forward<Args>(args)...);
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MEMORY_H_
