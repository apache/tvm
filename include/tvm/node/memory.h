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
 * \file tvm/node/memory.h
 * \brief Node memory management.
 */
#ifndef TVM_NODE_MEMORY_H_
#define TVM_NODE_MEMORY_H_

#include <utility>
#include "node.h"

namespace tvm {
/*!
 * \brief Allocate a node object.
 * \param args arguments to the constructor.
 * \tparam T the node type.
 * \return The NodePtr to the allocated object.
 */
template<typename T, typename... Args>
inline NodePtr<T> make_node(Args&&... args);

// Detail implementations after this
//
// The current design allows swapping the
// allocator pattern when necessary.
//
// Possible future allocator optimizations:
// - Arena allocator that gives ownership of memory to arena (deleter_= nullptr)
// - Thread-local object pools: one pool per size and alignment requirement.
// - Can specialize by type of object to give the specific allocator to each object.
//
template<typename T>
class SimpleNodeAllocator {
 public:
  template<typename... Args>
  static T* New(Args&&... args) {
    return new T(std::forward<Args>(args)...);
  }
  static NodeBase::FDeleter Deleter() {
    return Deleter_;
  }

 private:
  static void Deleter_(NodeBase* ptr) {
    delete static_cast<T*>(ptr);
  }
};

template<typename T, typename... Args>
inline NodePtr<T> make_node(Args&&... args) {
  using Allocator = SimpleNodeAllocator<T>;
  static_assert(std::is_base_of<NodeBase, T>::value,
                "make_node can only be used to create NodeBase");
  T* node = Allocator::New(std::forward<Args>(args)...);
  node->deleter_ = Allocator::Deleter();
  return NodePtr<T>(node);
}

}  // namespace tvm
#endif  // TVM_NODE_MEMORY_H_
