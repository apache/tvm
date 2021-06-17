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

#ifndef TVM_RUNTIME_THREAD_MAP_H_
#define TVM_RUNTIME_THREAD_MAP_H_

#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace runtime {

/*! \brief Container to hold one value per thread
 *
 * Similar to thread_local, but intended for use as a non-static or
 * non-block variable, such as class member variables.  All member
 * functions are thread-safe to call.  If only the current thread's
 * value is accessed, no additional synchronization is required.  If
 * another thread's stored values are accessed, external
 * synchronization may be required.
 *
 * Calls that only require access to already-existing values will not
 * block each other.  Calls that require constructing a new value will
 * block any other calls.
 *
 * \tparam T The object type to be held.  For instantiation of
 * ThreadMap<T> and for calls to ThreadMap<T>::Get, only a forward
 * declaration is required.  For calls to ThreadMap<T>::GetOrMake, a
 * full class definition is required.
 */
template <typename T>
class ThreadMap {
 public:
  ThreadMap() {}

  /*! \brief Return the current thread's stored object, if it exists.
   *
   * \return If it exists, a pointer to the stored object.  Otherwise,
   * returns nullptr.
   */
  const T* Get() const { return this->Get(std::this_thread::get_id()); }

  /*! \brief Return the stored object for a given thread, if it exists.
   *
   * \param id The thread whose object should be returned.
   *
   * \return If it exists, a pointer to the stored object.  Otherwise,
   * returns nullptr.
   */
  const T* Get(std::thread::id id) const {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    auto res = values_.find(id);
    if (res == values_.end()) {
      return nullptr;
    } else {
      return res->second.get();
    }
  }

  /*! \brief Return the current thread's stored object, if it exists.
   *
   * \return If it exists, a pointer to the stored object.  Otherwise,
   * returns nullptr.
   */
  T* Get() { return const_cast<T*>(const_cast<const ThreadMap<T>*>(this)->Get()); }

  /*! \brief Return the stored object for a given thread, if it exists.
   *
   * \param id The thread whose object should be returned.
   *
   * \return If it exists, a pointer to the stored object.  Otherwise,
   * returns nullptr.
   */
  T* Get(std::thread::id id) {
    return const_cast<T*>(const_cast<const ThreadMap<T>*>(this)->Get(id));
  }

  /*! \brief Return the current thread's stored object, making it if
   * necessary.
   *
   * Since this method can modify the stored map, there is no
   * non-const version available.
   *
   * \tparam Params Types of the stored object's constructor arguments
   *
   * \return A reference to the stored object
   */
  template <typename... Params>
  T& GetOrMake(Params&&... params) {
    return GetOrMake(std::this_thread::get_id(), std::forward<Params>(params)...);
  }

  /*! \brief Return the stored object for a given thread, making it if
   * necessary
   *
   * Since this method can modify the stored map, there is no
   * non-const version available.
   *
   * \tparam Params Types of the stored object's constructor arguments
   *
   * \param id The thread whose object should be returned.
   *
   * \param params Arguments to the stored object's constructor.  Only
   * used if the specified thread does not currently exist in the map.
   *
   * \return A reference to the stored object
   */
  template <typename... Params>
  T& GetOrMake(std::thread::id id, Params&&... params) {
    // Try to get stored value first, which would only require shared
    // access.
    if (T* output = Get(id)) {
      return *output;
    }

    // Not in map, need exclusive lock to write
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);

    // Check again, in case another thread got the unique lock first
    // and already constructed the object.
    auto res = values_.find(id);
    if (res != values_.end()) {
      return *res->second;
    }

    // No value exists, make one and return it.
    std::unique_ptr<T>& new_val = values_[id] =
        std::make_unique<T>(std::forward<Params>(params)...);
    return *new_val;
  }

  /*! \brief Clears all values held by the ThreadMap
   *
   * Calling Clear() invalidates any pointers/references previously
   * returned by Get/GetOrMake.
   *
   */
  void Clear() {
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    values_.clear();
  }

 private:
  //! \brief Mutex to protect values_
  mutable std::shared_timed_mutex mutex_;

  //! \brief Map containing stored values
  std::unordered_map<std::thread::id, std::unique_ptr<T>> values_;
};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_THREAD_MAP_H_
