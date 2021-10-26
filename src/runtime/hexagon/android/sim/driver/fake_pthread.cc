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

#include <cassert>
#include <cerrno>
#include <csetjmp>
#include <cstddef>
#include <cstdlib>
#include <map>
#include <vector>

#include "pthread.h"
#include "sched.h"

/*!
 * Implementation of a subset of pthread API for single-threaded execution.
 *
 * They main idea is that the thread function ("start_routine" in the call
 * to pthread_create) is executed immediately. When pthread_create returns,
 * the thread function has already finished.
 *
 * Since the thread routine can itself call pthread_create, it is possible
 * to have multiple threads existing at the same time, although only the
 * last one is running.
 *
 * There are two main things that need to be taken care of:
 * - thread-specific data, i.e. pthread_setspecific, pthread_getspecific,
 *   and the handling of thread keys,
 * - handling of thread return values.
 *
 * Threads are identified by thread ids (of type pthread_t). The main process
 * thread has the id of 0, the remaining threads have ids starting at 1 and
 * incrementing by 1. For each thread there is some data (thread_info_t)
 * associated with it, and stored in "thread_data" map. When a thread
 * terminates, the corresponding entry from "thread_data" cannot be removed
 * until the return value is claimed (pthread_join), unless it is explicitly
 * discarded (pthread_detach). When a new thread is created, it gets the
 * first available id for which there is no entry in "thread_data". This
 * could be an id that was never allocated, or an id that was used, but
 * has since been removed from the map.
 * A thread can terminate through thread_exit. This means that when the
 * thread function calls thread_exit, the execution should return to the
 * pthread_create call that ran it. This is implemented via setjmp/longjmp
 * (neither longjmp nor pthread_exit unwind the stack).
 *
 * Any mutexes or condition variables cannot block, or else it would cause
 * a deadlock. Since there is only one thread running at a time, locking
 * a mutex or waiting for a condition always succeeds (returns immediately).
 */

struct key_entry_t {
  key_entry_t(void* v, void (*d)(void*)) : value(v), dtor(d) {}
  void* value = nullptr;
  void (*dtor)(void*) = nullptr;
};

struct thread_info_t {
  thread_info_t() = default;
  std::map<pthread_key_t, key_entry_t> keys;
  std::jmp_buf env;
  void* ret_value = nullptr;
  bool finished = false;
  bool detached = false;
};

static pthread_t main_thread_id = 0;

static std::map<pthread_t, thread_info_t> thread_data = {
    // Reserve the 0th entry.
    {main_thread_id, {}}};

static std::vector<pthread_t> running_threads = {main_thread_id};

template <typename K, typename V>
K first_available_key(const std::map<K, V>& m) {
  auto i = m.begin(), e = m.end();
  K key = 1;
  for (; i != e && key == i->first; ++i, ++key) {
  }
  return key;
}

int pthread_cond_destroy(pthread_cond_t* cond) { return 0; }

int pthread_cond_init(pthread_cond_t* __restrict cond, const pthread_condattr_t* __restrict attr) {
  return 0;
}

int pthread_cond_signal(pthread_cond_t* cond) { return 0; }

int pthread_cond_broadcast(pthread_cond_t* cond) { return 0; }

int pthread_cond_timedwait(pthread_cond_t* __restrict cond, pthread_mutex_t* __restrict mutex,
                           const struct timespec* __restrict abstime) {
  return 0;
}

int pthread_cond_wait(pthread_cond_t* __restrict cond, pthread_mutex_t* __restrict mutex) {
  return 0;
}

int pthread_mutexattr_init(pthread_mutexattr_t* attr) { return 0; }

int pthread_mutexattr_destroy(pthread_mutexattr_t* attr) { return 0; }

int pthread_mutexattr_settype(pthread_mutexattr_t* attr, int type) { return 0; }

int pthread_mutexattr_gettype(const pthread_mutexattr_t* __restrict attr, int* __restrict type) {
  *type = PTHREAD_MUTEX_NORMAL;
  return 0;
}

int pthread_mutex_init(pthread_mutex_t* __restrict mutex,
                       const pthread_mutexattr_t* __restrict attr) {
  return 0;
}

int pthread_mutex_destroy(pthread_mutex_t* mutex) { return 0; }

int pthread_mutex_lock(pthread_mutex_t* mutex) { return 0; }

int pthread_mutex_trylock(pthread_mutex_t* mutex) { return 0; }

int pthread_mutex_unlock(pthread_mutex_t* mutex) { return 0; }

int pthread_once(pthread_once_t* once_control, void (*init_routine)(void)) {
  static_assert(PTHREAD_ONCE_INIT != PTHREAD_ONCE_DONE,
                "PTHREAD_ONCE_INIT must be different from PTHREAD_ONCE_DONE");
  if (*once_control == PTHREAD_ONCE_INIT) {
    init_routine();
    *once_control = PTHREAD_ONCE_DONE;
  }
  return 0;
}

int pthread_equal(pthread_t t1, pthread_t t2) { return t1 == t2; }

int pthread_create(pthread_t* thread, const pthread_attr_t* attr, void* (*start_routine)(void*),
                   void* arg) {
  std::jmp_buf& env = thread_data[pthread_self()].env;
  volatile pthread_t tid;
  if (setjmp(env) == 0) {
    tid = first_available_key(thread_data);
    *thread = tid;
    running_threads.push_back(pthread_t(tid));
    thread_info_t& thr = thread_data[pthread_t(tid)];
    thr.ret_value = start_routine(arg);
  }
  thread_info_t& thr = thread_data[pthread_t(tid)];
  thr.finished = true;
  running_threads.pop_back();

  // Destroy all keys.
  bool repeat = true;
  size_t iter = 0;
  while (repeat && iter++ < PTHREAD_DESTRUCTOR_ITERATIONS) {
    repeat = false;
    // Assume that destructors can create new keys (i.e. modify the map).
    for (size_t k = 0; k != PTHREAD_KEYS_MAX; ++k) {
      auto f = thr.keys.find(k);
      if (f == thr.keys.end()) {
        continue;
      }
      key_entry_t& key = f->second;
      if (key.dtor == nullptr || key.value == nullptr) {
        continue;
      }
      key.dtor(key.value);
      repeat = true;
    }
  }

  if (thr.detached) {
    thread_data.erase(pthread_t(tid));
  }

  return 0;
}

int pthread_join(pthread_t thread, void** retval) {
  auto f = thread_data.find(thread);
  if (f == thread_data.end()) {
    return ESRCH;
  }
  thread_info_t& thr = f->second;
  if (!thr.finished) {
    return EDEADLK;
  }
  if (retval != nullptr) {
    *retval = thr.ret_value;
  }
  thread_data.erase(f);
  return 0;
}

int pthread_detach(pthread_t thread) {
  auto f = thread_data.find(thread);
  if (f == thread_data.end()) {
    return ESRCH;
  }
  // Can discard the return value.
  f->second.detached = true;
  return 0;
}

void pthread_exit(void* retval) {
  pthread_t sid = pthread_self();
  if (sid != main_thread_id) {
    thread_info_t& self = thread_data[sid];
    self.ret_value = retval;
    self.finished = true;
    longjmp(self.env, 1);
  }
  exit(0);  // Only executes for the main thread, plus silences
            // the "should not return" warning.
}

int pthread_key_create(pthread_key_t* key, void (*destructor)(void*)) {
  if (key == nullptr) {
    return EINVAL;
  }
  auto& keys = thread_data[pthread_self()].keys;
  pthread_key_t k = first_available_key(keys);
  if (k >= PTHREAD_KEYS_MAX) {
    return EAGAIN;
  }
  *key = k;
  keys.emplace(k, key_entry_t{nullptr, destructor});
  return 0;
}

int pthread_key_delete(pthread_key_t key) {
  auto& keys = thread_data[pthread_self()].keys;
  auto f = keys.find(key);
  if (f == keys.end()) {
    return EINVAL;
  }
  // pthread_key_delete does not call key destructors.
  keys.erase(f);
  return 0;
}

int pthread_setspecific(pthread_key_t key, const void* value) {
  auto& keys = thread_data[pthread_self()].keys;
  auto f = keys.find(key);
  if (f == keys.end()) {
    return EINVAL;
  }
  f->second.value = const_cast<void*>(value);
  return 0;
}

void* pthread_getspecific(pthread_key_t key) {
  auto& keys = thread_data[pthread_self()].keys;
  auto f = keys.find(key);
  if (f != keys.end()) {
    return f->second.value;
  }
  return nullptr;
}

pthread_t pthread_self(void) { return running_threads.back(); }

int sched_yield(void) { return 0; }

#ifdef __cplusplus_
extern "C" int nanosleep(const struct timespec* req, struct timespec* rem);
#endif

int nanosleep(const struct timespec* req, struct timespec* rem) { return 0; }
