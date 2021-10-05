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

#ifndef TVM_RUNTIME_HEXAGON_SIM_DRIVER_PTHREAD_H_
#define TVM_RUNTIME_HEXAGON_SIM_DRIVER_PTHREAD_H_

#define _PROVIDE_POSIX_TIME_DECLS 1
#include <time.h>
#undef _PROVIDE_POSIX_TIME_DECLS

typedef int pthread_t;
typedef int pthread_attr_t;
typedef int pthread_cond_t;
typedef int pthread_condattr_t;
typedef int pthread_key_t;
typedef int pthread_mutex_t;
typedef int pthread_mutexattr_t;
typedef int pthread_once_t;

enum {
  PTHREAD_COND_INITIALIZER,
  PTHREAD_MUTEX_DEFAULT,
  PTHREAD_MUTEX_ERRORCHECK,
  PTHREAD_MUTEX_INITIALIZER,
  PTHREAD_MUTEX_NORMAL,
  PTHREAD_MUTEX_RECURSIVE,
  PTHREAD_ONCE_INIT = 0,  // Must be same as in QuRT
  PTHREAD_ONCE_DONE,      // Non-standard
};

const size_t PTHREAD_KEYS_MAX = 128;
const size_t PTHREAD_DESTRUCTOR_ITERATIONS = 4;

#ifdef __cplusplus
extern "C" {
#endif
int pthread_cond_destroy(pthread_cond_t* cond);
int pthread_cond_init(pthread_cond_t* __restrict cond, const pthread_condattr_t* __restrict attr);
int pthread_cond_signal(pthread_cond_t* cond);
int pthread_cond_broadcast(pthread_cond_t* cond);
int pthread_cond_timedwait(pthread_cond_t* __restrict cond, pthread_mutex_t* __restrict mutex,
                           const struct timespec* __restrict abstime);
int pthread_cond_wait(pthread_cond_t* __restrict cond, pthread_mutex_t* __restrict mutex);

int pthread_mutexattr_init(pthread_mutexattr_t* attr);
int pthread_mutexattr_destroy(pthread_mutexattr_t* attr);
int pthread_mutexattr_gettype(const pthread_mutexattr_t* __restrict attr, int* __restrict type);
int pthread_mutexattr_settype(pthread_mutexattr_t* attr, int type);

int pthread_mutex_init(pthread_mutex_t* __restrict mutex,
                       const pthread_mutexattr_t* __restrict attr);
int pthread_mutex_destroy(pthread_mutex_t* mutex);
int pthread_mutex_lock(pthread_mutex_t* mutex);
int pthread_mutex_trylock(pthread_mutex_t* mutex);
int pthread_mutex_unlock(pthread_mutex_t* mutex);

int pthread_once(pthread_once_t* once_control, void (*init_routine)(void));
int pthread_equal(pthread_t t1, pthread_t t2);

int pthread_create(pthread_t* thread, const pthread_attr_t* attr, void* (*start_routine)(void*),
                   void* arg);
int pthread_join(pthread_t thread, void** retval);
int pthread_detach(pthread_t thread);
void pthread_exit(void* retval) __attribute__((__noreturn__));

int pthread_key_create(pthread_key_t* key, void (*destructor)(void*));
int pthread_key_delete(pthread_key_t key);
int pthread_setspecific(pthread_key_t key, const void* value);
void* pthread_getspecific(pthread_key_t key);

pthread_t pthread_self(void);
#ifdef __cplusplus
}
#endif

#endif  // TVM_RUNTIME_HEXAGON_SIM_DRIVER_PTHREAD_H_
