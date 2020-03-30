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

#ifndef TVM_RUNTIME_HEXAGON_TARGET_HEXAGON_TARGET_LOG_H_
#define TVM_RUNTIME_HEXAGON_TARGET_HEXAGON_TARGET_LOG_H_
#ifdef __ANDROID__

#include <android/log.h>

#define TVM_LOGV(...) \
  __android_log_print(ANDROID_LOG_VERBOSE, "TVM", ##__VA_ARGS__)
#define TVM_LOGD(...) \
  __android_log_print(ANDROID_LOG_DEBUG, "TVM", ##__VA_ARGS__)
#define TVM_LOGI(...) \
  __android_log_print(ANDROID_LOG_INFO, "TVM", ##__VA_ARGS__)
#define TVM_LOGW(...) \
  __android_log_print(ANDROID_LOG_WARN, "TVM", ##__VA_ARGS__)
#define TVM_LOGE(...) \
  __android_log_print(ANDROID_LOG_ERROR, "TVM", ##__VA_ARGS__)
#define TVM_LOGF(...) \
  __android_log_print(ANDROID_LOG_FATAL, "TVM", ##__VA_ARGS__)

#endif  // __ANDROID__
#endif  // TVM_RUNTIME_HEXAGON_TARGET_HEXAGON_TARGET_LOG_H_
