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

#ifndef _SCOPE_TIMER_H_
#define _SCOPE_TIMER_H_

#include "esp_log.h"
#include "esp_timer.h"

static const char* TAG = "KWS";

class ScopeTimer {
  const char* msg;
  int64_t t1, t2;

 public:
  ScopeTimer(const char* msg) : msg(msg) { t1 = esp_timer_get_time(); }
  ~ScopeTimer() {
    t2 = esp_timer_get_time();
    ESP_LOGI(TAG, "%s: %lld us", msg, t2 - t1);
  }
};

#define LOG_TIME(x, name)    \
  t1 = esp_timer_get_time(); \
  x;                         \
  t2 = esp_timer_get_time(); \
  ESP_LOGI(TAG, "%s: %lld us", name, t2 - t1);

#endif  // _SCOPE_TIMER_H_
