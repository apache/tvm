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
 * \file aiTestTvmHelper.h
 * \brief STM32 Helper functions for STM32 AI test application
 */

#ifndef __AI_TEST_TVM_HELPER_H__
#define __AI_TEST_TVM_HELPER_H__

#include <stdint.h>

#include "ai_runtime_api.h"
#include "ai_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

void aiPlatformVersion(void);

void aiLogErr(const char *fct, const char *err);
void aiPrintLayoutBuffer(const char *msg, int idx, ai_tensor * tensor);
void aiPrintNetworkInfo(ai_model_info *nn, ai_handle hdl);

#if defined(NO_X_CUBE_AI_RUNTIME) && NO_X_CUBE_AI_RUNTIME == 1
#include "ai_platform.h"
void aiTvmToReport(ai_model_info *nn, ai_handle hdl, ai_network_report *report);
#endif

#ifdef __cplusplus
}
#endif

#endif /* __AI_TEST_TVM_HELPER_H__ */
