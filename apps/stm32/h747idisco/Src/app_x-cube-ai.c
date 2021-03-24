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
 * \file app_x-cube-ai.c
 * \brief AI program body
 */

#ifdef __cplusplus
 extern "C" {
#endif

#include <string.h>
#include "app_x-cube-ai.h"
#include "bsp_ai.h"
#ifdef USE_VALID
#include "aiValidation.h"
#else
#include "aiSystemPerformance.h"
#endif

/* USER CODE BEGIN includes */

/* USER CODE END includes */

/*************************************************************************
  *
  */
void MX_X_CUBE_AI_Init(void)
{
    MX_UARTx_Init();
#ifdef USE_VALID
    aiValidationInit();
#else
    aiSystemPerformanceInit();
#endif
    /* USER CODE BEGIN 0 */
    /* USER CODE END 0 */
}

void MX_X_CUBE_AI_Process(void)
{
#ifdef USE_VALID
    aiValidationProcess();
#else
    aiSystemPerformanceProcess();
#endif
    HAL_Delay(1000); /* delay 1s */
    /* USER CODE BEGIN 1 */
    /* USER CODE END 1 */
}
