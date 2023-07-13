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

#ifndef RTE_COMPONENTS_H
#define RTE_COMPONENTS_H

#ifdef M55_HP
#define CMSIS_device_header "M55_HP.h"
#elif defined M55_HE
#define CMSIS_device_header "M55_HE.h"
#else
#define CMSIS_device_header "ARMCM55.h"
#endif

#include CMSIS_device_header

#define RTE_Drivers_GPIO 1
#define RTE_Drivers_PINCONF 1
#define RTE_UART4 1
#define RTE_UART0 1

#define RTE_Drivers_CAMERA0 0
#define RTE_Drivers_I3C0 0
#define RTE_Drivers_SAI 0

#endif /* RTE_COMPONENTS_H */
