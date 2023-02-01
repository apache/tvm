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

#ifndef TVM_APPS_MICROTVM_ZEPHYR_HOST_DRIVEN_PLATFORM_H_
#define TVM_APPS_MICROTVM_ZEPHYR_HOST_DRIVEN_PLATFORM_H_

#include <zephyr/drivers/gpio.h>

#ifdef CONFIG_LED
#define LED0_NODE DT_ALIAS(led0)
// #define LED0 DT_GPIO_LABEL(LED0_NODE, gpios)
// #define LED0_PIN DT_GPIO_PIN(LED0_NODE, gpios)
// #define LED0_FLAGS DT_GPIO_FLAGS(LED0_NODE, gpios)
// static const struct device* led0_pin;
static const struct gpio_dt_spec led0 = GPIO_DT_SPEC_GET(LED0_NODE, gpios);
#endif  // CONFIG_LED

#endif /* TVM_APPS_MICROTVM_ZEPHYR_HOST_DRIVEN_PLATFORM_H_ */
