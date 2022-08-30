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

/* Please refer to http://www.freertos.org/a00110.html for refernce. */
#ifndef FREERTOS_CONFIG_H
#define FREERTOS_CONFIG_H
/******************************************************************************
 * Defines
 **********SYSTEM_CORE_CLOCK********************************************************************/
/* Hardware features */
#define configENABLE_MPU 0
#define configENABLE_FPU 0
#define configENABLE_TRUSTZONE 0
/* Scheduling */
#define configCPU_CLOCK_HZ 25000000
#define configUSE_PORT_OPTIMISED_TASK_SELECTION 0
#define configUSE_PREEMPTION 1
#define configUSE_TIME_SLICING 0
#define configMAX_PRIORITIES 5
#define configIDLE_SHOULD_YIELD 1
#define configUSE_16_BIT_TICKS 0
#define configRUN_FREERTOS_SECURE_ONLY 1
/* Stack and heap */
#define configMINIMAL_STACK_SIZE (uint16_t)128
#define configMINIMAL_SECURE_STACK_SIZE 1024
#define configTOTAL_HEAP_SIZE (size_t)(50 * 1024)
#define configMAX_TASK_NAME_LEN 12
/* OS features */
#define configUSE_MUTEXES 1
#define configUSE_TICKLESS_IDLE 1
#define configUSE_APPLICATION_TASK_TAG 0
#define configUSE_NEWLIB_REENTRANT 0
#define configUSE_CO_ROUTINES 0
#define configUSE_COUNTING_SEMAPHORES 1
#define configUSE_RECURSIVE_MUTEXES 1
#define configUSE_QUEUE_SETS 0
#define configUSE_TASK_NOTIFICATIONS 1
#define configUSE_TRACE_FACILITY 1
/* Hooks */
#define configUSE_IDLE_HOOK 0
#define configUSE_TICK_HOOK 0
#define configUSE_MALLOC_FAILED_HOOK 0
/* Debug features */
#define configCHECK_FOR_STACK_OVERFLOW 0
#define configASSERT(x)       \
  if ((x) == 0) {             \
    taskDISABLE_INTERRUPTS(); \
    for (;;)                  \
      ;                       \
  }
#define configQUEUE_REGISTRY_SIZE 0
/* Timers and queues */
#define configUSE_TIMERS 1
#define configTIMER_TASK_PRIORITY (configMAX_PRIORITIES - 1)
#define configTIMER_TASK_STACK_DEPTH configMINIMAL_STACK_SIZE
#define configTIMER_QUEUE_LENGTH 5
/* Task settings */
#define INCLUDE_vTaskPrioritySet 1
#define INCLUDE_uxTaskPriorityGet 1
#define INCLUDE_vTaskDelete 1
#define INCLUDE_vTaskCleanUpResources 0
#define INCLUDE_vTaskSuspend 1
#define INCLUDE_vTaskDelayUntil 1
#define INCLUDE_vTaskDelay 1
#define INCLUDE_uxTaskGetStackHighWaterMark 0
#define INCLUDE_xTaskGetIdleTaskHandle 0
#define INCLUDE_eTaskGetState 1
#define INCLUDE_xTaskResumeFromISR 0
#define INCLUDE_xTaskGetCurrentTaskHandle 1
#define INCLUDE_xTaskGetSchedulerState 0
#define INCLUDE_xSemaphoreGetMutexHolder 0
#define INCLUDE_xTimerPendFunctionCall 1
#define configUSE_STATS_FORMATTING_FUNCTIONS 1
#define configCOMMAND_INT_MAX_OUTPUT_SIZE 2048
#ifdef __NVIC_PRIO_BITS
#define configPRIO_BITS __NVIC_PRIO_BITS
#else
#define configPRIO_BITS 3
#endif
/* Interrupt settings */
#define configLIBRARY_LOWEST_INTERRUPT_PRIORITY 0x07
#define configLIBRARY_MAX_SYSCALL_INTERRUPT_PRIORITY 5
#define configKERNEL_INTERRUPT_PRIORITY \
  (configLIBRARY_LOWEST_INTERRUPT_PRIORITY << (8 - configPRIO_BITS))
#define configMAX_SYSCALL_INTERRUPT_PRIORITY \
  (configLIBRARY_MAX_SYSCALL_INTERRUPT_PRIORITY << (8 - configPRIO_BITS))
#ifndef __IASMARM__
#define configGENERATE_RUN_TIME_STATS 0
#define portCONFIGURE_TIMER_FOR_RUN_TIME_STATS()
#define portGET_RUN_TIME_COUNTER_VALUE() 0
#define configTICK_RATE_HZ (TickType_t)1000
#endif /* __IASMARM__ */
#define xPortPendSVHandler PendSV_Handler
#define vPortSVCHandler SVC_Handler
#define xPortSysTickHandler SysTick_Handler
#endif /* FREERTOS_CONFIG_H */
