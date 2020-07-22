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

.syntax unified
.cpu cortex-m7
.fpu softvfp
.thumb

.section .text.UTVMInit
.type UTVMInit, %function
UTVMInit:
  /* enable fpu */
  ldr r0, =0xE000ED88
  ldr r1, [r0]
  ldr r2, =0xF00000
  orr r1, r2
  str r1, [r0]
  dsb
  isb
  /* set stack pointer */
  ldr sp, =_utvm_stack_pointer_init
  bl UTVMMain
.size UTVMInit, .-UTVMInit
