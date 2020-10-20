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
 * \file micro_session.h
 * \brief session to manage multiple micro modules
 *
 * Each session consists of an interaction with a *single* logical device.
 * Within that interaction, multiple TVM modules can be loaded on the logical
 * device.
 *
 * Multiple sessions can exist simultaneously, but there is only ever one
 * *active* session. The idea of an active session mainly has implications for
 * the frontend, in that one must make a session active in order to allocate
 * new TVM objects on it. Aside from that, previously allocated objects can be
 * used even if the session which they belong to is not currently active.
 */
#ifndef TVM_RUNTIME_MICRO_MICRO_SESSION_H_
#define TVM_RUNTIME_MICRO_MICRO_SESSION_H_

#endif  // TVM_RUNTIME_MICRO_MICRO_SESSION_H_
