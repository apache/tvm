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
 * \file unset_log_macros.h
 * \brief Undef some macros to resolve conflicts between dmlc logging and pytorch
 */

#ifndef TVM_CONTRIB_TORCH_UNSET_LOG_MACROS_H_
#define TVM_CONTRIB_TORCH_UNSET_LOG_MACROS_H_
#ifdef LOG
#undef LOG
#endif
#ifdef LOG_IF
#undef LOG_IF
#endif
#ifdef CHECK
#undef CHECK
#endif
#ifdef CHECK_EQ
#undef CHECK_EQ
#endif
#ifdef CHECK_LT
#undef CHECK_LT
#endif
#ifdef CHECK_LE
#undef CHECK_LE
#endif
#ifdef CHECK_GT
#undef CHECK_GT
#endif
#ifdef CHECK_GE
#undef CHECK_GE
#endif
#ifdef CHECK_NE
#undef CHECK_NE
#endif
#ifdef CHECK_NOTNULL
#undef CHECK_NOTNULL
#endif
#ifdef LOG_EVERY_N
#undef LOG_EVERY_N
#endif

#ifdef DCHECK
#undef DCHECK
#endif
#ifdef DCHECK_EQ
#undef DCHECK_EQ
#endif
#ifdef DCHECK_LT
#undef DCHECK_LT
#endif
#ifdef DCHECK_LE
#undef DCHECK_LE
#endif
#ifdef DCHECK_GT
#undef DCHECK_GT
#endif
#ifdef DCHECK_GE
#undef DCHECK_GE
#endif
#ifdef DCHECK_NE
#undef DCHECK_NE
#endif
#ifdef VLOG
#undef VLOG
#endif
#ifdef DLOG
#undef DLOG
#endif
#ifdef DLOG_IF
#undef DLOG_IF
#endif

#endif  // TVM_CONTRIB_TORCH_UNSET_LOG_MACROS_H_