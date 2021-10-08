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
 *
 * \file utils.h
 * \brief Internal utilities for manipulating TE schedules.
 */

#ifndef TVM_DRIVER_UTILS_H_
#define TVM_DRIVER_UTILS_H_

#include <tvm/te/schedule.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/function.h>

#include <string>
#include <unordered_map>

namespace tvm {

/*!
 * \brief Create an PrimFunc out of a TE Schedule.
 *
 * Generated PrimFunc expresses reads/writes using
 * BufferLoad/BufferStore, with all Tensors and
 * ProducerLoad/ProducerStore having been replaced.
 *
 * Assumes that the schedule has already been normalized, either with
 * `te::Schedule::normalize` or
 * `te::Schedule::normalize_for_feature_extraction`.
 *
 * Does not apply lowering passes. If you want
 * to apply lowering passes as well, use LowerSchedule.
 *
 * \param sch The schedule
 * \param args The arguments to the function.
 * \param name The name of the lowered function.
 * \param binds Buffer assignments.
 * \return The result module.
 */
tir::PrimFunc ScheduleToPrimFunc(te::Schedule sch, const Array<ObjectRef>& args,
                                 const std::string& name,
                                 const std::unordered_map<te::Tensor, tir::Buffer>& binds);

}  // namespace tvm

#endif  // TVM_DRIVER_UTILS_H_
