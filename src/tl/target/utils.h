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
 * \file tl/target/utils.h
 * \brief helper functions for target attributes.
 *
 */

#ifndef TVM_TL_TARGET_UTILS_H_
#define TVM_TL_TARGET_UTILS_H_

#include <tvm/target/target.h>

namespace tvm {
namespace tl {

bool TargetIsCuda(Target target);
bool TargetIsRocm(Target target);

bool TargetIsVolta(Target target);
bool TargetIsTuring(Target target);
bool TargetIsAmpere(Target target);
bool TargetIsHopper(Target target);
bool TargetIsCDNA(Target target);
 
bool TargetHasAsyncCopy(Target target);
bool TargetHasLdmatrix(Target target);
bool TargetHasStmatrix(Target target);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TARGET_UTILS_H_
