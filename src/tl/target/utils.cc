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
 * \file tl/target/utils.cc
 * \brief helper functions for target attributes.
 */

#include "utils.h"

namespace tvm {
namespace tl {

bool TargetIsCuda(Target target) { return target->GetTargetDeviceType() == kDLCUDA; }
bool TargetIsRocm(Target target) { return target->GetTargetDeviceType() == kDLROCM; }

int GetArchInt(Target target) {
  auto s = target->GetAttr<String>("arch");
  ICHECK(s.defined());
  const char* arch_str = s.value().c_str();
  ICHECK_EQ(arch_str[0], 's');
  ICHECK_EQ(arch_str[1], 'm');
  ICHECK_EQ(arch_str[2], '_');
  return atoi(&arch_str[3]);
}

bool TargetIsVolta(Target target) {
  if (!TargetIsCuda(target)) return false;
  int arch = GetArchInt(target);
  return arch >= 70 && arch < 75;
}

bool TargetIsTuring(Target target) {
  if (!TargetIsCuda(target)) return false;
  int arch = GetArchInt(target);
  return arch >= 75 && arch < 80;
}

bool TargetIsAmpere(Target target) {
  if (!TargetIsCuda(target)) return false;
  int arch = GetArchInt(target);
  return arch >= 80 && arch < 90;
}

bool TargetIsHopper(Target target) {
  if (!TargetIsCuda(target)) return false;
  int arch = GetArchInt(target);
  return arch >= 90;
}

bool TargetHasAsyncCopy(Target target) {
  if (!TargetIsCuda(target)) return false;
  int arch = GetArchInt(target);
  return arch >= 80;
}

bool TargetHasLdmatrix(Target target) {
  if (!TargetIsCuda(target)) return false;
  int arch = GetArchInt(target);
  return arch >= 75;
}

bool TargetHasStmatrix(Target target) {
  if (!TargetIsCuda(target)) return false;
  int arch = GetArchInt(target);
  return arch >= 90;
}

}  // namespace tl
}  // namespace tvm
