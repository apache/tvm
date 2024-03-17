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

#ifndef _MFCC_PREPROCESSOR_H_
#define _MFCC_PREPROCESSOR_H_

#include "mfcc.h"
#include "spec.h"

class MfccPreprocessor {
 public:
  MfccPreprocessor();
  ~MfccPreprocessor();

  void Apply(const void* src, const size_t src_bytes, void* dst);
  int Errors();

 private:
  MFCC* mfcc_;
  float* mfccBufferTemplate_;
  int errors_;
};

#endif  // _MFCC_PREPROCESSOR_H_
