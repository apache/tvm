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
 * \file nnvm/base.h
 * \brief Configuration of nnvm as well as basic data structure.
 */
#ifndef NNVM_BASE_H_
#define NNVM_BASE_H_

#include <dmlc/base.h>
#include <dmlc/common.h>
#include <dmlc/any.h>
#include <dmlc/memory.h>
#include <dmlc/logging.h>
#include <dmlc/registry.h>
#include <dmlc/array_view.h>

namespace nnvm {

/*! \brief any type */
using dmlc::any;

/*! \brief array_view type  */
using dmlc::array_view;

/*!\brief getter function of any type */
using dmlc::get;

/*!\brief "unsafe" getter function of any type */
using dmlc::unsafe_get;

enum TypeFlag {
  kFloat32 = 0,
  kFloat64 = 1,
  kFloat16 = 2,
  kUint8 = 3,
  kInt32 = 4,
  kInt8  = 5,
  kInt64 = 6,
  // kBool = 7,
  // 7 is reserved for kBool, in order to keep consistency with MXNet TypeFlag defined in
  // https://github.com/apache/incubator-mxnet/blob/master/3rdparty/mshadow/mshadow/base.h#L314
  kInt16 = 8,
  kUint16 = 9,
  kUint32 = 10,
  kUint64 = 11,
  kBfloat16 = 12,
};

}  // namespace nnvm

// describe op registration point
#define NNVM_STRINGIZE_DETAIL(x) #x
#define NNVM_STRINGIZE(x) NNVM_STRINGIZE_DETAIL(x)
#define NNVM_DESCRIBE(...) describe(__VA_ARGS__ "\n\nFrom:" __FILE__ ":" NNVM_STRINGIZE(__LINE__))
#define NNVM_ADD_FILELINE "\n\nDefined in " __FILE__ ":L" NNVM_STRINGIZE(__LINE__)
#endif  // NNVM_BASE_H_
