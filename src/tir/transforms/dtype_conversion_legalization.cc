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
 * \file dtype_conversion_legalization.h
 * \brief Header file of data type conversion legalization routines.
 */
#include "dtype_conversion_legalization.h"

namespace tvm {
namespace tir {

PrimExpr FpToFp(PrimExpr src_value, FloatConfig src_fp, FloatConfig tgt_fp) {

}

PrimExpr IntToFp(PrimExpr src_value, FloatConfig tgt_fp) {

}

PrimExpr FpToInt(PrimExpr src_value, FloatConfig src_fp, DataType tgt_dtype) {

}

PrimExpr DTypeConversion(PrimExpr src_value, DataType tgt_dtype) {
  DataType src_dtype = src_value.dtype();
  auto is_floating_point = [](DataType dtype) {
    return dtype.is_float() || dtype.is_float8() || dtype.is_bfloat16();
  };
  auto is_integer = [](DataType dtype) {
    return dtype.is_int() || dtype.is_uint();
  };
  if (is_floating_point(src_dtype) && is_floating_point(tgt_dtype)) {
    return FpToFp(src_value, FloatConfig::FromDataType(src_dtype), FloatConfig::FromDataType(tgt_dtype));
  } else {
    if (is_integer(src_dtype) && is_floating_point(tgt_dtype)) {
      return IntToFp(src_value, FloatConfig::FromDataType(tgt_dtype));
    } else if (is_floating_point(src_dtype) && is_integer(src_dtype)) {
      return FpToInt(src_value, FloatConfig::FromDataType(src_dtype), tgt_dtype);
    } else {
      LOG(FATAL) << "Not Implemented yet";
    }
  }
}

}  // namespace tir
}  // namespace tvm
