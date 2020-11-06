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
 * \brief gotvm package source for common utilities
 * \file utils.go
 */

package gotvm

//#include "gotvm.h"
import "C"

import (
    "unsafe"
)

// Native string map for go string
type nativeGoString struct { p uintptr; n int32 }

func goStringFromNative (s string) (retStr string) {
    p := *(*nativeGoString)(unsafe.Pointer(&s))
    retStr = string((*[0x7fffffff]byte)(unsafe.Pointer(p.p))[:p.n])
    C.free(unsafe.Pointer(p.p))
    return
}
