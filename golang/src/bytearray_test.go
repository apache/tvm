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
 * \brief gotvm package
 * \file bytearray_test.go
 */


package gotvm

import (
    "testing"
    "math/rand"
)

// Check ByteArray creation from byte slice and verify the data.
func TestByteArrayGet(t *testing.T) {
    data := make([]byte, 1024)
    rand.Read(data)

    barr := newByteArray(data)
    dataRet := barr.getData()
    if len(data) != len(dataRet) {
            t.Errorf("Data expected Len: %v Got :%v\n", len(data), len(dataRet))
            return
    }
    for i := range data {
        if data[i] != dataRet[i] {
            t.Errorf("Data expected: %v Got :%v at : %v\n", data[i], dataRet[i], i)
            return
        }
    }
}
