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
 *  Copyright (c) 2018 by Contributors
 * \brief gotvm package
 * \file gotvm_test.go
 */


package gotvm

import (
    "testing"
    "reflect"
)

// Check TVMVersion API
func TestTVMVersion(t *testing.T) {
    if len(TVMVersion) == 0 {
        t.Error("TVMVersion not set\n")
    }
    if reflect.TypeOf(TVMVersion).Kind() != reflect.String {
        t.Error("TVMVersion type mismatch\n")
    }
}

// Check DLPackVersion API
func TestDLPackVersion(t *testing.T) {
    if reflect.TypeOf(DLPackVersion).Kind() != reflect.Int {
        t.Error("TVMVersion type mismatch\n")
    }
}
