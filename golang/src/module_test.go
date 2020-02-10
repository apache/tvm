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
 * \file module_test.go
 */


package gotvm

import (
    "testing"
    "reflect"
)

// Check module loading - dll
func TestModuleTestLoad1(t *testing.T) {
    // dll
    mod, err := LoadModuleFromFile("./deploy.so", "dll")
    if err != nil {
        t.Error(err.Error())
        return
    }
    if reflect.TypeOf(mod).Kind() != reflect.Ptr {
        t.Error("Module type mis matched\n")
        return
    }
}

// Check module loading - dylib
func TestModuleTestLoad2(t *testing.T) {
    // dylib
    mod, err := LoadModuleFromFile("./deploy.so", "dylib")
    if err != nil {
        t.Error(err.Error())
        return
    }
    if reflect.TypeOf(mod).Kind() != reflect.Ptr {
        t.Error("Module type mis matched\n")
        return
    }
}

func TestModuleTestLoad3(t *testing.T) {
    // dso
    mod, err := LoadModuleFromFile("./deploy.so", "dso")
    if err != nil {
        t.Error(err.Error())
        return
    }
    if reflect.TypeOf(mod).Kind() != reflect.Ptr {
        t.Error("Module type mis matched\n")
        return
    }
}

// Check module loading - so
func TestModuleTestLoad4(t *testing.T) {
    // so
    mod, err := LoadModuleFromFile("./deploy.so", "so")
    if err != nil {
        t.Error(err.Error())
        return
    }
    if reflect.TypeOf(mod).Kind() != reflect.Ptr {
        t.Error("Module type mis matched\n")
        return
    }
}

// Check module loading - default (so)
func TestModuleTestLoad5(t *testing.T) {
    // default type as so
    mod, err := LoadModuleFromFile("./deploy.so")
    if err != nil {
        t.Error(err.Error())
        return
    }
    if reflect.TypeOf(mod).Kind() != reflect.Ptr {
        t.Error("Module type mis matched\n")
        return
    }
}

// Check module loading err
func TestModuleTestLoadErr(t *testing.T) {
    // Unknown file should return error
    _, err := LoadModuleFromFile("xyzabc.so")
    if err == nil {
        t.Error("Expected an error, but not received\n")
        return
    }
}
