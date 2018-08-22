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

func TestTVMVersion(t *testing.T) {
    if len(TVMVersion) == 0 {
        t.Error("TVMVersion not set\n")
    }
    if reflect.TypeOf(TVMVersion).Kind() != reflect.String {
        t.Error("TVMVersion type mismatch\n")
    }
}

func TestDLPackVersion(t *testing.T) {
    if reflect.TypeOf(DLPackVersion).Kind() != reflect.Int {
        t.Error("TVMVersion type mismatch\n")
    }
}

func TestGoTVMVersion(t *testing.T) {
    if len(GoTVMVersion) == 0 {
        t.Error("GoTVMVersion not set\n")
    }
    if reflect.TypeOf(GoTVMVersion).Kind() != reflect.String {
        t.Error("GoTVMVersion type mismatch\n")
    }
}

