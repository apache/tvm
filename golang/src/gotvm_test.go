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
