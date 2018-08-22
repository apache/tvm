/*!
 *  Copyright (c) 2018 by Contributors
 * \brief gotvm package
 * \file gotvm_test.go
 */


package gotvm

import (
    "testing"
    "strings"
)

func TestErrorTest(t *testing.T) {
    _, err := LoadModuleFromFile("dummy.so")
    if err == nil {
        t.Error("Expected an error, but not received\n")
        return
    }

    errStr := err.Error()
    if !(strings.Contains(errStr, string("cannot open shared object"))) {
        t.Error("Ah! TVM didn't report an error\n")
    }
}

