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

func TestModuleTest(t *testing.T) {
    // dll
    mod, err := LoadModuleFromFile("./deploy.so", "dll")
    if err != nil {
        t.Error(err.Error())
        return
    }

    if reflect.TypeOf(mod).Kind() != reflect.Ptr {
        t.Error("Module type mis matched\n")
    }

    // dylib
    mod, err = LoadModuleFromFile("./deploy.so", "dylib")
    if err != nil {
        t.Error(err.Error())
        return
    }

    if reflect.TypeOf(mod).Kind() != reflect.Ptr {
        t.Error("Module type mis matched\n")
    }

    // dso
    mod, err = LoadModuleFromFile("./deploy.so", "dso")
    if err != nil {
        t.Error(err.Error())
        return
    }

    if reflect.TypeOf(mod).Kind() != reflect.Ptr {
        t.Error("Module type mis matched\n")
    }

    // so
    mod, err = LoadModuleFromFile("./deploy.so", "so")
    if err != nil {
        t.Error(err.Error())
        return
    }

    if reflect.TypeOf(mod).Kind() != reflect.Ptr {
        t.Error("Module type mis matched\n")
    }

    // default type as so
    mod, err = LoadModuleFromFile("./deploy.so")
    if err != nil {
        t.Error(err.Error())
        return
    }

    if reflect.TypeOf(mod).Kind() != reflect.Ptr {
        t.Error("Module type mis matched\n")
    }

    // Unknown file should return error
    _, err = LoadModuleFromFile("xyzabc.so")
    if err == nil {
        t.Error("Expected an error, but not received\n")
        return
    }
}

