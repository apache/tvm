/*!
 *  Copyright (c) 2018 by Contributors
 * \brief gotvm package
 * \file module_test.go
 */


package gotvm

import (
    "testing"
    "reflect"
)

func TestModuleTestLoad1(t *testing.T) {
    // dll
    mod, err := LoadModuleFromFile("./deploy.so", "dll")
    if err != nil {
        t.Error(err.Error())
        return
    }

    if reflect.TypeOf(mod).Kind() != reflect.Ptr {
        t.Error("Module type mis matched\n")
    }
}

func TestModuleTestLoad2(t *testing.T) {
    // dylib
    mod, err := LoadModuleFromFile("./deploy.so", "dylib")
    if err != nil {
        t.Error(err.Error())
        return
    }

    if reflect.TypeOf(mod).Kind() != reflect.Ptr {
        t.Error("Module type mis matched\n")
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
    }
}

func TestModuleTestLoad4(t *testing.T) {
    // so
    mod, err := LoadModuleFromFile("./deploy.so", "so")
    if err != nil {
        t.Error(err.Error())
        return
    }

    if reflect.TypeOf(mod).Kind() != reflect.Ptr {
        t.Error("Module type mis matched\n")
    }
}

func TestModuleTestLoad5(t *testing.T) {
    // default type as so
    mod, err := LoadModuleFromFile("./deploy.so")
    if err != nil {
        t.Error(err.Error())
        return
    }

    if reflect.TypeOf(mod).Kind() != reflect.Ptr {
        t.Error("Module type mis matched\n")
    }
}

func TestModuleTestLoadErr(t *testing.T) {
    // Unknown file should return error
    _, err := LoadModuleFromFile("xyzabc.so")
    if err == nil {
        t.Error("Expected an error, but not received\n")
        return
    }
}

