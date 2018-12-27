/*!
 *  Copyright (c) 2018 by Contributors
 * \brief gotvm package for TVMType interface
 * \file type.go
 */

package gotvm

//#include "gotvm.h"
import "C"

import (
    "fmt"
)

// pTVMType corresponding to data types.
type pTVMType struct {
    code uint8
    bits uint8
    lanes uint16
}

// data type to pTVMType mapping
var dtypeMap = map[string] pTVMType {
    "int8": pTVMType{0, 8, 1},
    "int16": pTVMType{0, 16, 1},
    "int32": pTVMType{0, 32, 1},
    "int64": pTVMType{0, 64, 1},
    "uint8": pTVMType{1, 8, 1},
    "uint16": pTVMType{1, 16, 1},
    "uint32": pTVMType{1, 32, 1},
    "uint64": pTVMType{1, 64, 1},
    "float32": pTVMType{2, 32, 1},
    "float64": pTVMType{2, 64, 1},
}

// dtypeFromTVMType return the pTVMType corresponding to given dtype
//
// `dtype` string for the given data type.
func dtypeFromTVMType(tvmtype pTVMType) (retVal string, err error) {
    for k, v := range dtypeMap {
        if v.code == tvmtype.code && v.bits == tvmtype.bits && v.lanes == tvmtype.lanes {
            retVal = k
            return
        }
    }

    err = fmt.Errorf("Cannot map TVMType:%v to dtype", tvmtype)
    return
}

// dtypeToTVMType return the pTVMType corresponding to given dtype
//
// `dtype` string for the given data type.
func dtypeToTVMType(args ...interface{}) (tvmtype pTVMType, err error) {
    dtype := args[0].(string)
    lanes := 1

    if len(args) == 2 {
        lanes = args[1].(int)
    }

    for k, v := range dtypeMap {
        if k == dtype {
            tvmtype = v
            tvmtype.lanes = uint16(lanes)
            return
        }
    }
    err = fmt.Errorf("Cannot map dtype:%v to TVMType", dtype)
    return
}
