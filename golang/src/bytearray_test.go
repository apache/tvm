/*!
 *  Copyright (c) 2018 by Contributors
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
