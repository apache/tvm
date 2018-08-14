/*!
 *  Copyright (c) 2018 by Contributors
 * \brief gotvm package source for TVMContext interface
 * \file context.go
 */

package gotvm

//#include "gotvm.h"
import "C"

// Context dtype corresponding to TVMContext aka DLContext
type Context struct {
    DeviceType int32
    DeviceID    int32
}
