using System;

namespace TVMRuntime
{
    /// <summary>
    /// TVM Type code.
    /// </summary>
    public enum TVMTypeCode
    {
        TVMInt = 0,

        TVMUInt = 1,

        TVMFloat = 2,

        TVMOpaqueHandle = 3,

        TVMNullptr = 4,

        TVMDataType = 5,

        TVMContext = 6,

        TVMDLTensorHandle = 7,

        TVMObjectHandle = 8,

        TVMModuleHandle = 9,

        TVMPackedFuncHandle = 10,

        TVMStr = 11,

        TVMBytes = 12,

        TVMNDArrayHandle = 13,

        TVMObjectRValueRefArg = 14,

        TVMExtBegin = 15,

        TVMNNVMFirst = 16,

        TVMNNVMLast = 20,

        TVMExtReserveEnd = 64,

        TVMExtEnd = 128,

        TVMCustomBegin = 129,
    }
}
