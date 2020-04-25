using System;


namespace TVMRuntime
{
    /// <summary>
    /// TVMT ype code.
    /// </summary>
    public enum TVMTypeCode
    {
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

    public static class PFManager
    {

    }
}
