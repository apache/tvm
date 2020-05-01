using System;


namespace TVMRuntime
{
    /// <summary>
    /// TVM Type code.
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

    //TODO: Auto type code inference/matching

    public static class PFManager
    {
        /// <summary>
        /// Runs the packed func.
        /// </summary>
        /// <param name="funcHandle">Func handle.</param>
        /// <param name="inputArgs">Input arguments.</param>
        public static void RunPackedFunc(IntPtr funcHandle, object [] inputArgs)
        {
            int numArgs = inputArgs.Length;
            int[] typeCodes = new int[numArgs];
            TVMValue [] args = new TVMValue[numArgs];

            for (int i=0; i < numArgs; i++)
            {
                Type t = inputArgs[i].GetType();
                if (t.Equals(typeof(byte)) || t.Equals(typeof(sbyte))
                    || t.Equals(typeof(int)) || t.Equals(typeof(long)))
                {
                    args[i].vInt64 = (long)inputArgs[i];
                    typeCodes[i] = (int)TVMDataTypeCode.Int;
                }
                else if (t.Equals(typeof(uint)))
                {
                    args[i].vInt64 = (long)inputArgs[i];
                    typeCodes[i] = (int)TVMDataTypeCode.UInt;
                }
                else if (t.Equals(typeof(float))
                    || t.Equals(typeof(double)))
                {
                    args[i].vFloat64 = (double)inputArgs[i];
                    typeCodes[i] = (int)TVMDataTypeCode.Float;
                }
                else if (t.Equals(typeof(NDArray)))
                {
                    args[i].handle = ((NDArray)inputArgs[i]).NDArrayHandle;
                    typeCodes[i] = (int)TVMTypeCode.TVMNDArrayHandle;
                }

                // TODO: UIntPtr/IntPtr does not mean only NDArray Handle
                //      so find a generic solution
                else if (t.Equals(typeof(IntPtr)))
                {
                    args[i].handle = (IntPtr)inputArgs[i];
                    typeCodes[i] = (int)TVMTypeCode.TVMNDArrayHandle;
                }
                else if (t.Equals(typeof(UIntPtr)))
                {
                    unsafe
                    {
                        args[i].handle = (IntPtr)((UIntPtr)inputArgs[i]).ToPointer();
                    }
                    typeCodes[i] = (int)TVMTypeCode.TVMNDArrayHandle;
                }
                else
                {
                    throw new System.ArgumentException(t + " not supported!");
                }
            }

            TVMValue retVal = new TVMValue();
            int retTypeCode = 0;

            UnamangedPFManagerWrapper.InvokeTVMRuntimePackedFunc(funcHandle,
                args, typeCodes, numArgs, ref retVal, ref retTypeCode);
        }

        /// <summary>
        /// Runs the packed func.
        /// </summary>
        /// <param name="funcHandle">Func handle.</param>
        /// <param name="inputArgs">Input arguments.</param>
        public static void RunPackedFunc(IntPtr funcHandle, string[] inputArgs)
        {
            int numArgs = inputArgs.Length;
            int[] typeCodes = new int[numArgs];

            for (int i = 0; i < numArgs; i++)
            {
                typeCodes[i] = (int)TVMTypeCode.TVMStr;
            }

            TVMValue retVal = new TVMValue();
            int retTypeCode = 0;

            UnamangedPFManagerWrapper.InvokeTVMRuntimePackedFuncStr(funcHandle,
                inputArgs, typeCodes, numArgs, ref retVal, ref retTypeCode);

        }

        /// <summary>
        /// Disposes the packed func.
        /// </summary>
        /// <param name="funcHandle">Func handle.</param>
        public static void DisposePackedFunc(IntPtr funcHandle)
        {
            UnamangedPFManagerWrapper.DisposeTVMRuntimeFuncHandle(funcHandle);
        }
    }
}
