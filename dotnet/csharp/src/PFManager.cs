using System;
using System.Runtime.InteropServices;
using Native;

namespace TVMRuntime
{
    public static class PFManager
    {
        /// <summary>
        /// Gets the TVM Runtime global packed func.
        /// </summary>
        /// <param name="funcName">Func name.</param>
        /// <param name="funcHandle">Func handle.</param>
        public static void GetGlobalPackedFunc(string funcName, ref IntPtr funcHandle)
        {
            int result = NativeImport.TVMFuncGetGlobal(funcName, ref funcHandle);
            Utils.CheckSuccess(0, result);
        }

        /// <summary>
        /// Disposes the TVM Runtime func handle.
        /// </summary>
        /// <param name="funcHandle">Func handle.</param>
        public static void DisposePackedFunc(IntPtr funcHandle)
        {
            int result = NativeImport.TVMFuncFree(funcHandle);
            Utils.CheckSuccess(0, result);
        }

        /// <summary>
        /// Runs the packed func.
        /// </summary>
        /// <param name="funcHandle">Func handle.</param>
        /// <param name="inputArgs">Input arguments.</param>
        public static TVMValue RunPackedFunc(IntPtr funcHandle, object[] inputArgs)
        {
            int numArgs = inputArgs.Length;
            int[] typeCodes = new int[numArgs];
            TVMValueStruct[] args = new TVMValueStruct[numArgs];

            // Temporary Marshalling Pointers
            IntPtr strValuePtr = IntPtr.Zero;
            IntPtr[] byteArrayPtr = new IntPtr[] { IntPtr.Zero };

            try
            {
                for (int i = 0; i < numArgs; i++)
                {
                    Type t = inputArgs[i].GetType();
                    if (t.Equals(typeof(byte)) || t.Equals(typeof(sbyte))
                        || t.Equals(typeof(int)) || t.Equals(typeof(long)))
                    {
                        long value = 0;
                        if (t.Equals(typeof(byte))) value = (long)((byte)inputArgs[i]);
                        else if (t.Equals(typeof(sbyte))) value = (long)((sbyte)inputArgs[i]);
                        else if (t.Equals(typeof(int))) value = (long)((int)inputArgs[i]);
                        else if (t.Equals(typeof(long))) value = (long)(inputArgs[i]);

                        args[i].vInt64 = value;
                        typeCodes[i] = (int)TVMDataTypeCode.Int;
                    }
                    else if (t.Equals(typeof(uint)))
                    {
                        args[i].vInt64 = (long)((uint)inputArgs[i]);
                        typeCodes[i] = (int)TVMDataTypeCode.UInt;
                    }
                    else if (t.Equals(typeof(float))
                        || t.Equals(typeof(double)))
                    {
                        double value = 0;
                        if (t.Equals(typeof(float))) value = (double)((float)inputArgs[i]);
                        else if (t.Equals(typeof(double))) value = (double)(inputArgs[i]);

                        args[i].vFloat64 = value;
                        typeCodes[i] = (int)TVMDataTypeCode.Float;
                    }
                    else if (t.Equals(typeof(string)))
                    {
                        // NOTE: String Marshalling is tricky as it can not go
                        //       in common structure as string, so send it as
                        //       pointer.
                        strValuePtr = Marshal.StringToHGlobalAuto((string)inputArgs[i]);
                        args[i].handle = strValuePtr;
                        typeCodes[i] = (int)TVMTypeCode.TVMStr;
                    }
                    else if (t.Equals(typeof(byte [])))
                    {
                        byteArrayPtr = GetByteArrayPtr((byte[])inputArgs[i]);
                        args[i].handle = byteArrayPtr[0];
                        typeCodes[i] = (int)TVMTypeCode.TVMBytes;
                    }
                    else if (t.Equals(typeof(NDArray)))
                    {
                        args[i].handle = ((NDArray)inputArgs[i]).NDArrayHandle;
                        typeCodes[i] = (int)TVMTypeCode.TVMNDArrayHandle;
                    }
                    else if (t.Equals(typeof(PackedFunction)))
                    {
                        args[i].handle = ((PackedFunction)inputArgs[i]).FuncHandle;
                        typeCodes[i] = (int)TVMTypeCode.TVMPackedFuncHandle;
                    }
                    else if (t.Equals(typeof(Module)))
                    {
                        args[i].handle = ((Module)inputArgs[i]).ModuleHandle;
                        typeCodes[i] = (int)TVMTypeCode.TVMModuleHandle;
                    }
                    else
                    {
                        throw new System.ArgumentException(t + " not supported!");
                    }
                }

                TVMValueStruct retVal = new TVMValueStruct();
                int retTypeCode = 0;
                int result = NativeImport.TVMFuncCall(funcHandle,
                    args, typeCodes, numArgs, ref retVal, ref retTypeCode);
                Utils.CheckSuccess(0, result);

                return GetTVMValueFromReturn(retVal, (TVMRuntime.TVMTypeCode)retTypeCode);
            }
            finally
            {
                // Free The Temporary Marshalling Pointers.
                if (!IntPtr.Zero.Equals(strValuePtr)) Marshal.FreeHGlobal(strValuePtr);
                foreach (IntPtr ptr in byteArrayPtr)
                {
                    if (!IntPtr.Zero.Equals(ptr)) Marshal.FreeHGlobal(ptr);
                }
            }
        }

        /// <summary>
        /// Gets the byte array pointer.
        /// </summary>
        /// <returns>The byte array ptr.</returns>
        /// <param name="input">Input.</param>
        private static IntPtr[] GetByteArrayPtr(byte[] input)
        {
            // Initialize unmanged memory to hold the struct.
            int lengthArray = Marshal.SizeOf(input[0]) * input.Length;
            IntPtr pnt = Marshal.AllocHGlobal(lengthArray);

            TVMRuntimeByteArray loadParamsArgs = new TVMRuntimeByteArray();
            loadParamsArgs.paramPtr = pnt;
            loadParamsArgs.size = input.Length;
            IntPtr pnt1 = Marshal.AllocHGlobal(Marshal.SizeOf(loadParamsArgs));

            // Copy the struct to unmanaged memory.
            Marshal.Copy(input, 0, pnt, lengthArray);
            Marshal.StructureToPtr(loadParamsArgs, pnt1, false);
            return new IntPtr[] { pnt1, pnt };
        }

        /// <summary>
        /// Gets the TVMValue from return.
        /// </summary>
        /// <returns>The TVMV alue from return.</returns>
        /// <param name="retVal">Ret value.</param>
        /// <param name="retTypeCode">Ret type code.</param>
        public static TVMValue GetTVMValueFromReturn(TVMValueStruct retVal, TVMTypeCode retTypeCode)
        {
            switch (retTypeCode)
            {
                case TVMTypeCode.TVMInt:
                case TVMTypeCode.TVMUInt:
                    return new TVMValue(retTypeCode, retVal.vInt64);
                case TVMTypeCode.TVMFloat:
                    return new TVMValue(retTypeCode, retVal.vFloat64);
                case TVMTypeCode.TVMOpaqueHandle:
                case TVMTypeCode.TVMObjectHandle:
                    return new TVMValue(retTypeCode, retVal.handle);
                case TVMTypeCode.TVMModuleHandle:
                    return new TVMValue(retTypeCode, new Module(retVal.handle));
                case TVMTypeCode.TVMPackedFuncHandle:
                    return new TVMValue(retTypeCode, new PackedFunction(retVal.handle));
                case TVMTypeCode.TVMDLTensorHandle:
                case TVMTypeCode.TVMNDArrayHandle:
                    NDArray ndArray = NDArray.Empty();
                    ndArray.NDArrayHandle = retVal.handle;
                    return new TVMValue(retTypeCode, ndArray);
                case TVMTypeCode.TVMBytes:
                    TVMRuntimeByteArray byteArray = Marshal.PtrToStructure<TVMRuntimeByteArray>(retVal.handle);
                    byte[] array = new byte[byteArray.size];
                    Marshal.Copy(byteArray.paramPtr, array, 0, array.Length);
                    return new TVMValue(retTypeCode, array);
                case TVMTypeCode.TVMStr:
                    return new TVMValue(retTypeCode, Marshal.PtrToStringAuto(retVal.handle));
                case TVMTypeCode.TVMNullptr:
                    return new TVMValue(retTypeCode, null);
                default:
                    throw new System.ArgumentException(retTypeCode.ToString() + " not supported!");
            }
        }
    }
}
