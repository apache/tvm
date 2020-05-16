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

        // TODO: String marshalling is tricky, currently assume only input has
        //       so divided the func call to two types,
        //       and there is no return value as string.
        //       Later based on need add the functionality.

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

        /// <summary>
        /// Runs the packed func.
        /// </summary>
        /// <param name="funcHandle">Func handle.</param>
        /// <param name="inputArgs">Input arguments.</param>
        public static TVMValue RunPackedFunc(IntPtr funcHandle, string[] inputArgs)
        {
            int numArgs = inputArgs.Length;
            int[] typeCodes = new int[numArgs];

            for (int i = 0; i < numArgs; i++)
            {
                typeCodes[i] = (int)TVMTypeCode.TVMStr;
            }

            TVMValueStruct retVal = new TVMValueStruct();
            int retTypeCode = 0;

            int result = NativeImport.TVMFuncCall(funcHandle,
                inputArgs, typeCodes, numArgs, ref retVal, ref retTypeCode);
            Utils.CheckSuccess(0, result);

            return GetTVMValueFromReturn(retVal, (TVMRuntime.TVMTypeCode)retTypeCode);

        }

        //TODO: TVM String as return value need to be handled, currently no need
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
                case TVMTypeCode.TVMNullptr:
                    return new TVMValue(retTypeCode, null);
                default:
                    throw new System.ArgumentException(retTypeCode.ToString() + " not supported!");
            }
        }
    }
}
