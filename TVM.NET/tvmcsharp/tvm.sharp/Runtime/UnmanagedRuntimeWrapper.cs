﻿using System;
using System.Runtime.InteropServices;
using static TVMRuntime.Utils;
using static TVMRuntime.UnamangedPFManagerWrapper;
using System.Text;

namespace TVMRuntime
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct TVMRuntimeCreateArgs
    {
        public string graphJsonString;
        public IntPtr moduleHandle;
        public int deviceType;
        public int deviceId;

    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct TVMRuntimeSetInputArgs
    {
        public string inputName;
        public IntPtr inputTensorHandle;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct TVMRuntimeLoadParamsArgs
    {
        public IntPtr paramPtr;
        public long size;
    }

    internal static class UnmanagedRuntimeWrapper
    {
        /// <summary>
        /// The global registry name of the tvm create func.
        /// </summary>
        private static string tvmCreateFuncName = "tvm.graph_runtime.create";

        private static IntPtr tvmCreateFuncHandle = IntPtr.Zero;

        /// <summary>
        /// Initializes the <see cref="T:TVMRuntime.UnmanagedRuntimeWrapper"/> class.
        /// </summary>
        static UnmanagedRuntimeWrapper()
        {
            UnamangedPFManagerWrapper.GetTVMRuntimeGlobalPackedFunc(tvmCreateFuncName,
                            ref tvmCreateFuncHandle);
        }

        /// <summary>
        /// TVM func call.
        /// </summary>
        /// <returns>The func call.</returns>
        /// <param name="funcHandle">Func handle.</param>
        /// <param name="args">Arguments.</param>
        /// <param name="argTypeCodes">Argument type codes.</param>
        /// <param name="numArgs">Number arguments.</param>
        /// <param name="retVal">Ret value.</param>
        /// <param name="retTypeCode">Ret type code.</param>
        [DllImport(Utils.libName)]
        private static extern int TVMFuncCall(IntPtr funcHandle,
            IntPtr args,
            [MarshalAs(UnmanagedType.LPArray)] int[] argTypeCodes,
            int numArgs,
            ref IntPtr retVal,
            ref int retTypeCode);

        /// <summary>
        /// TVM func call.
        /// </summary>
        /// <returns>The unc call.</returns>
        /// <param name="funcHandle">Func handle.</param>
        /// <param name="args">Arguments.</param>
        /// <param name="argTypeCodes">Argument type codes.</param>
        /// <param name="numArgs">Number arguments.</param>
        /// <param name="retVal">Ret value.</param>
        /// <param name="retTypeCode">Ret type code.</param>
        [DllImport(Utils.libName)]
        private static extern int TVMFuncCall(IntPtr funcHandle,
            ref IntPtr args,
            [MarshalAs(UnmanagedType.LPArray)] int[] argTypeCodes,
            int numArgs,
            ref IntPtr retVal,
            ref int retTypeCode);

        /// <summary>
        /// Invokes the TVM Runtime create packed func.
        /// </summary>
        /// <param name="funcHandle">Func handle.</param>
        /// <param name="args">Arguments.</param>
        /// <param name="argTypeCodes">Argument type codes.</param>
        /// <param name="numArgs">Number arguments.</param>
        /// <param name="retVal">Ret value.</param>
        /// <param name="retTypeCode">Ret type code.</param>
        private static void InvokeTVMRuntimeCreatePackedFunc(IntPtr funcHandle,
            IntPtr args, int[] argTypeCodes, int numArgs,
            ref IntPtr retVal, ref int retTypeCode)
        {
            int result = TVMFuncCall(funcHandle, args, argTypeCodes, numArgs,
                ref retVal, ref retTypeCode);

            Utils.CheckSuccess(0, result);
        }

        /// <summary>
        /// Invokes the TVM Runtime set input packed func.
        /// </summary>
        /// <param name="funcHandle">Func handle.</param>
        /// <param name="args">Arguments.</param>
        /// <param name="argTypeCodes">Argument type codes.</param>
        /// <param name="numArgs">Number arguments.</param>
        /// <param name="retVal">Ret value.</param>
        /// <param name="retTypeCode">Ret type code.</param>
        private static void InvokeTVMRuntimeSetInputPackedFunc(IntPtr funcHandle,
            IntPtr args, int[] argTypeCodes, int numArgs,
            ref IntPtr retVal, ref int retTypeCode)
        {
            int result = TVMFuncCall(funcHandle, args, argTypeCodes, numArgs,
                ref retVal, ref retTypeCode);

            Utils.CheckSuccess(0, result);

        }

        /// <summary>
        /// Invokes the TVM Runtime load parameter func.
        /// </summary>
        /// <param name="funcHandle">Func handle.</param>
        /// <param name="args">Arguments.</param>
        /// <param name="argTypeCodes">Argument type codes.</param>
        /// <param name="numArgs">Number arguments.</param>
        /// <param name="retVal">Ret value.</param>
        /// <param name="retTypeCode">Ret type code.</param>
        private static void InvokeTVMRuntimeLoadParamFunc(IntPtr funcHandle,
            ref IntPtr args, int[] argTypeCodes, int numArgs,
            ref IntPtr retVal, ref int retTypeCode)
        {
            int result = TVMFuncCall(funcHandle, ref args, argTypeCodes, numArgs,
                ref retVal, ref retTypeCode);

            Utils.CheckSuccess(0, result);
        }

        /// <summary>
        /// Creates the TVM Runtime.
        /// </summary>
        /// <param name="moduleHandle">Module handle.</param>
        /// <param name="graphJsonString">Graph json string.</param>
        /// <param name="ctx">Context.</param>
        /// <param name="runtimeHandle">Runtime handle.</param>
        public static void CreateTVMRuntime(IntPtr moduleHandle,
            string graphJsonString, TVMContext ctx, ref IntPtr runtimeHandle)
        {
            TVMRuntimeCreateArgs tvmCreateArgs = new TVMRuntimeCreateArgs();
            tvmCreateArgs.moduleHandle = moduleHandle;
            tvmCreateArgs.graphJsonString = graphJsonString;
            tvmCreateArgs.deviceType = (int)ctx.deviceType;
            tvmCreateArgs.deviceId = ctx.deviceId;

            int[] argTypeCodes = new int[] {
                    (int)TVMTypeCode.TVMStr, (int)TVMTypeCode.TVMModuleHandle,
                    (int)TVMDataTypeCode.Int,
                    (int)TVMDataTypeCode.Int };

            // Initialize unmanged memory to hold the struct.
            IntPtr pnt = Marshal.AllocHGlobal(Marshal.SizeOf(tvmCreateArgs));

            try
            {
                // Copy the struct to unmanaged memory.
                Marshal.StructureToPtr(tvmCreateArgs, pnt, false);

                int numArgs = 4;
                int retTypeCode = 0;

                InvokeTVMRuntimeCreatePackedFunc(tvmCreateFuncHandle,
                        pnt, argTypeCodes, numArgs,
                        ref runtimeHandle, ref retTypeCode);

                // Check Return type code
                Utils.CheckSuccess((int)TVMTypeCode.TVMModuleHandle, retTypeCode);
            }
            finally
            {
                // Free the unmanaged memory.
                Marshal.FreeHGlobal(pnt);
            }

        }

        /// <summary>
        /// Gets the TVM Runtime embeded func.
        /// </summary>
        /// <param name="funcName">Func name.</param>
        /// <param name="runtimeHandle">Runtime handle.</param>
        /// <param name="funcHandle">Func handle.</param>
        public static void GetTVMRuntimeEmbededFunc(string funcName,
                        IntPtr runtimeHandle, ref IntPtr funcHandle)
        {
            UnmanagedModuleWrapper.GetModuleEmbededFunc(runtimeHandle,
                funcName, 0, ref funcHandle);
        }

        /// <summary>
        /// Invokes the runtime run func.
        /// </summary>
        /// <param name="runFuncHandle">Run func handle.</param>
        public static void InvokeRuntimeRunFunc(IntPtr runFuncHandle)
        {
            int retTypeCode = 0;
            TVMValue emptyOutput = new TVMValue();
            UnamangedPFManagerWrapper.InvokeTVMRuntimePackedFunc(runFuncHandle,
                new TVMValue[] { }, new int[] { }, 0,
                ref emptyOutput, ref retTypeCode);

            // Check Return type code
            Utils.CheckSuccess((int)TVMTypeCode.TVMNullptr, retTypeCode);
        }

        /// <summary>
        /// Invokes the runtime set input func.
        /// </summary>
        /// <param name="setInputFuncHandle">Set input func handle.</param>
        /// <param name="inputIndex">Input index.</param>
        /// <param name="inputTensorHandle">Input tensor handle.</param>
        public static void InvokeRuntimeSetInputFunc(
                            IntPtr setInputFuncHandle,
                            int inputIndex,
                            IntPtr inputTensorHandle)
        {
            TVMValue arg0 = new TVMValue(inputIndex);

            TVMValue arg1 = new TVMValue(inputTensorHandle);
            
            int retTypeCode = 0;
            TVMValue emptyOutput = new TVMValue();

            UnamangedPFManagerWrapper.InvokeTVMRuntimePackedFunc(
                setInputFuncHandle,
                new TVMValue[] { arg0, arg1 },
                new int[] { (int)TVMDataTypeCode.Int,
                (int)TVMTypeCode.TVMNDArrayHandle}, 2,
                ref emptyOutput, ref retTypeCode);

            // Check Return type code
            Utils.CheckSuccess((int)TVMTypeCode.TVMNullptr, retTypeCode);
        }

        /// <summary>
        /// Invokes the runtime set input func.
        /// </summary>
        /// <param name="setInputFuncHandle">Set input func handle.</param>
        /// <param name="inputName">Input name.</param>
        /// <param name="inputTensorHandle">Input tensor handle.</param>
        public static void InvokeRuntimeSetInputFunc(
                            IntPtr setInputFuncHandle,
                            string inputName,
                            IntPtr inputTensorHandle)
        {
            TVMRuntimeSetInputArgs tvmSetInputArgs = new TVMRuntimeSetInputArgs();
            tvmSetInputArgs.inputName = inputName;
            tvmSetInputArgs.inputTensorHandle = inputTensorHandle;
            
            int[] argTypeCodes = new int[] {
                    (int)TVMTypeCode.TVMStr, (int)TVMTypeCode.TVMNDArrayHandle };

            // Initialize unmanged memory to hold the struct.
            IntPtr pnt = Marshal.AllocHGlobal(Marshal.SizeOf(tvmSetInputArgs));

            try
            {
                // Copy the struct to unmanaged memory.
                Marshal.StructureToPtr(tvmSetInputArgs, pnt, false);

                int numArgs = 2;
                int retTypeCode = 0;
                IntPtr retVal = IntPtr.Zero;

                InvokeTVMRuntimeSetInputPackedFunc(setInputFuncHandle,
                    pnt, argTypeCodes, numArgs,
                    ref retVal, ref retTypeCode);

                // Check Return type code
                Utils.CheckSuccess((int)TVMTypeCode.TVMNullptr, retTypeCode);
            }
            finally
            {
                // Free the unmanaged memory.
                Marshal.FreeHGlobal(pnt);
            }

        }

        /// <summary>
        /// Invokes the runtime get output func.
        /// </summary>
        /// <param name="getOutputFuncHandle">Get output func handle.</param>
        /// <param name="outputIndex">Output index.</param>
        /// <param name="outputTensor">Output tensor.</param>
        public static void InvokeRuntimeGetOutputFunc(
                            IntPtr getOutputFuncHandle,
                            int outputIndex,
                            ref NDArray outputTensor)
        {
            int retTypeCode = 0;
            TVMValue retOutput = new TVMValue();

            UnamangedPFManagerWrapper.InvokeTVMRuntimePackedFunc(getOutputFuncHandle,
                new TVMValue[] { new TVMValue(outputIndex) },
                new int[] { (int)TVMDataTypeCode.Int }, 1,
                ref retOutput, ref retTypeCode);

            // Check Return type code
            Utils.CheckSuccess((int)TVMTypeCode.TVMNDArrayHandle, retTypeCode);

            // Update the NDArray
            outputTensor.NDArrayHandle = retOutput.handle;
        }

        /// <summary>
        /// Invokes the runtime load parameter func.
        /// </summary>
        /// <param name="loadParamFuncHandle">Load parameter func handle.</param>
        /// <param name="paramDict">Parameter dict.</param>
        public static void InvokeRuntimeLoadParamFunc(
                            IntPtr loadParamFuncHandle,
                            byte [] paramDict)
        {
            // Initialize unmanged memory to hold the struct.
            int lengthArray = Marshal.SizeOf(paramDict[0]) * paramDict.Length;
            IntPtr pnt = Marshal.AllocHGlobal(lengthArray);

            TVMRuntimeLoadParamsArgs loadParamsArgs = new TVMRuntimeLoadParamsArgs();
            loadParamsArgs.paramPtr = pnt;
            loadParamsArgs.size = paramDict.Length;
            IntPtr pnt1 = Marshal.AllocHGlobal(Marshal.SizeOf(loadParamsArgs));
            
            try
            {
                // Copy the struct to unmanaged memory.
                Marshal.Copy(paramDict, 0, pnt, lengthArray);
                Marshal.StructureToPtr(loadParamsArgs, pnt1, false);

                int retTypeCode = 0;
                IntPtr retVal = IntPtr.Zero;

                InvokeTVMRuntimeLoadParamFunc(
                    loadParamFuncHandle,
                    ref pnt1,
                    new int[] { (int)TVMTypeCode.TVMBytes }, 1,
                    ref retVal, ref retTypeCode);

                // Check Return type code
                Utils.CheckSuccess((int)TVMTypeCode.TVMNullptr, retTypeCode);
            }
            finally
            {
                // Free the unmanaged memory.
                Marshal.FreeHGlobal(pnt);
                Marshal.FreeHGlobal(pnt1);
            }
        }

        /// <summary>
        /// Disposes the runtime.
        /// </summary>
        /// <param name="runtimeHandle">Runtime handle.</param>
        public static void DisposeRuntime(IntPtr runtimeHandle)
        {
            UnmanagedModuleWrapper.DisposeModule(runtimeHandle);
        }

        // TODO: Add other runtime member function as well
    }
}
