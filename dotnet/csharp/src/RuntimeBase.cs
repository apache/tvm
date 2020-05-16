using System;
using System.Runtime.InteropServices;
using static TVMRuntime.Utils;
using static TVMRuntime.PFManager;
using System.Text;
using Native;

namespace TVMRuntime
{
    // Structurs for special marshalling meant only for Runtime
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

    public class RuntimeBase
    {
        /// <summary>
        /// The global registry name of the tvm create func.
        /// </summary>
        private static string tvmCreateFuncName = "tvm.graph_runtime.create";

        private static IntPtr tvmCreateFuncHandle = IntPtr.Zero;

        protected IntPtr _runtimeHandle = IntPtr.Zero;

        /// <summary>
        /// Gets the runtime handle.
        /// </summary>
        /// <value>The runtime handle.</value>
        public IntPtr RuntimeHandle { get => _runtimeHandle; }

        /// <summary>
        /// Initializes the <see cref="T:TVMRuntime.UnmanagedRuntimeWrapper"/> class.
        /// </summary>
        static RuntimeBase()
        {
            PFManager.GetGlobalPackedFunc(tvmCreateFuncName,
                            ref tvmCreateFuncHandle);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TVMRuntime.RuntimeNative"/> class.
        /// </summary>
        protected RuntimeBase()
        {

        }

        /// <summary>
        /// Creates the TVM Runtime.
        /// </summary>
        /// <param name="moduleHandle">Module handle.</param>
        /// <param name="graphJsonString">Graph json string.</param>
        /// <param name="ctx">Context.</param>
        protected void CreateTVMRuntime(IntPtr moduleHandle,
            string graphJsonString, TVMContext ctx)
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

                int result = NativeImport.TVMFuncCall(tvmCreateFuncHandle, pnt, argTypeCodes, numArgs,
                ref _runtimeHandle, ref retTypeCode);

                Utils.CheckSuccess(0, result);

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
        /// <param name="funcHandle">Func handle.</param>
        protected void GetTVMRuntimeEmbededFunc(string funcName, ref IntPtr funcHandle)
        {
            int result = NativeImport.TVMModGetFunction(_runtimeHandle, funcName, 0,
                    ref funcHandle);
            Utils.CheckSuccess(0, result);
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
            TVMValueStruct arg0 = new TVMValueStruct(inputIndex);

            TVMValueStruct arg1 = new TVMValueStruct(inputTensorHandle);

            int retTypeCode = 0;
            TVMValueStruct emptyOutput = new TVMValueStruct();

            int result = NativeImport.TVMFuncCall(setInputFuncHandle,
                new TVMValueStruct[] { arg0, arg1 },
                new int[] { (int)TVMDataTypeCode.Int,
                (int)TVMTypeCode.TVMNDArrayHandle}, 2,
                ref emptyOutput, ref retTypeCode);
            Utils.CheckSuccess(0, result);

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

                int result = NativeImport.TVMFuncCall(setInputFuncHandle, pnt, argTypeCodes, numArgs,
                ref retVal, ref retTypeCode);

                Utils.CheckSuccess(0, result);

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
        /// Invokes the runtime load parameter func.
        /// </summary>
        /// <param name="loadParamFuncHandle">Load parameter func handle.</param>
        /// <param name="paramDict">Parameter dict.</param>
        public static void InvokeRuntimeLoadParamFunc(
                            IntPtr loadParamFuncHandle,
                            byte[] paramDict)
        {
            // Initialize unmanged memory to hold the struct.
            int lengthArray = Marshal.SizeOf(paramDict[0]) * paramDict.Length;
            IntPtr pnt = Marshal.AllocHGlobal(lengthArray);

            TVMRuntimeByteArray loadParamsArgs = new TVMRuntimeByteArray();
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

                int result = NativeImport.TVMFuncCall(
                    loadParamFuncHandle,
                    ref pnt1,
                    new int[] { (int)TVMTypeCode.TVMBytes }, 1,
                    ref retVal, ref retTypeCode);

                Utils.CheckSuccess(0, result);

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
        protected void DisposeRuntimeHandle()
        {
            if (!IntPtr.Zero.Equals(_runtimeHandle))
            {
                int result = NativeImport.TVMModFree(_runtimeHandle);
                Utils.CheckSuccess(0, result);
                _runtimeHandle = IntPtr.Zero;
            }
        }

        /// <summary>
        /// Releases unmanaged resources and performs other cleanup operations before the
        /// <see cref="T:TVMRuntime.RuntimeNative"/> is reclaimed by garbage collection.
        /// </summary>
        ~RuntimeBase()
        {
            DisposeRuntimeHandle();
        }
    }
}
