using System;
using System.Runtime.InteropServices;

namespace TVMRuntime
{
    /// <summary>
    ///
    /// </summary>
    /// <remarks>
    /// @brief Union type of values
    /// being passed through API and function calls.
    /// </remarks>
    [StructLayout(LayoutKind.Explicit)]
    public partial struct TVMValue
    {
        [FieldOffset(0)]
        public long vInt64;

        [FieldOffset(0)]
        public double vFloat64;

        [FieldOffset(0)]
        public IntPtr handle;

        public TVMValue(long inpInt64) : this()
        {
            vInt64 = inpInt64;
        }

        public TVMValue(double vinpFloat64) : this()
        {
            vFloat64 = vinpFloat64;
        }

        public TVMValue(IntPtr vinpHandle) : this()
        {
            handle = vinpHandle;
        }

    }

    public static class UnamangedPFManagerWrapper
    {
        /// <summary>
        ///
        /// </summary>
        /// <param name="name">The name of the function.</param>
        /// <param name="funcHandle">the result function pointer, NULL if it does not exist.</param>
        /// <remarks>
        /// @brief Get a global function.@note The function handle of global function is managed by TVM runtime,
        /// So TVMFuncFree should not be called when it get deleted.
        /// </remarks>
        [DllImport(Utils.libName)]
        private static extern int TVMFuncGetGlobal([MarshalAs(UnmanagedType.LPStr)] string name, ref UIntPtr funcHandle);


        /// <summary>
        ///
        /// </summary>
        /// <param name="funcHandle">The function handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        /// <remarks>
        /// @brief Free the function when it is no longer needed.
        /// </remarks>
        [DllImport(Utils.libName)]
        private static extern int TVMFuncFree(UIntPtr funcHandle);

        /// <summary>
        /// TVM func call.
        /// </summary>
        /// <returns>The func call.</returns>
        /// <param name="funcHandle">Func.</param>
        /// <param name="args">Arguments.</param>
        /// <param name="argTypeCodes">Argument type codes.</param>
        /// <param name="numArgs">Number arguments.</param>
        /// <param name="retVal">Ret value.</param>
        /// <param name="retTypeCode">Ret type code.</param>
        [DllImport(Utils.libName)]
        private static extern int TVMFuncCall(UIntPtr funcHandle,
            [MarshalAs(UnmanagedType.LPArray)] TVMValue[] args,
            [MarshalAs(UnmanagedType.LPArray)] int[] argTypeCodes,
            int numArgs,
            ref TVMValue retVal,
            ref int retTypeCode);

        /// <summary>
        /// TVM func call.
        /// </summary>
        /// <returns>The func call.</returns>
        /// <param name="funcHandle">Func.</param>
        /// <param name="args">Arguments.</param>
        /// <param name="argTypeCodes">Argument type codes.</param>
        /// <param name="numArgs">Number arguments.</param>
        /// <param name="retVal">Ret value.</param>
        /// <param name="retTypeCode">Ret type code.</param>
        [DllImport(Utils.libName)]
        private static extern int TVMFuncCall(UIntPtr funcHandle,
            [MarshalAs(UnmanagedType.LPArray)] string[] args,
            [MarshalAs(UnmanagedType.LPArray)] int[] argTypeCodes,
            int numArgs,
            ref TVMValue retVal,
            ref int retTypeCode);

        /// <summary>
        /// Gets the TVM Runtime global packed func.
        /// </summary>
        /// <param name="funcName">Func name.</param>
        /// <param name="funcHandle">Func handle.</param>
        public static void GetTVMRuntimeGlobalPackedFunc(string funcName, ref UIntPtr funcHandle)
        {
            TVMFuncGetGlobal(funcName, ref funcHandle);
        }

        /// <summary>
        /// Disposes the TVM Runtime func handle.
        /// </summary>
        /// <param name="funcHandle">Func handle.</param>
        public static void DisposeTVMRuntimeFuncHandle(UIntPtr funcHandle)
        {
            TVMFuncFree(funcHandle);
        }

        // TODO: String marshalling is tricky, currently assume only input has
        //       so divided the func call to two types,
        //       and there is no return value as string.
        //       Later based on need add the functionality.

        /// <summary>
        /// Invokes the TVM Runtime packed func.
        /// </summary>
        /// <param name="funcHandle">Func handle.</param>
        /// <param name="args">Arguments.</param>
        /// <param name="argTypeCodes">Argument type codes.</param>
        /// <param name="numArgs">Number arguments.</param>
        /// <param name="retVal">Ret value.</param>
        /// <param name="retTypeCode">Ret type code.</param>
        public static void InvokeTVMRuntimePackedFunc(UIntPtr funcHandle,
            TVMValue[] args, int[] argTypeCodes, int numArgs,
            ref TVMValue retVal, ref int retTypeCode)
        {
            TVMFuncCall(funcHandle, args, argTypeCodes, numArgs,
                ref retVal, ref retTypeCode);
        }

        /// <summary>
        /// Invokes the TVM Runtime packed func with string input.
        /// </summary>
        /// <param name="funcHandle">Func handle.</param>
        /// <param name="args">Arguments.</param>
        /// <param name="argTypeCodes">Argument type codes.</param>
        /// <param name="numArgs">Number arguments.</param>
        /// <param name="retVal">Ret value.</param>
        /// <param name="retTypeCode">Ret type code.</param>
        public static void InvokeTVMRuntimePackedFuncStr(UIntPtr funcHandle,
            string[] args, int[] argTypeCodes, int numArgs,
            ref TVMValue retVal, ref int retTypeCode)
        {
            TVMFuncCall(funcHandle, args, argTypeCodes, numArgs,
                ref retVal, ref retTypeCode);
        }
    }
}
