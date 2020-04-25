using System;
using System.Runtime.InteropServices;
using static TVMRuntime.Utils;
using TVMRuntime;

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
        public long v_int64;

        [FieldOffset(0)]
        public double v_float64;

        [FieldOffset(0)]
        public UIntPtr v_handle;

        public TVMValue(long vinp_int64) : this()
        {
            v_int64 = vinp_int64;
        }

        public TVMValue(double vinp_float64) : this()
        {
            v_float64 = vinp_float64;
        }

        public TVMValue(UIntPtr vinp_handle) : this()
        {
            v_handle = vinp_handle;
        }

    }

    public static class UnamangedPFManagerWrapper
    {
        /// <summary>
        ///
        /// </summary>
        /// <param name="name">The name of the function.</param>
        /// <param name="func_handle">the result function pointer, NULL if it does not exist.</param>
        /// <remarks>
        /// @brief Get a global function.@note The function handle of global function is managed by TVM runtime,
        /// So TVMFuncFree should not be called when it get deleted.
        /// </remarks>
        [DllImport(Utils.libName)]
        private static extern int TVMFuncGetGlobal([MarshalAs(UnmanagedType.LPStr)] string name, ref UIntPtr func_handle);


        /// <summary>
        ///
        /// </summary>
        /// <param name="func_handle">The function handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        /// <remarks>
        /// @brief Free the function when it is no longer needed.
        /// </remarks>
        [DllImport(Utils.libName)]
        private static extern int TVMFuncFree(UIntPtr func_handle);

        /// <summary>
        /// TVM func call.
        /// </summary>
        /// <returns>The func call.</returns>
        /// <param name="func_handle">Func.</param>
        /// <param name="args">Arguments.</param>
        /// <param name="arg_type_codes">Argument type codes.</param>
        /// <param name="num_args">Number arguments.</param>
        /// <param name="ret_val">Ret value.</param>
        /// <param name="ret_type_code">Ret type code.</param>
        [DllImport(Utils.libName)]
        private static extern int TVMFuncCall(UIntPtr func_handle,
            [MarshalAs(UnmanagedType.LPArray)] TVMValue[] args,
            [MarshalAs(UnmanagedType.LPArray)] int[] arg_type_codes,
            int num_args,
            ref TVMValue ret_val,
            ref int ret_type_code);

        /// <summary>
        /// TVM func call.
        /// </summary>
        /// <returns>The func call.</returns>
        /// <param name="func_handle">Func.</param>
        /// <param name="args">Arguments.</param>
        /// <param name="arg_type_codes">Argument type codes.</param>
        /// <param name="num_args">Number arguments.</param>
        /// <param name="ret_val">Ret value.</param>
        /// <param name="ret_type_code">Ret type code.</param>
        [DllImport(Utils.libName)]
        private static extern int TVMFuncCall(UIntPtr func_handle,
            [MarshalAs(UnmanagedType.LPArray)] string[] args,
            [MarshalAs(UnmanagedType.LPArray)] int[] arg_type_codes,
            int num_args,
            ref TVMValue ret_val,
            ref int ret_type_code);

        /// <summary>
        /// Gets the TVM Runtime global packed func.
        /// </summary>
        /// <param name="func_name">Func name.</param>
        /// <param name="func_handle">Func handle.</param>
        public static void GetTVMRuntimeGlobalPackedFunc(string func_name, ref UIntPtr func_handle)
        {
            TVMFuncGetGlobal(func_name, ref func_handle);
        }

        /// <summary>
        /// Disposes the TVM Runtime func handle.
        /// </summary>
        /// <param name="func_handle">Func handle.</param>
        public static void DisposeTVMRuntimeFuncHandle(UIntPtr func_handle)
        {
            TVMFuncFree(func_handle);
        }

        // TODO: String marshalling is tricky, currently assume only input has
        //       so divided the func call to two types,
        //       and there is no return value as string.
        //       Later based on need add the functionality.

        /// <summary>
        /// Invokes the TVM Runtime packed func.
        /// </summary>
        /// <param name="func_handle">Func handle.</param>
        /// <param name="args">Arguments.</param>
        /// <param name="arg_type_codes">Argument type codes.</param>
        /// <param name="num_args">Number arguments.</param>
        /// <param name="ret_val">Ret value.</param>
        /// <param name="ret_type_code">Ret type code.</param>
        public static void InvokeTVMRuntimePackedFunc(UIntPtr func_handle,
            TVMValue[] args, int[] arg_type_codes, int num_args,
            ref TVMValue ret_val, ref int ret_type_code)
        {
            TVMFuncCall(func_handle, args, arg_type_codes, num_args,
                ref ret_val, ref ret_type_code);
        }

        /// <summary>
        /// Invokes the TVM Runtime packed func with string input.
        /// </summary>
        /// <param name="func_handle">Func handle.</param>
        /// <param name="args">Arguments.</param>
        /// <param name="arg_type_codes">Argument type codes.</param>
        /// <param name="num_args">Number arguments.</param>
        /// <param name="ret_val">Ret value.</param>
        /// <param name="ret_type_code">Ret type code.</param>
        public static void InvokeTVMRuntimePackedFuncStr(UIntPtr func_handle,
            string[] args, int[] arg_type_codes, int num_args,
            ref TVMValue ret_val, ref int ret_type_code)
        {
            TVMFuncCall(func_handle, args, arg_type_codes, num_args,
                ref ret_val, ref ret_type_code);
        }
    }
}
