using System;
using System.Runtime.InteropServices;
using TVMRuntime;

namespace Native
{
    // Common Marshalling structures to be used between TVMRuntime
    /// <summary>
    ///
    /// </summary>
    /// <remarks>
    /// @brief Union type of values
    /// being passed through API and function calls.
    /// </remarks>
    [StructLayout(LayoutKind.Explicit)]
    public partial struct TVMValueStruct
    {
        [FieldOffset(0)]
        public long vInt64;

        [FieldOffset(0)]
        public double vFloat64;

        [FieldOffset(0)]
        public IntPtr handle;

        public TVMValueStruct(long vinpInt64) : this()
        {
            vInt64 = vinpInt64;
        }

        public TVMValueStruct(double vinpFloat64) : this()
        {
            vFloat64 = vinpFloat64;
        }

        public TVMValueStruct(IntPtr vinpHandle) : this()
        {
            handle = vinpHandle;
        }

    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct TVMRuntimeByteArray
    {
        public IntPtr paramPtr;
        public long size;
    }


    internal static class NativeImport
    {
        // Module Imports

        /// <summary>
        ///
        /// </summary>
        /// <param name="fileName">The file name to load the module from.</param>
        /// <param name="format">The format of the module.</param>
        /// <param name="outHandle">The result module</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        /// <remarks>
        /// @brief Load module from file.@note The resulting module do not contain import relation.
        /// It can be reconstructed by TVMModImport.
        /// </remarks>
        [DllImport(Utils.libName)]
        internal static extern int TVMModLoadFromFile([MarshalAs(UnmanagedType.LPStr)] string fileName,
            [MarshalAs(UnmanagedType.LPStr)] string format, ref IntPtr outHandle);


        /// <summary>
        ///
        /// </summary>
        /// <param name="mod">The module handle.</param>
        /// <param name="dep">The dependent module to be imported.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        /// <remarks>
        /// @brief Add dep to mod's dependency.
        /// This allows functions in this module to use modules.
        /// </remarks>
        [DllImport(Utils.libName)]
        internal static extern int TVMModImport(IntPtr mod, IntPtr dep);


        /// <summary>
        ///
        /// </summary>
        /// <param name="mod">The module handle.</param>
        /// <param name="funcName">The name of the function.</param>
        /// <param name="queryImports">Whether to query imported modules</param>
        /// <param name="outHandle">The result function, 
        /// can be NULL if it is not available.</param>
        /// <returns>0 when no error is thrown, -1 when failure happens</returns>
        /// <remarks>
        /// @brief Get function from the module.
        /// </remarks>
        [DllImport(Utils.libName)]
        internal static extern int TVMModGetFunction(IntPtr mod,
            [MarshalAs(UnmanagedType.LPStr)] string funcName,
            int queryImports, ref IntPtr outHandle);


        /// <summary>
        ///
        /// </summary>
        /// <param name="mod">The module to be freed.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        /// <remarks>
        /// @brief Free the Module@note This may not free up the module's resources.
        /// If there is active TVMFunctionHandle uses the module
        /// Or if this module is imported by another active module.
        /// The all functions remains valid until TVMFuncFree is called.
        /// </remarks>
        [DllImport(Utils.libName)]
        internal static extern int TVMModFree(IntPtr mod);

        // Packed Function related imports
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
        internal static extern int TVMFuncGetGlobal([MarshalAs(UnmanagedType.LPStr)] string name, ref IntPtr funcHandle);


        /// <summary>
        ///
        /// </summary>
        /// <param name="funcHandle">The function handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        /// <remarks>
        /// @brief Free the function when it is no longer needed.
        /// </remarks>
        [DllImport(Utils.libName)]
        internal static extern int TVMFuncFree(IntPtr funcHandle);

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
        internal static extern int TVMFuncCall(IntPtr funcHandle,
            [MarshalAs(UnmanagedType.LPArray)] TVMValueStruct[] args,
            [MarshalAs(UnmanagedType.LPArray)] int[] argTypeCodes,
            int numArgs,
            ref TVMValueStruct retVal,
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
        internal static extern int TVMFuncCall(IntPtr funcHandle,
            [MarshalAs(UnmanagedType.LPArray)] string[] args,
            [MarshalAs(UnmanagedType.LPArray)] int[] argTypeCodes,
            int numArgs,
            ref TVMValueStruct retVal,
            ref int retTypeCode);


        // Runtime specific customized imports

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
        internal static extern int TVMFuncCall(IntPtr funcHandle,
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
        internal static extern int TVMFuncCall(IntPtr funcHandle,
            ref IntPtr args,
            [MarshalAs(UnmanagedType.LPArray)] int[] argTypeCodes,
            int numArgs,
            ref IntPtr retVal,
            ref int retTypeCode);

        // NDArray specific imports
        /// <summary>
        /// TVM Array alloc.
        /// </summary>
        /// <returns>The rray alloc.</returns>
        /// <param name="shape">Shape.</param>
        /// <param name="ndim">Ndim.</param>
        /// <param name="dtypeCode">Dtype code.</param>
        /// <param name="dtypeBits">Dtype bits.</param>
        /// <param name="dtypeLanes">Dtype lanes.</param>
        /// <param name="deviceType">Device type.</param>
        /// <param name="deviceId">Device identifier.</param>
        /// <param name="output">Output.</param>
        [DllImport(Utils.libName)]
        internal static extern int TVMArrayAlloc(
                          [MarshalAs(UnmanagedType.LPArray)] long[] shape,
                          int ndim,
                          int dtypeCode,
                          int dtypeBits,
                          int dtypeLanes,
                          int deviceType,
                          int deviceId,
                          ref IntPtr output);

        /// <summary>
        /// TVM Array free.
        /// </summary>
        /// <returns>The rray free.</returns>
        /// <param name="tensorHandle">Tensor handle.</param>
        [DllImport(Utils.libName)]
        internal static extern int TVMArrayFree(
                          IntPtr tensorHandle);

        /// <summary>
        /// TVMArray copy from bytes.
        /// </summary>
        /// <returns>The NDArray copy from bytes.</returns>
        /// <param name="handle">Handle.</param>
        /// <param name="data">Data.</param>
        /// <param name="nbytes">Nbytes.</param>
        [DllImport(Utils.libName)]
        internal static extern int TVMArrayCopyFromBytes(IntPtr handle,
                                  IntPtr data,
                                  long nbytes);

        /// <summary>
        /// TVMArray copy to bytes.
        /// </summary>
        /// <returns>The NDArray copy to bytes.</returns>
        /// <param name="handle">Handle.</param>
        /// <param name="data">Data.</param>
        /// <param name="nbytes">Nbytes.</param>
        [DllImport(Utils.libName)]
        internal static extern int TVMArrayCopyToBytes(IntPtr handle,
                                  IntPtr data,
                                  long nbytes);
    }
}
