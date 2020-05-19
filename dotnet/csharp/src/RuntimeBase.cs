using System;
using System.Runtime.InteropServices;
using static TVMRuntime.Utils;
using static TVMRuntime.PFManager;
using System.Text;
using Native;

namespace TVMRuntime
{
    public class RuntimeBase
    {
        /// <summary>
        /// The global registry name of the tvm create func.
        /// </summary>
        private static string tvmCreateFuncName = "tvm.graph_runtime.create";

        private static PackedFunction tvmCreateFunc = new PackedFunction(IntPtr.Zero);

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
            IntPtr funcHandle = IntPtr.Zero;
            PFManager.GetGlobalPackedFunc(tvmCreateFuncName,
                            ref funcHandle);
            tvmCreateFunc.FuncHandle = funcHandle;
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
        /// <param name="module">Module.</param>
        /// <param name="graphJsonString">Graph json string.</param>
        /// <param name="ctx">Context.</param>
        protected void CreateTVMRuntime(Module module,
            string graphJsonString, TVMContext ctx)
        {
            _runtimeHandle = tvmCreateFunc.Invoke(graphJsonString, module,
                                (int)ctx.deviceType, ctx.deviceId).AsHandle();
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
