using System;
using System.IO;
using System.Runtime.InteropServices;
using static TVMRuntime.Utils;

namespace TVMRuntime
{
    public struct RuntimeParams
    {
        /// <summary>
        /// The mod lib path.
        /// </summary>
        public string modLibPath;

        /// <summary>
        /// (Optional) The mod lib format.
        /// </summary>
        public string modLibFormat;

        /// <summary>
        /// The graph json path.
        /// </summary>
        public string graphJsonPath;

        /// <summary>
        /// The parameter dict path.
        /// </summary>
        public string paramDictPath;

        /// <summary>
        /// The context.
        /// </summary>
        public TVMContext context;
    }


    public class Runtime
    {
        private Module _module = null;
        private byte[] _paramsDict;
        private string _graphJsonString;
        private bool _isInstantiated = false;
        private UIntPtr _runtimeHandle = UIntPtr.Zero;

        // all embeded func handles
        // TODO: Convert to Map[name, funchandles]
        private UIntPtr _runtimeRunFuncHandle = UIntPtr.Zero;
        private UIntPtr _runtimeSetInputFuncHandle = UIntPtr.Zero;
        private UIntPtr _runtimeLoadParamHandle = UIntPtr.Zero;
        private UIntPtr _runtimeGetOutputFuncHandle = UIntPtr.Zero;

        /// <summary>
        /// Creates the instance.
        /// </summary>
        /// <param name="runtimeParam">Runtime parameter.</param>
        private void CreateInstance(RuntimeParams runtimeParam)
        {
            string errMsg = "";
            if (!ValidateInputs(runtimeParam, ref errMsg))
            {
                throw new System.ArgumentException("Please provide valid path for ", errMsg);
            }
            _module = new Module(runtimeParam.modLibPath,
                                runtimeParam.modLibFormat);

            _paramsDict = Utils.ReadByteArrayFromFile(runtimeParam.paramDictPath);

            _graphJsonString = Utils.ReadStringFromFile(runtimeParam.graphJsonPath);

            UnmanagedRuntimeWrapper.CreateTVMRuntime(_module.ModuleHandle,
                                            _graphJsonString,
                                            runtimeParam.context,
                                            ref _runtimeHandle);

            // Load all embeded func handles
            UnmanagedRuntimeWrapper.GetTVMRuntimeEmbededFunc("run",
                _runtimeHandle, ref _runtimeRunFuncHandle);

            UnmanagedRuntimeWrapper.GetTVMRuntimeEmbededFunc("set_input",
                _runtimeHandle, ref _runtimeSetInputFuncHandle);

            UnmanagedRuntimeWrapper.GetTVMRuntimeEmbededFunc("load_params",
                _runtimeHandle, ref _runtimeLoadParamHandle);

            UnmanagedRuntimeWrapper.GetTVMRuntimeEmbededFunc("get_output",
                _runtimeHandle, ref _runtimeGetOutputFuncHandle);

            _isInstantiated = true;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TVMRuntime.Runtime"/> class.
        /// </summary>
        /// <param name="runtimeParam">Runtime parameter.</param>
        public Runtime(RuntimeParams runtimeParam)
        {
            CreateInstance(runtimeParam);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TVMRuntime.Runtime"/> class.
        /// </summary>
        public Runtime()
        {

        }

        /// <summary>
        /// Create the specified runtimeParam.
        /// </summary>
        /// <returns>The create.</returns>
        /// <param name="runtimeParam">Runtime parameter.</param>
        public Runtime Create(RuntimeParams runtimeParam)
        {
            if (!_isInstantiated) { CreateInstance(runtimeParam); }
            return this;
        }

        /// <summary>
        /// Gets the runtime handle.
        /// </summary>
        /// <value>The runtime handle.</value>
        public UIntPtr RuntimeHandle { get => _runtimeHandle; }

        /// <summary>
        /// Run this instance.
        /// </summary>
        public void Run ()
        {
            if (!_isInstantiated) { Console.WriteLine("Not instantiated yet!"); return; }

            UnmanagedRuntimeWrapper.InvokeRuntimeRunFunc(_runtimeRunFuncHandle);
        }

        /// <summary>
        /// Sets the input.
        /// </summary>
        /// <param name="inputName">Input name.</param>
        /// <param name="inputTensor">Input tensor handle.</param>
        public void SetInput(string inputName, NDArray inputTensor)
        {
            if (!_isInstantiated) { Console.WriteLine("Not instantiated yet!"); return; }

            UnmanagedRuntimeWrapper.InvokeRuntimeSetInputFunc(_runtimeSetInputFuncHandle,
                inputName, inputTensor.NDArrayHandle);
        }

        /// <summary>
        /// Loads the parameters.
        /// </summary>
        public void LoadParams()
        {
            if (!_isInstantiated) { Console.WriteLine("Not instantiated yet!"); return; }

            UnmanagedRuntimeWrapper.InvokeRuntimeLoadParamFunc(_runtimeLoadParamHandle,
                _paramsDict);
        }

        /// <summary>
        /// Gets the output.
        /// </summary>
        /// <param name="outputIndex">Output index.</param>
        /// <param name="outputTensor">Output tensor.</param>
        public void GetOutput(int outputIndex, ref NDArray outputTensor)
        {
            if (!_isInstantiated) { Console.WriteLine("Not instantiated yet!"); return; }

            UnmanagedRuntimeWrapper.InvokeRuntimeGetOutputFunc(_runtimeGetOutputFuncHandle,
                outputIndex, ref outputTensor);
        }

        /// <summary>
        /// Disposes the runtime.
        /// </summary>
        public void DisposeRuntime()
        {
            if (_isInstantiated)
            {
                // Release all resources from runtime module
                PFManager.DisposePackedFunc(_runtimeRunFuncHandle);
                PFManager.DisposePackedFunc(_runtimeSetInputFuncHandle);
                PFManager.DisposePackedFunc(_runtimeLoadParamHandle);
                PFManager.DisposePackedFunc(_runtimeGetOutputFuncHandle);
                UnmanagedRuntimeWrapper.DisposeRuntime(_runtimeHandle);
                _runtimeHandle = UIntPtr.Zero;
                _module.DisposeModule();
                _isInstantiated = false;
            }
        }

        /// <summary>
        /// Validates the inputs.
        /// </summary>
        /// <returns><c>true</c>, if inputs are valid, <c>false</c> otherwise.</returns>
        /// <param name="runtimeParams">Runtime parameters.</param>
        public static bool ValidateInputs(RuntimeParams runtimeParams, ref string errMsg)
        {
            if ((!File.Exists(runtimeParams.modLibPath)))
            {
                errMsg = "RuntimeParams.modLibPath : module(lib) file path.";
                return false;
            }
            if ((!File.Exists(runtimeParams.graphJsonPath)))
            {
                errMsg = "RuntimeParams.graphJsonPath : graph(json) file path.";
                return false;
            }
            if ((!File.Exists(runtimeParams.paramDictPath)))
            {
                errMsg = "RuntimeParams.paramDictPath : params file path.";
                return false;
            }
            return true;
        }


        /// <summary>
        /// Releases unmanaged resources and performs other cleanup operations before the
        /// <see cref="T:TVMRuntime.Runtime"/> is reclaimed by garbage collection.
        /// </summary>
        ~Runtime()
        {
            DisposeRuntime();
        }
    }
}
