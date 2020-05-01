using System;
using System.IO;
using System.Collections.Generic;

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
        private IntPtr _runtimeHandle = IntPtr.Zero;

        // all embeded func handles
        private Dictionary<string, IntPtr> _funcHandleDict = BuildDictionary();

        /// <summary>
        /// Builds the dictionary for embeded functions in runtime.
        /// </summary>
        /// <returns>The dictionary.</returns>
        private static Dictionary<string, IntPtr> BuildDictionary()
        {
            var elements = new Dictionary<string, IntPtr>();

            elements.Add(key: "run", value: new IntPtr());
            elements.Add(key: "set_input", value: new IntPtr());
            elements.Add(key: "load_params", value: new IntPtr());
            elements.Add(key: "get_output", value: new IntPtr());

            return elements;
        }

        /// <summary>
        /// Loads the func handles through dictionary.
        /// </summary>
        /// <param name="runtime">Runtime.</param>
        private static void LoadFuncHandlesThruDictionary(Runtime runtime)
        {
            List<string> keys = new List<string>(runtime._funcHandleDict.Keys);
            foreach (string funcName in keys)
            {
                IntPtr handle = IntPtr.Zero;

                UnmanagedRuntimeWrapper.GetTVMRuntimeEmbededFunc(funcName,
                runtime._runtimeHandle, ref handle);

                runtime._funcHandleDict[funcName] = handle;
            }
        }

        /// <summary>
        /// Releases the func handles through dictionary.
        /// </summary>
        /// <param name="runtime">Runtime.</param>
        private static void ReleaseFuncHandlesThruDictionary(Runtime runtime)
        {
            foreach (IntPtr funcHandle in runtime._funcHandleDict.Values)
            {
                PFManager.DisposePackedFunc(funcHandle);
            }
            runtime._funcHandleDict.Clear();
        }

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

            // Load Module
            _module = new Module(runtimeParam.modLibPath,
                                runtimeParam.modLibFormat);

            _paramsDict = Utils.ReadByteArrayFromFile(runtimeParam.paramDictPath);

            _graphJsonString = Utils.ReadStringFromFile(runtimeParam.graphJsonPath);

            // Create Runtime
            UnmanagedRuntimeWrapper.CreateTVMRuntime(_module.ModuleHandle,
                                            _graphJsonString,
                                            runtimeParam.context,
                                            ref _runtimeHandle);

            // Load all required embeded func handles
            LoadFuncHandlesThruDictionary(this);

            _isInstantiated = true;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TVMRuntime.Runtime"/> class.
        /// </summary>
        /// <param name="runtimeParam">Runtime parameter.</param>
        private Runtime(RuntimeParams runtimeParam)
        {
            CreateInstance(runtimeParam);
        }

        /// <summary>
        /// Create the specified runtimeParam.
        /// </summary>
        /// <returns>The create.</returns>
        /// <param name="runtimeParam">Runtime parameter.</param>
        public static Runtime Create(RuntimeParams runtimeParam)
        {
            return new Runtime(runtimeParam);
        }

        /// <summary>
        /// Gets the runtime handle.
        /// </summary>
        /// <value>The runtime handle.</value>
        public IntPtr RuntimeHandle { get => _runtimeHandle; }

        /// <summary>
        /// Run this instance.
        /// </summary>
        public void Run ()
        {
            if (!_isInstantiated) { throw new System.NullReferenceException("Runtime not initialized"); }

            UnmanagedRuntimeWrapper.InvokeRuntimeRunFunc(_funcHandleDict["run"]);
        }

        /// <summary>
        /// Sets the input.
        /// </summary>
        /// <param name="inputName">Input name.</param>
        /// <param name="inputTensor">Input tensor handle.</param>
        public void SetInput(string inputName, NDArray inputTensor)
        {
            if (!_isInstantiated) { throw new System.NullReferenceException("Runtime not initialized"); }

            UnmanagedRuntimeWrapper.InvokeRuntimeSetInputFunc(_funcHandleDict["set_input"],
                inputName, inputTensor.NDArrayHandle);
        }

        /// <summary>
        /// Loads the parameters.
        /// </summary>
        public void LoadParams()
        {
            if (!_isInstantiated) { throw new System.NullReferenceException("Runtime not initialized"); }

            UnmanagedRuntimeWrapper.InvokeRuntimeLoadParamFunc(_funcHandleDict["load_params"],
                _paramsDict);
        }

        /// <summary>
        /// Gets the output.
        /// </summary>
        /// <returns>The output.</returns>
        /// <param name="outputIndex">Output index.</param>
        public NDArray GetOutput(int outputIndex)
        {
            if (!_isInstantiated) { throw new System.NullReferenceException("Runtime not initialized"); }

            NDArray outputTensor = NDArray.Empty();
            UnmanagedRuntimeWrapper.InvokeRuntimeGetOutputFunc(_funcHandleDict["get_output"],
                outputIndex, ref outputTensor);

            return outputTensor;
        }

        /// <summary>
        /// Disposes the runtime.
        /// </summary>
        public void DisposeRuntime()
        {
            if (_isInstantiated)
            {
                // Release all resources from runtime module
                ReleaseFuncHandlesThruDictionary(this);
                UnmanagedRuntimeWrapper.DisposeRuntime(_runtimeHandle);
                _runtimeHandle = IntPtr.Zero;
                _module.DisposeModule();
                _paramsDict = null;
                _graphJsonString = null;
                _isInstantiated = false;
            }
        }

        /// <summary>
        /// Validates the inputs.
        /// </summary>
        /// <returns><c>true</c>, if inputs are valid, <c>false</c> otherwise.</returns>
        /// <param name="runtimeParams">Runtime parameters.</param>
        private static bool ValidateInputs(RuntimeParams runtimeParams, ref string errMsg)
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
