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


    public class Runtime : RuntimeBase
    {
        private byte[] _paramsDict;
        private string _graphJsonString;
        private bool _isInstantiated = false;

        // all embeded func handles
        private Dictionary<string, PackedFunction> _funcDict = BuildDictionary();

        /// <summary>
        /// Builds the dictionary for embeded functions in runtime.
        /// </summary>
        /// <returns>The dictionary.</returns>
        private static Dictionary<string, PackedFunction> BuildDictionary()
        {
            var elements = new Dictionary<string, PackedFunction>();

            elements.Add(key: "run", value: new PackedFunction(IntPtr.Zero));
            elements.Add(key: "set_input", value: new PackedFunction(IntPtr.Zero));
            elements.Add(key: "load_params", value: new PackedFunction(IntPtr.Zero));
            elements.Add(key: "get_output", value: new PackedFunction(IntPtr.Zero));

            return elements;
        }

        /// <summary>
        /// Loads the func handles through dictionary.
        /// </summary>
        /// <param name="runtime">Runtime.</param>
        private static void LoadFuncHandlesThruDictionary(Runtime runtime)
        {
            List<string> keys = new List<string>(runtime._funcDict.Keys);
            foreach (string funcName in keys)
            {
                IntPtr handle = IntPtr.Zero;

                runtime.GetTVMRuntimeEmbededFunc(funcName, ref handle);

                runtime._funcDict[funcName].FuncHandle = handle;
            }
        }

        /// <summary>
        /// Releases the func handles through dictionary.
        /// </summary>
        /// <param name="runtime">Runtime.</param>
        private static void ReleaseFuncHandlesThruDictionary(Runtime runtime)
        {
            foreach (PackedFunction func in runtime._funcDict.Values)
            {
                func.Dispose();
            }
            runtime._funcDict.Clear();
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
            Module module = new Module(runtimeParam.modLibPath,
                                runtimeParam.modLibFormat);

            _paramsDict = Utils.ReadByteArrayFromFile(runtimeParam.paramDictPath);

            _graphJsonString = Utils.ReadStringFromFile(runtimeParam.graphJsonPath);

            // Create Runtime
            CreateTVMRuntime(module, _graphJsonString, runtimeParam.context);

            // Release Module as, we no longer need it
            module.DisposeModule();

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
        /// Run this instance.
        /// </summary>
        public void Run ()
        {
            if (!_isInstantiated) { throw new System.NullReferenceException("Runtime not initialized"); }

            _funcDict["run"].Invoke();
        }

        /// <summary>
        /// Sets the input.
        /// </summary>
        /// <param name="inputName">Input name.</param>
        /// <param name="inputTensor">Input tensor handle.</param>
        public void SetInput(string inputName, NDArray inputTensor)
        {
            if (!_isInstantiated) { throw new System.NullReferenceException("Runtime not initialized"); }

            _funcDict["set_input"].Invoke(inputName, inputTensor);
        }

        /// <summary>
        /// Loads the parameters.
        /// </summary>
        public void LoadParams()
        {
            if (!_isInstantiated) { throw new System.NullReferenceException("Runtime not initialized"); }

            _funcDict["load_params"].Invoke(_paramsDict);
        }

        /// <summary>
        /// Gets the output.
        /// </summary>
        /// <returns>The output.</returns>
        /// <param name="outputIndex">Output index.</param>
        public NDArray GetOutput(int outputIndex)
        {
            if (!_isInstantiated) { throw new System.NullReferenceException("Runtime not initialized"); }

            return _funcDict["get_output"].Invoke(outputIndex).AsNDArray();
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
                DisposeRuntimeHandle();
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
