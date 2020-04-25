using System;
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
        private Module module = null;
        private byte[] paramsDict;
        private string graphJsonString;
        private bool isInstantiated = false;
        private UIntPtr runtimeHandle = UIntPtr.Zero;

        // all embeded func handles
        private UIntPtr runtimeRunFuncHandle = UIntPtr.Zero;
        private UIntPtr runtimeSetInputFuncHandle = UIntPtr.Zero;
        private UIntPtr runtimeLoadParamHandle = UIntPtr.Zero;
        private UIntPtr runtimeGetOutputFuncHandle = UIntPtr.Zero;

        /// <summary>
        /// Creates the instance.
        /// </summary>
        /// <param name="runtimeParam">Runtime parameter.</param>
        private void CreateInstance(RuntimeParams runtimeParam)
        {
            module = new Module(runtimeParam.modLibPath,
                                runtimeParam.modLibFormat);

            paramsDict = Utils.ReadByteArrayFromFile(runtimeParam.paramDictPath);

            graphJsonString = Utils.ReadStringFromFile(runtimeParam.graphJsonPath);

            UnmanagedRuntimeWrapper.CreateTVMRuntime(module.ModuleHandle,
                                            graphJsonString,
                                            runtimeParam.context,
                                            ref runtimeHandle);

            // Load all embeded func handles
            UnmanagedRuntimeWrapper.GetTVMRuntimeEmbededFunc("run",
                runtimeHandle, ref runtimeRunFuncHandle);

            UnmanagedRuntimeWrapper.GetTVMRuntimeEmbededFunc("set_input",
                runtimeHandle, ref runtimeSetInputFuncHandle);

            UnmanagedRuntimeWrapper.GetTVMRuntimeEmbededFunc("load_params",
                runtimeHandle, ref runtimeLoadParamHandle);

            UnmanagedRuntimeWrapper.GetTVMRuntimeEmbededFunc("get_output",
                runtimeHandle, ref runtimeGetOutputFuncHandle);

            isInstantiated = true;
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
            if (!isInstantiated) { CreateInstance(runtimeParam); }
            return this;
        }

        /// <summary>
        /// Run this instance.
        /// </summary>
        public void Run ()
        {
            if (!isInstantiated) { Console.WriteLine("Not instantiated yet!"); return; }

            UnmanagedRuntimeWrapper.InvokeRuntimeRunFunc(runtimeHandle);
        }

        /// <summary>
        /// Sets the input.
        /// </summary>
        /// <param name="inputName">Input name.</param>
        /// <param name="inputTensorHandle">Input tensor handle.</param>
        public void SetInput(string inputName, UIntPtr inputTensorHandle)
        {
            if (!isInstantiated) { Console.WriteLine("Not instantiated yet!"); return; }

            UnmanagedRuntimeWrapper.InvokeRuntimeSetInputFunc(runtimeSetInputFuncHandle,
                inputName, inputTensorHandle);
        }

        /// <summary>
        /// Loads the parameters.
        /// </summary>
        public void LoadParams()
        {
            if (!isInstantiated) { Console.WriteLine("Not instantiated yet!"); return; }

            UnmanagedRuntimeWrapper.InvokeRuntimeLoadParamFunc(runtimeLoadParamHandle,
                paramsDict);
        }

        /// <summary>
        /// Gets the output.
        /// </summary>
        /// <param name="outputIndex">Output index.</param>
        /// <param name="outputTensor">Output tensor.</param>
        public void GetOutput(int outputIndex, ref TVMTensor outputTensor)
        {
            if (!isInstantiated) { Console.WriteLine("Not instantiated yet!"); return; }

            UnmanagedRuntimeWrapper.InvokeRuntimeGetOutputFunc(runtimeGetOutputFuncHandle,
                outputIndex, ref outputTensor);
        }

        // TODO: Destructor
    }
}
