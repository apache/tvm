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
        public string mod_lib_path;

        /// <summary>
        /// (Optional) The mod lib format.
        /// </summary>
        public string mod_lib_format;

        /// <summary>
        /// The graph json path.
        /// </summary>
        public string graph_json_path;

        /// <summary>
        /// The parameter dict path.
        /// </summary>
        public string param_dict_path;

        /// <summary>
        /// The context.
        /// </summary>
        public TVMContext context;
    }


    public class Runtime
    {
        private Module module = null;
        private byte[] params_dict;
        private string graph_json_string;
        private bool isInstantiated = false;
        private UIntPtr runtime_handle = UIntPtr.Zero;

        // all embeded func handles
        private UIntPtr runtime_run_func_handle = UIntPtr.Zero;
        private UIntPtr runtime_set_input_func_handle = UIntPtr.Zero;
        private UIntPtr runtime_load_param_handle = UIntPtr.Zero;
        private UIntPtr runtime_get_output_func_handle = UIntPtr.Zero;

        private void CreateInstance(RuntimeParams runtimeParam)
        {
            Console.WriteLine("Jai Hanuman module create start!!!");
            module = new Module(runtimeParam.mod_lib_path,
                                runtimeParam.mod_lib_format);

            Console.WriteLine("Jai Hanuman module created!!!");

            params_dict = Utils.ReadByteArrayFromFile(runtimeParam.param_dict_path);

            Console.WriteLine("byte length: " + params_dict.Length);

            graph_json_string = Utils.ReadStringFromFile(runtimeParam.graph_json_path);

            Console.WriteLine("Module handle : " + module.Module_handle);
            UnmanagedRuntimeWrapper.CreateTVMRuntime(module.Module_handle,
                                            graph_json_string,
                                            runtimeParam.context,
                                            ref runtime_handle);

            Console.WriteLine("Jai Hanuman runtime created!!!");
            Console.WriteLine("Module handle : " + module.Module_handle);
            Console.WriteLine("Runtime handle : " + runtime_handle);

            // Load all embeded func handles
            UnmanagedRuntimeWrapper.GetTVMRuntimeEmbededFunc("run",
                runtime_handle, ref runtime_run_func_handle);

            UnmanagedRuntimeWrapper.GetTVMRuntimeEmbededFunc("set_input",
                runtime_handle, ref runtime_set_input_func_handle);

            UnmanagedRuntimeWrapper.GetTVMRuntimeEmbededFunc("load_params",
                runtime_handle, ref runtime_load_param_handle);

            UnmanagedRuntimeWrapper.GetTVMRuntimeEmbededFunc("get_output",
                runtime_handle, ref runtime_get_output_func_handle);

            isInstantiated = true;
        }

        public Runtime(RuntimeParams runtimeParam)
        {
            CreateInstance(runtimeParam);
        }

        public Runtime()
        {

        }

        public Runtime Create(RuntimeParams runtimeParam)
        {
            if (!isInstantiated) { CreateInstance(runtimeParam); }
            return this;
        }

        public void Run ()
        {
            if (!isInstantiated) { Console.WriteLine("Not instantiated yet!"); return; }

            UnmanagedRuntimeWrapper.InvokeRuntimeRunFunc(runtime_handle);
        }

        public void SetInput(string input_name, UIntPtr input_tensor_handle)
        {
            if (!isInstantiated) { Console.WriteLine("Not instantiated yet!"); return; }

            UnmanagedRuntimeWrapper.InvokeRuntimeSetInputFunc(runtime_set_input_func_handle,
                input_name, input_tensor_handle);
        }

        public void LoadParams()
        {
            if (!isInstantiated) { Console.WriteLine("Not instantiated yet!"); return; }
            Console.WriteLine("byte length: " + params_dict.Length);

            UnmanagedRuntimeWrapper.InvokeRuntimeLoadParamFunc(runtime_load_param_handle,
                params_dict);
        }

        public void GetOutput(int output_index, ref TVMTensor output_tensor)
        {
            if (!isInstantiated) { Console.WriteLine("Not instantiated yet!"); return; }

            UnmanagedRuntimeWrapper.InvokeRuntimeGetOutputFunc(runtime_get_output_func_handle,
                output_index, ref output_tensor);
        }

        // TODO: Destructor
    }
}
