using System;
using System.Runtime.InteropServices;
using static TVMRuntime.Utils;
using static TVMRuntime.UnamangedPFManagerWrapper;
using System.Text;

namespace TVMRuntime
{
    [StructLayout(LayoutKind.Sequential)]
    struct TVMRuntimeCreateArgs
    {
        public string graph_json_string;
        public UIntPtr module_handle;
        public int device_type;
        public int device_id;

    }

    [StructLayout(LayoutKind.Sequential)]
    struct TVMRuntimeSetInputArgs
    {
        public string input_name;
        public UIntPtr input_tensor_handle;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct TVMRuntimeLoadParamsArgs
    {
        public IntPtr param_ptr;
        //public char [] param_ptr;
        public long size;
    }

    public static class UnmanagedRuntimeWrapper
    {
        /// <summary>
        /// The global registry name of the tvm create func.
        /// </summary>
        private static string tvmCreateFuncName = "tvm.graph_runtime.create";

        private static UIntPtr tvmCreateFuncHandle = UIntPtr.Zero;

        static UnmanagedRuntimeWrapper()
        {
            UnamangedPFManagerWrapper.GetTVMRuntimeGlobalPackedFunc(tvmCreateFuncName,
                            ref tvmCreateFuncHandle);
        }

        /// <summary>
        /// TVM func call.
        /// </summary>
        /// <returns>The func call.</returns>
        /// <param name="func_handle">Func handle.</param>
        /// <param name="args">Arguments.</param>
        /// <param name="arg_type_codes">Argument type codes.</param>
        /// <param name="num_args">Number arguments.</param>
        /// <param name="ret_val">Ret value.</param>
        /// <param name="ret_type_code">Ret type code.</param>
        [DllImport(Utils.libName)]
        private static extern int TVMFuncCall(UIntPtr func_handle,
            IntPtr args,
            [MarshalAs(UnmanagedType.LPArray)] int[] arg_type_codes,
            int num_args,
            ref UIntPtr ret_val,
            ref int ret_type_code);

        [DllImport(Utils.libName)]
        private static extern int TVMFuncCall(UIntPtr func_handle,
            ref IntPtr args,
            [MarshalAs(UnmanagedType.LPArray)] int[] arg_type_codes,
            int num_args,
            ref UIntPtr ret_val,
            ref int ret_type_code);

        [DllImport(Utils.libName)]
        private static extern int TVMFuncCall(UIntPtr func_handle,
            [MarshalAs(UnmanagedType.LPArray)] byte[] args,
            [MarshalAs(UnmanagedType.LPArray)] int[] arg_type_codes,
            int num_args,
            ref TVMValue ret_val,
            ref int ret_type_code);

        [DllImport(Utils.libName)]
        private static extern int TVMFuncCall(UIntPtr func_handle,
            int args,
            [MarshalAs(UnmanagedType.LPArray)] int[] arg_type_codes,
            int num_args,
            ref TVMTensor ret_val,
            ref int ret_type_code);

        private static void InvokeTVMRuntimeCreatePackedFunc(UIntPtr func_handle,
            IntPtr args, int[] arg_type_codes, int num_args,
            ref UIntPtr ret_val, ref int ret_type_code)
        {
            TVMFuncCall(func_handle, args, arg_type_codes, num_args,
                ref ret_val, ref ret_type_code);
        }

        private static void InvokeTVMRuntimeSetInputPackedFunc(UIntPtr func_handle,
            IntPtr args, int[] arg_type_codes, int num_args,
            ref UIntPtr ret_val, ref int ret_type_code)
        {
            TVMFuncCall(func_handle, args, arg_type_codes, num_args,
                ref ret_val, ref ret_type_code);
        }

        private static void InvokeTVMRuntimeLoadParamFunc(UIntPtr func_handle,
            ref IntPtr args, int[] arg_type_codes, int num_args,
            ref UIntPtr ret_val, ref int ret_type_code)
        {
            TVMFuncCall(func_handle, ref args, arg_type_codes, num_args,
                ref ret_val, ref ret_type_code);
        }

        private static void InvokeTVMRuntimeGetOutputFunc(UIntPtr func_handle,
            int arg, int[] arg_type_codes, int num_args,
            ref TVMTensor ret_val, ref int ret_type_code)
        {
            TVMFuncCall(func_handle, arg, arg_type_codes, num_args,
                ref ret_val, ref ret_type_code);
        }

        public static void CreateTVMRuntime(UIntPtr module_handle,
            string graph_json_string, TVMContext ctx, ref UIntPtr runtime_handle)
        {
            Console.WriteLine("Jai hanuman runtime created inside");
            Console.WriteLine((int)ctx.device_type);
            Console.WriteLine(ctx.device_id);
            TVMRuntimeCreateArgs tvm_create_args = new TVMRuntimeCreateArgs();
            tvm_create_args.module_handle = module_handle;
            tvm_create_args.graph_json_string = graph_json_string;
            tvm_create_args.device_type = (int)ctx.device_type;
            tvm_create_args.device_id = ctx.device_id;

            int[] arg_type_codes = new int[] {
                    (int)TVMTypeCode.TVMStr, (int)TVMTypeCode.TVMModuleHandle,
                    (int)TVMDataTypeCode.Int,
                    (int)TVMDataTypeCode.Int };

            // Initialize unmanged memory to hold the struct.
            IntPtr pnt = Marshal.AllocHGlobal(Marshal.SizeOf(tvm_create_args));

            try
            {
                // Copy the struct to unmanaged memory.
                Marshal.StructureToPtr(tvm_create_args, pnt, false);

                // Create another point.
                /*TVMRuntimeCreateArgs anotherP;

                // Set this Point to the value of the
                // Point in unmanaged memory.
                anotherP = (TVMRuntimeCreateArgs)Marshal.PtrToStructure(pnt,
                            typeof(TVMRuntimeCreateArgs));*/

                int num_args = 4;

                int ret_type_code = 0;

                InvokeTVMRuntimeCreatePackedFunc(tvmCreateFuncHandle,
                        pnt, arg_type_codes, num_args,
                        ref runtime_handle, ref ret_type_code);
                Console.WriteLine("Jai hanuman runtime created inside 1");
                Console.WriteLine(tvmCreateFuncHandle);
                Console.WriteLine(ret_type_code);
                Console.WriteLine(runtime_handle);
            }
            finally
            {
                // Free the unmanaged memory.
                Marshal.FreeHGlobal(pnt);
            }

        }

        public static void GetTVMRuntimeEmbededFunc(string func_name,
                        UIntPtr runtime_handle, ref UIntPtr func_handle)
        {
            UnmanagedModuleWrapper.GetModuleEmbededFunc(runtime_handle,
                func_name, 0, ref func_handle);
        }

        public static void InvokeRuntimeRunFunc(UIntPtr run_func_handle)
        {
            int ret_type_code = 0;
            TVMValue empty_output = new TVMValue();
            UnamangedPFManagerWrapper.InvokeTVMRuntimePackedFunc(run_func_handle,
                new TVMValue[] { }, new int[] { }, 0,
                ref empty_output, ref ret_type_code);
        }

        public static void InvokeRuntimeSetInputFunc(
                            UIntPtr set_input_func_handle,
                            int input_index,
                            UIntPtr input_tensor_handle)
        {
            TVMValue arg0 = new TVMValue(input_index);

            TVMValue arg1 = new TVMValue(input_tensor_handle);
            
            int ret_type_code = 0;
            TVMValue empty_output = new TVMValue();

            UnamangedPFManagerWrapper.InvokeTVMRuntimePackedFunc(
                set_input_func_handle,
                new TVMValue[] { arg0, arg1 },
                new int[] { (int)TVMDataTypeCode.Int,
                (int)TVMTypeCode.TVMNDArrayHandle}, 2,
                ref empty_output, ref ret_type_code);
        }

        public static void InvokeRuntimeSetInputFunc(
                            UIntPtr set_input_func_handle,
                            string input_name,
                            UIntPtr input_tensor_handle)
        {
            TVMRuntimeSetInputArgs tvm_set_input_args = new TVMRuntimeSetInputArgs();
            tvm_set_input_args.input_name = input_name;
            tvm_set_input_args.input_tensor_handle = input_tensor_handle;
            
            int[] arg_type_codes = new int[] {
                    (int)TVMTypeCode.TVMStr, (int)TVMTypeCode.TVMNDArrayHandle };

            // Initialize unmanged memory to hold the struct.
            IntPtr pnt = Marshal.AllocHGlobal(Marshal.SizeOf(tvm_set_input_args));

            try
            {
                // Copy the struct to unmanaged memory.
                Marshal.StructureToPtr(tvm_set_input_args, pnt, false);

                // Create another point.
                /*TVMRuntimeCreateArgs anotherP;

                // Set this Point to the value of the
                // Point in unmanaged memory.
                anotherP = (TVMRuntimeCreateArgs)Marshal.PtrToStructure(pnt,
                            typeof(TVMRuntimeCreateArgs));*/

                int num_args = 2;

                int ret_type_code = 0;
                UIntPtr ret_val = UIntPtr.Zero;

                InvokeTVMRuntimeSetInputPackedFunc(tvmCreateFuncHandle,
                    pnt, arg_type_codes, num_args,
                    ref ret_val, ref ret_type_code);
            }
            finally
            {
                // Free the unmanaged memory.
                Marshal.FreeHGlobal(pnt);
            }

            /*int num_args = 2;

            int ret_type_code = 0;
            UIntPtr ret_val = UIntPtr.Zero;

            InvokeTVMRuntimeSetInputPackedFunc(tvmCreateFuncHandle,
                    ref tvm_set_input_args, arg_type_codes, num_args,
                    ref ret_val, ref ret_type_code);*/
        }

        public static void InvokeRuntimeGetOutputFunc(
                            UIntPtr get_output_func_handle,
                            int output_index,
                            ref TVMTensor output_tensor)
        {
            int ret_type_code = 0;
            
            InvokeTVMRuntimeGetOutputFunc(
                get_output_func_handle,
                output_index,
                new int[] { (int)TVMDataTypeCode.Int }, 1,
                ref output_tensor, ref ret_type_code);
        }

        public static void InvokeRuntimeLoadParamFunc(
                            UIntPtr load_param_func_handle,
                            byte [] param_dict)
        {
            //int ret_type_code = 0;
            TVMValue empty_output = new TVMValue();
            TVMRuntimeLoadParamsArgs load_params_args = new TVMRuntimeLoadParamsArgs();


            /*InvokeTVMRuntimeLoadParamFunc(
                load_param_func_handle,
                param_dict,
                new int[] { (int)TVMTypeCode.TVMBytes }, 1,
                ref empty_output, ref ret_type_code);*/

            // Initialize unmanged memory to hold the struct.
            char [] input = Encoding.Default.GetString(param_dict).ToCharArray();
            int lengthArray = Marshal.SizeOf(param_dict[0]) * param_dict.Length;
            IntPtr pnt = Marshal.AllocHGlobal(lengthArray);

            Console.WriteLine("lengthArray: " + lengthArray);

            load_params_args.param_ptr = pnt;
            load_params_args.size = param_dict.Length;
            IntPtr pnt1 = Marshal.AllocHGlobal(Marshal.SizeOf(load_params_args));
            Console.WriteLine("size of: " + Marshal.SizeOf(load_params_args));

            try
            {
                // Copy the struct to unmanaged memory.
                //Marshal.StructureToPtr(param_dict, pnt, false);
                Marshal.Copy(param_dict, 0, pnt, lengthArray);
                Marshal.StructureToPtr(load_params_args, pnt1, false);

                // Create another point.
                TVMRuntimeLoadParamsArgs anotherP;

                // Set this Point to the value of the
                // Point in unmanaged memory.
                anotherP = (TVMRuntimeLoadParamsArgs)Marshal.PtrToStructure(pnt1,
                            typeof(TVMRuntimeLoadParamsArgs));

                Console.WriteLine("Jai hanuman load param!!!");
                Console.WriteLine(anotherP.size);
                Console.WriteLine(anotherP.param_ptr);


                int num_args = 2;

                int ret_type_code = 0;
                UIntPtr ret_val = UIntPtr.Zero;

                InvokeTVMRuntimeLoadParamFunc(
                    load_param_func_handle,
                    ref pnt1,
                    new int[] { (int)TVMTypeCode.TVMBytes }, 1,
                    ref ret_val, ref ret_type_code);
            }
            finally
            {
                // Free the unmanaged memory.
                Marshal.FreeHGlobal(pnt);
                Marshal.FreeHGlobal(pnt1);
            }
        }

        // TODO: Add other runtime member function as well
    }
}
