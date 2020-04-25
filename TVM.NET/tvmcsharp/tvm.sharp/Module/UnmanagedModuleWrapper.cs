using System;
using System.Runtime.InteropServices;
using static TVMRuntime.Utils;

namespace TVMRuntime
{
    public static class UnmanagedModuleWrapper
    {
        /// <summary>
        ///
        /// </summary>
        /// <param name="file_name">The file name to load the module from.</param>
        /// <param name="format">The format of the module.</param>
        /// <param name="out_handle">The result module</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        /// <remarks>
        /// @brief Load module from file.@note The resulting module do not contain import relation.
        /// It can be reconstructed by TVMModImport.
        /// </remarks>
        [DllImport(Utils.libName)]
        private static extern int TVMModLoadFromFile([MarshalAs(UnmanagedType.LPStr)] string file_name,
            [MarshalAs(UnmanagedType.LPStr)] string format, ref UIntPtr out_handle);


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
        private static extern int TVMModImport(UIntPtr mod, UIntPtr dep);


        /// <summary>
        ///
        /// </summary>
        /// <param name="mod">The module handle.</param>
        /// <param name="func_name">The name of the function.</param>
        /// <param name="query_imports">Whether to query imported modules</param>
        /// <param name="out_handle">The result function, 
        /// can be NULL if it is not available.</param>
        /// <returns>0 when no error is thrown, -1 when failure happens</returns>
        /// <remarks>
        /// @brief Get function from the module.
        /// </remarks>
        [DllImport(Utils.libName)]
        private static extern int TVMModGetFunction(UIntPtr mod,
            [MarshalAs(UnmanagedType.LPStr)] string func_name,
            int query_imports, ref UIntPtr out_handle);


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
        private static extern int TVMModFree(UIntPtr mod);


        /// <summary>
        /// Loads the module from file.
        /// </summary>
        /// <param name="file_name">File name.</param>
        /// <param name="format">Format.</param>
        /// <param name="mod_handle">Module handle.</param>
        public static void LoadModuleFromFile(string file_name,
            string format, ref UIntPtr mod_handle)
        {
            // TODO: Error handling
            int result = TVMModLoadFromFile(file_name, format, 
                    ref mod_handle);
        }

        /// <summary>
        /// Imports the module.
        /// </summary>
        /// <param name="mod">Mod.</param>
        /// <param name="dep">Dep.</param>
        public static void ImportModule(UIntPtr mod, UIntPtr dep)
        {
            // TODO: Error handling
            int result = TVMModImport(mod, dep);
        }

        /// <summary>
        /// Gets the module embeded func.
        /// </summary>
        /// <param name="mod">Mod.</param>
        /// <param name="func_name">Func name.</param>
        /// <param name="query_imports">Query imports.</param>
        /// <param name="func_handle">Func handle.</param>
        public static void GetModuleEmbededFunc(UIntPtr mod, string func_name,
            int query_imports, ref UIntPtr func_handle)
        {
            // TODO: Error handling
            Console.WriteLine("Jai Hanuman GetModuleEmbededFunc start!!!");
            Console.WriteLine(mod);
            Console.WriteLine(func_name);
            Console.WriteLine(query_imports);
            Console.WriteLine(func_handle);
            int result = TVMModGetFunction(mod, func_name, query_imports,
                    ref func_handle);

        }

        /// <summary>
        /// Disposes the module.
        /// </summary>
        /// <param name="mod">Mod.</param>
        public static void DisposeModule(UIntPtr mod)
        {
            // TODO: Error handling
            int result = TVMModFree(mod);
        }
    }
}
