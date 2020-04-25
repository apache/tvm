using System;
using System.IO;

namespace TVMRuntime
{
    public static class Utils
    {
        //public const string libName = "native/libtvm_runtime.so";
        public const string libName = "native/libtvm_runtime_log.so";

        public static byte [] ReadByteArrayFromFile(string file_path)
        {
            if (!File.Exists(file_path))
            {
                Console.WriteLine(file_path + " does not exist.");
                return null;
            }

            //string text = File.ReadAllText(file_path);
            //byte[] byteArray = Convert.FromBase64String(text);

            return File.ReadAllBytes(file_path);
            //return byteArray;
        }

        public static string ReadStringFromFile(string file_path)
        {
            if (!File.Exists(file_path))
            {
                Console.WriteLine(file_path + " does not exist.");
                return null;
            }

            return File.ReadAllText(file_path);
        }
    }
}
