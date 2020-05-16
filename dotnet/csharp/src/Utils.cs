using System;
using System.IO;

namespace TVMRuntime
{
    public static class Utils
    {
	public const string libName = "tvm_runtime";

        /// <summary>
        /// Reads the byte array from file.
        /// </summary>
        /// <returns>The byte array from file.</returns>
        /// <param name="filePath">File path.</param>
        public static byte [] ReadByteArrayFromFile(string filePath)
        {
            if (!File.Exists(filePath))
            {
                Console.WriteLine(filePath + " does not exist.");
                return null;
            }

            return File.ReadAllBytes(filePath);
        }

        /// <summary>
        /// Reads the string from file.
        /// </summary>
        /// <returns>The string from file.</returns>
        /// <param name="filePath">File path.</param>
        public static string ReadStringFromFile(string filePath)
        {
            if (!File.Exists(filePath))
            {
                Console.WriteLine(filePath + " does not exist.");
                return null;
            }

            return File.ReadAllText(filePath);
        }

        /// <summary>
        /// Checks the result for successful operation.
        /// </summary>
        public static void CheckSuccess<T>(T expected, T actual)
        {
            if (!expected.Equals(actual))
            {
                throw new System.InvalidOperationException("Unsuccessful operation!");
            }
        }

        /// <summary>
        /// Gets the number of bytes.
        /// </summary>
        /// <returns>The number of bytes.</returns>
        /// <param name="bitLength">Bit length.</param>
        public static int GetNumOfBytes(int bitLength)
        {
            return (int)(bitLength / 8);
        }
    }
}
