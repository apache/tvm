using System;
using System.Runtime.InteropServices;
using static TVMRuntime.Utils;

namespace TVMRuntime
{
    [StructLayout(LayoutKind.Sequential)]
    /// <summary>
    /// Data type.
    /// </summary>
    internal struct TVMDataTypeUnmanaged
    {
        /// <summary>
        /// The code.
        /// </summary>
        public byte code;

        /// <summary>
        /// The bits.
        /// </summary>
        public byte bits;

        /// <summary>
        /// The lanes.
        /// </summary>
        public ushort lanes;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct TensorContextUnmanaged
    {
        /// <summary>
        /// The type of the device.
        /// </summary>
        public DeviceType deviceType;

        /// <summary>
        /// The device identifier.
        /// </summary>
        public int deviceId;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct TVMTensor
    {
        public IntPtr data;
        public TensorContextUnmanaged ctx;
        public int ndim;
        public TVMDataTypeUnmanaged dtype;
        public IntPtr shape;
        public IntPtr strides;
        public uint byte_offset;
    }


    internal static class UnmanagedNDArrayWrapper
    {
        /// <summary>
        /// TVM Array alloc.
        /// </summary>
        /// <returns>The rray alloc.</returns>
        /// <param name="shape">Shape.</param>
        /// <param name="ndim">Ndim.</param>
        /// <param name="dtypeCode">Dtype code.</param>
        /// <param name="dtypeBits">Dtype bits.</param>
        /// <param name="dtypeLanes">Dtype lanes.</param>
        /// <param name="deviceType">Device type.</param>
        /// <param name="deviceId">Device identifier.</param>
        /// <param name="output">Output.</param>
        [DllImport(Utils.libName)]
        private static extern int TVMArrayAlloc(
                          [MarshalAs(UnmanagedType.LPArray)] long[] shape,
                          int ndim,
                          int dtypeCode,
                          int dtypeBits,
                          int dtypeLanes,
                          int deviceType,
                          int deviceId,
                          ref IntPtr output);

        /// <summary>
        /// TVM Array free.
        /// </summary>
        /// <returns>The rray free.</returns>
        /// <param name="tensorHandle">Tensor handle.</param>
        [DllImport(Utils.libName)]
        private unsafe static extern int TVMArrayFree(
                          IntPtr tensorHandle);

        /// <summary>
        /// TVMArray copy from bytes.
        /// </summary>
        /// <returns>The NDArray copy from bytes.</returns>
        /// <param name="handle">Handle.</param>
        /// <param name="data">Data.</param>
        /// <param name="nbytes">Nbytes.</param>
        [DllImport(Utils.libName)]
        private static extern int TVMArrayCopyFromBytes(IntPtr handle,
                                  IntPtr data,
                                  long nbytes);

        /// <summary>
        /// TVMArray copy to bytes.
        /// </summary>
        /// <returns>The NDArray copy to bytes.</returns>
        /// <param name="handle">Handle.</param>
        /// <param name="data">Data.</param>
        /// <param name="nbytes">Nbytes.</param>
        [DllImport(Utils.libName)]
        private static extern int TVMArrayCopyToBytes(IntPtr handle,
                                  IntPtr data,
                                  long nbytes);

        /// <summary>
        /// Creates the NDArray.
        /// </summary>
        /// <param name="shape">Shape.</param>
        /// <param name="ndim">Ndim.</param>
        /// <param name="dtypeCode">Dtype code.</param>
        /// <param name="dtypeBits">Dtype bits.</param>
        /// <param name="dtypeLanes">Dtype lanes.</param>
        /// <param name="deviceType">Device type.</param>
        /// <param name="deviceId">Device identifier.</param>
        /// <param name="arrayHandle">Array handle.</param>
        public static void CreateNDArray(long[] shape,
                          int ndim,
                          int dtypeCode,
                          int dtypeBits,
                          int dtypeLanes,
                          int deviceType,
                          int deviceId,
                          ref IntPtr arrayHandle)
        {
            TVMArrayAlloc(shape, ndim, dtypeCode, dtypeBits, dtypeLanes,
                    deviceType, deviceId, ref arrayHandle);
        }

        /// <summary>
        /// Gets the NDArray shape.
        /// </summary>
        /// <returns>The NDArray shape.</returns>
        /// <param name="arrayHandle">Array handle.</param>
        public static long [] GetNDArrayShape(IntPtr arrayHandle)
        {
            TVMTensor tensor = Marshal.PtrToStructure<TVMTensor>(arrayHandle);
            int ndim = tensor.ndim;
            long[] shapeOut = new long[ndim];
            
            Marshal.Copy(tensor.shape, shapeOut, 0, shapeOut.Length);

            return shapeOut;
        }

        /// <summary>
        /// Gets the NDArray ndim.
        /// </summary>
        /// <returns>The NDArray ndim.</returns>
        /// <param name="arrayHandle">Array handle.</param>
        public static int GetNDArrayNdim(IntPtr arrayHandle)
        {
            return Marshal.PtrToStructure<TVMTensor>(arrayHandle).ndim;
        }

        /// <summary>
        /// Gets the NDArray dtype.
        /// </summary>
        /// <param name="arrayHandle">Array handle.</param>
        /// <param name="dataType">Data type.</param>
        public static void GetNDArrayDtype(IntPtr arrayHandle, ref TVMDataType dataType)
        {
            TVMTensor tensor = Marshal.PtrToStructure<TVMTensor>(arrayHandle);
            dataType.code = tensor.dtype.code;
            dataType.bits = tensor.dtype.bits;
            dataType.lanes = tensor.dtype.lanes;
            dataType.numOfBytes = (byte)(tensor.dtype.bits / 8);
        }

        /// <summary>
        /// Gets the offset.
        /// </summary>
        /// <returns>The offset.</returns>
        /// <param name="numOfBytes">Number of bytes.</param>
        /// <param name="currIndex">Curr index.</param>
        private static int GetOffset(int numOfBytes, int currIndex)
        {
            return numOfBytes * currIndex;
        }

        /// <summary>
        /// Gets the NDArray element.
        /// </summary>
        /// <returns>The NDArray element.</returns>
        /// <param name="array">Array.</param>
        /// <param name="index">Index.</param>
        public static object GetNDArrayElem(NDArray array, int index)
        {
            if (index < 0 || index >= array.Size)
            {
                throw new System.ArgumentOutOfRangeException("Index should be in range [0, " + (array.Size - 1) + "].");
            }

            TVMTensor tensor = Marshal.PtrToStructure<TVMTensor>(array.NDArrayHandle);
            byte[] to = array.InternalBuffer;
            Marshal.Copy(tensor.data, to, 0, to.Length);


            if (array.DataType.IsFloat32())
            {
                return BitConverter.ToSingle(to, GetOffset(array.DataType.numOfBytes, index));
            }
            else if (array.DataType.IsFloat64())
            {
                return BitConverter.ToDouble(to, GetOffset(array.DataType.numOfBytes, index));
            }
            else if (array.DataType.IsInt32())
            {
                return BitConverter.ToInt32(to, GetOffset(array.DataType.numOfBytes, index));
            }
            else if (array.DataType.IsInt64())
            {
                return BitConverter.ToInt64(to, GetOffset(array.DataType.numOfBytes, index));
            }
            else if (array.DataType.IsInt16())
            {
                return BitConverter.ToInt16(to, GetOffset(array.DataType.numOfBytes, index));
            }
            else if (array.DataType.IsUint16())
            {
                return BitConverter.ToChar(to, GetOffset(array.DataType.numOfBytes, index));
            }
            else if (array.DataType.IsInt8())
            {
                return to[GetOffset(array.DataType.numOfBytes, index)];
            }
            else
            {
                throw new System.ArrayTypeMismatchException("Unknown type");
            }
        }

        /// <summary>
        /// Copies the float data to NDArray.
        /// </summary>
        /// <param name="array">Array.</param>
        /// <param name="from">From.</param>
        public static void CopyFloatDataToNDArray(NDArray array, float [] from)
        {
            int total_size = from.Length * array.DataType.numOfBytes;
            byte [] byteArray = new byte[total_size];

            Buffer.BlockCopy(from, 0, byteArray, 0, byteArray.Length);

            CopyByteDataToNDArray(array.NDArrayHandle, byteArray);
        }

        /// <summary>
        /// Gets the float data from NDArray.
        /// </summary>
        /// <returns>The float data from NDArray.</returns>
        /// <param name="array">Array.</param>
        public static float [] GetFloatDataFromNDArray(NDArray array)
        {
            long totalSize = array.Size * array.DataType.numOfBytes;
            byte[] fromArray = new byte[totalSize];

            GetByteDataFromNDArray(array.NDArrayHandle, fromArray);

            float[] toArray = new float[array.Size];
            Buffer.BlockCopy(fromArray, 0, toArray, 0, fromArray.Length);
            return toArray;
        }

        /// <summary>
        /// Copies the double data to NDArray.
        /// </summary>
        /// <param name="array">Array.</param>
        /// <param name="from">From.</param>
        public static void CopyDoubleDataToNDArray(NDArray array, double[] from)
        {
            int totalSize = from.Length * array.DataType.numOfBytes;
            byte[] byteArray = new byte[totalSize];

            Buffer.BlockCopy(from, 0, byteArray, 0, byteArray.Length);

            CopyByteDataToNDArray(array.NDArrayHandle, byteArray);
        }

        /// <summary>
        /// Gets the double data from NDArray.
        /// </summary>
        /// <returns>The double data from NDArray.</returns>
        /// <param name="array">Array.</param>
        public static double[] GetDoubleDataFromNDArray(NDArray array)
        {
            long totalSize = array.Size * array.DataType.numOfBytes;
            byte[] fromArray = new byte[totalSize];

            GetByteDataFromNDArray(array.NDArrayHandle, fromArray);

            double[] toArray = new double[array.Size];
            Buffer.BlockCopy(fromArray, 0, toArray, 0, fromArray.Length);
            return toArray;
        }

        /// <summary>
        /// Copies the int data to NDArray.
        /// </summary>
        /// <param name="array">Array.</param>
        /// <param name="from">From.</param>
        public static void CopyIntDataToNDArray(NDArray array, int[] from)
        {
            int totalSize = from.Length * array.DataType.numOfBytes;
            byte[] byteArray = new byte[totalSize];

            Buffer.BlockCopy(from, 0, byteArray, 0, byteArray.Length);

            CopyByteDataToNDArray(array.NDArrayHandle, byteArray);
        }

        /// <summary>
        /// Gets the int data from NDArray.
        /// </summary>
        /// <returns>The int data from NDArray.</returns>
        /// <param name="array">Array.</param>
        public static int[] GetIntDataFromNDArray(NDArray array)
        {
            long totalSize = array.Size * array.DataType.numOfBytes;
            byte[] fromArray = new byte[totalSize];

            GetByteDataFromNDArray(array.NDArrayHandle, fromArray);

            int[] toArray = new int[array.Size];
            Buffer.BlockCopy(fromArray, 0, toArray, 0, fromArray.Length);
            return toArray;
        }

        /// <summary>
        /// Copies the long data to NDArray.
        /// </summary>
        /// <param name="array">Array.</param>
        /// <param name="from">From.</param>
        public static void CopyLongDataToNDArray(NDArray array, long[] from)
        {
            int totalSize = from.Length * array.DataType.numOfBytes;
            byte[] byteArray = new byte[totalSize];

            Buffer.BlockCopy(from, 0, byteArray, 0, byteArray.Length);

            CopyByteDataToNDArray(array.NDArrayHandle, byteArray);
        }

        /// <summary>
        /// Gets the long data from NDArray.
        /// </summary>
        /// <returns>The long data from NDArray.</returns>
        /// <param name="array">Array.</param>
        public static long[] GetLongDataFromNDArray(NDArray array)
        {
            long totalSize = array.Size * array.DataType.numOfBytes;
            byte[] fromArray = new byte[totalSize];

            GetByteDataFromNDArray(array.NDArrayHandle, fromArray);

            long[] toArray = new long[array.Size];
            Buffer.BlockCopy(fromArray, 0, toArray, 0, fromArray.Length);
            return toArray;
        }

        /// <summary>
        /// Copies the short data to NDArray.
        /// </summary>
        /// <param name="array">Array.</param>
        /// <param name="from">From.</param>
        public static void CopyShortDataToNDArray(NDArray array, short[] from)
        {
            int totalSize = from.Length * array.DataType.numOfBytes;
            byte[] byteArray = new byte[totalSize];

            Buffer.BlockCopy(from, 0, byteArray, 0, byteArray.Length);

            CopyByteDataToNDArray(array.NDArrayHandle, byteArray);
        }

        /// <summary>
        /// Gets the short data from NDArray.
        /// </summary>
        /// <returns>The short data from NDArray.</returns>
        /// <param name="array">Array.</param>
        public static short[] GetShortDataFromNDArray(NDArray array)
        {
            long totalSize = array.Size * array.DataType.numOfBytes;
            byte[] fromArray = new byte[totalSize];

            GetByteDataFromNDArray(array.NDArrayHandle, fromArray);

            short[] toArray = new short[array.Size];
            Buffer.BlockCopy(fromArray, 0, toArray, 0, fromArray.Length);
            return toArray;
        }

        /// <summary>
        /// Copies the char data to NDArray.
        /// </summary>
        /// <param name="array">Array.</param>
        /// <param name="from">From.</param>
        public static void CopyCharDataToNDArray(NDArray array, char[] from)
        {
            int totalSize = from.Length * array.DataType.numOfBytes;
            byte[] byteArray = new byte[totalSize];

            Buffer.BlockCopy(from, 0, byteArray, 0, byteArray.Length);

            CopyByteDataToNDArray(array.NDArrayHandle, byteArray);
        }

        /// <summary>
        /// Gets the char data from NDArray.
        /// </summary>
        /// <returns>The char data from NDArray.</returns>
        /// <param name="array">Array.</param>
        public static char[] GetCharDataFromNDArray(NDArray array)
        {
            long totalSize = array.Size * array.DataType.numOfBytes;
            byte[] fromArray = new byte[totalSize];

            GetByteDataFromNDArray(array.NDArrayHandle, fromArray);

            char[] toArray = new char[array.Size];
            Buffer.BlockCopy(fromArray, 0, toArray, 0, fromArray.Length);
            return toArray;
        }

        /// <summary>
        /// Copies the byte data to NDArray.
        /// </summary>
        /// <param name="array">Array.</param>
        /// <param name="from">From.</param>
        public static void CopyByteDataToNDArray(NDArray array, byte[] from)
                            => CopyByteDataToNDArray(array.NDArrayHandle, from);

        /// <summary>
        /// Gets the byte data from NDArray.
        /// </summary>
        /// <returns>The byte data from NDArray.</returns>
        /// <param name="array">Array.</param>
        public static byte[] GetByteDataFromNDArray(NDArray array)
        {
            long totalSize = array.Size * array.DataType.numOfBytes;
            byte[] fromArray = new byte[totalSize];

            GetByteDataFromNDArray(array.NDArrayHandle, fromArray);
            return fromArray;
        }

        /// <summary>
        /// Copies the byte data to NDArray.
        /// </summary>
        /// <param name="arrayHandle">Array handle.</param>
        /// <param name="from">From.</param>
        private static void CopyByteDataToNDArray(IntPtr arrayHandle, byte[] from)
        {
            IntPtr unmanagedPointer = Marshal.AllocHGlobal(from.Length);
            try
            {
                Marshal.Copy(from, 0, unmanagedPointer, from.Length);

                TVMArrayCopyFromBytes(arrayHandle, unmanagedPointer, from.LongLength);
            }
            finally
            {
                // Free the unmanaged memory.
                Marshal.FreeHGlobal(unmanagedPointer);
            }
        }

        /// <summary>
        /// Gets the byte data from NDArray.
        /// </summary>
        /// <param name="arrayHandle">Array handle.</param>
        /// <param name="to">To.</param>
        private static void GetByteDataFromNDArray(IntPtr arrayHandle, byte[] to)
        {
            IntPtr unmanagedPointer = Marshal.AllocHGlobal(to.Length);
            try
            {
                TVMArrayCopyToBytes(arrayHandle, unmanagedPointer, to.LongLength);

                Marshal.Copy(unmanagedPointer, to, 0, to.Length);
            }
            finally
            {
                // Free the unmanaged memory.
                Marshal.FreeHGlobal(unmanagedPointer);
            }
        }

        /// <summary>
        /// Disposes the NDArray.
        /// </summary>
        /// <param name="arrayHandle">Array handle.</param>
        public static void DisposeNDArray(IntPtr arrayHandle)
        {
            TVMArrayFree(arrayHandle);
        }
    }
}
