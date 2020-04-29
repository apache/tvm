using System;
using System.Runtime.InteropServices;
using static TVMRuntime.Utils;

namespace TVMRuntime
{
    [StructLayout(LayoutKind.Sequential)]
    /// <summary>
    /// Data type.
    /// </summary>
    public struct TVMDataTypeUnmanaged
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
    public struct TensorContextUnmanaged
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
    public unsafe struct TVMTensor
    {
        public UIntPtr data;
        public TensorContextUnmanaged ctx;
        public int ndim;
        public TVMDataTypeUnmanaged dtype;
        public IntPtr shape;
        public IntPtr strides;
        public uint byte_offset;
    }


    public static class UnmanagedNDArrayWrapper
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
                          [MarshalAs(UnmanagedType.LPArray)] int[] shape,
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
        public static void CreateNDArray(int[] shape,
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
        public static int [] GetNDArrayShape(IntPtr arrayHandle)
        {
            TVMTensor tensor = Marshal.PtrToStructure<TVMTensor>(arrayHandle);
            int ndim = tensor.ndim;
            long[] shapeOut = new long[ndim];
            
            Marshal.Copy(tensor.shape, shapeOut, 0, shapeOut.Length);

            return Array.ConvertAll<long, int>(shapeOut,
                            delegate (long i) {return (int)i;});
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
        }

        /// <summary>
        /// Gets the NDArray element.
        /// </summary>
        /// <returns>The NDArray element.</returns>
        /// <param name="arrayHandle">Array handle.</param>
        /// <param name="index">Index.</param>
        /// <param name="arraySize">Array size.</param>
        public static object GetNDArrayElem(IntPtr arrayHandle, int index, int arraySize)
        {
            if (index < 0 || index >= arraySize)
            {
                Console.WriteLine("Index should be in range [0, " + (arraySize - 1) + "].");
                return null;
            }
            unsafe
            {
                byte code = ((TVMTensor*)arrayHandle)->dtype.code;
                if (code == (byte)TVMDataTypeCode.Int)
                {
                    return ((int*)((TVMTensor*)arrayHandle)->data)[index];
                }
                else if (code == (byte)TVMDataTypeCode.UInt)
                {
                    return ((uint*)((TVMTensor*)arrayHandle)->data)[index];
                }
                else if (code == (byte)TVMDataTypeCode.Float)
                {
                    return ((float*)((TVMTensor*)arrayHandle)->data)[index];
                }
                else
                {
                    Console.WriteLine(code + " not supported!");
                    return null;
                }
            }
        }

        /// <summary>
        /// Sets the NDArray element.
        /// </summary>
        /// <param name="arrayHandle">Array handle.</param>
        /// <param name="index">Index.</param>
        /// <param name="value">Value.</param>
        /// <param name="arraySize">Array size.</param>
        public static void SetNDArrayElem(IntPtr arrayHandle, int index,
            object value, int arraySize)
        {
            if (index < 0 || index >= arraySize)
            {
                Console.WriteLine("Index should be in range [0, " + (arraySize - 1) + "].");
            }

            unsafe
            {
                byte code = ((TVMTensor*)arrayHandle)->dtype.code;
                if (code == (byte)TVMDataTypeCode.Int)
                {
                    ((int*)((TVMTensor*)arrayHandle)->data)[index] = (int)value;
                }
                else if (code == (byte)TVMDataTypeCode.UInt)
                {
                    ((uint*)((TVMTensor*)arrayHandle)->data)[index] = (uint)value;
                }
                else if (code == (byte)TVMDataTypeCode.Float)
                {
                    ((float*)((TVMTensor*)arrayHandle)->data)[index] = (float)value;
                }
                else
                {
                    Console.WriteLine(code + " not supported!");
                }
            }
        }

        /// <summary>
        /// Copies the float data to NDArray.
        /// </summary>
        /// <param name="arrayHandle">Array handle.</param>
        /// <param name="from">From.</param>
        public static void CopyFloatDataToNDArray(IntPtr arrayHandle, float [] from)
        {
            int total_size = from.Length * 4;
            byte [] byteArray = new byte[total_size];

            Buffer.BlockCopy(from, 0, byteArray, 0, byteArray.Length);

            IntPtr unmanagedPointer = Marshal.AllocHGlobal(byteArray.Length);
            try
            {
                Marshal.Copy(byteArray, 0, unmanagedPointer, byteArray.Length);

                TVMArrayCopyFromBytes(arrayHandle, unmanagedPointer, byteArray.LongLength);
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
