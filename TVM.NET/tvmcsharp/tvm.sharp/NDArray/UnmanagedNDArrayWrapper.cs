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
        public DeviceType device_type;

        /// <summary>
        /// The device identifier.
        /// </summary>
        public int device_id;
    }

    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct TVMTensor
    {
        public UIntPtr data;
        public TensorContextUnmanaged ctx;
        public int ndim;
        public TVMDataTypeUnmanaged dtype;
        public int* shape;
        public int* strides;
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
        private unsafe static extern int TVMArrayAlloc(
                          [MarshalAs(UnmanagedType.LPArray)] int[] shape,
                          int ndim,
                          int dtypeCode,
                          int dtypeBits,
                          int dtypeLanes,
                          int deviceType,
                          int deviceId,
                          TVMTensor** output);

        /// <summary>
        /// TVM Array free.
        /// </summary>
        /// <returns>The rray free.</returns>
        /// <param name="tensorHandle">Tensor handle.</param>
        [DllImport(Utils.libName)]
        private unsafe static extern int TVMArrayFree(
                          TVMTensor* tensorHandle);

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
                          ref UIntPtr arrayHandle)
        {
            unsafe
            {
                TVMTensor* tensorHandle = null;
                TVMArrayAlloc(shape, ndim, dtypeCode, dtypeBits, dtypeLanes,
                    deviceType, deviceId, &tensorHandle);

                arrayHandle = (UIntPtr)tensorHandle;
            }
        }

        /// <summary>
        /// Gets the NDArray shape.
        /// </summary>
        /// <returns>The NDA rray shape.</returns>
        /// <param name="arrayHandle">Array handle.</param>
        public static int [] GetNDArrayShape(UIntPtr arrayHandle)
        {
            unsafe
            {
                int ndim = ((TVMTensor*)arrayHandle)->ndim;
                int[] shapeOut = new int[ndim];
                for (int i = 0; i < ndim; i++)
                {
                    shapeOut[i] = ((TVMTensor*)arrayHandle)->shape[i];
                }
                return shapeOut;
            }
        }

        /// <summary>
        /// Gets the NDArray element.
        /// </summary>
        /// <returns>The NDA rray element.</returns>
        /// <param name="arrayHandle">Array handle.</param>
        /// <param name="index">Index.</param>
        /// <param name="arraySize">Array size.</param>
        public static object GetNDArrayElem(UIntPtr arrayHandle, int index, int arraySize)
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
        public static void SetNDArrayElem(UIntPtr arrayHandle, int index,
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
        /// Disposes the NDArray.
        /// </summary>
        /// <param name="arrayHandle">Array handle.</param>
        public static void DisposeNDArray(UIntPtr arrayHandle)
        {
            unsafe
            {
                TVMArrayFree((TVMTensor*)arrayHandle);
            }
        }
    }
}
