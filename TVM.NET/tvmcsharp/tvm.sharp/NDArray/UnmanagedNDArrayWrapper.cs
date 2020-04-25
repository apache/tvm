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
        [DllImport(Utils.libName)]
        private unsafe static extern int TVMArrayAlloc(
                          [MarshalAs(UnmanagedType.LPArray)] int[] shape,
                          int ndim,
                          int dtype_code,
                          int dtype_bits,
                          int dtype_lanes,
                          int device_type,
                          int device_id,
                          TVMTensor** output);

        [DllImport(Utils.libName)]
        private unsafe static extern int TVMArrayFree(
                          TVMTensor* tensor_handle);

        public static void CreateNDArray(int[] shape,
                          int ndim,
                          int dtype_code,
                          int dtype_bits,
                          int dtype_lanes,
                          int device_type,
                          int device_id,
                          ref UIntPtr array_handle)
        {
            unsafe
            {
                TVMTensor* tensor_handle = null;
                TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &tensor_handle);

                array_handle = (UIntPtr)tensor_handle;
            }
        }

        public static int [] GetNDArrayShape(UIntPtr array_handle)
        {
            unsafe
            {
                int ndim = ((TVMTensor*)array_handle)->ndim;
                int[] shape_out = new int[ndim];
                for (int i = 0; i < ndim; i++)
                {
                    shape_out[i] = ((TVMTensor*)array_handle)->shape[i];
                }
                return shape_out;
            }
        }

        public static object GetNDArrayElem(UIntPtr array_handle, int index, int arraySize)
        {
            if (index < 0 || index >= arraySize)
            {
                Console.WriteLine("Index should be in range [0, " + (arraySize - 1) + "].");
                return null;
            }
            unsafe
            {
                byte code = ((TVMTensor*)array_handle)->dtype.code;
                if (code == (byte)TVMDataTypeCode.Int)
                {
                    return ((int*)((TVMTensor*)array_handle)->data)[index];
                }
                else if (code == (byte)TVMDataTypeCode.UInt)
                {
                    return ((uint*)((TVMTensor*)array_handle)->data)[index];
                }
                else if (code == (byte)TVMDataTypeCode.Float)
                {
                    return ((float*)((TVMTensor*)array_handle)->data)[index];
                }
                else
                {
                    Console.WriteLine(code + " not supported!");
                    return null;
                }
            }
        }

        public static void SetNDArrayElem(UIntPtr array_handle, int index,
            object value, int arraySize)
        {
            if (index < 0 || index >= arraySize)
            {
                Console.WriteLine("Index should be in range [0, " + (arraySize - 1) + "].");
            }

            unsafe
            {
                byte code = ((TVMTensor*)array_handle)->dtype.code;
                if (code == (byte)TVMDataTypeCode.Int)
                {
                    ((int*)((TVMTensor*)array_handle)->data)[index] = (int)value;
                }
                else if (code == (byte)TVMDataTypeCode.UInt)
                {
                    ((uint*)((TVMTensor*)array_handle)->data)[index] = (uint)value;
                }
                else if (code == (byte)TVMDataTypeCode.Float)
                {
                    ((float*)((TVMTensor*)array_handle)->data)[index] = (float)value;
                }
                else
                {
                    Console.WriteLine(code + " not supported!");
                }
            }
        }

        public static void DisposeNDArray(UIntPtr array_handle)
        {
            unsafe
            {
                TVMArrayFree((TVMTensor*)array_handle);
            }
        }
    }
}
