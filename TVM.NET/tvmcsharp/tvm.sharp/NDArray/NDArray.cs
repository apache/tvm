using System;

namespace TVMRuntime
{
    enum TVMDataTypeCode
    {
        Int = 0,
        UInt = 1,
        Float = 2,
    }

    public struct TVMDataType
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

        public TVMDataType(string dtype) : this()
        {
            string lower_dtype = dtype.ToLower();

            // TODO: support user input for lanes
            lanes = 1;
            if (lower_dtype.StartsWith("i") && lower_dtype.Contains("int"))
            {
                code = (byte)TVMDataTypeCode.Int;
                bits = (byte)Int32.Parse(lower_dtype.Split("int")[1]);
            }
            else if (lower_dtype.StartsWith("u") && lower_dtype.Contains("uint"))
            {
                code = (byte)TVMDataTypeCode.UInt;
                bits = (byte)Int32.Parse(lower_dtype.Split("uint")[1]);
            }
            else if (lower_dtype.StartsWith("f") && lower_dtype.Contains("float"))
            {
                code = (byte)TVMDataTypeCode.Float;
                bits = (byte)Int32.Parse(lower_dtype.Split("float")[1]);
            }
            else if (lower_dtype.StartsWith("b") && lower_dtype.Contains("bool"))
            {
                code = (byte)TVMDataTypeCode.UInt;
                bits = 1;
            }
            else
            {
                Console.WriteLine(dtype + " not supported!");
            }
        }
    }

    public class NDArray
    {
        UIntPtr array_handle = UIntPtr.Zero;
        int ndim = 0;
        int[] shape;
        int arraySize = 0;


        public NDArray(int[] shapeInp,
                          int ndimInp,
                          TVMDataType dataType,
                          TVMContext ctx)
        {
            UnmanagedNDArrayWrapper.CreateNDArray(shape, ndim, dataType.code,
                dataType.bits, dataType.lanes, (int)ctx.device_type,
                 ctx.device_id, ref array_handle);
            shape = shapeInp;
            ndim = ndimInp;
            arraySize = 1;
            Array.ForEach(shape, i => arraySize *= i);
        }

        public NDArray(int[] shapeInp,
                          int ndimInp,
                          int dtype_code,
                          int dtype_bits,
                          int dtype_lanes,
                          TVMContext ctx)
        {
            UnmanagedNDArrayWrapper.CreateNDArray(shape, ndim, dtype_code,
                dtype_bits, dtype_lanes, (int)ctx.device_type,
                 ctx.device_id, ref array_handle);
            shape = shapeInp;
            ndim = ndimInp;
            arraySize = 1;
            Array.ForEach(shape, i => arraySize *= i);
        }

        public NDArray(int[] shapeInp,
                          int ndimInp,
                          string dataTypeStr,
                          TVMContext ctx)
        {
            TVMDataType dataType = new TVMDataType(dataTypeStr);
            UnmanagedNDArrayWrapper.CreateNDArray(shape, ndim, dataType.code,
                dataType.bits, dataType.lanes, (int)ctx.device_type,
                 ctx.device_id, ref array_handle);
            shape = shapeInp;
            ndim = ndimInp;
            arraySize = 1;
            Array.ForEach(shape, i => arraySize *= i);
        }

        public UIntPtr NDArray_handle { get => array_handle; }

        public object this[int i]
        {
            get { return UnmanagedNDArrayWrapper.GetNDArrayElem(array_handle, i, arraySize); }
            set { UnmanagedNDArrayWrapper.SetNDArrayElem(array_handle, i, value, arraySize); }
        }

        ~NDArray()
        {
            UnmanagedNDArrayWrapper.DisposeNDArray(array_handle);
        }
    }
}
