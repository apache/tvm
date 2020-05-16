using System;

namespace TVMRuntime
{
    /// <summary>
    /// TVM Data type code.
    /// </summary>
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

        /// <summary>
        /// The number of bytes.
        /// </summary>
        public byte numOfBytes;

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TVMRuntime.TVMDataType"/> struct.
        /// </summary>
        /// <param name="dtype">Dtype.</param>
        public TVMDataType(string dtype) : this()
        {
            string lowerDtype = dtype.ToLower();
            bits = 0;

            // TODO: support user input for lanes
            lanes = 1;
            if (lowerDtype.StartsWith("i") && lowerDtype.Contains("int"))
            {
                code = (byte)TVMDataTypeCode.Int;
                bits = (byte)Int32.Parse(lowerDtype.Split("int")[1]);
            }
            else if (lowerDtype.StartsWith("u") && lowerDtype.Contains("uint"))
            {
                code = (byte)TVMDataTypeCode.UInt;
                bits = (byte)Int32.Parse(lowerDtype.Split("uint")[1]);
            }
            else if (lowerDtype.StartsWith("f") && lowerDtype.Contains("float"))
            {
                code = (byte)TVMDataTypeCode.Float;
                bits = (byte)Int32.Parse(lowerDtype.Split("float")[1]);
            }
            else if (lowerDtype.StartsWith("b") && lowerDtype.Contains("bool"))
            {
                code = (byte)TVMDataTypeCode.UInt;
                bits = 1;
            }

            if (((bits & (bits - 1)) != 0) || (bits < 8))
            {
                throw new System.ArgumentException(dtype + " not supported!");
            }

            numOfBytes = (byte)Utils.GetNumOfBytes(bits);
        }

        // Byte type
        public bool IsInt8()
        {
            if ((code == (byte)TVMDataTypeCode.Int) && (bits == 8)) return true;
            return false;
        }

        // Char type
        public bool IsUint16()
        {
            if ((code == (byte)TVMDataTypeCode.UInt) && (bits == 16)) return true;
            return false;
        }

        // Short type
        public bool IsInt16()
        {
            if ((code == (byte)TVMDataTypeCode.Int) && (bits == 16)) return true;
            return false;
        }

        // Int type
        public bool IsInt32()
        {
            if ((code == (byte)TVMDataTypeCode.Int) && (bits == 32)) return true;
            return false;
        }

        // Long type
        public bool IsInt64()
        {
            if ((code == (byte)TVMDataTypeCode.Int) && (bits == 64)) return true;
            return false;
        }

        // Float type
        public bool IsFloat32()
        {
            if ((code == (byte)TVMDataTypeCode.Float) && (bits == 32)) return true;
            return false;
        }

        // Double type
        public bool IsFloat64()
        {
            if ((code == (byte)TVMDataTypeCode.Float) && (bits == 64)) return true;
            return false;
        }
    }
}
