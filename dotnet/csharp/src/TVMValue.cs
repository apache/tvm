using System;

namespace TVMRuntime
{
    public class TVMValue
    {
        TVMTypeCode _typeCode;
        Object _value;

        public TVMValue(TVMTypeCode typeCode, Object value)
        {
            _typeCode = typeCode;
            _value = value;
        }

        private bool IsLongValue()
        {
            switch(_typeCode)
            {
                case TVMTypeCode.TVMInt:
                case TVMTypeCode.TVMUInt:
                    return true;
                default:
                    return false;
            }
        }

        public long AsLong()
        {
            if(!IsLongValue()) throw new NotSupportedException();
            return (long)_value;
        }

        public double AsDouble()
        {
            if(_typeCode != TVMTypeCode.TVMFloat) throw new NotSupportedException();
            return (double)_value;
        }

        public byte[] AsBytes()
        {
            if (_typeCode != TVMTypeCode.TVMBytes) throw new NotSupportedException();
            return (byte[])_value;
        }

        public Module AsModule()
        {
            if (_typeCode != TVMTypeCode.TVMModuleHandle) throw new NotSupportedException();
            return (Module)_value;
        }

        public PackedFunction AsFunction()
        {
            if (_typeCode != TVMTypeCode.TVMPackedFuncHandle) throw new NotSupportedException();
            return (PackedFunction)_value;
        }

        private bool IsNDArrayValue()
        {
            switch (_typeCode)
            {
                case TVMTypeCode.TVMDLTensorHandle:
                case TVMTypeCode.TVMNDArrayHandle:
                    return true;
                default:
                    return false;
            }
        }

        public NDArray AsNDArray()
        {
            if (!IsNDArrayValue()) throw new NotSupportedException();
            return (NDArray)_value;
        }

        public string AsString()
        {
            if (_typeCode != TVMTypeCode.TVMStr) throw new NotSupportedException();
            return (string)_value;
        }

        private bool IsHandleValue()
        {
            switch (_typeCode)
            {
                case TVMTypeCode.TVMOpaqueHandle:
                case TVMTypeCode.TVMObjectHandle:
                case TVMTypeCode.TVMModuleHandle:
                case TVMTypeCode.TVMPackedFuncHandle:
                case TVMTypeCode.TVMDLTensorHandle:
                case TVMTypeCode.TVMNDArrayHandle:
                    return true;
                default:
                    return false;
            }
        }

        public IntPtr AsHandle()
        {
            if (!IsHandleValue()) throw new NotSupportedException();
            return (IntPtr)_value;
        }

        public void DisposeTVMValue()
        {
            switch (_typeCode)
            {
                case TVMTypeCode.TVMInt:
                case TVMTypeCode.TVMUInt:
                case TVMTypeCode.TVMFloat:
                case TVMTypeCode.TVMOpaqueHandle:
                case TVMTypeCode.TVMBytes:
                case TVMTypeCode.TVMNullptr:
                    _value = null;
                    break;
                case TVMTypeCode.TVMModuleHandle:
                    ((Module)_value).DisposeModule();
                    break;
                case TVMTypeCode.TVMPackedFuncHandle:
                    ((PackedFunction)_value).Dispose();
                    break;
                case TVMTypeCode.TVMDLTensorHandle:
                case TVMTypeCode.TVMNDArrayHandle:
                    ((NDArray)_value).Dispose();
                    break;
                default:
                    throw new System.ArgumentException(_typeCode.ToString() + " not supported!");
            }

        }

        ~TVMValue()
        {
            DisposeTVMValue();
        }
    }
}