using System;
using Xunit;
using TVMRuntime;

namespace Tests
{
    public class UnitTestNDArray
    {
        [Fact]
        // J
        public void TestNDArrayCreateSuccess()
        {
            NDArray testND = NDArray.Empty();

            Assert.Equal(testND.NDArrayHandle, IntPtr.Zero);

            testND.Dispose();
        }

        [Fact]
        // A
        public void TestNDArrayFloatTypeOneDimDataSuccess()
        {
            long[] shape = { 3 };
            TVMContext ctx = new TVMContext(0);
            NDArray testND = NDArray.Empty(shape, "float32", ctx);
            Assert.NotEqual(testND.NDArrayHandle, IntPtr.Zero);

            float[] data = { 1.3F, 2.3F, 3.3F };

            testND.CopyFrom(data);

            Assert.Equal(data, testND.AsFloatArray());

            // Check Indexer wise Set, Get
            for (int i = 0; i < data.Length; i++)
            {
                testND[i] = data[i];
            }

            for (int i = 0; i < data.Length; i++)
            {
                Assert.Equal(data[i], testND[i]);
            }

            testND.Dispose();
        }

        [Fact]
        // I
        public void TestNDArrayFloatTypeMultiDimDataSuccess()
        {
            long[] shape = { 224, 224, 3 };
            TVMContext ctx = new TVMContext(0);
            NDArray testND = NDArray.Empty(shape, "float32", ctx);
            Assert.NotEqual(testND.NDArrayHandle, IntPtr.Zero);

            float[] data = new float[testND.Size];

            for (int i = 0; i < testND.Size; i++)
            {
                data[i] = (float)i / 0.3F;
            }

            testND.CopyFrom(data);

            Assert.Equal(data, testND.AsFloatArray());

            // Check Indexer wise Set, Get
            for (int i = 0; i < data.Length; i++)
            {
                testND[i] = data[i];
            }

            for (int i = 0; i < testND.Size; i++)
            {
                Assert.Equal(data[i], testND[i]);
            }

            testND.Dispose();
        }

        [Fact]
        // H
        public void TestNDArrayDoubleTypeOneDimDataSuccess()
        {
            long[] shape = { 3 };
            TVMContext ctx = new TVMContext(0);
            NDArray testND = NDArray.Empty(shape, "float64", ctx);
            Assert.NotEqual(testND.NDArrayHandle, IntPtr.Zero);

            double[] data = { 1.333333, 2.333333, 3.333333 };

            testND.CopyFrom(data);

            Assert.Equal(data, testND.AsDoubleArray());

            // Check Indexer wise Set, Get
            for (int i = 0; i < data.Length; i++)
            {
                testND[i] = data[i];
            }

            for (int i = 0; i < data.Length; i++)
            {
                Assert.Equal(data[i], testND[i]);
            }

            testND.Dispose();
        }

        [Fact]
        // A
        public void TestNDArrayDoubleTypeMultiDimDataSuccess()
        {
            long[] shape = { 3, 3, 3 };
            TVMContext ctx = new TVMContext(0);
            NDArray testND = NDArray.Empty(shape, "float64", ctx);
            Assert.NotEqual(testND.NDArrayHandle, IntPtr.Zero);

            double[] data = new double[testND.Size];

            for (int i = 0; i < testND.Size; i++)
            {
                data[i] = (double)i / 0.3;
            }

            testND.CopyFrom(data);

            Assert.Equal(data, testND.AsDoubleArray());

            // Check Indexer wise Set, Get
            for (int i = 0; i < data.Length; i++)
            {
                testND[i] = data[i];
            }

            for (int i = 0; i < testND.Size; i++)
            {
                Assert.Equal(data[i], testND[i]);
            }

            testND.Dispose();
        }

        [Fact]
        // N
        public void TestNDArrayIntTypeOneDimDataSuccess()
        {
            long[] shape = { 3 };
            TVMContext ctx = new TVMContext(0);
            NDArray testND = NDArray.Empty(shape, "int32", ctx);
            Assert.NotEqual(testND.NDArrayHandle, IntPtr.Zero);

            int[] data = { 1, 2, 3 };

            testND.CopyFrom(data);

            Assert.Equal(data, testND.AsIntArray());

            // Check Indexer wise Set, Get
            for (int i = 0; i < data.Length; i++)
            {
                testND[i] = data[i];
            }

            for (int i = 0; i < data.Length; i++)
            {
                Assert.Equal(data[i], testND[i]);
            }

            testND.Dispose();
        }

        [Fact]
        // U
        public void TestNDArrayIntTypeMultiDimDataSuccess()
        {
            long[] shape = { 3, 3, 3 };
            TVMContext ctx = new TVMContext(0);
            NDArray testND = NDArray.Empty(shape, "int32", ctx);
            Assert.NotEqual(testND.NDArrayHandle, IntPtr.Zero);

            int[] data = new int[testND.Size];

            for (int i = 0; i < testND.Size; i++)
            {
                data[i] = i;
            }

            testND.CopyFrom(data);

            Assert.Equal(data, testND.AsIntArray());

            // Check Indexer wise Set, Get
            for (int i = 0; i < data.Length; i++)
            {
                testND[i] = data[i];
            }

            for (int i = 0; i < testND.Size; i++)
            {
                Assert.Equal(data[i], testND[i]);
            }

            testND.Dispose();
        }

        [Fact]
        // M
        public void TestNDArrayLongTypeOneDimDataSuccess()
        {
            long[] shape = { 3 };
            TVMContext ctx = new TVMContext(0);
            NDArray testND = NDArray.Empty(shape, "int64", ctx);
            Assert.NotEqual(testND.NDArrayHandle, IntPtr.Zero);

            long[] data = { 1L, 2L, 3L };

            testND.CopyFrom(data);

            Assert.Equal(data, testND.AsLongArray());

            // Check Indexer wise Set, Get
            for (int i = 0; i < data.Length; i++)
            {
                testND[i] = data[i];
            }

            for (int i = 0; i < data.Length; i++)
            {
                Assert.Equal(data[i], testND[i]);
            }

            testND.Dispose();
        }

        [Fact]
        // A
        public void TestNDArrayLongTypeMultiDimDataSuccess()
        {
            long[] shape = { 3, 3, 3 };
            TVMContext ctx = new TVMContext(0);
            NDArray testND = NDArray.Empty(shape, "int64", ctx);
            Assert.NotEqual(testND.NDArrayHandle, IntPtr.Zero);

            long[] data = new long[testND.Size];

            for (int i = 0; i < testND.Size; i++)
            {
                data[i] = i * 333333L;
            }

            testND.CopyFrom(data);

            Assert.Equal(data, testND.AsLongArray());

            // Check Indexer wise Set, Get
            for (int i = 0; i < data.Length; i++)
            {
                testND[i] = data[i];
            }

            for (int i = 0; i < testND.Size; i++)
            {
                Assert.Equal(data[i], testND[i]);
            }

            testND.Dispose();
        }

        [Fact]
        // N
        public void TestNDArrayShortTypeOneDimDataSuccess()
        {
            long[] shape = { 3 };
            TVMContext ctx = new TVMContext(0);
            NDArray testND = NDArray.Empty(shape, "int16", ctx);
            Assert.NotEqual(testND.NDArrayHandle, IntPtr.Zero);

            short[] data = { 1, 2, 3 };

            testND.CopyFrom(data);

            Assert.Equal(data, testND.AsShortArray());

            // Check Indexer wise Set, Get
            for (int i = 0; i < data.Length; i++)
            {
                testND[i] = data[i];
            }

            for (int i = 0; i < data.Length; i++)
            {
                Assert.Equal(data[i], testND[i]);
            }

            testND.Dispose();
        }

        [Fact]
        public void TestNDArrayShortTypeMultiDimDataSuccess()
        {
            long[] shape = { 3, 3, 3 };
            TVMContext ctx = new TVMContext(0);
            NDArray testND = NDArray.Empty(shape, "int16", ctx);
            Assert.NotEqual(testND.NDArrayHandle, IntPtr.Zero);

            short[] data = new short[testND.Size];

            for (int i = 0; i < testND.Size; i++)
            {
                data[i] = (short)i;
            }

            testND.CopyFrom(data);

            Assert.Equal(data, testND.AsShortArray());

            // Check Indexer wise Set, Get
            for (int i = 0; i < data.Length; i++)
            {
                testND[i] = data[i];
            }

            for (int i = 0; i < testND.Size; i++)
            {
                Assert.Equal(data[i], testND[i]);
            }

            testND.Dispose();
        }

        [Fact]
        public void TestNDArrayCharTypeOneDimDataSuccess()
        {
            long[] shape = { 3 };
            TVMContext ctx = new TVMContext(0);
            NDArray testND = NDArray.Empty(shape, "uint16", ctx);
            Assert.NotEqual(testND.NDArrayHandle, IntPtr.Zero);

            char[] data = { 'A', 'N', 'S' };

            testND.CopyFrom(data);

            Assert.Equal(data, testND.AsCharArray());

            // Check Indexer wise Set, Get
            for (int i = 0; i < data.Length; i++)
            {
                testND[i] = data[i];
            }

            for (int i = 0; i < data.Length; i++)
            {
                Assert.Equal(data[i], testND[i]);
            }

            testND.Dispose();
        }

        [Fact]
        public void TestNDArrayCharTypeMultiDimDataSuccess()
        {
            long[] shape = { 3, 3, 3 };
            TVMContext ctx = new TVMContext(0);
            NDArray testND = NDArray.Empty(shape, "uint16", ctx);
            Assert.NotEqual(testND.NDArrayHandle, IntPtr.Zero);

            char[] data = new char[testND.Size];

            for (int i = 0; i < testND.Size; i++)
            {
                data[i] = (char)(i % 65536);
            }

            testND.CopyFrom(data);

            Assert.Equal(data, testND.AsCharArray());

            // Check Indexer wise Set, Get
            for (int i = 0; i < data.Length; i++)
            {
                testND[i] = data[i];
            }

            for (int i = 0; i < testND.Size; i++)
            {
                Assert.Equal(data[i], testND[i]);
            }

            testND.Dispose();
        }

        [Fact]
        public void TestNDArrayByteTypeOneDimDataSuccess()
        {
            long[] shape = { 3 };
            TVMContext ctx = new TVMContext(0);
            NDArray testND = NDArray.Empty(shape, "int8", ctx);
            Assert.NotEqual(testND.NDArrayHandle, IntPtr.Zero);

            byte[] data = { 1, 2, 3 };

            testND.CopyFrom(data);

            Assert.Equal(data, testND.AsByteArray());

            // Check Indexer wise Set, Get
            for (int i = 0; i < data.Length; i++)
            {
                testND[i] = data[i];
            }

            for (int i = 0; i < data.Length; i++)
            {
                Assert.Equal(data[i], testND[i]);
            }

            testND.Dispose();
        }

        [Fact]
        public void TestNDArrayByteTypeMultiDimDataSuccess()
        {
            long[] shape = { 3, 3, 3 };
            TVMContext ctx = new TVMContext(0);
            NDArray testND = NDArray.Empty(shape, "int8", ctx);
            Assert.NotEqual(testND.NDArrayHandle, IntPtr.Zero);

            byte[] data = new byte[testND.Size];

            for (int i = 0; i < testND.Size; i++)
            {
                data[i] = (byte)(i % 256);
            }

            testND.CopyFrom(data);

            Assert.Equal(data, testND.AsByteArray());

            // Check Indexer wise Set, Get
            for (int i = 0; i < data.Length; i++)
            {
                testND[i] = data[i];
            }

            for (int i = 0; i < testND.Size; i++)
            {
                Assert.Equal(data[i], testND[i]);
            }

            testND.Dispose();
        }
    }
}
