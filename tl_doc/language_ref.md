# TVM.TL language reference

## T.Kernel
args: the grid size (0-3 dimension) and the num_threads.

returns: the blockIdx variables

launch a kernel, it must be used in a with statement. There can be multiple kernels launched sequentially inside a prim function.

## T.alloc_shared
args: shape, dtype

returns: Buffer

Allocate buffer on shared memory, It must be used within T.Kernel scope and should be allocated at the top of the scope.

Dynamic shared memory is used.

## T.alloc_fragment
args: shape, dtype

returns: Buffer

Allocate buffer on register memory, It must be used within T.Kernel scope and should be allocated at the top of the scope.

The shape represents the whole shape of the buffer. Each element in the buffer is distributed stored on each threads, this storage partition will be inferred by the compiler.

## T.copy
args: src, dst

Copys data from src to dst, src and dst can be one of (Buffer, BufferLoad, BufferRegion). If you use BufferLoad that represents a single starting point, the other params should not be BufferLoad, since we need to know the copy region.

Zero will be padded if we detect the load is out of boundary.

## T.gemm
args: A, B, C, transpose_A, transpose_B, policy

Performs gemm operation on A, B and C. C must be a fragment, B must be on shared memory, A can be either a fragment or shared.

Note that the current implementation has some shape and dtype constraints, for example, the length of reduction axis must be a multiple of 32 for fp16 multiplicand case, we will update this later.

## T.reduce_max T.reduce_sum
args: src, dst, dim

Performs a reduce operation from src to dst on dimension dim. Currently we only support src and dst to be a fragment.

## T.Parallel
You can use T.Parallel to write a loop. The loop will be partitioned to all the threads by the compiler (The compiler will consider vectorize size, the fragment's thread mapping ... ). Note that this is the only way you can perform arbitary operation on fragments.

## T.Pipelined
args: start, stop, num_stages

Pipeline the loop, copy from the global memory will be converted to async operations and reordered to the point after it is consumed. num_stages is the number of buffer between producer-consumer. (e.g. Double buffer when num_stages=2)

## T.clear T.fill
nothing special, they will be converted to T.Parallel

## T.use_swizzle
Optimization for L2 cache. The launch of blockIdx.x and blockIdx.y will be serpentined.

You need to add it in a kernel after buffer is all allocated.
