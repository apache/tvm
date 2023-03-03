import tvm
from tvm import topi
import math
from typing import List, Sequence, Tuple

from tvm.tir.schedule import BlockRV, ExprRV, LoopRV
from collections import deque


def tile_layout_transform(
    sch: tvm.tir.Schedule,
    block_write: BlockRV,
    src_layout: str,
    dst_layout: str,
    input_shape: List[int],
    tile_size: ExprRV,
):
    """
    High level tiling for layout transform block.
    """
    
    ## Tiling layout transforms:
    # Assume we have an input shape of [A, B, C, D] and want to layout transform
    # ABCD --> DBAC so the output shape would be [D, B, A, C].
    #
    # Consider reading from the input buffer in a cache-friendly fashion on CPU. We would
    # expect a loop structure like:
    # lAr, lBr, lCr, lDr = T.grid(A, B, C, D)
    #
    # Meanwhile consider writing to the output buffer in a cache-friendly fashion on CPU:
    # lDw, lBw, lAw, lCw = T.grid(D, B, A, C)
    #
    # Clearly in many scenarios it is impossible to guarantee contiguous writes and reads
    # within a single loop. Due to non-adjacent dimensions. Instead we work on transposing some 
    # small sub-tensor of our input writing and then reading from shared memory. We must now 
    # construct our submatrix so that reading and writing can both be done with some contiguous  
    # access in global memory.
    #
    # Consider the case of a 2D transpose. For example [1024, 2048] -> [2048, 1024].
    # We note that if we deal with a submatrix of shape [32, 32] which corresponds 
    # to the dimension of our input tensor, then rows of the submatrix are contiguous
    # in the input tensor. Meanwhile, columns of our submatrix are contiguous in our 
    # output vector. Therefore, with this tile shape we have opportunity to read
    # contiguously in our input tensor and write to shared memory, and write contiguously
    # to our output tensor.
    #
    # The multiple dimensional case has a similar analogue. We want to allocate shared
    # memory per block of [`tile_size`, `tile_size`]. We want the inner most dimension
    # of our shared memory to correspond to contiguous reads from the input tensor and
    # the outer dimension to correspond to contiguous writes into the output tensor.
    # 
    # In terms of the loop structure reading from the input tensor, the inner most loops
    # of our tile must correspond to the inner most dimensions of the input shape,  
    # while the outer dimensions correspond to the inner most dimensions of the output shape.
    # To obtain an inner tile with this loop structure we factor out a contiguous `tile_size`
    # chunk of our loop in the shape of interest. 
    # 
    # An example is probably best to show this idea:
    # Let's say we want a layout transform of ABCD --> DCAB. With shape
    # [1024_a, 2_b, 32_c, 8_d] --> [8_d, 32_c, 1024_a, 2_b]
    #
    # And tile size 32.
    #
    # Then we initially have a coalesced-read loop pattern of:
    # T.grid(1024_a, 2_b, 32_c, 8_d)
    # 
    # To obtain an inner tile of 32, we factor 4 from 32_c and 8 from 8_d:
    # T.grid(1024_a, 2_b, 8_c1, 1_d1, 4_c2t, 8_d2t)
    # T.grid(1024_a, 2_b, 8_cr, 1_dr, 32_dim1)
    #
    # To obtain an outer tile of 32, we factor from B then A to follow contiguous write 
    # pattern:
    #
    # T.grid(64_a1, 1_b1, 8_cr, 1_dr, 16_a2t, 2_b2t, 32_dim1)
    # T.grid(64_ar, 1_br, 8_cr, 1_dr, 32_dim0, 32_dim1)
    #
    # Which allows us to read a tile with our wanted properties.
    # For writing we use the existing analysis infrastructure to generate the proper structure for writing.

    def pad_dimension_to_at_least_number(loop: LoopRV, requested_size: int):
        """E.g. if loop has extant of 8 but we want 10, returns size 10 loop with padding."""
        l1, l2 = sch.split(loop, [None, requested_size])
        return sch.fuse(l1, l2)

    def pad_dimension_to_factor_of_tile_size(
        loop: LoopRV, initial_size: int, tile_size: int = tile_size
    ) -> Tuple[LoopRV, int]:
        """
        Pads loop of given size until it is divisble into tile_size.
        If the given size of the loop is greater than tile size. Do not pad.

        example, loop_size = 5, tile_size = 32. loop_size --> 8
                loop_size = 5, tile_size = 36. loop_size --> 6
                loop_size = 8, tile_size = 32. loop_size --> 8
                loop_size = 33, tile_size = 32. loop_size --> 33

        Returns padded loopRV and the new size
        """
        if tile_size % initial_size == 0:
            return loop, int(initial_size)

        if initial_size > tile_size or initial_size == tile_size:
            return loop, int(initial_size)

        # if initial_size > tile_size return without change, factor = 1
        size = initial_size
        while (tile_size % size) % tile_size > 0:
            size += 1

        return pad_dimension_to_at_least_number(loop, size), int(size)

    def spin_out_factor(
        loops: List[LoopRV], loop_extants: List[int], index: int, factor_needed: int
    ) -> Tuple[List[LoopRV], List[int], int]:
        """
        Factor out loop dimensions to reach the requested factor. Updates the schedule in-place.

        E.g. say we want to factors which eventually multiply to 32 (factor_needed).

        Say we have the index we chose is a loop with an extant of 8.
        E.g. loops / loop_extants = [3, 32, 6, 8], factor_needed = 32, index = 3
            - 8 divides into 32 so we just split up the loop into two loops with extants 1 and 8.
            - we then keep the 1-loop in place and move the new 8-loop to back of the list of loops
            - ending loops / loop_extants = [3, 32, 6, 1, 8], remaining_factor_needed = 32 / 8 = 4

        E.g. loops / loop_extants = [3, 32, 6, 8], factor_needed=32, index = 0
            - 3 does not divide 32, so we pad until the extant divides 32, e.g. 4
            - we then split up the loop into extants 1 and 4, moving the 4 to the back
            - ending loops / loop_extants = [1, 32, 6, 8, 4], remaining_factor_needed = 32 / 4 = 8

        E.g. loops / loop_extants = [3, 32, 6, 8], factor_needed=5, index = 3
            - 8 is larger than 5 so we immediately do the splitting routine.
            - the 8 extant loop becomes loops with extants 2 and 5
            - ending loops / loop_extants = [1, 32, 6, 2, 5], remaining_factor_needed = 5 / 5 = 1

        After updating loop ordering in place, returns the new list of loops, extants, and the
        remaining factor needed.
        """
        cur_loop = loops[index]
        cur_extant = loop_extants[index]

        # Pad loops to divide evenly for factors needed, and split
        new_loop, new_size = pad_dimension_to_factor_of_tile_size(
            cur_loop, cur_extant, tile_size=factor_needed
        )

        split_factor = min(new_size, factor_needed)
        new_loop_split, factored_loop = sch.split(new_loop, [None, split_factor])
        factor_needed = factor_needed // split_factor

        # update caching
        loops[index] = new_loop_split
        loops.append(factored_loop)

        loop_extants[index] = math.ceil(new_size / split_factor)
        loop_extants.append(split_factor)

        sch.reorder(*loops)
        return loops, loop_extants, factor_needed

    def factor_dim_in_order(
        indices: Sequence[int],
        loops: List[LoopRV],
        cur_loop_extants: List[int],
        work_needed_inner_loop: int = tile_size,
    ):
        """Factors out the loops in the order of indices until we reach needed work.
        
        Adds new loop factors to the back in reverse order of access.
        """
        for i in indices:
            loops, cur_loop_extants, work_needed_inner_loop = spin_out_factor(
                loops, cur_loop_extants, i, work_needed_inner_loop
            )
            if work_needed_inner_loop == 1:
                break
        return loops, cur_loop_extants

    def get_high_level_loop_structure(block):
        """Runs the factorization described above."""
        # index 0 ... rank - 1 will always correspond to original loops
        # perhaps after they have been factored.
        loops = sch.get_loops(block)
        cur_loop_extants = list(input_shape)

        # Factor dim0 tile size and fuse things together
        loops, cur_loop_extants = factor_dim_in_order(
            range(rank - 1, -1, -1),
            loops,
            cur_loop_extants,
            work_needed_inner_loop=tile_size,
        )
        # The factors which multiply to tile_size are now in back of our
        # list of loops. However because we added them by traversing the inner
        # dimensions, they are actually reversed order to guarantee the best access
        # so reorder so reorder before fusing.
        loops = loops[:rank] + loops[rank:][::-1]
        cur_loop_extants = cur_loop_extants[:rank] + cur_loop_extants[rank::-1]
        sch.reorder(*loops)
        dim0_loop_tiled = sch.fuse(*loops[rank:])
        loops = loops[:rank]
        loops.append(dim0_loop_tiled)
        cur_loop_extants = cur_loop_extants[:rank]
        cur_loop_extants.append(tile_size)

        # Same thing with dim1
        # [:rank + 1], since we placed dim0_loop_tiled in the end which we want to keep
        loops, cur_loop_extants = factor_dim_in_order(
            (
                src_layout.index(dst_layout[loop_index_dst])
                for loop_index_dst in range(rank - 1, -1, -1)
            ),
            loops,
            cur_loop_extants,
            work_needed_inner_loop=tile_size,
        )
        loops = loops[: rank + 1] + loops[rank + 1 :][::-1]
        cur_loop_extants = cur_loop_extants[: rank + 1] + cur_loop_extants[rank + 1 :: -1]
        sch.reorder(*loops)
        dim1_loop_tiled = sch.fuse(*loops[rank + 1 :])
        loops = loops[: rank + 1]
        loops.append(dim1_loop_tiled)
        cur_loop_extants = cur_loop_extants[: rank + 1]
        cur_loop_extants.append(tile_size)

    rank = len(src_layout)

    # Outer loop structure of read block matches that of src_layout
    # E.g. if input_shape is [4, 6, 8]. Loops for read block will be
    # for i, j, k in T.grid(4, 6, 8):
    #     ...
    # Read block will read from global memory coalesced at the start
    # Assume write to output global memory is coalesced in block_write
    block_read = sch.cache_read(block_write, 0, "shared")

    # Here we have [loop1, loop2, loop3 ... dim0_tiled, dim1_tiled]
    get_high_level_loop_structure(block_read)
    loops = sch.get_loops(block_read)

    # If there are insufficient elements, than dim1_tiled or dim0_tiled might be too small
    # In all likelihood you should use a smaller tile, but I don't want things to crash.
    loops[-1] = pad_dimension_to_at_least_number(loops[-1], tile_size)
    loops[-2] = pad_dimension_to_at_least_number(loops[-2], tile_size)

    # We want the dim0 and dim1 parent loops to be the inner most. Right now dim1 is inner-msot
    # and we just need to move dim0 in (last dimension of dst).
    # Recall right now structure is at least [l1 l2 ... ln, dim0_tiled, dim1_tiled]
    # where n >= 2.
    dim0_loop_index = src_layout.index(dst_layout[-1])
    dim0_loop = loops.pop(dim0_loop_index)
    loops = loops[:-3] + [dim0_loop, loops[-3]] + loops[-2:]
    sch.reorder(*loops)

    # After this: [outer_loop (block binding), dim0_tiled, dim1_tiled]
    outer_loop = sch.fuse(*loops[:-2])

    # Now that we have the high level loop structure, we can use reverse_compute_at magic
    # To get the proper loop structure for writing! This is also as coalesced as possible
    # already.
    sch.reverse_compute_at(block_write, outer_loop)

    # Fuse all inner loops for the write into 2 loops, grab inner loops for both read
    # and write block which have locality (we will bind these to threadIdx)
    fused_write_loop = sch.fuse(*sch.get_loops(block_write)[1:])
    _, inner_write_loop = sch.split(fused_write_loop, [None, tile_size])
    inner_read_loop = sch.get_loops(block_read)[-2]

    sch.bind(loop=outer_loop, thread_axis="blockIdx.x")
    sch.bind(loop=inner_write_loop, thread_axis="threadIdx.x")
    sch.bind(loop=inner_read_loop, thread_axis="threadIdx.x")

def auto_inline(sch, start_block):
    # Autoinlines given block into consumers, and repeats process for consumer of block
    # Done by default for injective schedules.
    fringe = deque([start_block])
    visited = set()
    while len(fringe) > 0:
        cur_block = fringe.popleft()
        if cur_block in visited:
            continue
        else:
            visited.add(cur_block)

        consumer_blocks = sch.get_consumers(cur_block)
        if len(consumer_blocks) >= 1:
            fringe.extend(consumer_blocks)
            sch.compute_inline(cur_block)
        else:
            # Found output block, no more inlining needed
            return cur_block


@tvm.register_func("meta_schedule.cuda.layout_transform")
def cuda_layout_transform_schedule_rule(sch, block):
    # params: input_buffer, output_buffer
    params = sch.mod["main"].params
    input_buffer = sch.mod["main"].buffer_map[params[0]]
    output_buffer = sch.mod["main"].buffer_map[params[1]]
    
    # Info needed for tiling
    input_shape = [int(dim) for dim in input_buffer.shape]
    output_shape = [int(dim) for dim in output_buffer.shape]
    src_layout = sch.get_sref(block).stmt.annotations["src_layout"]
    dst_layout = sch.get_sref(block).stmt.annotations["dst_layout"]

    # For each schedule we also want to inline each stage as would be done in normal circumstances
    # to prevent extraneous memory access.
    block = auto_inline(sch, block)

    schedules = []

    # Always include the default schedules which will be handled via AutoBind schedule rule
    schedules.append(sch)

    # Tile size 2,3,4...32 as tile size of 1 has no coaslescing.
    for tile_size in range(2, 33):
        cur_sch = sch.copy()
        tile_layout_transform(cur_sch, block, src_layout, dst_layout, input_shape, tile_size)
        schedules.append(cur_sch)

    return schedules
