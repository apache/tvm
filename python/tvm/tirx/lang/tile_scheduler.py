# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Reusable tile scheduler helpers for TIR tests/kernels.

These classes emit TIR via @T.inline. Decorate with @T.meta_class so that
instances are automatically treated as meta values inside @T.prim_func.
"""

from tvm.script import tirx as T


@T.meta_class
class BaseTileScheduler:
    """Base class for tile schedulers with common state and macros."""

    def __init__(self, prefix: str):
        self.m_idx = T.local_scalar("int32")
        self.n_idx = T.local_scalar("int32")
        self.linear_idx = T.local_scalar("int32")

    @T.inline
    def update_current_m_n_idx(self, linear_idx):
        # To be implemented by subclasses
        pass

    @T.inline
    def init(self, linear_init):
        self.linear_idx = linear_init
        self.update_current_m_n_idx(linear_init)

    @T.inline
    def next_tile(self, step):
        self.linear_idx = self.linear_idx + step
        self.update_current_m_n_idx(self.linear_idx)

    def valid(self, total_tiles):
        return self.linear_idx < total_tiles


class ClusterPersistentScheduler2D(BaseTileScheduler):
    """
    Tile scheduler for cluster-based persistent kernels.

    Distributes a 2D tile grid across persistent clusters using group-major ordering
    for L2 cache locality. Each cluster starts at its cluster_id and strides by
    num_clusters to process tiles.

    Tile Ordering (group-major for L2 locality):
    - Tiles are grouped into "L2 groups" of `l2_group_size` rows
    - Within a group, tiles are visited in column-major order within the group
    - Groups are processed in row-major order

    Example with 4x4 tiles, l2_group_size=2:
        Group 0 (rows 0-1):  0  2  4  6
                             1  3  5  7
        Group 1 (rows 2-3):  8 10 12 14
                             9 11 13 15

    Serpentine Mode (serpentine=True):
    - Uses CUTLASS-style 2D block swizzle with serpentine traversal
    - Grid is divided into swizzle_size x swizzle_size blocks
    - Within each block, tiles are visited in row-major order
    - Blocks are traversed in serpentine order (even block-rows forward, odd backward)
    - This provides better L2 locality by reusing both A and B tiles

    Example with 4x4 tiles, swizzle_size=2, serpentine=True:
        Block layout:
          Block(0,0)  Block(0,1)
          Block(1,0)  Block(1,1)

        Tile numbering with serpentine:
               n=0  n=1  n=2  n=3
          m=0   0    1   14   15
          m=1   2    3   12   13
          m=2   4    5   10   11
          m=3   6    7    8    9

        Traversal: Block(0,0) -> Block(1,0) -> Block(1,1) -> Block(0,1)
                   (serpentine: down in col 0, then up in col 1)

    Parameters
    ----------
    prefix : str
        Prefix for TIR variable names
    num_m_tiles : int | T.ExprLike
        Total number of tiles in M dimension (can be runtime expression)
    num_n_tiles : int
        Total number of tiles in N dimension
    num_clusters : int
        Number of persistent clusters (determines stride)
    l2_group_size : int
        Number of M-tile rows per L2 locality group (default: 8)
        When serpentine=True, this is used as swizzle_size for 2D blocks
    cluster_m : int
        Cluster dimension in M for hierarchical scheduling (default: 1)
    cluster_n : int
        Cluster dimension in N for hierarchical scheduling (default: 1)
    serpentine : bool
        If True, use CUTLASS-style 2D block swizzle with serpentine traversal (default: False)

    Attributes
    ----------
    m_idx : T.local_scalar
        Current M tile index (output)
    n_idx : T.local_scalar
        Current N tile index (output)
    work_idx : T.local_scalar
        Global work item index for this cluster
    tile_count : T.local_scalar
        Number of tiles processed by this cluster so far

    Usage
    -----
    ```python
    scheduler = ClusterPersistentScheduler2D(
        "sched", num_m_tiles=M_TILES, num_n_tiles=N_TILES,
        num_clusters=NUM_CLUSTERS, l2_group_size=8
    )
    scheduler.init(cluster_id)  # cluster_id = cta_idx // CLUSTER_SIZE

    while scheduler.valid():
        m = T.meta_var(scheduler.m_idx)  # current M tile
        n = T.meta_var(scheduler.n_idx)  # current N tile
        # ... process tile (m, n) ...
        scheduler.next_tile()
    ```

    Examples
    --------
    Example 1: Basic persistent kernel
    ```
    num_m_tiles=4, num_n_tiles=4, num_clusters=3, l2_group_size=2
    cluster_m=1, cluster_n=1 (default, no tile subdivision)

    Group-major tile numbering (l2_group_size=2):
           n=0  n=1  n=2  n=3
      m=0   0    2    4    6   ┐ L2 group 0
      m=1   1    3    5    7   ┘
      m=2   8   10   12   14   ┐ L2 group 1
      m=3   9   11   13   15   ┘

    Work distribution (cluster starts at cluster_id, strides by num_clusters=3):
      cluster 0: work_idx 0,3,6,9,12,15  -> tiles 0,3,6,9,12,15
      cluster 1: work_idx 1,4,7,10,13    -> tiles 1,4,7,10,13
      cluster 2: work_idx 2,5,8,11,14    -> tiles 2,5,8,11,14

    Tile grid (which cluster handles each tile):
           n=0  n=1  n=2  n=3
      m=0   C0   C2   C1   C0   ┐ L2 group 0
      m=1   C1   C0   C2   C1   ┘
      m=2   C2   C1   C0   C2   ┐ L2 group 1
      m=3   C0   C2   C1   C0   ┘

    Tile sequence per cluster (in execution order):
      cluster 0: (0,0)->(1,1)->(0,3)->(2,0)->(2,3)->(3,3)
      cluster 1: (1,0)->(0,2)->(1,3)->(2,1)->(3,2)
      cluster 2: (0,1)->(1,2)->(2,0)->(3,1)->(2,3)
    ```

    Example 2: 2SM GEMM (typical B200 config)
    ```
    M=1024, N=512, CTA_M=128, MMA_N=128, CLUSTER_M=2, CLUSTER_N=1
    => M_TILES=8, N_TILES=4
    => CLUSTER_M_TILES=4, CLUSTER_N_TILES=4 (scheduler at cluster granularity)

    Scheduler params:
      num_m_tiles=4, num_n_tiles=4, num_clusters=74, l2_group_size=8
      cluster_m=1, cluster_n=1

    Key: Scheduler outputs CLUSTER-level tiles.
         All CTAs in same cluster get SAME (m_idx, n_idx) from scheduler.
         CTAs differentiate via cluster_rank (computed OUTSIDE scheduler):
           cluster_rank = cta_idx % CLUSTER_SIZE
           cb_m = cluster_rank % CLUSTER_M   # 0 or 1 for 2SM
           cb_n = cluster_rank // CLUSTER_M  # 0 for 2SM

    Final CTA tile:
      cta_m = m_idx * CLUSTER_M + cb_m
      cta_n = n_idx * CLUSTER_N + cb_n

    Example: cluster 5 gets scheduler tile (1,2)
      CTA rank=0 (cb_m=0): actual tile (2,2)
      CTA rank=1 (cb_m=1): actual tile (3,2)
    ```
    """

    def __init__(
        self,
        prefix: str,
        num_m_tiles,
        num_n_tiles: int,
        num_clusters: int,
        l2_group_size: int = 8,
        cluster_m: int = 1,
        cluster_n: int = 1,
        serpentine: bool = False,
    ):
        super().__init__(prefix)
        self._num_m_tiles = num_m_tiles
        self._num_n_tiles = num_n_tiles
        self._num_clusters = num_clusters
        self._l2_group_size = l2_group_size
        self._cluster_m = cluster_m
        self._cluster_n = cluster_n
        self._serpentine = serpentine

        # Rename internal state for clarity
        self.work_idx = self.linear_idx  # alias: global work item index
        self.tile_count = T.local_scalar("int32")
        self.tile_idx = self.tile_count  # alias for backward compatibility

        is_static_m = isinstance(num_m_tiles, int)

        # Number of tile columns after accounting for cluster_n
        n_tile_cols = (num_n_tiles + cluster_n - 1) // cluster_n
        self._N_TILE_COLS = n_tile_cols

        if is_static_m:
            self._M_TILE_ROWS = (num_m_tiles + cluster_m - 1) // cluster_m
            self._FULL_GROUPS = self._M_TILE_ROWS // l2_group_size
        else:
            # Dynamic expressions for runtime M
            self._M_TILE_ROWS = T.truncdiv(self._num_m_tiles + self._cluster_m - 1, self._cluster_m)
            self._FULL_GROUPS = T.truncdiv(self._M_TILE_ROWS, self._l2_group_size)

        self._TAIL_ROWS = self._M_TILE_ROWS - self._FULL_GROUPS * l2_group_size
        self._TOTAL_TILES = self._M_TILE_ROWS * n_tile_cols * cluster_m * cluster_n

        # For serpentine mode: precompute block counts
        if serpentine:
            self._N_BLOCKS = n_tile_cols // l2_group_size  # full blocks in N
            self._M_BLOCKS = (
                self._M_TILE_ROWS // l2_group_size
                if is_static_m
                else T.truncdiv(self._M_TILE_ROWS, l2_group_size)
            )
            self._BLOCK_SIZE = l2_group_size * l2_group_size  # tiles per block
            self._FULL_BLOCK_TILES = self._M_BLOCKS * self._N_BLOCKS * self._BLOCK_SIZE
            # Residual tiles (not covered by full blocks)
            self._RESIDUAL_N = n_tile_cols - self._N_BLOCKS * l2_group_size
            self._RESIDUAL_M = self._M_TILE_ROWS - self._M_BLOCKS * l2_group_size

    # fmt: off
    @T.inline
    def update_current_m_n_idx(self, work_idx):
        """Convert global work index to (m_idx, n_idx) tile coordinates."""
        CLUSTER_M = T.meta_var(self._cluster_m)
        CLUSTER_N = T.meta_var(self._cluster_n)

        # Extract hierarchical cluster-local offsets
        cluster_m_offset = T.meta_var(work_idx % CLUSTER_M)
        t = T.meta_var(work_idx // CLUSTER_M)
        cluster_n_offset = T.meta_var(t % CLUSTER_N)
        tile_linear = T.meta_var(t // CLUSTER_N)

        @T.inline
        def set_tile_coords(tile_row, tile_col):
            self.m_idx = tile_row * CLUSTER_M + cluster_m_offset
            self.n_idx = tile_col * CLUSTER_N + cluster_n_offset

        if self._serpentine:
            self._update_serpentine(tile_linear, set_tile_coords)
        else:
            self._update_group_major(tile_linear, set_tile_coords)

    def _update_group_major(self, tile_linear, set_tile_coords):
        """Group-major ordering with parse-time pruning of statically-dead branches.

        The TIR script parser does not constant-fold ``if False: ...``, so a
        Python-literal ``FULL_GROUPS == 0`` would otherwise produce
        ``T.bitwise_and(T.bool(False), tile_linear < 0)`` IR plus the dead
        then-leg.  Branch in plain Python here and only invoke the inline
        emitter that can actually fire.
        """
        full_zero = isinstance(self._FULL_GROUPS, int) and self._FULL_GROUPS == 0
        tail_zero = isinstance(self._TAIL_ROWS, int) and self._TAIL_ROWS == 0
        if full_zero and tail_zero:
            self._gm_emit_zero(set_tile_coords)
        elif full_zero:
            self._gm_emit_tail_only(tile_linear, set_tile_coords)
        elif tail_zero:
            self._gm_emit_full_only(tile_linear, set_tile_coords)
        else:
            self._gm_emit_full_and_tail(tile_linear, set_tile_coords)

    @T.inline
    def _gm_emit_zero(self, set_tile_coords):
        set_tile_coords(0, 0)

    @T.inline
    def _gm_emit_full_only(self, tile_linear, set_tile_coords):
        FULL_GROUPS = T.meta_var(self._FULL_GROUPS)
        GROUP_SIZE = T.meta_var(self._l2_group_size)
        GROUP_SPAN = T.meta_var(self._l2_group_size * self._N_TILE_COLS)
        if (FULL_GROUPS > 0) & (tile_linear < FULL_GROUPS * GROUP_SPAN):
            group_id: T.let = tile_linear // GROUP_SPAN
            within_group: T.let = tile_linear % GROUP_SPAN
            tile_row: T.let = group_id * GROUP_SIZE + (within_group % GROUP_SIZE)
            tile_col: T.let = within_group // GROUP_SIZE
            set_tile_coords(tile_row, tile_col)
        else:
            set_tile_coords(0, 0)

    @T.inline
    def _gm_emit_tail_only(self, tile_linear, set_tile_coords):
        FULL_GROUPS = T.meta_var(self._FULL_GROUPS)
        TAIL_ROWS = T.meta_var(self._TAIL_ROWS)
        GROUP_SIZE = T.meta_var(self._l2_group_size)
        GROUP_SPAN = T.meta_var(self._l2_group_size * self._N_TILE_COLS)
        if TAIL_ROWS > 0:
            rem: T.let = tile_linear - FULL_GROUPS * GROUP_SPAN
            tile_row: T.let = FULL_GROUPS * GROUP_SIZE + (rem % TAIL_ROWS)
            tile_col: T.let = rem // TAIL_ROWS
            set_tile_coords(tile_row, tile_col)
        else:
            set_tile_coords(0, 0)

    @T.inline
    def _gm_emit_full_and_tail(self, tile_linear, set_tile_coords):
        FULL_GROUPS = T.meta_var(self._FULL_GROUPS)
        TAIL_ROWS = T.meta_var(self._TAIL_ROWS)
        GROUP_SIZE = T.meta_var(self._l2_group_size)
        GROUP_SPAN = T.meta_var(self._l2_group_size * self._N_TILE_COLS)
        if (FULL_GROUPS > 0) & (tile_linear < FULL_GROUPS * GROUP_SPAN):
            group_id: T.let = tile_linear // GROUP_SPAN
            within_group: T.let = tile_linear % GROUP_SPAN
            tile_row: T.let = group_id * GROUP_SIZE + (within_group % GROUP_SIZE)
            tile_col: T.let = within_group // GROUP_SIZE
            set_tile_coords(tile_row, tile_col)
        elif TAIL_ROWS > 0:
            rem: T.let = tile_linear - FULL_GROUPS * GROUP_SPAN
            tile_row: T.let = FULL_GROUPS * GROUP_SIZE + (rem % TAIL_ROWS)
            tile_col: T.let = rem // TAIL_ROWS
            set_tile_coords(tile_row, tile_col)
        else:
            set_tile_coords(0, 0)

    @T.inline
    def _update_serpentine(self, tile_linear, set_tile_coords):
        """CUTLASS-style 2D block swizzle with serpentine traversal.

        Algorithm:
        1. Divide grid into swizzle_size x swizzle_size blocks
        2. Within each block, visit tiles in row-major order
        3. Blocks are traversed column by column (along N)
        4. Within each column of blocks, use serpentine:
           - Even columns: top to bottom
           - Odd columns: bottom to top

        This maximizes L2 reuse for both A and B matrices.
        """
        S = T.meta_var(self._l2_group_size)  # swizzle_size
        M_BLOCKS = T.meta_var(self._M_BLOCKS)
        N_BLOCKS = T.meta_var(self._N_BLOCKS)
        BLOCK_SIZE = T.meta_var(self._BLOCK_SIZE)  # S * S
        FULL_BLOCK_TILES = T.meta_var(self._FULL_BLOCK_TILES)
        M_TILE_ROWS = T.meta_var(self._M_TILE_ROWS)
        T.meta_var(self._N_TILE_COLS)
        RESIDUAL_N = T.meta_var(self._RESIDUAL_N)
        RESIDUAL_M = T.meta_var(self._RESIDUAL_M)

        # Check if we're in the full block region
        if (M_BLOCKS > 0) & (N_BLOCKS > 0) & (tile_linear < FULL_BLOCK_TILES):
            # Which block (in linear order along columns of blocks)
            block_linear: T.let = tile_linear // BLOCK_SIZE
            within_block: T.let = tile_linear % BLOCK_SIZE

            # Block column and row
            block_col: T.let = block_linear // M_BLOCKS
            block_row_raw: T.let = block_linear % M_BLOCKS

            # Serpentine: odd columns go bottom-to-top
            block_row: T.let = T.Select(
                block_col % 2 == 0,
                block_row_raw,
                M_BLOCKS - 1 - block_row_raw
            )

            # Position within block (row-major within block)
            local_row: T.let = within_block // S
            local_col: T.let = within_block % S

            tile_row: T.let = block_row * S + local_row
            tile_col: T.let = block_col * S + local_col
            set_tile_coords(tile_row, tile_col)

        elif RESIDUAL_N > 0:
            # Residual tiles in the rightmost partial column of blocks
            # These are tiles where n >= N_BLOCKS * S
            rem: T.let = tile_linear - FULL_BLOCK_TILES

            # First handle the right residual strip (full M height, partial N width)
            right_strip_tiles: T.let = M_TILE_ROWS * RESIDUAL_N
            if rem < right_strip_tiles:
                # Row-major within the right strip
                tile_row: T.let = rem // RESIDUAL_N
                tile_col: T.let = N_BLOCKS * S + (rem % RESIDUAL_N)
                set_tile_coords(tile_row, tile_col)
            elif RESIDUAL_M > 0:
                # Bottom residual strip (already covered in right strip overlap)
                # This handles corner case - shouldn't normally reach here
                # as right strip already covers full M height
                set_tile_coords(0, 0)
            else:
                set_tile_coords(0, 0)

        elif RESIDUAL_M > 0:
            # Bottom residual strip only (no right residual)
            rem: T.let = tile_linear - FULL_BLOCK_TILES
            bottom_strip_tiles: T.let = RESIDUAL_M * (N_BLOCKS * S)
            if rem < bottom_strip_tiles:
                tile_row: T.let = M_BLOCKS * S + (rem % RESIDUAL_M)
                tile_col: T.let = rem // RESIDUAL_M
                set_tile_coords(tile_row, tile_col)
            else:
                set_tile_coords(0, 0)
        else:
            # Fallback
            set_tile_coords(0, 0)

    @T.inline
    def init(self, cluster_id):
        """Initialize scheduler for a given cluster.

        Parameters
        ----------
        cluster_id : int
            The cluster's index (typically cta_idx // CLUSTER_SIZE)
        """
        self.linear_idx = cluster_id
        self.tile_count = 0
        self.update_current_m_n_idx(cluster_id)

    @T.inline
    def next_tile(self):
        """Advance to the next tile for this cluster."""
        self.linear_idx = self.linear_idx + self._num_clusters
        self.tile_count = self.tile_count + 1
        self.update_current_m_n_idx(self.linear_idx)

    @T.inline
    def next_tile_stride(self, stride: int):
        """Advance by a custom stride (for non-standard scheduling)."""
        self.linear_idx = self.linear_idx + stride
        self.tile_count = self.tile_count + 1
        self.update_current_m_n_idx(self.linear_idx)
    # fmt: on

    def valid(self):
        """Check if this cluster has more tiles to process."""
        return self.linear_idx < self._TOTAL_TILES


class GroupMajor3D(BaseTileScheduler):
    """
    3D grouped-row scheduler (M,N,K) with tail handling on M.

    Args
    ----
    prefix: str
    m_tiles: int | T PrimExpr   # tiles along M (static or runtime)
    n_tiles: int                # tiles along N (static)
    k_tiles: int                # tiles along K (static)
    group_rows: int             # rows per group along M
    step: int = 1               # default stride for next_tile()
    """

    def __init__(
        self, prefix: str, m_tiles, n_tiles: int, k_tiles: int, group_rows: int, step: int = 1
    ):
        super().__init__(prefix)
        self._step = step
        self.tile_idx = T.local_scalar("int32")
        self.k_idx = T.local_scalar("int32")

        # ---- constants / primexprs baked once ----
        self._G = group_rows
        self._N = n_tiles
        self._K = k_tiles

        if isinstance(m_tiles, int):
            self._GROUPS = m_tiles // group_rows
            self._FINAL_ROWS = m_tiles - self._GROUPS * group_rows
            self._SAFE_FINAL_ROWS = max(self._FINAL_ROWS, 1)
            self._GROUP_SIZE = group_rows * n_tiles * k_tiles
            self._TOTAL = m_tiles * n_tiles * k_tiles
        else:
            self._GROUPS = T.truncdiv(m_tiles, group_rows)
            self._FINAL_ROWS = m_tiles - self._GROUPS * group_rows
            self._SAFE_FINAL_ROWS = T.max(self._FINAL_ROWS, 1)
            self._GROUP_SIZE = self._G * self._N * self._K
            self._TOTAL = m_tiles * n_tiles * k_tiles

        # handy composites used in macro
        self._FULL_BOUND = self._GROUPS * self._GROUP_SIZE
        self._HAS_FULL = self._GROUPS > 0
        self._HAS_TAIL = self._FINAL_ROWS > 0

    # fmt: off
    @T.inline
    def update_current_m_n_idx(self, linear_idx):
        # full-group formulas
        full_m: T.let = T.floordiv(linear_idx, self._GROUP_SIZE) * self._G + T.floormod(
            linear_idx, self._G
        )
        full_n: T.let = T.floormod(T.floordiv(linear_idx, self._G), self._N)
        full_k: T.let = T.floordiv(T.floormod(linear_idx, self._GROUP_SIZE), self._G * self._N)

        # tail formulas (relative to FULL_BOUND)
        # Use _SAFE_FINAL_ROWS (max(FINAL_ROWS, 1)) to avoid divide-by-zero when there is no tail
        rem: T.let = linear_idx - self._FULL_BOUND
        tail_m: T.let = self._GROUPS * self._G + T.floormod(rem, self._SAFE_FINAL_ROWS)
        tail_n: T.let = T.floordiv(rem, self._SAFE_FINAL_ROWS) % self._N
        tail_k: T.let = T.floordiv(rem, self._SAFE_FINAL_ROWS * self._N)

        # choose phase
        if self._HAS_FULL & (linear_idx < self._FULL_BOUND):
            self.m_idx = full_m
            self.n_idx = full_n
            self.k_idx = full_k
        elif self._HAS_TAIL:
            self.m_idx = tail_m
            self.n_idx = tail_n
            self.k_idx = tail_k
        else:
            self.m_idx = 0
            self.n_idx = 0
            self.k_idx = 0

    @T.inline
    def init(self, linear_init):
        self.linear_idx = linear_init
        self.tile_idx = 0
        self.update_current_m_n_idx(linear_init)

    @T.inline
    def next_tile(self):
        self.linear_idx = self.linear_idx + self._step
        self.tile_idx = self.tile_idx + 1
        self.update_current_m_n_idx(self.linear_idx)

    @T.inline
    def next_tile_stride(self, stride: int):
        self.linear_idx = self.linear_idx + stride
        self.tile_idx = self.tile_idx + 1
        self.update_current_m_n_idx(self.linear_idx)
    # fmt: on

    def valid(self):
        return self.linear_idx < self._TOTAL


class RankAwareGroupMajorTileScheduler(BaseTileScheduler):
    """
    Group-major scheduler that applies a rank-aware remapping (remote rows first).
    Kept as a thin adapter because it depends on NVSHMEM rank at device-side.
    """

    def __init__(
        self, prefix: str, m_clusters: int, n_clusters: int, group_size: int, world_size: int
    ):
        super().__init__(prefix)
        self._m_clusters = m_clusters
        self._n_clusters = n_clusters
        self._group_size = group_size
        self._world_size = world_size

    @T.inline
    def update_current_m_n_idx(self, linear_idx):
        my_rank: T.let = T.nvshmem.my_pe()
        remote_m_clusters: T.let = self._m_clusters - self._m_clusters // self._world_size
        group_rows: T.let = (remote_m_clusters // self._group_size) * self._group_size
        final_rows: T.let = remote_m_clusters - group_rows
        group_repeat: T.let = self._group_size * self._n_clusters
        if linear_idx < group_rows * self._n_clusters and group_rows > 0:
            self.m_idx = (
                (linear_idx // group_repeat) * self._group_size
                + (linear_idx % self._group_size)
                + (my_rank + 1) * self._m_clusters // self._world_size
            ) % self._m_clusters
            self.n_idx = (linear_idx % group_repeat) // self._group_size
        elif linear_idx < remote_m_clusters * self._n_clusters:
            remainder_idx: T.let = linear_idx - group_rows * self._n_clusters
            self.m_idx = (
                group_rows
                + remainder_idx % final_rows
                + (my_rank + 1) * self._m_clusters // self._world_size
            ) % self._m_clusters
            self.n_idx = remainder_idx // final_rows
        else:
            remainder_idx: T.let = linear_idx - remote_m_clusters * self._n_clusters
            self.m_idx = (
                remote_m_clusters
                + remainder_idx % (self._m_clusters // self._world_size)
                + (my_rank + 1) * self._m_clusters // self._world_size
            ) % self._m_clusters
            self.n_idx = remainder_idx // (self._m_clusters // self._world_size)

    @T.inline
    def next_tile(self, stride: int):
        self.linear_idx = self.linear_idx + stride
        self.update_current_m_n_idx(self.linear_idx)

    def valid(self):
        return self.linear_idx < self._m_clusters * self._n_clusters


class IndexedTripleTileScheduler(BaseTileScheduler):
    """Scheduler that maps linear_idx to (b_idx, h_idx, q_idx) via index lists."""

    def __init__(self, prefix: str, b_indices, h_indices, q_indices, tiles_indptr):
        super().__init__(prefix)
        self.b_indices = b_indices
        self.h_indices = h_indices
        self.q_indices = q_indices
        self.tiles_indptr = tiles_indptr
        self.q_idx = T.local_scalar("int32")
        self.h_idx = T.local_scalar("int32")
        self.b_idx = T.local_scalar("int32")
        self.linear_lim = T.local_scalar("int32")

    @T.inline
    def _load(self):
        self.q_idx = self.q_indices[self.linear_idx]
        self.h_idx = self.h_indices[self.linear_idx]
        self.b_idx = self.b_indices[self.linear_idx]

    @T.inline
    def init(self, sm):
        self.linear_idx = self.tiles_indptr[sm]
        self.linear_lim = self.tiles_indptr[sm + 1]
        self._load()

    @T.inline
    def next_tile(self):
        self.linear_idx = self.linear_idx + 1
        self._load()

    def valid(self):
        return self.linear_idx < self.linear_lim


class FlashAttentionLinearScheduler(BaseTileScheduler):
    """Linear 3D scheduler for flash attention (batch, head, m_block).

    Used for non-causal attention with simple linear decomposition.
    Maps linear_idx -> (batch_idx, head_idx, m_block_idx) using:
        batch = linear_idx // (num_heads * num_m_blocks)
        head = (linear_idx % (num_heads * num_m_blocks)) // num_m_blocks
        m_block = linear_idx % num_m_blocks

    Parameters
    ----------
    prefix : str
        Prefix for TIR variable names
    num_batches : int
        Number of batches
    num_heads : int
        Number of KV heads
    num_m_blocks : int
        Number of Q blocks (M dimension tiles)
    num_ctas : int
        Number of CTAs for persistent kernel stride
    """

    def __init__(
        self, prefix: str, num_batches: int, num_heads: int, num_m_blocks: int, num_ctas: int
    ):
        super().__init__(prefix)
        self._num_batches = num_batches
        self._num_heads = num_heads
        self._num_m_blocks = num_m_blocks
        self._num_ctas = num_ctas
        self._total_tasks = num_batches * num_heads * num_m_blocks

        # Output indices
        self.batch_idx = T.local_scalar("int32")
        self.head_idx = T.local_scalar("int32")
        self.m_block_idx = T.local_scalar("int32")

    # fmt: off
    @T.inline
    def update_current_m_n_idx(self, linear_idx):
        """Convert linear index to (batch, head, m_block) coordinates."""
        NUM_HEADS = T.meta_var(self._num_heads)
        NUM_M_BLOCKS = T.meta_var(self._num_m_blocks)
        HEAD_M_PRODUCT = T.meta_var(NUM_HEADS * NUM_M_BLOCKS)

        self.batch_idx = linear_idx // HEAD_M_PRODUCT
        self.head_idx = (linear_idx % HEAD_M_PRODUCT) // NUM_M_BLOCKS
        self.m_block_idx = linear_idx % NUM_M_BLOCKS

    @T.inline
    def init(self, cta_id):
        """Initialize scheduler with CTA ID."""
        self.linear_idx = cta_id
        self.update_current_m_n_idx(cta_id)

    @T.inline
    def next_tile(self):
        """Advance to next tile by striding by num_ctas."""
        self.linear_idx = self.linear_idx + self._num_ctas
        self.update_current_m_n_idx(self.linear_idx)
    # fmt: on

    def valid(self):
        """Check if there are more tiles to process."""
        return self.linear_idx < self._total_tasks


class FlashAttentionLPTScheduler(BaseTileScheduler):
    """LPT scheduler with L2 swizzle for causal flash attention.

    Processes high-work Q blocks (with more KV blocks to attend to) first using
    Longest Processing Time (LPT) scheduling. Also applies L2 cache swizzle
    for better cache locality across batch*head dimensions.

    The LPT aspect comes from reversing m_block order: lower Q blocks have more
    KV blocks to process due to causal masking, so processing them first balances load.

    The scheduler is only applied to non-persistent kernels.

    L2 Swizzle: Groups consecutive batch*head indices together for L2 locality.

    Parameters
    ----------
    prefix : str
        Prefix for TIR variable names
    num_batches : int
        Number of batches
    num_heads : int
        Number of KV heads
    num_m_blocks : int
        Number of Q blocks (M dimension tiles)
    num_ctas : int
        Number of CTAs (should equal total_tasks for causal)
    l2_swizzle : int
        L2 swizzle factor for cache locality
    """

    def __init__(
        self, prefix: str, num_batches: int, num_heads: int, num_m_blocks: int, l2_swizzle: int
    ):
        super().__init__(prefix)
        self._num_batches = num_batches
        self._num_heads = num_heads
        self._num_m_blocks = num_m_blocks
        self._l2_swizzle = l2_swizzle
        self._total_tasks = num_batches * num_heads * num_m_blocks

        # Derived constants for L2 swizzle
        self._num_hb = num_batches * num_heads
        self._l2_major = l2_swizzle * num_m_blocks
        self._num_hb_quotient = self._num_hb // l2_swizzle

        # Output indices
        self.batch_idx = T.local_scalar("int32")
        self.head_idx = T.local_scalar("int32")
        self.m_block_idx = T.local_scalar("int32")

    # fmt: off
    @T.inline
    def update_current_m_n_idx(self, linear_idx):
        """Convert linear index to (batch, head, m_block) with LPT + L2 swizzle."""
        L2_SWIZZLE = T.meta_var(self._l2_swizzle)
        L2_MAJOR = T.meta_var(self._l2_major)
        NUM_HB_QUOTIENT = T.meta_var(self._num_hb_quotient)
        NUM_HB = T.meta_var(self._num_hb)
        NUM_HEADS = T.meta_var(self._num_heads)
        NUM_M_BLOCKS = T.meta_var(self._num_m_blocks)

        # L2 swizzle decomposition
        bidhb: T.let = linear_idx // L2_MAJOR
        l2_mod: T.let = linear_idx % L2_MAJOR

        # Handle residual section (last partial swizzle group)
        num_hb_remainder: T.let = T.max(NUM_HB % L2_SWIZZLE, 1)
        m_block_raw: T.let = T.Select(bidhb < NUM_HB_QUOTIENT, l2_mod // L2_SWIZZLE, l2_mod // num_hb_remainder)  # noqa: E501
        bidhb_residual: T.let = T.Select(bidhb < NUM_HB_QUOTIENT, l2_mod % L2_SWIZZLE, l2_mod % num_hb_remainder)  # noqa: E501
        bidhb_actual: T.let = bidhb * L2_SWIZZLE + bidhb_residual

        self.batch_idx = bidhb_actual // NUM_HEADS
        self.head_idx = bidhb_actual % NUM_HEADS

        # LPT: Reverse block order so high-work blocks are processed first
        self.m_block_idx = (NUM_M_BLOCKS - 1) - m_block_raw

    @T.inline
    def init(self, cta_id):
        """Initialize scheduler with CTA ID."""
        self.linear_idx = cta_id
        self.update_current_m_n_idx(cta_id)

    @T.inline
    def next_tile(self):
        """Advance to next tile by striding by num_ctas."""
        self.linear_idx = self._total_tasks
    # fmt: on

    def valid(self):
        """Check if there are more tiles to process."""
        return self.linear_idx < self._total_tasks
