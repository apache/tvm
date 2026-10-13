..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

Explicit CUDA tile patterns
================================================================================

Choose an instruction, prepare zero-copy operand views, then apply it::

    T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], registers, memory,
                      scope="thread")

The instruction and modifiers determine the atom contract. The caller supplies
regions with the canonical ``group × repeat × atom`` coordinates described
below. Lowering checks the contract and emits one specified PTX
instruction per repeat coordinate, in row-major repeat order. Lowering
unrolls static repeats. It does not infer a different opcode, width, matrix
count, transpose mode, or coordinate permutation.

CUDA exposes two tile operations: ``apply`` for an explicitly selected PTX
instruction and ``ewise`` for an explicitly supplied scalar TIRx function.
The raw ``T.ptx`` namespace remains available for scalar/address operands.
There is no automatic instruction selection or fallback to the retired CUDA
dispatch algorithms. Trainium's API and lowering are unchanged.

This page is the contract reference. The :doc:`tile_ops` instruction
guides provide complete kernels, caller views, synchronization, and numerical
checks, organized by PTX instruction family.

Group, repeat, and atom axes
--------------------------------------------------------------------------------

A thread instruction at ``scope="thread"`` has no group prefix. At
``scope="warp"`` it has a leading lane group axis; at ``scope="warpgroup"``
its groups are ``(warp_in_group, lane)``; at ``scope="cta"`` they are
``(warp_in_cta, lane)``. Extents come from the active execution scope, including
its explicitly guarded subrange. Warp-collective instructions consume the
lane axis inside their atom instead: their group prefix is empty at warp
scope, one warp axis at warpgroup/CTA scope. All 32 lanes must be active.
Cluster scope requires an explicit launch cluster and prepends its CTA axes
in z/y/x order, omitting undeclared leading axes. A one-dimensional cluster
therefore has ``(cta_x, warp, lane)`` groups for thread instructions.
Local tensor ownership uses ``cbx/cby/cbz`` for these CTA coordinates.

For example, 64 threads, three 128b loads per thread::

    # A and R both have shape (2, 32, 3, 4).
    registers = TileLayout(S[(2,32,3,4):(1 @ warpid,1 @ laneid,8,2)])
    T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], R, A, scope="cta")

The first two coordinates bind hardware IDs, the third generates three
unrolled calls per thread, and the last selects four words of one instruction.
The separated register slots are legal; the four memory words must be aligned
and contiguous. A changed opcode or modifier is always written at the call site.

Views such as ``sub``, ``permute``, ``view(dtype)``, and ``rearrange`` prepare
these coordinates without moving data. A failed contract requires a different
explicit view, caller-written preparation, or a different explicit instruction.
It never invokes a permutation/factorization solver.

128-bit global load
--------------------------------------------------------------------------------

``ld.global.v4.b32``, ``scope="thread"``:

* Both regions have shape ``(*repeat, 4)`` and scalar 32-bit elements.
* The source is global memory. For each repeat, its four words have addresses
  ``base + 0, base + 1, base + 2, base + 3`` (in words); the base is 16B aligned.
* The destination is four distinct registers of the executing thread. Their
  storage offsets may be separated or permuted. No thread axes are accepted.
* The operation copies raw bits; it performs no numeric conversion.

The earlier example keeps its original float32 tensor layout::

    original = TileLayout(S[(15, 2, 7, 2):(4, 1, 60, 2)])
    A_std = A.rearrange("i b j d -> j i (d b)")
    R_std = R.rearrange("i b j d -> j i (d b)")
    T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], R_std, A_std,
                      scope="thread")

``A_std`` has shape ``(7,15,4)`` and addresses ``60*j + 4*i + word``.
The repeat axes are ``j`` and ``i``: this emits 7*15 = 105 vector loads.
The destination may instead use original strides ``(8,2,120,4)``; each
instruction then writes slots separated by two, without a remap copy.

ldmatrix with interleaved rows and separated registers
--------------------------------------------------------------------------------

``ldmatrix.sync.aligned.m8n8.x2.shared.b16``, ``scope="warp"``:

* Both regions have shape ``(*repeat, 2, 8, 4)``. The suffix is
  ``(matrix, row, word)``.
* Each word is a 32-bit container holding two adjacent 16-bit matrix values.
  The pattern accepts float32/int32/uint32 containers as raw bits.
* In shared memory, each ``(matrix,row)`` is four consecutive words with a
  16B-aligned start. Different rows can be separated; they need not form a
  contiguous 8x8 allocation.
* Destination element ``(matrix,row,word)`` belongs to lane ``4*row+word``.
  Each lane receives two words; the register slots are selected by its layout.
* All 32 lanes must execute the instruction. Shared initialization and any
  preceding synchronization belong to the caller.

Use the original shared shape ``(16,4,2)`` and strides ``(8,1,128)``::

    S_std = SM.rearrange("(r m) w k -> k m r w", r=8, m=2)
    registers = TileLayout(S[(2,2,8,4):(8,2,4 @ laneid,1 @ laneid)])
    # R is declared over 12 caller-allocated words per thread, using registers.
    T.cuda.tile.apply(T.ptx["ldmatrix.sync.aligned.m8n8.x2.shared.b16"],
                      R, S_std, scope="warp")

``S_std[k,m,r,w] = SM[2*r+m,w,k]``. Its strides are ``(128,8,16,1)``:
first matrix = even original rows; second matrix = odd original rows.
For each ``k``, lanes 0..7 provide the eight matrix-0 row addresses, lanes
8..15 provide the eight matrix-1 row addresses, and lanes 16..31 duplicate
those valid addresses. These address providers are distinct from result owners.

For lane ``l``, the complete output mapping is::

    R_raw[0]  = SM[2*(l//4),   l%4, 0]
    R_raw[2]  = SM[2*(l//4)+1, l%4, 0]
    R_raw[8]  = SM[2*(l//4),   l%4, 1]
    R_raw[10] = SM[2*(l//4)+1, l%4, 1]

There are two repeat coordinates, ``k=0,1``, so this is exactly two ``.x2``
instructions. Slots 1,3,4,5,6,7,9,11 remain untouched. Lowering binds a
physical view of the caller's registers and passes these slots directly as
PTX output operands; it allocates no intermediate tile.

Memory and register instructions
--------------------------------------------------------------------------------

``ld`` / ``st`` use ``(*group,*repeat,word)``, where ``word`` has the vector
length selected in the opcode (one for scalar instructions). Both operands use
the written PTX storage width; use an explicit dtype view for packed words.
Global, shared, and local address spaces are explicit. Cache operators,
eviction policies, prefetch widths, and vector widths stay in the PTX spelling.
``pred`` and an opcode's optional ``cache_policy`` are ordinary runtime operands.
A predicated load can request ``preserve_dst=True``.

``ldmatrix`` / ``stmatrix`` support m8n8.b16 with x1/x2/x4 and optional
``.trans``. The canonical suffix is ``(matrix,8,4)`` in b32 containers.
For ``.trans``, the register image is the transpose of each shared b16 matrix;
its lane ownership is still ``4*row+word``. The shared address provider and
register owner are different maps.

Register instructions (mov, add/sub/mul/div, max/min, fma, cvt, sqrt, ex2, lg2,
rcp, abs, neg) end each tensor operand with the native register-group length.
A scalar arithmetic operand therefore uses suffix ``(1,)``. A read-only scalar
expression broadcasts over repeats. Packed ``f16x2`` / ``bf16x2`` / ``f32x2``
use b32 / b32 / b64 storage views, respectively. Packing is explicit; lowering
never chooses packed arithmetic because a layout happens to permit it.

Scalar compositions
--------------------------------------------------------------------------------

Define exactly the scalar expression to apply::

    @T.function(private=True)
    def silu_affine(x: T.float32, scale: T.float32, bias: T.float32) -> T.float32:
        y = x / (T.float32(1) + T.exp(-x))
        return y * scale + bias

    T.cuda.tile.ewise(silu_affine, R, R, T.float32(2), T.float32(1), scope="cta")

The atom is one scalar element. Tensor operands share the group/repeat shape;
scalar inputs broadcast. A tensor broadcast is an explicit zero-stride view.
Types must match the function parameters and result. The scalar function may
contain scalar bindings and returns, including conditional returns. Its typed
expression is serialized as ``LambdaExpr`` and inlined at each register store.
No shared/global staging, synchronization, or packed-vector selection occurs.
Shared-memory compositions explicitly load into registers, run ``ewise``, and
store back, with caller-owned synchronization.

Asynchronous copy
--------------------------------------------------------------------------------

``cp.async.{ca,cg}.shared.global`` uses ``(*group,*repeat,element)``.
``chunk_bytes=4/8/16`` fixes the last axis's byte count; .cg requires 16 bytes.
The corresponding source and destination chunks are contiguous and naturally
aligned. ``src_size`` or ``ignore_src`` uses the hardware's zero-fill operand;
``pred`` gates execution and has different semantics. Commit and wait are
caller instructions.

``cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes``
uses the same suffix with positive ``chunk_bytes`` divisible by 16. ``mbar``
and ``remote_cta_id`` are required. Source, destination, and barrier arguments
name local shared addresses; lowering maps the destination and barrier with
``mapa`` before issuing the requested bulk copy. Election, barrier initialization,
transaction accounting, and cluster synchronization belong to the caller.

TMA
--------------------------------------------------------------------------------

Select the rank, direction, tile/gather4/scatter4 mode, reduction operator,
multicast and cache qualifiers in the PTX opcode. The global operand is the
rectangular tensor view described by the TensorMap. Without interleave its
rank is exactly the instruction rank; shape, strides, starting region and
dtype define the map. Alternatively, pass an ordinary address expression for
a descriptor encoded by the caller. Its fields must agree with the pattern's
box, format and swizzle; the dispatcher does not inspect opaque descriptors.

Required keyword operands are ``box`` and ``coords``, both in outer-to-inner
logical axis order. The shared operand is ``(*group,*repeat,*box_image)``.
Every atom's unswizzled origin is 128-byte aligned, including unswizzled boxes.
Swizzled allocation bases are aligned to the full swizzle period; a subregion
uses its unswizzled origin so that the instruction applies its phase once.
``coord_steps`` has one coordinate vector per group/repeat axis. Lowering adds
these explicit steps to ``coords``, encodes the descriptor on the host, and
issues one instruction at each repeat. It does not fold dimensions, promote
dtypes, infer a box, prefetch a descriptor, select a source, or elect a thread.
Use ordinary control flow, or select explicitly encoded descriptor addresses,
to choose between sources. Prefetch is an explicit ``T.ptx.prefetch.tensormap``.

Descriptor operands are ``format`` (storage dtype by default, or ``tf32``),
``swizzle=0/1/2/3``, ``l2_promotion=0`` and ``oob_fill=0``. ``element_strides``
has one entry in 1..8 per descriptor dimension; without interleave the inner
entry is one. The non-interleaved shared image has extents
``ceil(box[i]/element_strides[i])`` and compact storage before swizzling.
Symbolic global dimensions and outer strides pass to runtime descriptor
encoding; statically invalid values are rejected when lowering. A load requires
``mbar``; a multicast also requires ``cta_mask``; a cache-hint form requires
``cache_policy``. All are Call arguments and participate in substitution.

Gather4/scatter4 use a 2D descriptor, ``box=(1,width)``, a shared atom of
``(4,width)``, and four explicit row coordinates in ``gather4``. Dynamic row
coordinates are allowed. No row-selection algorithm is hidden in lowering.

``format="float4_e2m1fn"`` selects CUDA's ``16U4_ALIGN16B`` representation.
Its inner global dimension is a multiple of 128, its inner box is 128, and
the global base and outer byte strides are 32-byte aligned. Each sixteen
nibbles occupy the low eight bytes of a sixteen-byte shared block. The other
eight bytes are undefined padding. The mbarrier counts **payload bytes**,
not padding. This representation supports loads with swizzle 0 or 3; gather4
uses the same packed shared image.

For ``interleave=1`` (16B) or ``2`` (32B), an opcode of rank 3..5 takes a
global view ``(N,C_slices,*spatial,Ci)``. ``Ci`` is the number of scalar
elements in one 16B/32B slice. The descriptor omits this last, implicit axis;
``box``, ``coords`` and ``coord_steps`` have the opcode rank. The W stride
is exactly one slice; Ci is contiguous. Other strides are explicit and can
include padding. Mode 2 requires swizzle 1 and 32-byte global alignment.

One interleaved tiled issue visits **one C slice**, selected by ``coords[1]``.
The C box and traversal-stride fields do not iterate additional slices. Its
shared image is ``(ceil(boxN/stepN),1,*ceil(boxSpatial/stepSpatial),Ci)``
before swizzling. To visit several channel slices, add a repeat axis and a
coordinate step in C. For example, rank-3 uint16 with interleave 2 uses
``Ci=16``; ``box=(1,1,8)`` copies a ``(1,1,8,16)`` shared atom. A shared
operand of shape ``(3,1,1,8,16)`` and ``coord_steps=((0,1,0),)`` issues three
copies for three successive channel slices. Gather4/scatter4 do not interleave.

TMEM transfers
--------------------------------------------------------------------------------

``tcgen05.ld/st.sync.aligned.<shape>.xN.b32`` use suffix ``(row,column)``
in b32 storage on both sides. Each atom is a warp instruction. The four
supported shapes and register images (lane ``l``, register ``r``) are:

.. list-table::
   :header-rows: 1

   * - Shape
     - Atom shape
     - Register coordinate (row, column)
   * - 32x32b
     - (32,N)
     - (l,r)
   * - 16x64b
     - (16,2N)
     - (l//4+8*(l%2), l//2%2+2*r)
   * - 16x128b
     - (16,4N)
     - (l//4+8*(r%2), l%4+4*(r//2))
   * - 16x256b
     - (16,8N)
     - (l//4+8*(r//2%2), r%2+2*(l%4)+8*(r//4))

TMEM coordinates are consecutive rows/columns in the executing warp's 32-lane
slab. Explicit group axes distribute atoms across warps. Register storage can
be strided.

``.pack::16b`` / ``.unpack::16b`` keep the same b32 register image. Their TMEM
operand instead has suffix ``(row,column,half)`` with ``half=2`` and a 16-bit
dtype. The two halves of register word ``(row,column)`` correspond to the low
16 bits of TMEM cells ``(row,2*column)`` and ``(row,2*column+1)``. In 16-bit
layout units this means TCol strides four for ``column`` and two for ``half``.
This is different from reinterpreting two adjacent 16-bit elements in one
dense b32 cell. The hardware performs the packing; lowering adds no conversion.

``.16x32bx2`` has register suffix ``(access,row,column)=(2,16,N)``. Lane ``l``
owns ``(l//16,l%16,r)`` for register ``r``. Its two TMEM access regions are
explicit operands, each with suffix ``(16,N)`` (or ``(16,N,2)`` for packing)::

    T.cuda.tile.apply(
        T.ptx["tcgen05.ld.sync.aligned.16x32bx2.x4.b32"],
        registers, memory[:, :16, :4], memory[:, 16:32, 8:12],
        imm_half_splitoff=(16 << 16) + 8, scope="warpgroup",
    )

For a store the operand order is ``memory_low, memory_high, registers``.
Both regions share the same backing TMEM tensor. The second region's encoded
base address must equal the first plus ``imm_half_splitoff``. There is no
implicit sixteen-row offset for the second access. The immediate can select
different rows, columns, or both, within the executing warp's TMEM partition.
Each repeat emits exactly one instruction, including both half-accesses.
Wait, fence, allocation and deallocation instructions remain at the call site.

``tcgen05.cp`` uses ``(row,b32_word)`` according to its explicit shape. Multicast
replicas must exactly match the written qualifier: warpx4 adds four replicas
32 lanes apart, 02_13 adds two 64 lanes apart, 01_23 adds two 32 lanes apart.
4x256b maps its four rows to TLanes 0/32/64/96; the other shapes follow their
named contiguous slabs (01_23 uses slabs 0 and 2).

Native decompression uses the same b32-word footprint and explicit qualifiers:
``.b8x16.b4x16_p64`` or ``.b8x16.b6x16_p32``. Each 16-byte source block holds
sixteen packed 4-bit or 6-bit values followed by eight or four bytes of padding,
respectively. The destination holds sixteen bytes: the 4-bit values gain two
low zero bits, while the 6-bit values gain two high zero bits. A 256-bit row
contains two such blocks. These are raw packed-word views, not sixteen
ordinary byte elements on the source side. Each repeat still emits one cp;
there is no software unpacking or implicit dtype conversion.

Shared matrix descriptors
--------------------------------------------------------------------------------

``tcgen05.cp`` takes ``ldo_bytes``, ``sdo_bytes`` and ``swizzle``.
MMA takes ``a_descriptor=(ldo_bytes,sdo_bytes,swizzle)`` and the analogous B
operand. These are explicit hardware descriptor fields, not tuning hints.
They are mechanically encoded, with no layout search.

For ``tcgen05.cp``, an 8-row group has row byte pitch ``16 << swizzle``.
Consecutive row groups are separated by ``sdo_bytes``. Without swizzling,
16-byte chunks within a row are separated by ``ldo_bytes``. With swizzling,
words are contiguous before swizzling and hardware ignores LDO; changing
that field does not make gapped words legal.

For MMA, ``row`` means M/N in K-major order and K in MN-major order. The
contiguous coordinate advances in units of the operand dtype. Its low bits
occupy a ``16 << swizzle`` byte row for modes 0..3. The offsets for each subsequent 8-row
group and each subsequent contiguous chunk are:

.. list-table:: MMA descriptor strides in bytes
   :header-rows: 1

   * - Mode
     - Next 8-row group
     - Next contiguous chunk
   * - K-major
     - SDO
     - LDO (unswizzled; swizzled instruction atoms fit within one row)
   * - MN-major, swizzled
     - SDO
     - LDO
   * - MN-major, unswizzled
     - LDO
     - SDO

MMA mode ``swizzle=4`` is the distinct 128B span / 32B atomicity format
for 32-bit MN-major operands. It has 128-byte rows and groups **four** K rows
per SDO step. Consecutive 128-byte M/N chunks use LDO. Its byte permutation
is ``offset ^ ((offset >> 2) & 96)`` with a 512-byte period, unlike mode 3's
``offset ^ ((offset >> 3) & 112)``. The mode is explicit in the descriptor;
it is never selected from a tensor layout. The same TF32 format in K-major
order uses modes 0..3.

This distinction follows PTX's canonical shared layouts; in particular,
unswizzled MN-major exchanges the descriptor fields' roles. The selected
swizzle permutes the resulting physical address. Patterns check this walk
against the supplied view, including its start and element offset.
Descriptor origins must be 16B aligned; a swizzled backing allocation must
align to its full period (256/512/1024 bytes). The descriptor encodes the
unswizzled start address; hardware applies the permutation to that start and
the subsequent accesses. In particular, cp may start in the middle of a
swizzle atom if its entire address walk matches the requested region.
Descriptor base-offset and absolute-LDO modes are not inferred.
Allocate a complete descriptor storage block even for a
partial logical atom; in particular, a 4x256b copy still needs an 8-row shared
backing block. The logical region may contain only the four copied rows.

Warp MMA
--------------------------------------------------------------------------------

``mma.sync.aligned.m16n8k{8,16}.row.col.f32.{f16,bf16}.{f16,bf16}.f32``
uses these canonical operands, after optional group and batch prefixes:

* D/C: ``(M_repeat,N_repeat,16,8)`` in f32.
* A: ``(M_repeat,K_repeat,16,K/2)`` in b32 words packing two K elements.
* B: ``(N_repeat,K_repeat,8,K/2)`` in the same representation.

Lane ``l`` owns A word ``(l//4+8*(r%2), l%4+4*(r//2))``, B word
``(l//4,l%4+4*r)``, and D/C element ``(l//4+8*(r//2),2*(l%4)+r%2)``.
Repeat order is batch, M, N, K. The first K step uses C (tensor or scalar
broadcast); later steps use D. Alpha/beta preparation and logical transpose
views are caller code.

tcgen05 MMA
--------------------------------------------------------------------------------

Select dense, .ws, or block_scale and CTA group in the opcode. Specify
``instruction_shape=(M,N,K)``; M is the CTA-group total. Per-CTA canonical
suffixes after group/batch axes are:

* D: ``(M_repeat,N_repeat,M/cta_group,N)``.
* A: ``(M_repeat,K_repeat,M/cta_group,K)``.
* B: ``(N_repeat,K_repeat,N/cta_group,K)``.
* Block scale adds SFA/SFB with the corresponding A/B repeat axes and atom
  ``(M/cta_group,scale_vec)`` / ``(N,scale_vec)``. SFB contains the
  full instruction N range in each CTA, including two-CTA operations.

A may be shared or TMEM; B is shared; D and scales are TMEM. D uses
16-bit or 32-bit accumulation storage as required by the selected format.
Each accumulator occupies one b32 TMEM cell; f16 occupies its low half
(``TCol`` stride two in f16 element units). A-in-TMEM remains packed. ``a_major`` and
``b_major`` explicitly select ``"k"`` (default) or ``"mn"``. Storage formats
come from the tensor dtype unless ``a_format`` / ``b_format`` is supplied
(for example, ``tf32`` over f32 storage). ``accum=False`` zeros the first K
step's accumulator; later K steps accumulate. Negation, saturation, output-lane
masks, .ws zero-column mask, and ``pred`` are explicit operands.

The MMA TMEM datapath follows ``tmem_mma_operand_layout`` in
``tvm.backend.cuda.layout`` for the chosen instruction shape and mode.
Weight-stationary M=128 uses Layout D, like non-.ws M=128.
Weight-stationary M=64 A-in-TMEM includes an explicit leading bank axis of
size two. Scale rows use lane ``row%32``, byte-column ``4*(row//32)+sf`` and
four 32-lane replicas. The scale ID is encoded from each scale tensor's own
byte-column origin. Preparation, CTA election, fences, commit and wait remain
caller-owned. Layouts never select .ws or a different M/N/K instruction shape.

Splitting an M64 .ws N128 operation into two N64 instructions changes the
bank boundary inside each instruction from 64 columns to 32. The caller must
account for this: ``test_tcgen05_mma_ws_split_n.py`` preserves the original
banked A and packed C, explicitly stages B in each instruction's column order,
and permutes the C view. Simply cutting B and C into consecutive N64 logical
slices does not preserve that operation.

Choosing instructions
--------------------------------------------------------------------------------

Choose each instruction and its canonical view at the call site:

.. list-table:: Instruction choices
   :header-rows: 1
   :widths: 25 75

   * - Operation
     - Instruction or composition
   * - ld/st, register copy
     - ``apply`` with a written ld/st width or mov opcode.
   * - ldmatrix/stmatrix
     - ``apply`` with matrix count and transpose qualifier in the opcode.
   * - Scalar/packed arithmetic and casts
     - ``apply`` with the exact arithmetic/cvt form and native register representation.
   * - exp, silu, scale/bias combinations
     - ``ewise`` with the caller's scalar TIRx function and explicit constants.
   * - Arithmetic on shared memory
     - Explicit shared load, register ``apply``/``ewise``, then shared store.
   * - cp.async, DSMEM, TMA
     - ``apply`` with explicit chunk/box/coordinates and runtime completion operands.
   * - tcgen05.ld/st/cp
     - ``apply`` with explicit shape, repeat count and multicast qualifiers.
   * - mma/tcgen05.mma
     - ``apply`` with explicit matrix mode, canonical M/N/K repeats and descriptors.

Keep barriers, thread election, packing, scale-factor preparation and pipeline
scheduling in caller code. Changing a view changes only logical coordinates;
when the desired physical arrangement requires a copy, write that copy
explicitly. Unsupported hardware modes remain available through scalar
``T.ptx`` calls and can acquire their own documented pattern separately.

Implementation and extension
--------------------------------------------------------------------------------

``python/tvm/backend/cuda/tile/patterns/`` has one module per fixed instruction
family. ``common.py`` evaluates canonical coordinates, validates group/repeat
shapes and physical storage, and emits repeat loops. ``descriptors.py`` only
encodes/checks explicit hardware fields. There are no imports of retired
CUDA dispatch implementations.

To add a pattern:

1. Define its canonical operand suffixes, ownership and physical address walk,
   all semantic qualifiers, and required caller preparation in this document.
2. Resolve the exact instruction through ``ptx.instruction_forms``. Reuse the
   existing PTX table for legality and native operand emission.
3. Use ``Context`` and ``Region`` to check the supplied views at those fixed
   coordinates. Reject mismatches; do not search for another layout or opcode.
4. Emit one selected instruction inside ``Context.finish``. Put algorithms,
   packing, synchronization and thread election at the call site.
5. Add positive/negative contracts, script/JSON roundtrips, exact PTX spelling
   and issue-count checks, and GPU results reconstructed independently from
   the documented hardware mapping.

The common IR is ``Call(opcode, operands, parameters)`` with only execution
scope in ``ScopeAttrs``. Runtime and static descriptor parameters both live in
ordinary arguments. ``ewise`` stores its typed scalar body as ``LambdaExpr``.
No process-local Python callable survives serialization.

Region extents are positive static integers. Starts may be symbolic when their
bounds and alignment can be proved. General physical-equivalence discovery is
outside the contract. Gapped allocation covers the layout's physical span,
including the address range introduced by a composed swizzle.
