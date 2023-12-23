import functools
import math
from queue import PriorityQueue
from typing import Iterable, Dict, List

import numpy as np
import tvm


from ..arch import Arch
from ..bestfit import BestFit
from ..config import Config, Stride, TileDict
from .common import coalesced_factor, coalesced_tensor_shape, factorize, get_all_factors
from ..node import PrimFuncNode
import logging

logger = logging.getLogger(__name__)


class DefaultPolicy:
    def __init__(self, func: tvm.tir.PrimFunc, arch: Arch, tags:Dict = {}) -> None:
        self.arch = arch
        self.prim_func_node = PrimFuncNode(func, tags)
        self.ordered_nodes = [self.prim_func_node]
        self.output_nodes = [self.prim_func_node]

    def emit_config(self, topk: int) -> List[Config]:
        base_tile = self.get_base_tile()
        if base_tile is None:
            return []

        rstep_map = self._assign_reduce_step(self.prim_func_node)
        smem_tile_condidates = self.DFS_smem_tile(base_tile, topk, rstep_map)
        results = []
        for td in smem_tile_condidates:
            if not self.check_tile_shape_isvalid(td):
                continue
            block_orders = self._assign_block_order(td)
            if block_orders is False:
                continue
            self._expand_reduce_axis(td)
            for codegen_dicts in self.assign_block_size(td):
                # handle cases where block is not ordinal (e.g. transpose)
                for _, block_order in block_orders.items():
                    codegen_dicts.block_order = block_order
                results.append(codegen_dicts)
                if len(results) >= topk:
                    break
            if len(results) >= topk:
                break
        return results

    def DFS_smem_tile(self, init_tile, topk, rstep_map) -> Iterable[TileDict]:
        _steps = [get_all_factors(n) for n in self.prim_func_node.get_space_dim()]
        steps = [step[step.index(t) :] for step, t in zip(_steps, init_tile)]
        for i in range(len(steps)):
            added = list(
                filter(
                    lambda s: s < steps[i][-1] and s > steps[i][0] and s not in steps[i],
                    [2, 4, 8, 16, 32],
                )
            )
            steps[i].extend(added)
            steps[i] = sorted(steps[i])
        visited_tiles = {}
        queue = PriorityQueue()

        def prio(td: TileDict):
            return (td.traffic + 1) * td.num_wave  # * (td.block_per_SM ** 0.5)

        def add_to_queue(tile):
            if tuple(tile) in visited_tiles:
                return
            td = self.compute_tile_dict(tile, rstep_map)
            visited_tiles[tuple(tile)] = td
            if td.valid:
                queue.put([prio(td), tile])

        add_to_queue(init_tile)
        while not (queue.empty() or len(visited_tiles) > 2000):
            _, tile = queue.get()
            dim_ids = [step.index(t) for step, t in zip(steps, tile)]
            for i in reversed(range(len(dim_ids))):
                if dim_ids[i] + 1 < len(steps[i]):
                    new_tile = tile.copy()
                    new_tile[i] = steps[i][dim_ids[i] + 1]
                    add_to_queue(new_tile)

        visited_tiles = filter(lambda td: td.valid, visited_tiles.values())
        sorted_tiles = sorted(visited_tiles, key=lambda td: prio(td))
        return sorted_tiles

    # get the minimum tile that could satisfy no redundancy computation
    def get_base_tile(self):
        shape = self.prim_func_node.get_space_dim()
        base_tile = [1 for _ in shape]

        return base_tile

    # handles multiple output cases
    def _get_output_tile_map(self, tile):
        tile_map = {}
        tile_map[self.prim_func_node] = [
            tile[i]
            * self.prim_func_node.get_space_dim()[i]
            // self.output_nodes[0].get_space_dim()[i]
            for i in range(len(tile))
        ]
        return tile_map

    def score_block_size(self, n):
        num_wrap = (n + self.arch.warp_size - 1) // self.arch.warp_size
        r1 = max(num_wrap / self.arch.sm_partition, self.arch.sm_partition / num_wrap)
        r2 = (num_wrap * self.arch.warp_size - n) / n
        return (r1, r2)

    def get_block_size(self, n):
        factors = get_all_factors(n)
        factors = list(filter(lambda x: x <= 1024, factors))
        factor_ordered = sorted(factors, key=self.score_block_size)
        return factor_ordered[0]

    def get_node_reduce_step_candidates(self, node: PrimFuncNode):
        # general idea : use factor first, since it does not require extra boundary check
        #                for large prime number, which is rare case, use power of 2.
        results = {}
        for k_iter in node.raxis:
            all_factors = get_all_factors(k_iter.dom)
            if len(all_factors) == 2 and k_iter.dom > 64:
                all_factors = [1]
                while all_factors[-1] * 2 < k_iter.dom:
                    all_factors.append(all_factors[-1] * 2)
            results[k_iter.var.name] = all_factors
        return results

    def _assign_reduce_step(self, node: PrimFuncNode):
        if node.reduction_block is None:
            return {}

        raxis = node.raxis
        tile = [1] * len(node.get_space_dim())
        all_steps = self.get_node_reduce_step_candidates(node)

        def sim(a, b):
            return (2 * a * b) / (a * a + b * b)

        def _score(rstep_id):
            rstep = {k: all_steps[k][rstep_id[k]] for k in rstep_id}
            score = 0
            shape = node.propogate_inputs(tile, rstep=rstep)
            for i, input_buffer in enumerate(node.input_buffers):
                read_transaction_elements = self.arch.transaction_size[1] // (
                    (node.get_buffer_dtype(input_buffer).bits + 7) // 8
                )
                score += sim(
                    coalesced_factor(shape[i], input_buffer.shape),
                    read_transaction_elements,
                )
            return score

        def _enlarge(rstep_id):
            candidates = []
            candidates.append((rstep_id, _score(rstep_id)))
            for ax in rstep_id:
                if rstep_id[ax] + 1 == len(all_steps[ax]):
                    continue
                r = rstep_id.copy()
                r[ax] += 1
                candidates.append((r, _score(r)))
            best = max(candidates, key=lambda x: x[1])
            return best

        # enlarge rstep to ensure read is coaleased
        cur_rstep_id = {ax.var.name: 0 for ax in raxis}
        cur_score = _score(cur_rstep_id)
        while True:
            if cur_score == 0:
                break
            new_rstep, new_score = _enlarge(cur_rstep_id)
            if new_score <= cur_score:
                break
            else:
                cur_rstep_id, cur_score = new_rstep, new_score
        rstep = {k: all_steps[k][cur_rstep_id[k]] for k in cur_rstep_id}
        return rstep

    def _expand_reduce_axis(self, td: TileDict):
        smem_limit = min(self.arch.max_smem_usage // td.block_per_SM, self.arch.smem_cap)
        rstep_map = td.rstep_map.copy()

        def _optimize(node, rstep):
            all_steps = self.get_node_reduce_step_candidates(node)
            for k in all_steps:
                all_steps[k] = list(filter(lambda x: x % rstep[k] == 0, all_steps[k]))

            def _score(rstep_id):
                rstep = {
                    k.var.name: all_steps[k.var.name][rstep_id[k.var.name]] for k in node.raxis
                }
                score = 0
                shape = node.propogate_inputs(td.get_tile(node), rstep=rstep)
                for i, input_buffer in enumerate(node.input_buffers):
                    score += coalesced_factor(shape[i], input_buffer.shape)
                return score

            def _enlarge(rstep_id):
                candidates = []
                for ax in rstep_id:
                    if rstep_id[ax] + 1 == len(all_steps[ax]):
                        continue
                    r = rstep_id.copy()
                    r[ax] += 1
                    candidates.append((r, _score(r)))
                if len(candidates) == 0:
                    return None
                return max(candidates, key=lambda x: x[1])[0]

            cur_rstep_id = {
                k.var.name: all_steps[k.var.name].index(rstep[k.var.name]) for k in node.raxis
            }
            new_rstep_map = rstep_map.copy()
            while True:
                new_rstep_id = _enlarge(cur_rstep_id)
                if new_rstep_id is None:
                    break
                new_rstep_map = {
                    k.var.name: all_steps[k.var.name][new_rstep_id[k.var.name]] for k in node.raxis
                }
                old_rstep_map = td.rstep_map
                td.rstep_map = new_rstep_map
                smem_usage, _ = self._compute_shared_memory_usage(td)
                td.rstep_map = old_rstep_map
                if smem_usage > smem_limit:
                    break
                else:
                    cur_rstep_id = new_rstep_id
            rstep = {
                k.var.name: all_steps[k.var.name][cur_rstep_id[k.var.name]] for k in node.raxis
            }
            return rstep

        for node in self.ordered_nodes:
            if len(node.raxis) > 0:
                rstep = _optimize(node, rstep_map)
                rstep_map = rstep
        td.rstep_map = rstep_map
        td.smem_cost, td.cached_tensors_map = self._compute_shared_memory_usage(td)

    def _compute_memory_traffic(self, output_tile):
        op_tile_map = self._get_output_tile_map(output_tile)
        traffic = 0
        return traffic, op_tile_map

    def infer_node_smem_usage(self, td: TileDict, node: PrimFuncNode):
        return node.footprint(td.get_tile(node), td.get_rstep(node), td.tensor_strides_map[node])

    def _compute_shared_memory_usage(self, td: TileDict):
        self._compute_stride_map(td)
        allocator = BestFit()
        block_map = {}
        cached_tensors_map = {}

        node_internal_bytes, cached_tensors_map[self.prim_func_node] = self.infer_node_smem_usage(
            td, self.prim_func_node
        )
        block = allocator.malloc(node_internal_bytes)
        allocator.free(block)
        assert len(block_map) == 0
        return allocator.limit, cached_tensors_map

    def compute_node_stride_map(self, node: PrimFuncNode, td: TileDict):
        output_strides = {
            int(i + len(node.input_buffers)): Stride() for i, _ in enumerate(node.output_buffers)
        }
        tensor_strides = {}
        return output_strides, tensor_strides

    def _compute_stride_map(self, td: TileDict):
        output_strides_map = {}
        tensor_strides_map = {}
        for node in self.ordered_nodes:
            output_strides_map[node], tensor_strides_map[node] = self.compute_node_stride_map(
                node, td
            )
        td.output_strides_map, td.tensor_strides_map = output_strides_map, tensor_strides_map

    def get_dtype_bits(self):
        return 16

    def compute_tile_dict(self, output_tile: List[int], rstep_map) -> TileDict:
        td = TileDict(output_tile)
        td.rstep_map = rstep_map
        td.traffic, td.tile_map = self._compute_memory_traffic(output_tile)
        td.smem_cost, td.cached_tensors_map = self._compute_shared_memory_usage(td)
        if td.smem_cost > self.arch.smem_cap:
            td.valid = False
            return td
        output_shape = self.output_nodes[0].get_space_dim()
        td.grid_size = int(np.prod([(y + x - 1) // x for x, y in zip(output_tile, output_shape)]))
        # estimated reg usage
        reg_usage = int(
            2
            * max(
                [
                    np.prod(td.get_tile(node)) * node.get_dtype().bits / 32
                    for node in self.ordered_nodes
                ]
            )
        )
        if reg_usage > self.arch.reg_cap:
            td.valid = False
            return td
        td.block_per_SM = min(
            self.arch.max_smem_usage // max(td.smem_cost, 1),
            self.arch.reg_cap // max(reg_usage, 1),
            self.arch.sm_partition,
        )
        td.num_wave = int(np.ceil(td.grid_size / int(td.block_per_SM * self.arch.compute_max_core)))
        return td

    def check_tile_shape_isvalid(self, td: TileDict):
        for node in self.ordered_nodes:
            if np.prod(td.get_tile(node)) == 0:
                return False
            node_grid_size = np.prod(
                [(y + x - 1) // x for x, y in zip(td.get_tile(node), node.get_space_dim())]
            )
            if node_grid_size != td.grid_size:
                return False
            if (
                hasattr(node, "reduce_op")
                and node.reduce_op is not None
                and len(node.reduce_op.axis) == len(td.output_tile)
            ):
                for i, tile_extent in enumerate(td.output_tile):
                    if node.reduce_op.axis[i].dom.extent % tile_extent:
                        return False

        return True

    def recommend_block_size(self, td: TileDict) -> List[int]:
        node_space_sizes = [int(np.prod(td.get_tile(node))) for node in self.ordered_nodes]
        max_block_size = functools.reduce(math.gcd, node_space_sizes)

        if max_block_size < self.arch.warp_size * self.arch.sm_partition and max_block_size == min(
            node_space_sizes
        ):
            node_reduce_sizes = [
                int(np.prod(list(td.get_rstep(node).values()))) for node in self.ordered_nodes
            ]
            total_sizes = [x * y for x, y in zip(node_space_sizes, node_reduce_sizes)]
            max_possible_size = functools.reduce(math.gcd, total_sizes)
            possible_block_sizes = list(
                filter(
                    lambda x: x % max_block_size == 0 and x <= 1024,
                    get_all_factors(max_possible_size),
                )
            )
            possible_block_sizes = list(
                filter(  # either be a factor of space or cover fully cover the space
                    lambda x: all([x % s == 0 or s % x == 0 for s in node_space_sizes]),
                    possible_block_sizes,
                )
            )
            factor_ordered = sorted(possible_block_sizes, key=self.score_block_size)
            return factor_ordered
        else:
            possible_block_sizes = get_all_factors(max_block_size)
            possible_block_sizes = list(filter(lambda x: x <= 1024, possible_block_sizes))
        factor_ordered = sorted(possible_block_sizes, key=self.score_block_size)
        return factor_ordered

    def assign_block_size(self, td: TileDict, topk=1):
        block_size_ordered = self.recommend_block_size(td)
        for block_size in block_size_ordered:
            result = {}
            failed = False
            result = self._assign_block_size(self.prim_func_node, td, block_size)
            if result is None:
                failed = True
                break
            if failed:
                continue
            else:
                yield result
                topk -= 1
                if topk == 0:
                    break

    def _assign_block_order(self, td: TileDict):
        block_idx = tvm.te.var("block_idx")
        analyzer = tvm.arith.Analyzer()
        analyzer.update(block_idx, tvm.arith.ConstIntBound(0, td.grid_size - 1))
        expr_map = {node: block_idx for node in self.output_nodes}
        result = {}
        for node in reversed(self.ordered_nodes):
            expr = expr_map[node]
            if not (expr.same_as(block_idx) or isinstance(expr, tvm.tir.expr.ConstExpr)):
                result[node] = expr
        return result

    def _assign_block_size(self, node: PrimFuncNode, td: TileDict, block_size: int):
        tile, rsteps = td.get_tile(node), td.get_rstep(node)
        factors = factorize(block_size)
        cur_threads = [1 for _ in tile]
        reduce_thread = {k: 1 for k in rsteps}
        ndim = len(tile)

        def _score(node, thread):  # small is better
            score = 0
            block_tile = [int(np.ceil(tile[i] / thread[i])) for i in range(ndim)]
            shape = node.propogate_inputs(block_tile)
            for i, buffer in enumerate(node.input_buffers):
                score += np.prod(shape[i]) / self.arch.bandwidth[1]
            for buffer in node.output_buffers:
                score += coalesced_tensor_shape(thread, buffer.shape, 8) / self.arch.bandwidth[0]
            return score

        for factor in reversed(factors):
            score_map = {}
            for i in range(ndim):
                if cur_threads[i] >= tile[i]:
                    continue
                if (tile[i] % (cur_threads[i] * factor)) != 0:
                    continue
                cur_threads[i] *= factor
                score_map[i] = (_score(node, cur_threads), i)
                cur_threads[i] //= factor
            if len(score_map) > 0:
                # assign to space axis
                dim_order = sorted(score_map.keys(), key=lambda x: score_map[x])
                cur_threads[dim_order[0]] *= factor
            else:
                # assign to reduce axis
                target_ax = None
                for ax, ax_len in reversed(list(rsteps.items())):
                    if ax_len % (reduce_thread[ax] * factor) == 0:
                        target_ax = ax
                        break
                assert target_ax
                reduce_thread[target_ax] *= factor

        codegen_dict = Config()
        codegen_dict.compute_capability = self.arch.compute_capability
        codegen_dict.block = tile
        codegen_dict.thread = cur_threads
        codegen_dict.rstep = [rsteps[ax.var.name] for ax in node.raxis]
        codegen_dict.reduce_thread = [reduce_thread[ax.var.name] for ax in node.raxis]
        codegen_dict.cached_tensors = td.cached_tensors_map[node]

        if node.get_dtype().bits == 16:  # set step=2 for fp16 case
            codegen_dict._step = [1 for _ in range(ndim)]
            for i in reversed(range(ndim)):
                if codegen_dict.block[i] // codegen_dict.thread[i] % 2 == 0:
                    codegen_dict._step[i] = 2
                    break
        elif node.get_dtype().bits == 8:  # set step=4 for 8bit case
            codegen_dict._step = [1 for _ in range(ndim)]
            for i in reversed(range(ndim)):
                if codegen_dict.block[i] // codegen_dict.thread[i] % 4 == 0:
                    codegen_dict._step[i] = 4
                    break
        # Plan vectorize
        codegen_dict.vectorize = self._plan_vectorize(node, td, block_size)
        return codegen_dict

    def _plan_vectorize(self, node: PrimFuncNode, td: TileDict, block_size: int):
        def is_cont(shape, vec):
            if len(shape) == 0:
                return vec == 1
            last = shape[-1]
            if last == 1:
                return is_cont(shape[0:-1], vec // last)
            else:
                return last % vec == 0

        def is_shape_aligned(shape, factor):
            return int(np.prod(shape)) % factor == 0

        def is_type_allowed(dtype, vec):
            return dtype.bits * vec <= 128

        vectorize_sizes = [16, 8, 4, 2]
        dtypes = node.get_reduce_inputs_dtype()
        shapes = node.propogate_reduction_inputs(td.get_tile(node), td.get_rstep(node))
        vectorize_result = {}
        for tensor, shape in shapes.items():
            for v in vectorize_sizes:
                if (
                    is_shape_aligned(shape, block_size * v)
                    and is_cont(shape, v)
                    and is_type_allowed(dtypes[tensor], v)
                ):
                    vectorize_result[tensor] = v
                    break
        return vectorize_result

    def plan_rasterization(self, td: TileDict):
        raise NotImplementedError()
