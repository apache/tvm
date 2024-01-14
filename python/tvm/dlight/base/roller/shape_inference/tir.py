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
from typing import Dict, List, Tuple
from tvm.tir.schedule.schedule import BlockRV
from tvm import arith, tir


class Statement:
    def __init__(self, block_analyzer, block: BlockRV):
        self.block_analyzer = block_analyzer
        self.block = block
        # assume one tir block only has one output buffer
        self.dep_name = block_analyzer.get_output_buffers(block)[0].name
        self.dependent_region = _extract_dependent_region(block_analyzer, block)

        self.reverse_bound_inference = {}

    def make_reverse(self, input_name: str, input_iter: List[tir.PrimExpr]):
        if len(self.block_analyzer.get_reduce_axis(self.block)) > 0:
            return None
        if len(self.dependent_region[input_name]) != 1:
            return None
        indices = self.dependent_region[input_name][0]
        iter_map_range = {
            _iter.var: _iter.dom for _iter in self.block_analyzer.get_spatial_axis(self.block)
        }
        iter_map_result = arith.detect_iter_map(
            indices,
            iter_map_range,
            check_level=arith.iter_affine_map.IterMapLevel.Surjective,
            simplify_trivial_iterators=False,
        )
        if len(iter_map_result.errors) > 0:
            return None
        results = arith.iter_affine_map.inverse_affine_iter_map(iter_map_result.indices, input_iter)
        output_indices = []
        for _iter in self.block_analyzer.get_spatial_axis(self.block):
            if _iter.var in results:
                output_indices.append(results[_iter.var])
            else:
                # not Bijective mapping case
                output_indices.append(tir.Var("undefined", dtype="int32") % int(_iter.dom.extent))
        return output_indices


def _merge_two_bounds(x: arith.ConstIntBound, y: arith.ConstIntBound):
    return arith.ConstIntBound(min(x.min_value, y.min_value), max(x.max_value, y.max_value))


class TensorDepNode(object):
    """
    For tensor dependency analysis.
    """

    def __init__(self, name):
        self.name = name
        self._next = []
        self._prev = []

    def add_next(self, node):
        self._next.append(node)
        self.deduplicate(self._next)

    def add_prev(self, node):
        self._prev.append(node)
        self.deduplicate(self._prev)

    def deduplicate(self, lst):
        seen = set()
        lst[:] = [n for n in lst if not (n in seen or seen.add(n))]

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class DependencyAnalysis(object):
    def __init__(self, deps):
        self.deps = deps
        self.name2dep = {
            dep.block_analyzer.get_output_buffers(dep.block)[0].name: dep for dep in self.deps
        }
        self.mapping = {}  # name -> TensorDepNode

    def get_or_create_node(self, name):
        if name not in self.mapping:
            self.mapping[name] = TensorDepNode(name)
        return self.mapping[name]

    def traverse_dependencies(self, compute):
        if isinstance(compute, Statement):
            node = self.get_or_create_node(
                compute.block_analyzer.get_output_buffers(compute.block)[0].name
            )
            # Loop through input tensors
            for input_buffer in compute.block_analyzer.get_input_buffers(compute.block):
                # Get the input node
                input_node = self.traverse_dependencies(input_buffer)
                input_node.add_next(node)
                node.add_prev(input_node)
        elif isinstance(compute, tir.Buffer):
            node = self.get_or_create_node(compute.name)
        return node

    def analyze(self):
        # Starting point for traversal
        for _, compute in self.name2dep.items():
            self.traverse_dependencies(compute)

    def print_dependencies(self):
        for name, node in self.mapping.items():
            print(f"{name} depends on {', '.join([prev.name for prev in node._prev])}")

    def find_path_from_source(self, start_name, target_name):
        """
        Finds the path (if it exists) from a starting node (source) to a target node.
        Returns the path as a list of nodes.
        """
        visited = set()
        path = []
        if self._find_path_recursive(self.mapping[start_name], target_name, visited, path):
            return path
        return []

    def _find_path_recursive(self, current_node, target_name, visited, path):
        """
        Recursive helper function for find_path_from_source.
        """
        if current_node.name == target_name:
            path.append(current_node)
            return True

        if current_node.name in visited:
            return False

        visited.add(current_node.name)
        path.append(current_node)

        for next_node in current_node._next:
            if self._find_path_recursive(next_node, target_name, visited, path):
                return True

        path.pop()
        return False


class InputShapeInference:
    def __init__(self, deps: List[Statement]):
        self.deps = deps
        self.target_mapping = {}
        self.buffer_mapping = {}
        self.reduce_axes = []
        for dep in self.deps:
            for ax in dep.block_analyzer.get_reduce_axis(dep.block):
                self.reduce_axes.append(ax)
        self.dep_analysis = DependencyAnalysis(self.deps)
        self.dep_analysis.analyze()

    def construct_dependency_target(self, targets: Tuple[str]):
        if targets in self.target_mapping:
            return self.target_mapping[targets]
        # should be buffer name instead of block name
        name2dep = {
            dep.block_analyzer.get_output_buffers(dep.block)[0].name: dep for dep in self.deps
        }
        mapping = {}
        input_vars = []
        for target in targets:
            vars = [
                iter.var
                for iter in name2dep[target].block_analyzer.get_spatial_axis(name2dep[target].block)
            ]
            input_vars.append(vars)
            mapping[target] = [vars]
        ana = arith.Analyzer()

        for dep in self.deps:
            for name in dep.dependent_region:
                if name not in mapping:
                    continue
                dep_name = dep.dep_name
                indices = mapping[name][0]
                output_indices = dep.make_reverse(name, indices)
                if dep_name in targets:
                    continue
                if dep_name not in mapping:
                    mapping[dep_name] = [output_indices]
                elif not region_exist_in_list(
                    output_indices, mapping[dep_name]
                ):
                    mapping[dep_name].append(output_indices)

        for dep in reversed(self.deps):
            indices_list = mapping[dep.dep_name]
            ax_vars = [iter.var for iter in dep.block_analyzer.get_spatial_axis(dep.block)]
            for input_name, regions in dep.dependent_region.items():
                if input_name in targets:
                    continue
                if input_name not in mapping:
                    mapping[input_name] = []
                for indices in indices_list:
                    for region in regions:
                        vmap = {
                            k: (tir.Cast(k.dtype, v) if v.dtype != k.dtype else v)
                            for k, v in zip(ax_vars, indices)
                        }
                        region = [
                            ana.simplify(tir.stmt_functor.substitute(ax, vmap)) for ax in region
                        ]
                        if not region_exist_in_list(region, mapping[input_name]):
                            mapping[input_name].append(region)
        buffers = []
        for dep in self.deps:
            for buffer in dep.block_analyzer.get_buffers(dep.block):
                buffers.append(buffer)

        for buffer in buffers:
            self.buffer_mapping[buffer.name] = buffer

        self.target_mapping[targets] = input_vars, mapping
        return input_vars, mapping

    def infer(
        self, shape: Dict[str, List[arith.ConstIntBound]], rstep: Dict[str, int] = {}, targets=None
    ):
        compute_targets = tuple(shape.keys())
        input_vars, mapping = self.construct_dependency_target(compute_targets)
        ana = arith.Analyzer()
        results = {}
        intermediate_bind = {}
        for vars, bounds in zip(input_vars, shape.values()):
            for var, bound in zip(vars, bounds):
                ana.update(var, bound, True)
        for ax in self.reduce_axes:
            # assume the dom.min is always 0, maybe we can extend the IterInfo to include the min value.
            if ax.var.name in rstep:
                bound = arith.ConstIntBound(
                    int(ax.dom.min), int(ax.dom.min + min(ax.dom.extent, rstep[ax.var.name]) - 1)
                )
            else:
                bound = arith.ConstIntBound(int(ax.dom.min), int(ax.dom.min + ax.dom.extent - 1))
            ana.update(ax.var, bound, True)

        for name, regions in mapping.items():
            if targets is not None and name not in targets:
                continue
            if compute_targets[0:1] == compute_targets:
                (compute_target,) = compute_targets
                path = self.dep_analysis.find_path_from_source(name, compute_target)
                if len(path) > 2:
                    intermediate_nodes = path[1:-1]
                    for node in intermediate_nodes:
                        iters = mapping[node.name]
                        if len(iters) != len(regions) or len(iters) != 1:
                            continue
                        if len(*iters) != len(*regions):
                            break
                        regions = iters
                        intermediate_bind[name] = compute_target

                for region in regions:
                    bound = [ana.const_int_bound(indice) for indice in region]
                    if name in results:  # simply merge two bounds
                        bound = [_merge_two_bounds(x, y) for x, y in zip(results[name], bound)]
                    results[name] = bound
            else:
                for region in regions:
                    bound = [ana.const_int_bound(indice) for indice in region]
                    if name in results:  # simply merge two bounds
                        bound = [_merge_two_bounds(x, y) for x, y in zip(results[name], bound)]
                    results[name] = bound

        for name, bounds in results.items():
            results[name] = [c.max_value - c.min_value + 1 for c in bounds]
        return results, intermediate_bind

    def get_input_exprs(self, output_exprs):
        input_vars, mapping = self.construct_dependency_target(tuple(output_exprs.keys()))
        ana = arith.Analyzer()
        for ax in self.reduce_axes:
            ana.bind(ax.var, 0)
        vmap = {}
        for vars, exprs in zip(input_vars, output_exprs.values()):
            for var, expr in zip(vars, exprs):
                if expr.dtype != var.dtype:
                    expr = tir.Cast(var.dtype, expr)
                vmap[var] = expr
        result = {}

        for name, regions in mapping.items():
            region = regions[0]
            result[name] = [
                ana.simplify(tir.stmt_functor.substitute(index, vmap)) for index in region
            ]
        return result


def region_exist_in_list(a, list) -> bool:
    def expr_is_same(a, b) -> bool:
        if isinstance(a, tir.IntImm) and isinstance(b, tir.IntImm):
            return a.value == b.value
        return a.same_as(b)

    def region_is_same(a, b) -> bool:
        for indice_a, indice_b in zip(a, b):
            if not expr_is_same(indice_a, indice_b):
                return False
        return True

    return any([region_is_same(a, x) for x in list])


def walk_indice(expr):
    if isinstance(expr, tir.expr.BinaryOpExpr):
        a = walk_indice(expr.a)
        b = walk_indice(expr.b)
        if a is not None and b is not None:
            return expr
        else:
            return None
    elif isinstance(expr, tir.expr.ConstExpr):
        return expr
    elif isinstance(expr, tir.Var):
        return expr
    elif isinstance(expr, tir.ProducerLoad):
        return None
    elif isinstance(expr, tir.Cast):
        a = walk_indice(expr.value)
        if a is not None:
            return expr
        return None
    elif isinstance(expr, tir.Call):
        return None
    else:
        raise Exception("Unhandled node type in walk_indice(): %s" % expr)


def _extract_dependent_region(block_analyzer, block: BlockRV) -> Dict[str, List[tir.PrimExpr]]:
    input_buffers = block_analyzer.get_input_buffers(block)
    dependent_region = {buffer.name: [] for buffer in input_buffers}

    def fvisit(x):
        if not isinstance(x, tir.BufferLoad):
            return
        if x.buffer.name not in dependent_region:
            return
        index = []
        for indice, shape_limit in zip(x.indices, x.buffer.shape):
            expr = walk_indice(indice)
            if expr is None:
                expr = tir.Var("undefined") % shape_limit
            index.append(expr)
        if not region_exist_in_list(index, dependent_region[x.buffer.name]):
            dependent_region[x.buffer.name].append(index)

    stmt = block_analyzer.sch.get_sref(block).stmt
    tir.stmt_functor.post_order_visit(stmt, fvisit=fvisit)
    return dependent_region


def get_analyzer_by_tir(block_analyzer, args) -> InputShapeInference:
    deps = [Statement(block_analyzer, block) for block in args]

    return InputShapeInference(deps)
