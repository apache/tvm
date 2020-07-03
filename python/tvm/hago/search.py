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
#pylint: disable=unused-argument
"""Automatic quantization toolkit."""
from __future__ import absolute_import

from .base import * 
from . import _quantize
from . import quantize as qtz
from .. import relay
from .threshold import threshold_estimate
from .hardware import *
from .record import *
from .topology import Topology, analyze_topology
from .analysis import compare

import tvm
import random
import math
import itertools
import numpy as np
import pickle
import scipy
from collections import namedtuple

#TODO(ziheng): unify topology and constraints


def generate_choices(graph, hardware, topology):
    def get_in_bits(hardware, node):
        in_bits = [None] * len(node.args)
        int_cstrs = integer_constraints(hardware[node.op])
        for cstr in int_cstrs:
            cur_in_bits = [dtype.bits for dtype in cstr.idtypes]
            in_bits = [max_with_none(v1, v2) for v1, v2 
                       in zip(in_bits, cur_in_bits)]
        return in_bits

    def get_out_bit(hardware, node):
        if not isinstance(node, relay.Call):
            return None
        int_cstrs = integer_constraints(hardware[node.op])
        if not int_cstrs:
            return None
        out_bit = max([cstr.odtype(0).bits for cstr in int_cstrs])
        return out_bit

    bits = []
    node2idx = build_node_index(graph)
    edge2idx = build_edge_index(graph)
    def fvisit(node):
        if isinstance(node, relay.Call):
            if not topology.node_conds[node2idx[node]]:
                return 

            cur_in_bits = get_in_bits(hardware, node)
            for idx, src in enumerate(node.args):
                eidx = edge2idx[(src, node)]
                if topology.edge_conds[eidx]:
                    src_out_bit = get_out_bit(hardware, src)
                    bit = min_with_none(cur_in_bits[idx], src_out_bit)
                    bits.append(bit)
    relay.analysis.post_order_visit(graph, fvisit)

    print('bit limit')
    edge2bit = build_edge_dict(graph, bits, topology.edge_conds)
    print_edge_dict(graph, edge2bit)

    choices = [list(reversed(range(4, bit + 1))) for bit in bits]
    # print('bit choices')
    # edge2choices = complete_dict(choices, topology.edge2cond)
    # print_edge_dict(graph, edge2choices)
    return choices


# TODO(ziheng) consider general structural search
class SearchSpace(object):
    def __init__(self):
        pass

    def random_sample(self):
        pass

    def next(self):
        pass

    def verify(self, combination):
        pass

    def neighbour(self, point, distance, portion):
        pass



# Primitive FVerify:
# def verify(op, bits):
# """bits = [in_bits] + [out_bits]"""

# Composed FVerify:
# bits


def create_search_space(graph, topology, choices):
    def group_cstrs(cstrs):
        """ quick union find algorithm """
        edge2idx = build_edge_index(graph)
        eidx2father = {} # edge index to parent edge index
        # initial
        print('initial')
        for cstr in cstrs:
            for edge in cstr.edges:
                eidx = edge2idx[edge]
                eidx2father[eidx] = eidx

        def find_root(eidx):
            father_idx = eidx2father[eidx]
            if father_idx == eidx:
                return eidx
            return find_root(father_idx)

        print('union')
        # union edges 
        for cstr in cstrs:
            roots = [find_root(edge2idx[edge]) for edge in cstr.edges]
            root = min(roots)
            for edge in cstr.edges:
                eidx2father[find_root(edge2idx[edge])] = root

        print('group')
        # group constraints
        root2gidx = {}
        groups = [] 
        for cstr in cstrs:
            root = find_root(edge2idx[cstr.edges[0]])
            print('root: {0}'.format(root))
            if root not in root2gidx:
                root2gidx[root] = len(groups)
                groups.append([])
            gidx = root2gidx[root]
            print('gidx: {}'.format(gidx))
            groups[gidx].append(cstr)
        return groups

    def merge_cstrs(cstrs):
        print('begin group constraints')
        groups = group_cstrs(cstrs)
        node2idx = build_node_index(graph)
        print('finished group constraints')
        for gidx, group in enumerate(groups):
            print('group {}:'.format(gidx))
            edges = set(edge for cstr in group for edge in cstr.edges)
            for edge in edges:
                print(edge_str(edge, node2idx))

        raise ValueError
        new_cstrs = []
        for group in groups:
            if len(group) == 1:
                new_cstrs.append(cstr)
            else:
                composed_cstr = group[0]
                # TODO(ziheng): change to two-way merge
                for cstr in group[1:]:
                    edges = (composed_cstr.edges, cstr.edges)
                    fverify = lambda atuple: composed_cstr.fverify(atuple[0]) and cstr.fverify(atuple[1]) 
                new_cstrs.append(Constraint(edges, fverify))
        return new_cstrs

    Constraint = namedtuple('Constraint', ['edges', 'fverify'])
    edge2choices = build_edge_dict(graph, choices, topology.edge_conds)
    node2edges = build_node2edges(graph)
    constraints = []
    # [([edge0, edge1], [edge2]) -> fverify0, 
    #  ([edge2, edge3], [edge4]) -> fverify1,
    #  ...
    # ]
    def fvisit(node):
        if isinstance(node, relay.Call):
            fverify = node.op.get_attr('FHagoVerify')
            if fverify:
                in_edges = [(src, node) for src in node.args]
                out_edges = node2edges[node]
                constraints.append(Constraint(in_edges + out_edges, fverify))
    relay.analysis.post_order_visit(graph, fvisit)
    grouped_cstrs = merge_cstrs(constraints)
    # [binary_tree_of_edge -> fverify
    #   where fveify is:
    #     lambda tuple: fverify0(tree(0)) and fverify(tree(1)))
    #  ...
    # ]

    # realize to space
    def flatten(tree_edges):
        """edges can be a tree""" 
        if isinstance(tree_edges, list):
            # base case
            return set(tree_edges)
        lset = flatten(tree_edges[0])
        rset = flatten(tree_edges[1])
        return lset.union(rset)

    def build_tree(tree_edges, edge2choice):
        if isinstance(item, list):
            # base case
            return list(map(lambda x: edge2choice[x], item))
        ltree = build_tree(tree_edges[0], edge2choice)
        rtree = build_tree(tree_edges[1], edge2choice)
        return (ltree, rtree)

    def combine(edge_list, edge2choices):
        choices_list = list(map(lambda x: edge2choices[edge], edge_list))
        return itertools.product(*choices_list)

    space = dict() 
    combined_edges = []
    for item in grouped_cstrs:
        list_edges = list(flatten(item.edges))
        combined_edges.append(list_edges)
        # verify feasible combinations
        combinations = []
        for comb in combine(list_edges, edge2choices):
            edge2choice = []
            for edge, choice in zip(list_edges, comb):
                edge2choice[edge] = choice
            tree_args = build_tree(item.edges, edge2choice)
            if item.fverify(tree_args):
                combinations.append(comb)
        space[list_edges].append(combinations)

    for edge, choices in edge2choices.items():
        if edge not in combined_edges:
            space[edge] = choices

    # space = [
    #     edge0: [4, 5, ... 8],
    #     (edge1, edge2, edge5): [(4, 4, 9), (4, 4, 10), ...],
    #     edge3: [4, 5, ... 8],
    #     ]

    # random_sample
    # choice = [
    #   edge0: 4
    #   edge1: 5
    #   edge2: 5
    #   edge5: 11
    #   edge3: 8
    #   ...
    #   ]
    return 0


def grid_search(f, domains, args, max_iter=1000):
    num_iter = 0
    best_guess = None
    best_cost = 0

    for guess in itertools.product(*domains):
        if num_iter >= max_iter:
            break
        cost = f(guess, *args)
        if cost >= best_cost:
            best_cost = cost
            best_guess = guess
        num_iter += 1
        print('niter: {}, acc: {}, best acc: {}'.format(num_iter, cost, best_cost))
        num_iter += 1
    return best_guess, best_cost


def random_guess(domains):
    while True:
        guess = []
        for domain in domains:
            guess.append(random.choice(domain))
        yield guess


def random_search(fcost, domains, args, max_iter=1000):
    num_iter = 0
    best_guess = None
    best_cost = 0

    for guess in random_guess(domains):
        print('iteration: {0}'.format(num_iter))
        if num_iter >= max_iter:
            break
        cost = fcost(guess, *args)
        if cost > best_cost:
            best_cost = cost
            best_guess = guess
        print('niter: {}, acc: {}, best acc: {}'.format(num_iter, cost, best_cost))
        num_iter += 1
    return best_guess, best_cost


def simulated_annealing(fcost, domains, args, T=0.5, Tmin=0.0005, cool=0.99, portion=0.10, step=2):
    def neighbour(origin, portion):
        num_changed = int(portion * len(origin))
        dims = random.sample(range(0, len(origin)), num_changed)
        new = origin.copy()
        for dim in dims:
            disturbance = random.choice(range(-step, step + 1))
            print('choose dimension {0}'.format(dim))
            print('change from {} to {}'.format(new[dim], new[dim]+disturbance))
            new[dim] += disturbance
            if new[dim] < min(domains[dim]):
                new[dim] = min(domains[dim])
            if new[dim] > max(domains[dim]):
                new[dim] = max(domains[dim])
        return new

    previous_guess, previous_cost = None, 0
    best_guess, best_cost = None, 0
    num_iter = 0
    # init with random guess
    guess = next(random_guess(domains))
    while T > Tmin:
        cost = fcost(guess, *args)
        if cost >= best_cost:
            # stored as best guess 
            best_guess = guess
            best_cost = cost
        if cost >= previous_cost or random.random() < math.exp(- (previous_cost - cost) / T):
            print('accept guess')
            # accept the guess
            previous_guess = guess
            previous_cost = cost
        T = T * cool
        print('niter: {}, acc: {}, best acc: {}'.format(num_iter, cost, best_cost))
        num_iter += 1
        # make new guess
        guess = neighbour(previous_guess, portion)
    return best_guess, best_cost


def greedy_squash(fcost, domains, args, tolerance=0.0, max_iter=3000):
    cfg = qtz.current_qconfig()
    best_guess, best_cost = None, float("inf")
    num_iter = 0
    # init with maximum bit setting
    guess = [choices[0] for choices in domains]
    stop_guess = [choices[-1] for choices in domains]
    dim_idx = 0
    last_update_idx = 0
    while num_iter < max_iter: 
        cost = fcost(guess, *args)
        if cost <= best_cost:
            # stored as best guess 
            best_guess = guess
            best_cost = cost

        if (cost - best_cost) <= tolerance:
            previous_guess = guess
            previous_cost = cost
            last_update_idx = dim_idx
        else:
            # move to next dimension
            dim_idx += 1

        if dim_idx - last_update_idx > len(domains):
            # early stopping
            break

        # make new guess
        guess = previous_guess.copy()
        while guess != stop_guess:
            dim = dim_idx % len(domains)
            if guess[dim] == min(domains[dim]):
                dim_idx += 1
            else:
                break
        guess[dim] -= 1
        print('niter: {}, acc: {}, best acc: {}'.format(num_iter, cost, best_cost))
        num_iter += 1
    return best_guess, best_cost


def search_bits_strategy(eval_func, bit_choices, graph, hardware, topology, dataset):
    cfg = qtz.current_qconfig()

    args = (graph, hardware, topology, dataset)
    if cfg.search_strategy == 'random_search':
        best_bits, best_acc = random_search(eval_func, bit_choices, args)
    elif cfg.search_strategy == 'default_setting':
        best_bits = [choices[0] for choices in bit_choices]
        # sim acc: 71.1, qtz acc: 71.1, imagenet: 68.7
        # best_bits = [6, 8, 24, 21, 24, 24, 8, 8, 21, 18, 21, 8, 7, 27, 23, 30, 32, 26, 8, 8, 22, 20, 22, 8, 8, 22, 24, 32, 32, 32, 8, 8, 32, 32, 8, 8, 32, 32, 32, 8, 8, 32, 32, 32, 32, 32, 8, 8, 32, 32, 32, 8, 8, 32, 32, 32, 32, 32, 8, 8, 32, 32, 8, 8, 32, 32, 32, 8, 8, 32, 32, 32, 32, 32, 8, 8, 32, 32, 32, 8, 8, 32, 32, 32, 32, 32, 8, 8, 32, 32, 8, 8, 32, 32, 32, 8, 8, 32, 32, 32, 32, 32, 8, 8, 32, 32, 32, 8, 8, 32, 32, 32, 32, 32]
        # sim acc: 71.9  qtz acc: 71.9, imagenet: 68.7
        # best_bits = [6, 8, 24, 21, 24, 24, 8, 8, 21, 18, 21, 8, 7, 27, 23, 30, 32, 26, 8, 8, 22, 20, 22, 8, 8, 22, 19, 22, 21, 22, 8, 7, 21, 19, 8, 8, 23, 21, 23, 8, 8, 22, 20, 31, 22, 22, 8, 8, 21, 19, 20, 8, 8, 24, 21, 24, 23, 24, 8, 8, 17, 16, 8, 8, 22, 20, 22, 8, 8, 23, 20, 29, 23, 23, 8, 8, 19, 16, 18, 8, 8, 18, 16, 18, 16, 18, 8, 8, 13, 11, 8, 8, 30, 32, 32, 8, 8, 32, 32, 32, 32, 32, 8, 8, 32, 32, 32, 8, 8, 32, 32, 32, 32, 32]
        best_acc = eval_func(best_bits, *args)
        return best_bits, best_acc
    elif cfg.search_strategy == 'grid_search':
        best_bits, best_acc = grid_search(eval_func, bit_choices, args)
    elif cfg.search_strategy == 'simulated_annealing':
        best_bits, best_acc = simulated_annealing(eval_func, bit_choices, args)
    elif cfg.search_strategy == 'greedy_squash':
        best_bits, best_acc = greedy_squash(eval_func, bit_choices, args)
    else:
        raise ValueError('unknown search strategy: {}'.format(cfg.search_strategy))

    return best_bits, best_acc


def softmax_with_temperature(x, temp=1.0, axis=1):
    e_x = np.exp((x - np.amax(x, axis=axis, keepdims=True)) / temp)
    return e_x / e_x.sum(axis=axis, keepdims=True)


def calculate_kl(out_x, out_y):
    out_x = softmax_with_temperature(out_x, temp=8.0, axis=1)
    out_y = softmax_with_temperature(out_y, temp=8.0, axis=1)
    num_samples = out_x.shape[0]
    kl = 0.
    for i in range(num_samples):
        entropy = scipy.stats.entropy(out_x[i], out_y[i])
        kl += entropy
    kl = kl / num_samples
    return kl



def search_quantize_strategy(mod, hardware, dataset=None):
    graph = mod['main']
    fout = open(current_qconfig().log_file, 'w+', buffering=1)
    origin_out, origin_acc = eval_acc(graph, dataset)
    print('original acc: {}'.format(origin_acc))
    topology = analyze_topology(graph, hardware)
    choices = generate_choices(graph, hardware, topology)
    # search_space = create_search_space(graph, topology, choices)
    model_hash = tvm.ir.structural_hash(graph)


    # search for bits settings with learning method
    def eval_func(bits, graph, hardware, topology, dataset):
        edge2bit = build_edge_dict(graph, bits, topology.edge_conds)
        print('bits')
        print_edge_dict(graph, edge2bit)
        # coarse-grained threshold estimate
        thresholds = threshold_estimate(graph, topology, bits, dataset)

        strategy = Strategy(model_hash, topology, bits, thresholds)
        quantizer = qtz.create_quantizer(graph, hardware, strategy)
        simulated_graph = quantizer.simulate()
        # print('simulated_graph')
        # print(simulated_graph)
        simulated_out, simulated_acc = eval_acc(simulated_graph, dataset)
        # [optional] calibrate threshold estimation
        quantized_graph = quantizer.quantize()
        quantized_out, quantized_acc = eval_acc(quantized_graph, dataset)

        kl_divergence = calculate_kl(origin_out, quantized_out)
        # logging
        print('simulated_acc: {}, quantized_acc: {}, kl_divergence: {}\n\n'.format(simulated_acc, quantized_acc, kl_divergence))
        result = MeasureResult(sim_acc=simulated_acc, quant_acc=quantized_acc, kl_divergence=kl_divergence)
        measure = Measure(strategy, result)
        fout.write(serialize(measure))
        fout.write('\n')
        return kl_divergence

    best_bits, best_acc = search_bits_strategy(eval_func, choices, graph, hardware, topology, dataset)
    print('finished search')
    print('best_acc: {0}'.format(best_acc))
    best_thresholds = threshold_estimate(graph, topology, best_bits, dataset)
    best_strategy = Strategy(model_hash, topology, best_bits, best_thresholds)
    fout.close()
    return best_strategy, best_acc


def group_same_graph_guesses(graph, hardware, topology, decided, choices, default):
    """group guesses which can share the same graph"""
    constraints = []
    for choice in choices:
        bits = decided + [choice] + default
        cstrs = qtz.select_constraint(graph, hardware, topology, bits)
        constraints.append((cstrs, bits))

    def group_by_key(pairs):
        m = defaultdict(list)
        for p in pairs:
            m[str(p[0])].append(p)
        ret = []
        for _, arr in m.items():
            key = arr[0][0]
            vals = [p[1] for p in arr]
            ret.append((key, vals))
        return ret
    constraints = group_by_key(constraints)

    groups = []
    for cstrs, grouped_guesses in constraints:
        simulator = qtz.Simulator(graph, topology, cstrs)
        print(simulator.simulated_graph)
        groups.append((simulator, grouped_guesses))
    return groups


def batched_search_quantize_strategy(mod, hardware, dataset=None):
    # (data_0, scale_0, clip_min_0)
    # (data_1, scale_0, clip_min_0)
    # (data_2, scale_0, clip_min_0)
    # accuracy

    # (data_0, scale_1, clip_min_1)
    # (data_1, scale_1, clip_min_1)
    # (data_2, scale_1, clip_min_1)
    # accuracy
    # ...
    # (data_0, scale_0, clip_min_0)
    # (data_1, scale_0, clip_min_0)
    # (data_2, scale_0, clip_min_0)
    # accuracy

    # generate best scale and clip_min
    # jump to the next domain
    graph = mod['main']
    fout = open(current_qconfig().log_file, 'w+', buffering=1)
    origin_out, origin_acc = eval_acc(graph, dataset)
    print('original acc: {}'.format(origin_acc))
    topology = analyze_topology(graph, hardware)
    choices = generate_choices(graph, hardware, topology)
    # search_space = create_search_space(graph, topology, choices)
    model_hash = tvm.ir.structural_hash(graph)

    print(choices)
    dim_idx = 0
    decided = []
    default = [bits[0] for bits in choices]
    while dim_idx < len(choices):
        groups = group_same_graph_guesses(graph, hardware, topology,
                                          decided, choices[dim_idx], default[dim_idx+1:])

        results = []
        for simulator, grouped_guesses in groups:
            constraints = simulator.constraints
            for bits in grouped_guesses:
                thresholds = threshold_estimate(graph, topology, bits, dataset)
                params = qtz.calculate_params(graph, topology, constraints, bits, thresholds)
                _, simulated_acc = simulator.eval(dataset, params)
                strategy = Strategy(model_hash, topology, bits, thresholds)
                results.append((strategy, simulated_acc))
                measure = Measure(strategy, MeasureResult(sim_acc=simulated_acc))
                fout.write(serialize(measure))
                fout.write('\n')
        best_strategy, best_acc = max(results, key=lambda item: (item[1], -sum(item[0].bits)))
        best_bit = strategy.bits[dim_idx]
        print('choose {} for the {}th bit, accuracy is: {}'.format(best_bit, dim_idx, best_acc))
        decided.append(best_bit)
        dim_idx += 1
    fout.close()
    return best_strategy, best_acc
