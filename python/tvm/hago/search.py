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
from .threshold import threshold_estimate, threshold_rectify

import tvm
import random
import math
import itertools
import time
import logging
import sys
import numpy as np
import multiprocessing as mp
try:
    import scipy
except ImportError:
    scipy = None

# TODOs:
# - experiments
# - add overflow check

# Hardware Specific
# simulation mostly for accuracy
# track scale for every tensor

# During Search Phase:
# search num_bit for every connection
#   - estimate threshold for every connection by calibration
#   - calculate constant parameters of simulated quantize
#     1. calculate scale, clip_min, clip_max of simulated quantize with (num_bit, threshold)
#     2. inferred scale of every connection
#     2. calculate overflow_min, overflow_max with inferred scale of every connection
#   - build simulated graph with those constant parameters
#   - acc = evaluate it on validation set
#   - TODO: refine threshold after this procedure
# choose num_bit setting

# During Realize Phase:
# infer scale of every connection


# sq(data, scale, clip_min, clip_max, upper_bound, lower_bound, signed=True, rounding='round')
# scale infer: map[op] -> scale

# simulated_qunatize(conv_out, 8, threshold, upper_bound=16):
# - 1. we want requantize it into 8bit
# - 2. we make sure it can fit into 16 bit before quantize

# map: call(sq_op, data, scale, clip_min, clip_max) -> (num_bit, threshold)

# Two representation for tensors:
# - REAL number reprentation
# - INTEGER number reprentation
# can be transformed:
#   scale = real_threshold / integer_range
#   INTEGER = REAL / scale
#   REAL = INTEGER * scale
# 
# Behavior of SimulatedQuantize:
# def simulated_quantize(data, scale, clip_min, clip_max, overflow_min_real, overflow_max_real):
#     # simulated overflow error
#     # because scale here is output scale, you cannot it to recover the quant of last op's output
#     data = overflow_truncate(data, overflow_min_real, overflow_max_real)
# 
#     # transform from real to integer(simulated)
#     quant = data / scale
#
#     # simulated rounding error
#     quant = round(quant)
#     # simulated clipping error
#     quant = clip(quant, clip_min, clip_max)
#
#     # transform from integer to real
#     data = quant * scale
#     return data


def generate_bit_choices(graph, hw_desc):
    _DEFAULT_BIT_LIMIT = 32
    def get_inputs_bit_limit(node):
        inputs_bit_limit = [None] * len(node.args)
        if isinstance(node, relay.Call) and node.op.name in hw_desc.ops:
            constraints = hw_desc[node.op.name]
            # init with the first constraint
            inputs_bit_limit = [dtype.bits for dtype in constraints[0].idtypes]
            for cstr in constraints:
                inputs_bit = [dtype.bits for dtype in cstr.idtypes]
                inputs_bit_limit = [max(v1, v2) for v1, v2
                                    in zip(inputs_bit, inputs_bit_limit)]
        return inputs_bit_limit

    def get_output_bit_limit(node):
        output_bit_limit = None
        if isinstance(node, relay.Call) and node.op.name in hw_desc.ops:
            constraints = hw_desc[node.op.name]
            output_bit_limit = max([cstr.odtype(0).bits for cstr in constraints])
        return output_bit_limit

    def smaller_limit(olimit, ilimit):
        # handle None
        if olimit is None:
            return ilimit
        if ilimit is None:
            return olimit
        return min(olimit, ilimit)

    edge2idx, num_edges = build_edge_index(graph)

    # analysis maximum num of bit on every tensor/edge
    bit_limits = [None] * num_edges
    def fvisit_bit_limits(e):
        if isinstance(e, relay.Call):
            dest = e
            inputs_bit_limit = get_inputs_bit_limit(dest)
            for src_idx, src in enumerate(dest.args):
                # consider output constraint of src node and input
                # constraint of dest node, take smaller one between them
                output_bit_limit = get_output_bit_limit(src)
                bit_limit = smaller_limit(output_bit_limit, inputs_bit_limit[src_idx])
                eidx = edge2idx[(src, dest)]
                bit_limits[eidx]= bit_limit
        else:
            return
    relay.analysis.post_order_visit(graph, fvisit_bit_limits)

    for idx, limit in enumerate(bit_limits):
        if limit is None:
            bit_limits[idx] = _DEFAULT_BIT_LIMIT

    print('bit limits')
    print_bits_info(graph, bit_limits)
    bit_choices = [list(reversed(range(4, limit + 1))) for limit in bit_limits]
    return bit_choices


def eval_acc(func, dataset):
    with relay.transform.build_config(opt_level=2):
        graph, lib, params = relay.build_module.build(func, target="llvm")
    outputs = []
    runtime = tvm.contrib.graph_runtime.create(graph, lib, tvm.cpu())
    runtime.set_input(**params)

    num_outputs = runtime.get_num_outputs()
    assert num_outputs == 1
    outputs = []

    num_correct = 0
    num_samples = 0
    for batch_id, batch in enumerate(dataset):
        runtime.set_input(0, batch['data'])
        runtime.run()
        output = runtime.get_output(0).asnumpy()
        predict = np.argmax(output, axis=1)
        label = batch['label']
        num_correct += np.sum(predict == label)
        num_samples += output.shape[0]
        outputs.append(output)
    # flatten outputs
    outputs = np.concatenate(outputs).reshape(-1)
    acc = num_correct / num_samples
    return outputs, acc


def grid_search(f, domains, args, max_iter):
    num_iter = 0
    best_guess = None
    best_cost = 0
    fout = open('qsearch_grid_search.log', 'w+', buffering=1)

    for guess in itertools.product(*domains):
        if num_iter >= max_iter:
            break
        cost = f(guess, *args)
        if cost > best_cost:
            best_cost = cost
            best_guess = guess
        num_iter += 1
        print('niter: {}, acc: {}, best acc: {}'.format(num_iter, cost, best_cost))
        fout.write(str(guess))
        fout.write('\n')
        fout.write("{}, {:.3f}, {:.3f}".format(num_iter, cost, best_cost))
        fout.write('\n')
        num_iter += 1
    return best_guess, best_cost


def random_guess(domains):
    while True:
        guess = []
        for domain in domains:
            guess.append(random.choice(domain))
        yield guess


def random_search(fcost, domains, args, max_iter):
    num_iter = 0
    best_guess = None
    best_cost = 0
    fout = open('qsearch_random_search.log', 'w+', buffering=1)


    for guess in random_guess(domains):
        print('iteration: {0}'.format(num_iter))
        if num_iter >= max_iter:
            break
        cost = fcost(guess, *args)
        if cost > best_cost:
            best_cost = cost
            best_guess = guess
        print('niter: {}, acc: {}, best acc: {}'.format(num_iter, cost, best_cost))
        fout.write(str(guess))
        fout.write('\n')
        fout.write("{}, {:.3f}, {:.3f}".format(num_iter, cost, best_cost))
        fout.write('\n')
        num_iter += 1
    fout.close()
    return best_guess, best_cost


def simulated_annealing(fcost, domains, args, T=0.5, Tmin=0.0005, cool=0.99, portion=0.10, step=2):
    fout = open('qsearch_simulated_annealing.log', 'w+', buffering=1)
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
        print('niter: {}, acc: {}, best acc: {}\n\n'.format(num_iter, cost, best_cost))
        fout.write(str(guess))
        fout.write('\n')
        fout.write("{}, {:.3f}, {:.3f}".format(num_iter, cost, best_cost))
        fout.write('\n')
        num_iter += 1
        # make new guess
        guess = neighbour(previous_guess, portion)
    return best_guess, best_cost


def search_bits_strategy(eval_func, bit_choices, graph, hw_desc, dataset):
    cfg = qtz.current_qconfig()

    args = (graph, hw_desc, dataset)
    if cfg.search_strategy == 'random_search':
        best_bits, best_acc = random_search(eval_func, bit_choices, args, 1000)
    elif cfg.search_strategy == 'grid_search':
        best_bits, best_acc = grid_search(eval_func, bit_choices, args, 1000)
    elif cfg.search_strategy == 'simulated_annealing':
        best_bits, best_acc = simulated_annealing(eval_func, bit_choices, args)
    else:
        raise ValueError('unknown search strategy: {}'.format(cfg.search_strategy))

    return best_bits, best_acc


def search_quantize_strategy(mod, hw_desc, dataset=None):
    graph = mod['main']
    print('original acc: {}'.format(eval_acc(graph, dataset)[1]))
    bit_choices = generate_bit_choices(graph, hw_desc)
    # bit_choices = [[choices[0]] for choices in bit_choices]
    print(bit_choices)
    print('number of bits: {}'.format(len(bit_choices)))

    # search for bits settings with learning method
    def eval_func(bits, graph, hw_desc, dataset):
        print('bits: {0}'.format(bits))
        # coarse-grained threshold estimate
        thresholds = threshold_estimate(graph, bits, dataset)
        # print('\nafter threshold estimate')
        # for thold in thresholds:
        #     print(type(thold))

        thresholds = threshold_rectify(graph, bits, thresholds)
        # print('\nafter threshold rectify')
        # for thold in thresholds:
        #     print(type(thold))
        # print('thresholds: {0}'.format(thresholds))
        op_params = qtz.calculate_params(graph, bits, thresholds, hw_desc)
        simulated_graph = qtz.simulate(graph, op_params)
        # print('simulated_graph')
        # print(simulated_graph)
        _, acc = eval_acc(simulated_graph, dataset)
        # [optional] calibrate threshold estimation
        return float(acc)

    best_bits, best_acc = search_bits_strategy(eval_func, bit_choices, graph, hw_desc, dataset)
    # # load config directly
    # with open('/sampa/home/ziheng/best_config.log', 'r') as fin:
    #     lines = [line.rstrip('\n') for line in fin]
    #     bits, info = lines
    #     best_bits = list(map(int, bits.strip('][').split(', ')))
    #     info = info.split(', ')
    #     best_acc = float(info[2])

    print('finished search')
    print('best_acc: {0}'.format(best_acc))
    best_thresholds = threshold_estimate(graph, best_bits, dataset)
    best_thresholds = threshold_rectify(graph, best_bits, best_thresholds)
    return best_bits, best_thresholds, best_acc
