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

from . import _quantize
from . import quantize as qtz
from .. import relay
from .threshold import threshold_estimate

import tvm
import random
import math
import itertools
import time
import logging
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



def printrelay(e):
    if isinstance(e, (relay.Var)):
        print(e.name_hint)
    if isinstance(e, relay.Constant):
        print('constant')
    if isinstance(e, relay.Call):
        print(e.op.name)
    return None

def generate_bit_choices(graph, hw_desc):
    """
    Question:
      - do we need to consider output nbit constraint?
    """
    # build indexing map
    expr2idx = {}  # expr(var/call) to idx
    def fvisit_build_index(e):
        if isinstance(e, (relay.Var, relay.Constant, relay.Call)):
            expr2idx[e] = fvisit_build_index.idx_cnt 
            fvisit_build_index.idx_cnt += 1
    fvisit_build_index.idx_cnt = 0
    relay.analysis.post_order_visit(graph, fvisit_build_index)

    # analysis maximum num of bit on every tensor/edge
    max_bits = [32 for _ in range(fvisit_build_index.idx_cnt)]
    def fvisit_max_bits(e):
        if isinstance(e, (relay.Var, relay.Constant)):
            # use 8 bit for variables/weights
            idx = expr2idx[e]
            max_bits[idx] = 8
        elif isinstance(e, relay.Call):
            if e.op.name in hw_desc.ops:
                print("op: "  + e.op.name)
                constraints = hw_desc[e.op.name]
                # consider output constraint of current op
                max_output_bit = max(instr[1][0] for instr in hw_desc[e.op.name])
                idx = expr2idx[e]
                max_bits[idx] = max(max_bits[idx], max_output_bit)
                print(max_bits[idx])

                # consider input constraint of followering op
                max_inputs_bit = constraints[0][0]
                for (inputs_bit, outputs_bit) in constraints:
                    max_inputs_bit = (max(v1, v2) for v1, v2
                                      in zip(inputs_bit, max_inputs_bit))
                for (inputrelay, max_input_bit) in zip(e.args, max_inputs_bit):
                    in_idx = expr2idx[inputrelay]
                    max_bits[in_idx] = max(max_bits[in_idx], max_input_bit)
        else:
            return
    relay.analysis.post_order_visit(graph, fvisit_max_bits)

    print(max_bits)
    exprs = []
    def fvisit_test(e):
        if isinstance(e, (relay.Var)):
            exprs.append(e)
        if isinstance(e, relay.Constant):
            exprs.append('constant')
        if isinstance(e, relay.Call):
            exprs.append(e.op.name)
    relay.analysis.post_order_visit(graph, fvisit_test)

    for name, bit in zip(exprs, max_bits):
        print(name, bit)
    bit_choices = [list(reversed(range(4, max_bit + 1))) for max_bit in max_bits]
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
    best_cost = sys.float_info.max

    for guess in itertools.product(*domains):
        if num_iter >= max_iter:
            break
        cost = f(guess, *args)
        if cost > best_cost:
            best_cost = cost
            best_guess = guess
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

    current_guess = next(random_guess(domains))
    current_cost = fcost(current_guess, *args)
    best_guess, best_cost = current_guess, current_cost
    num_iter = 0
    while T > Tmin:
        guess = neighbour(best_guess, portion)
        cost = fcost(guess, *args)
        if cost >= best_cost:
            # store as best guess 
            best_guess = guess
            best_cost = cost
        if cost >= current_cost or random.random() < math.exp(- (current_cost - cost) / T):
            print('accept guess')
            # accept the guess
            current_guess = guess
            current_cost = cost
        T = T * cool
        print('niter: {}, acc: {}, best acc: {}'.format(num_iter, cost, best_cost))
        fout.write(str(guess))
        fout.write('\n')
        fout.write("{}, {:.3f}, {:.3f}".format(num_iter, cost, best_cost))
        fout.write('\n')
        num_iter += 1
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
        # print('thresholds: {0}'.format(thresholds))
        op_params = qtz.calculate_quantize_op_params(graph, bits, thresholds, hw_desc)
        simulated_graph = qtz.simulate(graph, op_params)
        _, acc = eval_acc(simulated_graph, dataset)
        # [optional] calibrate threshold estimation
        return float(acc)

    best_bits, best_acc = search_bits_strategy(eval_func, bit_choices, graph, hw_desc, dataset)
    print('finished search')
    print('best_acc: {0}'.format(best_acc))
    best_thresholds = threshold_estimate(graph, best_bits, dataset)
    return best_bits, best_thresholds, best_acc
