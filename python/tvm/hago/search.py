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
from . import _ffi_api
from . import quantize as qtz
from .. import relay
from .threshold import threshold_estimate
from .hardware import *
from .record import *
from .topology import Topology, analyze_topology
from . import analysis

import tvm
import random
import math
import itertools
import numpy as np
import pickle
import scipy
from collections import namedtuple

#TODO(ziheng): unify topology and constraints

def generate_search_space(graph, hardware):
    topology = analyze_topology(graph, hardware)
    return topology.generate_search_space()


############################################################
# TODO: old search pipeline, need to move to new tuner API
#
# def grid_search(f, domains, args, max_iter=1000):
#     num_iter = 0
#     best_guess = None
#     best_cost = 0
# 
#     for guess in itertools.product(*domains):
#         if num_iter >= max_iter:
#             break
#         cost = f(guess, *args)
#         if cost >= best_cost:
#             best_cost = cost
#             best_guess = guess
#         num_iter += 1
#         print('niter: {}, acc: {}, best acc: {}'.format(num_iter, cost, best_cost))
#         num_iter += 1
#     return best_guess, best_cost
# 
# 
# def random_guess(domains):
#     while True:
#         guess = []
#         for domain in domains:
#             guess.append(random.choice(domain))
#         yield guess
# 
# 
# def random_search(fcost, domains, args, max_iter=1000):
#     num_iter = 0
#     best_guess = None
#     best_cost = 0
# 
#     for guess in random_guess(domains):
#         print('iteration: {0}'.format(num_iter))
#         if num_iter >= max_iter:
#             break
#         cost = fcost(guess, *args)
#         if cost > best_cost:
#             best_cost = cost
#             best_guess = guess
#         print('niter: {}, acc: {}, best acc: {}'.format(num_iter, cost, best_cost))
#         num_iter += 1
#     return best_guess, best_cost
# 
# 
# def simulated_annealing(fcost, domains, args, T=0.5, Tmin=0.0005, cool=0.99, portion=0.10, step=2):
#     def neighbour(origin, portion):
#         num_changed = int(portion * len(origin))
#         dims = random.sample(range(0, len(origin)), num_changed)
#         new = origin.copy()
#         for dim in dims:
#             disturbance = random.choice(range(-step, step + 1))
#             print('choose dimension {0}'.format(dim))
#             print('change from {} to {}'.format(new[dim], new[dim]+disturbance))
#             new[dim] += disturbance
#             if new[dim] < min(domains[dim]):
#                 new[dim] = min(domains[dim])
#             if new[dim] > max(domains[dim]):
#                 new[dim] = max(domains[dim])
#         return new
# 
#     previous_guess, previous_cost = None, 0
#     best_guess, best_cost = None, 0
#     num_iter = 0
#     # init with random guess
#     guess = next(random_guess(domains))
#     while T > Tmin:
#         cost = fcost(guess, *args)
#         if cost >= best_cost:
#             # stored as best guess 
#             best_guess = guess
#             best_cost = cost
#         if cost >= previous_cost or random.random() < math.exp(- (previous_cost - cost) / T):
#             print('accept guess')
#             # accept the guess
#             previous_guess = guess
#             previous_cost = cost
#         T = T * cool
#         print('niter: {}, acc: {}, best acc: {}'.format(num_iter, cost, best_cost))
#         num_iter += 1
#         # make new guess
#         guess = neighbour(previous_guess, portion)
#     return best_guess, best_cost
#
# 
# def softmax_with_temperature(x, temp=1.0, axis=1):
#     e_x = np.exp((x - np.amax(x, axis=axis, keepdims=True)) / temp)
#     return e_x / e_x.sum(axis=axis, keepdims=True)
# 
# 
# def calculate_kl(out_x, out_y):
#     out_x = softmax_with_temperature(out_x, temp=8.0, axis=1)
#     out_y = softmax_with_temperature(out_y, temp=8.0, axis=1)
#     num_samples = out_x.shape[0]
#     kl = 0.
#     for i in range(num_samples):
#         entropy = scipy.stats.entropy(out_x[i], out_y[i])
#         kl += entropy
#     kl = kl / num_samples
#     return kl
# 

def _accuracy_as_measure(func, dataset, outputs, ctx, target):
    # return a MeasureResult
    num_samples = 0
    num_correct = 0
    for idx, batch in enumerate(dataset):
        assert 'label' in batch
        label = batch['label'].asnumpy()
        sim_out = outputs[idx].asnumpy()
        sim_pred = np.argmax(sim_out, axis=1)
        # pre-calculated label in the provided dataset
        num_correct += np.sum(sim_pred == label) 
        num_samples += label.shape[0]
    acc = num_correct / num_samples
    return MeasureResult(accuracy=acc)


def get_measure_func(kind):
    # return a function: (graph, dataset, simulated_out) -> MeasureResult
    mapping = {
        MeasureKind.Accuracy: _accuracy_as_measure,
    }
    assert kind in mapping, 'not exist measure: {}'.format(kind)
    return mapping[kind]


def _group_same_graph(graph, hardware, topology, bits_list):
    """group guesses which can share the same graph"""
    constraints = []
    for bits in bits_list:
        cstrs = qtz.select_desc(graph, hardware, topology, bits)
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
        print("Simulated graph", simulator.simulated_graph)
        groups.append((simulator, grouped_guesses))
    return groups


# TODO(tvm-team): unify hago.tuner and autotvm.tuner to a general combinatorial
# optimization framework
class Tuner(object):
    def __init__(self, space, objective, max_trials=None):
        # support different objective: accuracy, kl, transfer_learning_loss
        self.space = space
        self.measure_kind = objective
        if isinstance(self.measure_kind, str):
            self.measure_kind = MeasureKind.str_to_enum(self.measure_kind)
        self.measure_func = get_measure_func(self.measure_kind) 
        self.best_measure = None
        self.max_trials = max_trials
        if max_trials is None:
            self.max_trials = math.inf

    def has_next(self):
        pass

    def next_trials(self):
        pass

    def update(self, measures):
        pass

    def tune(self, graph, hardware, dataset, ctx, target, fout=None):
        self.graph = graph
        self.hardware = hardware
        self.model_hash = tvm.ir.structural_hash(graph)
        self.dataset = dataset
        self.ctx = ctx
        self.target = target
        self.topology = analyze_topology(graph, hardware)
        self.stats = analysis.collect_stats(graph, self.topology,
            dataset, ctx, target)

        num_trials = 0
        while num_trials < self.max_trials:
            if not self.has_next():
                break

            trials = self.next_trials()
            measures = self._measure(trials)

            if fout is not None:
                self._write_to_file(fout, measures)
            self.update(measures)
            num_trials += len(measures)
        return self.best_measure

    def _write_to_file(self, fout, measures):
        for m in measures:
            fout.write(serialize(m))
            fout.write('\n')

    def _update_best_measure(self, measures):
        old_measure = self.best_measure
        if self.best_measure is None:
            self.best_measure = best_measure(measures, self.measure_kind)
        else:
            temp = [m for m in measures]
            temp.append(self.best_measure)
            self.best_measure = best_measure(temp, self.measure_kind)

        print('measures')
        for m in measures:
            print(m)
        print('best_measure')
        print(self.best_measure)
        return self.best_measure

    def _measure(self, bits_list):
        # support single sample measure and batched measure
        # [bits] -> [Measure(strategy, MeasureResult)]
        results = []
        if isinstance(bits_list, list):
            groups = _group_same_graph(self.graph, self.hardware, self.topology, bits_list)
            for simulator, grouped_guesses in groups:
                constraints = simulator.constraints
                for bits in grouped_guesses:
                    # TODO(ziheng) move thresholds outside
                    thresholds = threshold_estimate(self.graph, self.topology, self.stats, bits)
                    simulated_out = simulator.eval(bits, thresholds, self.dataset, self.ctx, self.target)
                    measure_result = self.measure_func(self.graph, self.dataset, simulated_out,
                                                       self.ctx, self.target)
                    strategy = Strategy(self.model_hash, bits, thresholds)
                    results.append(Measure(strategy, measure_result))
            return results
        else:
            raise ValueError


class DefaultSetting(Tuner):
    def __init__(self, space, objective, bits=None):
        super(DefaultSetting, self).__init__(space, objective, max_trials=1)
        if bits is None:
            self.bits = [choices[0] for choices in self.space]
        else:
            self.bits = bits

    def has_next(self):
        return True

    def next_trials(self):
        return [self.bits]

    def update(self, measures):
        self._update_best_measure(measures)


class RandomSearchTuner(Tuner):
    def __init__(self, space, objective, max_trials=None):
        if max_trials is None:
            max_trials = len(space)
        super(RandomSearchTuner, self).__init__(space, objective, max_trials)

    def has_next(self):
        return True

    def next_trials(self):
        return [[random.choice(choices) for choices in self.space]]

    def update(self, measures):
        self._update_best_measure(measures)


class GreedySearchTuner(Tuner):
    def __init__(self, space, objective, max_trials=None):
        super(GreedySearchTuner, self).__init__(space, objective, max_trials)
        self.dim_idx = 0
        self.bit_idx = 0
        self.decided = []
        self.default = [choices[0] for choices in space]

    def has_next(self):
        return self.dim_idx < len(self.space)

    def next_trials(self):
        choice = self.space[self.dim_idx][self.bit_idx]
        trials = [self.decided + [choice] + self.default[self.dim_idx+1:]]
        return trials

    def update(self, measures):
        best_measure = self._update_best_measure(measures)
        self.bit_idx += 1
        if measures[0].result.accuracy < best_measure.result.accuracy or \
            self.bit_idx >= len(self.space[self.dim_idx]):
            # move to next dimension
            best_bit = best_measure.strategy.bits[self.dim_idx]
            self.decided.append(best_bit)
            self.dim_idx += 1
            self.bit_idx = 0

    def _measure(self, bits_list):
        assert len(bits_list) == 1
        bits = bits_list[0]
        thresholds = threshold_estimate(self.graph, self.topology, self.stats, bits)
        quantizer = qtz.Quantizer(self.graph, self.hardware, self.topology, bits, thresholds)
        sgraph = quantizer.simulate()
        qgraph = quantizer.quantize()
        # print('original graph')
        # print(self.graph)
        # print('simulated graph')
        # print(sgraph)
        # print('quantized graph')
        # print(qgraph)
        # lowered_qgraph = relay.qnn.transform.CanonicalizeOps()(tvm.IRModule.from_expr(qgraph))
        # print('lowered quantized graph')
        # print(lowered_qgraph)
        # raise ValueError

        runtime = relay.create_executor("graph", ctx=self.ctx, target=self.target).evaluate(qgraph)
        input_keys = [str(param.name_hint) for param in qgraph.params]
        outputs = []
        for batch_id, batch in enumerate(self.dataset):
            inputs = {}
            for key in input_keys:
                assert key in batch
                inputs[key] = batch[key]
            out = runtime(**inputs)
            outputs.append(out)
        measure_result = self.measure_func(self.graph, self.dataset, outputs, self.ctx, self.target)
        strategy = Strategy(self.model_hash, bits, thresholds)
        result = Measure(strategy, measure_result)
        print(result)
        return [result]


class BatchedGreedySearchTuner(Tuner):
    def __init__(self, space, objective, max_trials=None):
        super(BatchedGreedySearchTuner, self).__init__(space, objective, max_trials)
        self.dim_idx = 0
        self.decided = []
        self.default = [choices[0] for choices in space]

    def has_next(self):
        return self.dim_idx < len(self.space)

    def next_trials(self):
        trials = [self.decided + [choice] + self.default[self.dim_idx+1:]
                  for choice in self.space[self.dim_idx]]
        return trials

    def update(self, measures):
        ms = self._update_best_measure(measures)
        best_bit = ms.strategy.bits[self.dim_idx]
        self.decided.append(best_bit)
        self.dim_idx += 1


def list_ops(graph):
    ops = set()
    def fvisit(node):
        if isinstance(node, relay.Call):
            ops.add(node.op.name)
    relay.analysis.post_order_visit(graph, fvisit)
    return ops

def search_quantize_strategy(graph, hardware, dataset, tuner, ctx, target):
    assert isinstance(graph, relay.Function)
    assert isinstance(dataset, qtz.CalibrationDataset)
    print('ops in graph:')
    print(list_ops(graph))
    qconfig = current_qconfig()
    origin_out = evaluate(graph, dataset, ctx, target)[0]
    origin_acc = calculate_accuracy(dataset, origin_out)
    # origin_out, origin_acc = eval_acc(graph, dataset, ctx, target)
    print('original acc: {}'.format(origin_acc))

    with open(qconfig.log_file, 'w+', buffering=1) as fout:
        measure = tuner.tune(graph, hardware, dataset, ctx, target, fout)
    return measure.strategy, measure.result
