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
from . import calibrate as _calibrate
from .. import expr as _expr
from .. import analysis as _analysis
from .. import transform as _transform
from .. import build_module as _build_module

import tvm
import sys
import random
import itertools
import functools
import numpy as np
import multiprocessing as mp
import logging
try:
    import scipy
except ImportError:
    scipy = None


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



from collections import namedtuple, defaultdict

TensorReprentation = namedtuple('TensorRepresentation', ['num_bit', 'threshold'])
SimulatedQuantizeParams = namedtuple("SimulatedQuantizeParams", ['scale',
                                                                 'clip_min',
                                                                 'clip_max',
                                                                 'overflow_min',
                                                                 'overflow_max'])
######################################################
# TODOs:
# - integrate kl distance threshold estimation into it 
# - add overflow check
# - simulated annealing

# for different hardwares, we need to consider instructions that it support. Reflect on graph level:
# - dtype constraint
# - shape constraint
# - layout constraint
# - op/subgraph combination
# - detect graph pattern, consider regex
# - check auto graph
# Consider:
# - Similarities with:
#   - TypeInfer of Op
#   - TensorIntrinsic
# - VTA, GPU:TensorCore, Quantization, LayoutTransform
class HardwareDescription(object):
    def __init__(self):
        self._op_constraints = defaultdict(list)

    def __getitem__(self, op_name):
        return self._op_constraints[op_name]

    @property
    def ops(self):
        return self._op_constraints.keys()


def create_accelerator_description():
    # TODO: change to DataType
    desc = HardwareDescription()
    desc['add'].append(([8, 8], [16]))
    desc['add'].append(([8, 8], [32]))
    desc['nn.conv2d'].append(([8, 8], [16]))
    # desc['nn.conv2d'].append(([8, 8], [32]))
    return desc


def generate_bit_choices(graph, hw_desc):
    """
    Question:
      - do we need to consider output nbit constraint?
    """
    # build indexing map
    expr2idx = {}  # expr(var/call) to idx
    def fvisit_build_index(e):
        if isinstance(e, (_expr.Var, _expr.Constant, _expr.Call)):
            expr2idx[e] = fvisit_build_index.idx_cnt 
            fvisit_build_index.idx_cnt += 1
    fvisit_build_index.idx_cnt = 0
    _analysis.post_order_visit(graph, fvisit_build_index)

    # analysis maximum num of bit on every tensor/edge
    max_bits = [32 for _ in range(fvisit_build_index.idx_cnt)]
    def fvisit_max_bits(e):
        if isinstance(e, (_expr.Var, _expr.Constant)):
            # use 8 bit for variables/weights
            idx = expr2idx[e]
            max_bits[idx] = 8
        elif isinstance(e, _expr.Call):
            if e.op.name in hw_desc.ops:
                constraints = hw_desc[e.op.name]
                # consider output constraint of current op
                max_output_bit = max(instr[1][0] for instr in hw_desc[e.op.name])
                idx = expr2idx[e]
                max_bits[idx] = max(max_bits[idx], max_output_bit)

                # consider input constraint of followering op
                max_inputs_bit = constraints[0][0]
                for (inputs_bit, outputs_bit) in constraints:
                    max_inputs_bit = (max(v1, v2) for v1, v2
                                      in zip(inputs_bit, max_inputs_bit))
                for (input_expr, max_input_bit) in zip(e.args, max_inputs_bit):
                    in_idx = expr2idx[input_expr]
                    max_bits[idx] = max(max_bits[in_idx], max_input_bit)
        else:
            return
    _analysis.post_order_visit(graph, fvisit_max_bits)

    print(max_bits)
    bit_choices = [list(reversed(range(4, max_bit + 1))) for max_bit in max_bits]
    return bit_choices


def threshold_estimate(graph, bits, dataset=None):
    # check bit setting
    # exprs = []
    # def fvisit_test(e):
    #     if isinstance(e, (_expr.Var)):
    #         exprs.append(e)
    #     if isinstance(e, _expr.Constant):
    #         exprs.append('constant')
    #     if isinstance(e, _expr.Call):
    #         exprs.append(e.op.name)
    # _analysis.post_order_visit(graph, fvisit_test)

    stats = _calibrate.collect_stats(graph, dataset)
    assert scipy is not None, "scipy need to be installed for \
    utilizing kl calibration during quantization"
    with mp.Pool() as pool:
        logging.info("finding threshold with kl for calibration...")
        thresholds = list(pool.map(_calibrate._find_scale_by_kl, stats))
    return thresholds


    # for (e, b) in zip(exprs, bits):
    #     print(e, b)

    thresholds = [4.0 for _ in exprs]
    return thresholds


def calculate_quantize_op_params(graph, bits, thresholds, hw_desc):
    # map: tensor -> (num_bit, threshold)
    # set scale, clip_min, clip_max for every tensor
    # integer_range = 2 ^ (num_bit - sign) 
    # scale = threshold / integer_range
    # clip_min = - (integer_range - 1)
    # clip_max =   (integer_range - 1)
    #
    # prepare:
    # overflow_num_bit = 16bit (hardware_constrait)
    # input_scale
    #
    # overflow_lower_bound_integer = - (2 ^ (overflow_num_bit - 1) - 1)
    # overflow_lower_bound_real    = overflow_lower_bound_quant * input_scale
    # overflow_upper_bound_integer = (2 ^ (overflow_num_bit - 1) - 1)
    # overflow_upper_bound_real    = overflow_upper_bound_quant * input_scale
    assert len(bits) == len(thresholds)
    print('num of tensors: {}'.format(len(bits)))
    sign = 1

    expr2idx = {}  # expr(var/call) to idx
    def fvisit_build_index(e):
        if isinstance(e, (_expr.Var, _expr.Constant, _expr.Call)):
            expr2idx[e] = fvisit_build_index.idx_cnt 
            fvisit_build_index.idx_cnt += 1
    fvisit_build_index.idx_cnt = 0
    _analysis.post_order_visit(graph, fvisit_build_index)

    op_params = []
    def fvisit(e):
        if isinstance(e, (_expr.Var, _expr.Constant, _expr.Call)):
            # if isinstance(e, _expr.Var):
            #     print(e)
            # if isinstance(e, _expr.Constant):
            #     print('constant')
            # if isinstance(e, _expr.Call):
            #     print(e.op.name)

            bit = bits[fvisit.idx_cnt]
            threshold = thresholds[fvisit.idx_cnt]
            integer_range = pow(2, bit - sign)
            scale = threshold / integer_range
            clip_min = - float(integer_range - 1)
            clip_max =   float(integer_range - 1)

            overflow_min = np.array([- sys.float_info.max], dtype=np.float32)
            overflow_max = np.array([sys.float_info.max], dtype=np.float32)

            # consider hardware constraint to detect overflow
            if isinstance(e, _expr.Call) and e.op.name in hw_desc.ops:
                # scale of inputs
                input_scales = [op_params[expr2idx[arg]].scale for arg in e.args]
                # print('input scales')
                # print(input_scales)
                # calculate op's output scale with input scales
                # TODO(ziheng): different rules for different op
                if e.op.name == 'nn.conv2d':
                    input_scale = functools.reduce(lambda x, y: x*y, input_scales, 1.0)
                elif e.op.name == 'add':
                    input_scale = max(input_scales)
                else:
                    raise ValueError('not support {0} yet.'.format(e.op.name))

                # print(hw_desc[e.op.name])
                overflow_num_bit = max(instr[1][0] for instr in hw_desc[e.op.name])
                overflow_min_integer = - (2 ^ (overflow_num_bit - sign) - 1)
                overflow_max_integer = (2 ^ (overflow_num_bit - sign) - 1)
                overflow_min = overflow_min_integer * input_scale
                overflow_max = overflow_max_integer * input_scale
                # TODO(ziheng) support scalar for extern function
                overflow_min = np.array([overflow_min], dtype=np.float32)
                overflow_max = np.array([overflow_max], dtype=np.float32)


            param = SimulatedQuantizeParams(scale, clip_min, clip_max,
                                            overflow_min,
                                            overflow_max)
            # print('op_param')
            # print(param)
            op_params.append(param)
            fvisit.idx_cnt += 1
    fvisit.idx_cnt = 0
    _analysis.post_order_visit(graph, fvisit)
    return op_params


def simulate(graph, op_params):
    class Simulator(tvm.relay.ExprMutator):
        def __init__(self):
            super().__init__()

        def create_simulated_graph(self, graph, op_params):
            self._op_params = op_params
            self._expr2idx = {}  # expr(var/call) to idx
            def fvisit_build_index(e):
                if isinstance(e, (_expr.Var, _expr.Constant, _expr.Call)):
                    self._expr2idx[e] = fvisit_build_index.idx_cnt 
                    fvisit_build_index.idx_cnt += 1
            fvisit_build_index.idx_cnt = 0
            _analysis.post_order_visit(graph, fvisit_build_index)
            return self.visit(graph)

        def visit(self, e):
            new_e = super().visit(e)
            if isinstance(e, (_expr.Var, _expr.Constant, _expr.Call)):
                param = self._op_params[self._expr2idx[e]]
                ret = _quantize.simulated_quantize(new_e,
                                                   tvm.relay.const(param.scale),
                                                   tvm.relay.const(param.clip_min),
                                                   tvm.relay.const(param.clip_max),
                                                   tvm.relay.const(param.overflow_min),
                                                   tvm.relay.const(param.overflow_max),
                                                   True,
                                                   "round")
                return ret
            return new_e

        def visit_function(self, fn):
            # skip params
            new_body = self.visit(fn.body)
            return _expr.Function(
                fn.params,
                new_body,
                fn.ret_type,
                fn.type_params,
                fn.attrs)

    # print('before simulating')
    # print(graph)
    # print('creare simulated graph')
    simulated_graph = Simulator().create_simulated_graph(graph, op_params)
    # print(simulated_graph)
    return simulated_graph


def eval_acc(func, dataset):
    with _transform.build_config(opt_level=2):
        graph, lib, params = _build_module.build(func, target="llvm")
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
        num_correct = np.sum(predict == label)
        num_samples += output.shape[0]
        outputs.append(output)
    # flatten outputs
    print(len(outputs))
    outputs = np.concatenate(outputs).reshape(-1)
    acc = num_correct / num_samples
    return outputs, acc


def random_guess(domains):
    while True:
        guess = []
        for domain in domains:
            guess.append(random.choice(domain))
        yield guess


def random_search(f, domains, args, max_iter):
    num_iter = 0
    best_guess = None
    best_cost = sys.float_info.max

    for guess in random_guess(domains):
        print('iteration: {0}'.format(num_iter))
        if num_iter >= max_iter:
            break
        cost = f(guess, *args)
        if cost < best_cost:
            best_cost = cost
            best_guess = guess
        num_iter += 1
    return best_guess, best_cost


def simulated_annealing(fcost, domains, args, T=1.0, Tmin=0.001, cool=0.9, step=1):
    def neighbour(origin):
        dim = random.randint(0, len(origin) - 1)
        disturbance = random.randint(-step, step)

        guess = copy(origin)
        guess[dim] += disturbance
        if guess[dim] < domains[dim][0]:
            guess[dim] = domains[dim][0]
        if guess[dim] > domains[dim][-1]:
            guess[dim] = domains[dim][-1]

    best_guess = random_guess(domains).next()
    best_cost = fcost(best_guess)
    while T > Tmin:
        guess = neighbour(best_guess)
        cost = fcost(guess, args)
        if cost < best_cost or random.random() < math.exp(-(cost - best_cost) / T):
            best_guess = guess
        T = T * cool
    return best_guess, best_cost


def search(mod, hw_desc, dataset=None):
    # cfg = current_qconfig()
    graph = mod['main']
    bit_choices = generate_bit_choices(graph, hw_desc)

    # search for bits settings with learning method
    def eval_func(bits, graph, hw_desc, dataset):
        print('bits: {0}'.format(bits))
        # coarse-grained threshold estimate
        thresholds = threshold_estimate(graph, bits, dataset)
        print('thresholds: {0}'.format(thresholds))
        op_params = calculate_quantize_op_params(graph, bits, thresholds, hw_desc)
        simulated_graph = simulate(graph, op_params)
        _, acc = eval_acc(simulated_graph, dataset)
        print('acc: {0}'.format(acc))
        # [optional] calibrate threshold estimation
        return acc

    args = (graph, hw_desc, dataset)
    best_bits, best_acc = random_search(eval_func, bit_choices, args, 1000)
    best_thresholds = threshold_estimate(graph, best_bits, dataset)
    return best_bits, best_thresholds, best_acc
