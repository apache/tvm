import json
import os
import time
from random import getrandbits

import numpy as np

import tvm
import topi
from tvm import autotvm

def save_curve(device, backend, workload_type, workload,
               tuner, template_key, value, outfile='vis.tsv'):
    with open(outfile, 'a') as fout:
        fout.write("\t".join([str(x) for x in 
            (device, backend, workload_type, workload,
             tuner, template_key, json.dumps(value), time.time())]) + '\n')

def save_point(curve_id, device, backend, tuner, template,
               workload_type, workload, workload_op_count, iter_num, iter_total,
               time_cost, timestamp, measure_pair, outfile='vis.tsv'):
    if not os.path.isfile(outfile):
        with open(outfile, 'w') as fout:
            fout.write("\t".join(["curve_id", "device", "backend",
                                  "tuner", "template", "workload_type", "workload", "workload_op_count", 
                                  "iter", "iter_total", "time_cost", "timestamp", "measure_pair"]) + "\n")
    with open(outfile, 'a') as fout:
            fout.write("\t".join([str(x) for x in [curve_id, device, backend, tuner, template, workload_type,
                          workload, workload_op_count, iter_num, iter_total, time_cost, timestamp, measure_pair]]) + "\n")


class VisLogger(object): 
    def __init__(self, task, device, tuner, template, iter_total, outfile, curve_id=None):
        self.device = device
        self.backend = str(task.target).split(" ")[0]
        self.tuner = tuner
        self.template = template
        self.workload_type = 'op'
        self.workload = str(task.workload)
        self.workload_op_count = task.flop
        self.iter_total = iter_total
        self.curve_id = "%0x" % getrandbits(128)
        self.outfile = outfile

        self.ct = 0

    def __call__(self, tuner, inputs, results):
        for inp, res in zip(inputs, results):
            if res.error_no == 0:
                cost = np.mean(res.costs)
            else:
                cost = float("+inf")

            save_point(self.curve_id, self.device, self.backend,
                       self.tuner, self.template, self.workload_type, self.workload,
                       self.workload_op_count, self.ct, self.iter_total, cost, time.time(),
                       json.dumps({'input': None, 'result': None}),
                       outfile=self.outfile)

            self.ct += 1

