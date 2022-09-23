import argparse
import os
import sys

from loguru import logger
import numpy as np
import tvm
import tvm.relay as relay
from tvm import rpc
from tvm import autotvm
import tvm.relay.testing
import pickle

from tvm.relay.function import Function
from tvm.relay.expr import Call, Constant, Tuple, GlobalVar, Var, TupleGetItem
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
import tvm.contrib.graph_executor as runtime
# from tvm.autotvm.measure.measure_methods import set_cuda_target_arch

if __name__ == '__main__':
    relay_path = sys.argv[1]
    with open(relay_path, "r") as relay_fn:
        relay_text = relay_fn.read()
    mod = tvm.parser.fromtext(relay_text)
    print(mod)