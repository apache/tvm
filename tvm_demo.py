import numpy as np
import onnx
import tvm
import time
from tvm import relay
from tvm.contrib import graph_runtime
import tvm.contrib.graph_executor as runtime


print(tvm.__version__)
