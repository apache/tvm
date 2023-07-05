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
"""
Deploy a Hugging Face Pruned Model on CPU
=========================================
**Author**: `Josh Fromm <https://github.com/jwfromm>`_

This tutorial demonstrates how to take any pruned model, in this case `PruneBert
from Hugging Face
<https://huggingface.co/huggingface/prunebert-base-uncased-6-finepruned-w-distil-squad>`_,
and use TVM to leverage the model's sparsity support to produce real speedups. Although
the primary purpose of this tutorial is to realize speedups on already pruned
models, it may also be useful to estimate how fast a model would be *if* it were
pruned. To this end, we also provide a function that takes an unpruned model and
replaces its weights
with random and pruned weights at a specified sparsity. This may be a useful
feature when trying to decide if a model is worth pruning or not.

Before we get into the code, it's useful to discuss sparsity and pruning
and dig into the two
different types of sparsity: **structured** and **unstructured**.

Pruning is a technique primarily used to reduce the parameter size of a model
by replacing weight values with 0s. Although many methods exist for choosing which
weights should be set to 0, the most straight forward is by picking the
weights with the smallest value. Typically, weights are pruned to a desired
sparsity percentage. For example, a 95% sparse model would have only 5% of
its weights non-zero. Pruning to very high sparsities often requires
fine-tuning or full retraining as it tends to be a lossy approximation.
Although parameter size benefits are quite easy to obtain from a pruned model
through simple compression, leveraging sparsity to yield runtime speedups
is more complicated.

In structured sparsity weights are pruned with the goal of clustering
pruned weights together. In other words, they are pruned using both their
value and location. The benefit of bunching up pruned weights is that it allows
an algorithm such as matrix multiplication to skip entire blocks. It turns out
that some degree of *block sparsity* is very important to realizing significant
speedups on most hardware available today.
This is because when loading memory in most CPUs or GPUs,
it doesn't save any work to skip reading a single value at a time, instead an entire
chunk or tile is read in and executed using something like vectorized instructions.

Unstructured sparse weights are those that are pruned only on the value of
the original weights. They may appear to be scattered randomly throughout
a tensor rather than in chunks like we'd see in block sparse weights.
At low sparsities, unstructured pruning techniques are difficult to
accelerate. However, at high sparsities many blocks of all 0 values
will naturally appear, making it possible to accelerate.

This tutorial interacts with both structured and unstructured sparsity.
Hugging Face's PruneBert model is unstructured but 95% sparse, allowing us
to apply TVM's block sparse optimizations to it, even if not optimally.
When generating random sparse weights for an unpruned model, we do so with structured
sparsity. A fun exercise is comparing the real speed of PruneBert with the block
sparse speed using fake weights to see the benefit of structured sparsity.
"""


###############################################################################
# Load Required Modules
# ---------------------
# Other than TVM, scipy, the latest transformers, and
# tensorflow 2.2+ are required.
import os
import tvm
import time
import itertools
import numpy as np
import tensorflow as tf
from tvm import relay, runtime
from tvm.contrib import graph_executor
from tvm.relay import data_dep_optimization as ddo
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)
import scipy.sparse as sp


# Ask tensorflow to limit its GPU memory to what's actually needed
# instead of gobbling everything that's available.
# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
# This way this tutorial is a little more friendly to sphinx-gallery.
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("tensorflow will use experimental.set_memory_growth(True)")
    except RuntimeError as e:
        print("experimental.set_memory_growth option is not available: {}".format(e))


###############################################################################
# Configure Settings
# ------------------
# Let's start by defining some parameters that define the type of model
# and sparsity to run.

# The name of the transformer model to download and run.
name = "huggingface/prunebert-base-uncased-6-finepruned-w-distil-squad"
# The number of batches in an input.
batch_size = 1
# The length of each input sequence.
seq_len = 128
# TVM platform identifier. Note that best cpu performance can be achieved by setting -mcpu
# appropriately for your specific machine. CUDA and ROCm are also supported.
target = "llvm"
# Which device to run on. Should be one of tvm.cpu() or tvm.cuda().
dev = tvm.cpu()
# If true, then a sparse variant of the network will be run and
# benchmarked.
measure_sparse = True
# The block size of structured sparsity to convert weight tensors
# into. Changing this parameter may yield speedups for some platforms.
bs_r = 1
# For models besides PruneBert (which is 95% sparse), this parameter
# determines how sparse the generated weights should be. The higher
# the sparsity, the faster the result.
sparsity = 0.85


###############################################################################
# Download and Convert Transformers Model
# ---------------------------------------
# Now we'll grab a model from the transformers module, download it,
# convert it into a TensorFlow graphdef in preperation for converting that graphdef into
# a relay graph that we can optimize and deploy.
def load_keras_model(module, name, seq_len, batch_size, report_runtime=True):
    model = module.from_pretrained(name)
    dummy_input = tf.keras.Input(shape=[seq_len], batch_size=batch_size, dtype="int32")
    dummy_out = model(dummy_input)  # Propagate shapes through the keras model.
    if report_runtime:
        np_input = np.random.uniform(size=[batch_size, seq_len], low=0, high=seq_len).astype(
            "int32"
        )
        start = time.time()
        repeats = 50
        for i in range(repeats):
            np_out = model(np_input)
        end = time.time()
        print("Keras Runtime: %f ms." % (1000 * ((end - start) / repeats)))
    return model


def convert_to_graphdef(model, batch_size, seq_len):
    model_func = tf.function(lambda x: model(x))
    input_dict = model._saved_model_inputs_spec
    input_spec = input_dict[list(input_dict.keys())[0]]
    model_func = model_func.get_concrete_function(
        tf.TensorSpec([batch_size, seq_len], input_spec.dtype)
    )
    frozen_func = convert_variables_to_constants_v2(model_func)
    return frozen_func.graph.as_graph_def()


def download_model(name, batch_size, seq_len):
    import transformers

    module = getattr(transformers, "TFBertForSequenceClassification")
    model = load_keras_model(module, name=name, batch_size=batch_size, seq_len=seq_len)
    return convert_to_graphdef(model, batch_size, seq_len)


###############################################################################
# Convert to Relay Graph
# ----------------------
# We now have all the tooling to get a transformers model in the right format
# for relay conversion. Let's import it! In the following function we
# save the imported graph in relay's json format so that we dont have
# to reimport from tensorflow each time this script is run.
def import_graphdef(
    name,
    batch_size,
    seq_len,
    save_relay=True,
    relay_file="model.json",
    relay_params="model.params",
):
    abs_path = os.path.dirname(os.path.abspath(__file__))
    shape_dict = {"input_1": (batch_size, seq_len)}
    relay_file = ("%s_%d_%d_%s" % (name, batch_size, seq_len, relay_file)).replace("/", "_")
    relay_params = ("%s_%d_%d_%s" % (name, batch_size, seq_len, relay_params)).replace("/", "_")
    if os.path.exists(os.path.join(abs_path, relay_file)) and os.path.exists(
        os.path.join(abs_path, relay_params)
    ):
        with open(os.path.join(abs_path, relay_file), "r") as fi:
            mod = tvm.ir.load_json(fi.read())
        with open(os.path.join(abs_path, relay_params), "rb") as fi:
            params = relay.load_param_dict(fi.read())
    else:
        graph_def = download_model(name, batch_size, seq_len)

        mod, params = relay.frontend.from_tensorflow(graph_def, shape=shape_dict)

        if save_relay:
            with open(os.path.join(abs_path, relay_file), "w") as fo:
                fo.write(tvm.ir.save_json(mod))
            with open(os.path.join(abs_path, relay_params), "wb") as fo:
                fo.write(runtime.save_param_dict(params))

    return mod, dict(params.items()), shape_dict


###############################################################################
# Run the Dense Graph
# -------------------
# Let's run the default version of the imported model. Note that even if
# the weights are sparse, we won't see any speedup because we are using
# regular dense matrix multiplications on these dense (but mostly zero)
# tensors instead of sparse aware kernels.
def run_relay_graph(mod, params, shape_dict, target, dev):
    with relay.build_config(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    input_shape = shape_dict["input_1"]
    dummy_data = np.random.uniform(size=input_shape, low=0, high=input_shape[1]).astype("int32")

    m = graph_executor.GraphModule(lib["default"](dev))
    m.set_input(0, dummy_data)
    m.run()
    tvm_output = m.get_output(0)

    print(m.benchmark(dev, repeat=5, number=5))
    return tvm_output


def run_dense(mod, params, shape_dict, target, dev):
    print("Dense Model Benchmark:")
    return run_relay_graph(mod, params, shape_dict, target, dev)


###############################################################################
# Run the Sparse Graph
# --------------------
# Next we'll convert the graph into a sparse representation and generate
# fake sparse weights if needed. Then we'll use the same benchmarking
# script as dense to see how much faster we go! We apply a few relay passes
# to the graph to get it leveraging sparsity. First we use
# `simplify_fc_transpose` to use transposes on the weights of dense layers
# into the parameters. This makes it easier to convert to matrix multiplies
# to sparse versions. Next we apply `bsr_dense.convert` to identify all
# weight matrices that can be sparse, and automatically replace them.
#
# The `bsr_dense.convert` call below is doing the heavy lifting of identifying
# which weights in the model can be made sparse by checking if they are
# at least `sparsity_threshold` percent sparse. If so, it converts those
# weights into *Block Compressed Row Format (BSR)*. BSR is essentially
# a representation that indexes into the nonzero chunks of the tensor,
# making it easy for an algorithm to load those non-zero chunks and ignore
# the rest of the tensor. Once the sparse weights are in BSR format,
# `relay.transform.DenseToSparse` is applied to actually replace
# `relay.dense` operations with `relay.sparse_dense` calls that can be
# run faster.
def random_bsr_matrix(M, N, BS_R, BS_C, density, dtype="float32"):
    Y = np.zeros((M, N), dtype=dtype)
    assert M % BS_R == 0
    assert N % BS_C == 0
    nnz = int(density * M * N)
    num_blocks = int(nnz / (BS_R * BS_C)) + 1
    candidate_blocks = np.asarray(list(itertools.product(range(0, M, BS_R), range(0, N, BS_C))))
    assert candidate_blocks.shape[0] == M // BS_R * N // BS_C
    chosen_blocks = candidate_blocks[
        np.random.choice(candidate_blocks.shape[0], size=num_blocks, replace=False)
    ]
    for i in range(len(chosen_blocks)):
        r, c = chosen_blocks[i]
        Y[r : r + BS_R, c : c + BS_C] = np.random.uniform(-0.1, 0.1, (BS_R, BS_C))
    s = sp.bsr_matrix(Y, blocksize=(BS_R, BS_C))
    assert s.data.shape == (num_blocks, BS_R, BS_C)
    assert s.data.size >= nnz
    assert s.indices.shape == (num_blocks,)
    assert s.indptr.shape == (M // BS_R + 1,)
    return s.todense()


def random_sparse_bert_params(func, params, density, BS_R, BS_C):
    def deepcopy(param_dic):
        ret = {}
        for k, v in param_dic.items():
            ret[k] = tvm.nd.array(v.numpy())
        return ret

    new_params = deepcopy(params)
    dense_weight_names = relay.analysis.sparse_dense._search_dense_op_weight(func)
    for item in dense_weight_names:
        name = str(item)
        shape = new_params[name].shape
        if shape[0] % BS_R == 0 and shape[1] % BS_C == 0:
            new_w = random_bsr_matrix(shape[0], shape[1], BS_R, BS_C, density)
            new_params[name] = tvm.nd.array(new_w)
    return new_params


def run_sparse(mod, params, shape_dict, target, dev, bs_r, sparsity, gen_weights):
    mod, params = ddo.simplify_fc_transpose.convert(mod["main"], params)
    if gen_weights:
        params = random_sparse_bert_params(mod, params, BS_R=bs_r, BS_C=1, density=1 - sparsity)
    mod, params = ddo.bsr_dense.convert(mod, params, (bs_r, 1), sparsity_threshold=0.8)
    print("Block Sparse Model with {blocksize}x1 blocks:".format(blocksize=bs_r))
    return run_relay_graph(mod, params, shape_dict, target, dev)


###############################################################################
# Run All the Code!
# -----------------
# And that's it! Now we'll simply call all the needed function to benchmark
# the model according to the set parameters. Note that to run this code
# you'll need to uncomment the last line first.
def benchmark():
    mod, params, shape_dict = import_graphdef(name, batch_size, seq_len)
    run_dense(mod, params, shape_dict, target, dev)
    if measure_sparse:
        gen_weights = "prune" not in name
        run_sparse(mod, params, shape_dict, target, dev, bs_r, sparsity, gen_weights)


# benchmark()

###############################################################################
# Sample Output
# -------------
# For reference, below is the output of the script when run on an AMD CPU
# and shows about a 2.5X speedup from using sparsity.

# Dense Model Benchmark:
# Cannot find config for target=llvm, workload=('dense_nopack.x86', ('TENSOR', (1, 768), 'float32'), ('TENSOR', (2, 768), 'float32'), None, 'float32'). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=llvm, workload=('dense_nopack.x86', ('TENSOR', (1, 768), 'float32'), ('TENSOR', (768, 768), 'float32'), None, 'float32'). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=llvm, workload=('dense_nopack.x86', ('TENSOR', (128, 3072), 'float32'), ('TENSOR', (768, 3072), 'float32'), None, 'float32'). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=llvm, workload=('dense_nopack.x86', ('TENSOR', (128, 768), 'float32'), ('TENSOR', (3072, 768), 'float32'), None, 'float32'). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=llvm, workload=('dense_nopack.x86', ('TENSOR', (128, 768), 'float32'), ('TENSOR', (768, 768), 'float32'), None, 'float32'). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=llvm, workload=('batch_matmul.x86', ('TENSOR', (12, 128, 128), 'float32'), ('TENSOR', (12, 64, 128), 'float32')). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=llvm, workload=('batch_matmul.x86', ('TENSOR', (12, 128, 64), 'float32'), ('TENSOR', (12, 128, 64), 'float32')). A fallback configuration is used, which may bring great performance regression.
# Runtime:             165.26 ms           (12.83 ms)
# Block Sparse Model with 1x1 blocks:
# Runtime:             67.75 ms            (8.83 ms)

# Here is the output of this script on a GPU (GTX 1070) with the target "cuda -libs=cublas".
#
# Dense Model Benchmark:
# Cannot find config for target=cuda -keys=cuda,gpu -libs=cublas -max_num_threads=1024 -thread_warp_size=32, workload=('dense_cublas.cuda', ('TENSOR', (1, 768), 'float32'), ('TENSOR', (2, 768), 'float32'), None, 'float32'). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=cuda -keys=cuda,gpu -libs=cublas -max_num_threads=1024 -thread_warp_size=32, workload=('dense_cublas.cuda', ('TENSOR', (1, 768), 'float32'), ('TENSOR', (768, 768), 'float32'), None, 'float32'). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=cuda -keys=cuda,gpu -libs=cublas -max_num_threads=1024 -thread_warp_size=32, workload=('dense_cublas.cuda', ('TENSOR', (128, 3072), 'float32'), ('TENSOR', (768, 3072), 'float32'), None, 'float32'). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=cuda -keys=cuda,gpu -libs=cublas -max_num_threads=1024 -thread_warp_size=32, workload=('dense_cublas.cuda', ('TENSOR', (128, 768), 'float32'), ('TENSOR', (3072, 768), 'float32'), None, 'float32'). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=cuda -keys=cuda,gpu -libs=cublas -max_num_threads=1024 -thread_warp_size=32, workload=('dense_cublas.cuda', ('TENSOR', (128, 768), 'float32'), ('TENSOR', (768, 768), 'float32'), None, 'float32'). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=cuda -keys=cuda,gpu -libs=cublas -max_num_threads=1024 -thread_warp_size=32, workload=('batch_matmul_cublas.cuda', ('TENSOR', (12, 128, 128), 'float32'), ('TENSOR', (12, 64, 128), 'float32'), (12, 128, 64)). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=cuda -keys=cuda,gpu -libs=cublas -max_num_threads=1024 -thread_warp_size=32, workload=('batch_matmul_cublas.cuda', ('TENSOR', (12, 128, 64), 'float32'), ('TENSOR', (12, 128, 64), 'float32'), (12, 128, 128)). A fallback configuration is used, which may bring great performance regression.
# Runtime:             10.64 ms            (0.29 ms)
# Block Sparse Model with 1x1 blocks:
# Runtime:             6.46 ms             (0.05 ms)
