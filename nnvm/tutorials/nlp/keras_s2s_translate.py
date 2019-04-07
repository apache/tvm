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
Keras LSTM Sequence to Sequence Model for Translation
=================================
**Author**: `Siju Samuel <https://siju-samuel.github.io/>`_

This script demonstrates how to implement a basic character-level sequence-to-sequence model.
We apply it to translating short English sentences into short French sentences,
character-by-character.

# Summary of the algorithm

- We start with input sequences from a domain (e.g. English sentences)
    and corresponding target sequences from another domain
    (e.g. French sentences).
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    Is uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.

This script loads the s2s.h5 model saved in repository
https://github.com/dmlc/web-data/raw/master/keras/models/s2s_translate/lstm_seq2seq.py
and generates sequences from it.  It assumes that no changes have been made (for example:
latent_dim is unchanged, and the input data and model architecture are unchanged).

# References

- Sequence to Sequence Learning with Neural Networks
    https://arxiv.org/abs/1409.3215
- Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation
    https://arxiv.org/abs/1406.1078

See lstm_seq2seq.py for more details on the model architecture and how it is trained.
"""

from keras.models import Model, load_model
from keras.layers import Input
import random
import os
import numpy as np
import keras
import tvm
import nnvm

######################################################################
# Download required files
# -----------------------
# Download files listed below from dmlc web-data repo.
model_file = "s2s_translate.h5"
data_file = "fra-eng.txt"

# Base location for model related files.
repo_base = 'https://github.com/dmlc/web-data/raw/master/keras/models/s2s_translate/'
model_url = os.path.join(repo_base, model_file)
data_url = os.path.join(repo_base, data_file)

# Download files listed below.
from tvm.contrib.download import download_testdata
model_path = download_testdata(model_url, model_file, module='keras')
data_path = download_testdata(data_url, data_file, module='data')

latent_dim = 256  # Latent dimensionality of the encoding space.
test_samples = 10000  # Number of samples used for testing.

######################################################################
# Process the data file
# ---------------------
# Vectorize the data.  We use the same approach as the training script.
# NOTE: the data must be identical, in order for the character -> integer
# mappings to be consistent.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
test_samples = min(test_samples, len(lines))
max_encoder_seq_length = 0
max_decoder_seq_length = 0
for line in lines[:test_samples]:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    max_encoder_seq_length = max(max_encoder_seq_length, len(input_text))
    max_decoder_seq_length = max(max_decoder_seq_length, len(target_text))
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

# Reverse-lookup token index to decode sequences back to something readable.
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

######################################################################
# Load Keras Model
# ----------------
# Restore the model and construct the encoder and decoder.
model = load_model(model_path)
encoder_inputs = model.input[0]   # input_1

encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]   # input_2
decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

######################################################################
# Compile both encoder and decoder model on NNVM
# ----------------------------------------------
# Creates NNVM graph definition from keras model file.
from tvm.contrib import graph_runtime
target = 'llvm'
ctx = tvm.cpu(0)

# Parse Encoder model
sym, params = nnvm.frontend.from_keras(encoder_model)
inp_enc_shape = (1, max_encoder_seq_length, num_encoder_tokens)
shape_dict = {'input_1': inp_enc_shape}

# Build Encoder model
with nnvm.compiler.build_config(opt_level=2):
    enc_graph, enc_lib, enc_params = nnvm.compiler.build(sym, target, shape_dict, params=params)
print("Encoder build ok.")

# Create graph runtime for encoder model
tvm_enc = graph_runtime.create(enc_graph, enc_lib, ctx)
tvm_enc.set_input(**enc_params)

# Parse Decoder model
inp_dec_shape = (1, 1, num_decoder_tokens)
shape_dict = {'input_2': inp_dec_shape,
              'input_3': (1, latent_dim),
              'input_4': (1, latent_dim)}

# Build Decoder model
sym, params = nnvm.frontend.from_keras(decoder_model)
with nnvm.compiler.build_config(opt_level=2):
    dec_graph, dec_lib, dec_params = nnvm.compiler.build(sym, target, shape_dict, params=params)
print("Decoder build ok.")

# Create graph runtime for decoder model
tvm_dec = graph_runtime.create(dec_graph, dec_lib, ctx)
tvm_dec.set_input(**dec_params)

# Decodes an input sequence.
def decode_sequence(input_seq):
    # Set the input for encoder model.
    tvm_enc.set_input('input_1', input_seq)

    # Run encoder model
    tvm_enc.run()

    # Get states from encoder network
    h = tvm_enc.get_output(0).asnumpy()
    c = tvm_enc.get_output(1).asnumpy()

    # Populate the first character of target sequence with the start character.
    sampled_token_index = target_token_index['\t']

    # Sampling loop for a batch of sequences
    decoded_sentence = ''
    while True:
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens), dtype='float32')
        # Update the target sequence (of length 1).
        target_seq[0, 0, sampled_token_index] = 1.

        # Set the input and states for decoder model.
        tvm_dec.set_input('input_2', target_seq)
        tvm_dec.set_input('input_3', h)
        tvm_dec.set_input('input_4', c)
        # Run decoder model
        tvm_dec.run()

        output_tokens = tvm_dec.get_output(0).asnumpy()
        h = tvm_dec.get_output(1).asnumpy()
        c = tvm_dec.get_output(2).asnumpy()

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]

        # Exit condition: either hit max length or find stop character.
        if sampled_char == '\n':
            break

        # Update the sentence
        decoded_sentence += sampled_char
        if len(decoded_sentence) > max_decoder_seq_length:
            break
    return decoded_sentence

def generate_input_seq(input_text):
    input_seq = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    for t, char in enumerate(input_text):
        input_seq[0, t, input_token_index[char]] = 1.
    return input_seq

######################################################################
# Run the model
# -------------
# Randonly take some text from test samples and translate
for seq_index in range(100):
    # Take one sentence randomly and try to decode.
    index = random.randint(1, test_samples)
    input_text, _ = lines[index].split('\t')
    input_seq = generate_input_seq(input_text)
    decoded_sentence = decode_sequence(input_seq)
    print((seq_index + 1), ": ", input_text,  "==>", decoded_sentence)
