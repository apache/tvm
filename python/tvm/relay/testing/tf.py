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
# pylint: disable=invalid-name, unused-variable, unused-argument, no-init
"""
Tensorflow Model Helpers
========================
Some helper definitions for tensorflow models.
"""
import re
import os.path
import collections
import numpy as np

# Tensorflow imports
import tensorflow as tf
from tensorflow.core.framework import graph_pb2

from tvm.contrib.download import download_testdata

######################################################################
# Some helper functions
# ---------------------

def ProcessGraphDefParam(graph_def):
    """Type-checks and possibly canonicalizes `graph_def`.

    Parameters
    ----------
    graph_def : Obj
        tensorflow graph definition.

    Returns
    -------
    graph_def : Obj
        tensorflow graph devinition

    """

    if not isinstance(graph_def, graph_pb2.GraphDef):
        # `graph_def` could be a dynamically-created message, so try a duck-typed
        # approach
        try:
            old_graph_def = graph_def
            graph_def = graph_pb2.GraphDef()
            graph_def.MergeFrom(old_graph_def)
        except TypeError:
            raise TypeError('graph_def must be a GraphDef proto.')
    return graph_def


def AddShapesToGraphDef(session, out_node):
    """ Add shapes attribute to nodes of the graph.
        Input graph here is the default graph in context.

    Parameters
    ----------
    session : tf.Session
        Tensorflow session
    out_node : String
        Final output node of the graph.

    Returns
    -------
    graph_def : Obj
        tensorflow graph definition with shapes attribute added to nodes.

    """

    graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        session,
        session.graph.as_graph_def(add_shapes=True),
        [out_node],
        )
    return graph_def

class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
                 label_lookup_path=None,
                 uid_lookup_path=None):
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.

        Parameters
        ----------
        label_lookup_path: String
            File containing String UID to integer node ID mapping .

        uid_lookup_path: String
            File containing String UID to human-readable string mapping.

        Returns
        -------
        node_id_to_name: dict
            dict from integer node ID to human-readable string.

        """
        if not tf.compat.v1.io.gfile.exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.compat.v1.io.gfile.exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.compat.v1.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.compat.v1.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]

def get_workload_official(model_url, model_sub_path):
    """ Import workload from tensorflow official

    Parameters
    ----------
    model_url: str
        URL from where it will be downloaded.

    model_sub_path:
        Sub path in extracted tar for the ftozen protobuf file.

    Returns
    -------
    model_path: str
        Full path to saved model file

    """

    model_tar_name = os.path.basename(model_url)
    model_path = download_testdata(model_url, model_tar_name, module=['tf', 'official'])
    dir_path = os.path.dirname(model_path)

    import tarfile
    if model_path.endswith("tgz") or model_path.endswith("gz"):
        tar = tarfile.open(model_path)
        tar.extractall(path=dir_path)
        tar.close()
    else:
        raise RuntimeError('Could not decompress the file: ' + model_path)
    return os.path.join(dir_path, model_sub_path)

def get_workload(model_path, model_sub_path=None):
    """ Import workload from frozen protobuf

    Parameters
    ----------
    model_path: str
        model_path on remote repository to download from.

    model_sub_path: str
        Model path in the compressed archive.

    Returns
    -------
    graph_def: graphdef
        graph_def is the tensorflow workload.

    """

    if model_sub_path:
        path_model = get_workload_official(model_path, model_sub_path)
    else:
        repo_base = 'https://github.com/dmlc/web-data/raw/master/tensorflow/models/'
        model_url = os.path.join(repo_base, model_path)
        path_model = download_testdata(model_url, model_path, module='tf')

    # Creates graph from saved graph_def.pb.
    with tf.compat.v1.gfile.FastGFile(path_model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name='')
        return graph_def

#######################################################################
# PTB LSTMBlockCell Model
# -----------------------

class PTBSmallConfig(object):
    """Small config.
    This configurations are used when training the model
    """
    num_layers = 2
    num_steps = 1
    hidden_size = 200
    batch_size = 1
    vocab_size = 10000
    init_scale = 0.1

def get_config():
    """Configuration used for training the model"""
    return PTBSmallConfig()

def pick_from_weight(weight, pows=1.0):
    """Identify token from Softmax output.
    This token will be mapped to word in the vocabulary.
    """
    weight = weight**pows
    t = np.cumsum(weight)
    s = np.sum(weight)
    return int(np.searchsorted(t, 0.5 * s))

def do_tf_sample(session, data, in_states, num_samples):
    """Sampled from the model"""
    samples = []
    sample = None
    #Cell inputs c and h should be passed for each layer explicitly.
    state_input_name = ['Model/MultiRNNCellZeroState/LSTMBlockCellZeroState/zeros:0',
                        'Model/MultiRNNCellZeroState/LSTMBlockCellZeroState/zeros_1:0',
                        'Model/MultiRNNCellZeroState/LSTMBlockCellZeroState_1/zeros:0',
                        'Model/MultiRNNCellZeroState/LSTMBlockCellZeroState_1/zeros_1:0']
    state = session.run(state_input_name)

    #Graph nodes to be fetched as run output. Tensorflow LSTMBlockCell create internal
    #nodes for intermediate operations (gates) in the cell during run.
    #Cell state (c) is ':1'and cell output (h) is ':6' for each layer.
    fetches = [['Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell:1',
                'Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell:6',
                'Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_1:1',
                'Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_1:6'],
               'Model/Softmax:0']

    def _get_feed_dict(input_name, input_data):
        """Create feed dict"""
        feed_dict = {}
        if isinstance(input_data, list):
            for i, e in enumerate(input_name):
                feed_dict[e] = input_data[i]
        else:
            feed_dict[input_name] = input_data
        return feed_dict

    for x in data:
        feed_dict = _get_feed_dict(state_input_name, state)
        feed_dict['Model/Placeholder:0'] = [[x]]
        state, probs = session.run(fetches, feed_dict)
        sample = pick_from_weight(probs[0])
    if sample is not None:
        samples.append(sample)
    else:
        samples.append(0)

    k = 1
    while k < num_samples:
        feed_dict = _get_feed_dict(state_input_name, state)
        feed_dict['Model/Placeholder:0'] = [[samples[-1]]]
        state, probs = session.run(fetches, feed_dict)
        sample = pick_from_weight(probs[0])
        samples.append(sample)
        k += 1
    return samples, state

def _create_ptb_vocabulary(data_dir):
    """Read the PTB sample data input to create vocabulary"""
    data_path = os.path.join(data_dir, 'simple-examples/data/')
    file_name = 'ptb.train.txt'
    def _read_words(filename):
        """Read the data for creating vocabulary"""
        with tf.compat.v1.gfile.GFile(filename, "r") as f:
            return f.read().encode("utf-8").decode("utf-8").replace("\n", "<eos>").split()

    def _build_vocab(filename):
        """Create vocabulary"""
        data = _read_words(filename)
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(len(words))))
        #for python 3.x
        id_to_word = dict((v, k) for k, v in word_to_id.items())
        return word_to_id, id_to_word

    def ptb_raw_data(data_path, file_name):
        """Read the sample data and create vocabulary"""
        train_path = os.path.join(data_path, file_name)
        word_to_id, id_2_word = _build_vocab(train_path)
        return word_to_id, id_2_word
    return ptb_raw_data(data_path, file_name)

def get_workload_ptb():
    """ Import ptb workload from frozen protobuf

    Parameters
    ----------
        Nothing.

    Returns
    -------
    graph_def: graphdef
        graph_def is the tensorflow workload for ptb.

    word_to_id : dict
        English word to integer id mapping

    id_to_word : dict
        Integer id to English word mapping
    """
    sample_repo = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/'
    sample_data_file = 'simple-examples.tgz'
    sample_url = sample_repo+sample_data_file
    ptb_model_file = 'RNN/ptb/ptb_model_with_lstmblockcell.pb'

    import tarfile
    file_path = download_testdata(sample_url, sample_data_file, module=['data', 'ptb_data'])
    dir_path = os.path.dirname(file_path)
    t = tarfile.open(file_path, 'r')
    t.extractall(dir_path)

    word_to_id, id_to_word = _create_ptb_vocabulary(dir_path)
    return word_to_id, id_to_word, get_workload(ptb_model_file)
