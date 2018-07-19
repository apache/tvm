# pylint: disable=invalid-name, unused-variable, unused-argument, no-init
"""
Tensorflow Model Helpers
========================
Some helper definitions for tensorflow models.
"""
import re
import os.path

# Tensorflow imports
import tensorflow as tf
from tensorflow.core.framework import graph_pb2

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
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
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

def get_workload(model_path):
    """ Import workload from frozen protobuf

    Parameters
    ----------
    model_path: str
        model_path on remote repository to download from.

    Returns
    -------
    graph_def: graphdef
        graph_def is the tensorflow workload for mobilenet.

    """

    repo_base = 'https://github.com/dmlc/web-data/raw/master/tensorflow/models/'
    model_name = os.path.basename(model_path)
    model_url = os.path.join(repo_base, model_path)

    from mxnet.gluon.utils import download
    download(model_url, model_name)

    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join("./", model_name), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name='')
        return graph_def
