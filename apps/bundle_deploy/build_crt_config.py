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

"""Creates a config header file for MISRA-C runtime."""

import argparse
import os
import logging
import json

def build_crt_config(fname, graph):
    with open(fname, 'w') as fp:
        graph = json.loads(graph)
        node_max_inputs = max([len(node.get('inputs')) for node in graph['nodes']])
        max_nodes = len(graph['nodes'])
        max_input_nodes = len(graph['nodes']) # TODO: get input nodes only
        max_node_row_ptr = len(graph['node_row_ptr'])
        max_outputs = len(graph['heads'])
        fp.write("/* This is an auto-generated header file. Please do NOT modify\n")
        fp.write(" * the content of this file unless you're aware what you're doing.\n")
        fp.write(" */\n")
        fp.write("\n")
        fp.write("/*! Maximum inputs in a GraphRuntimeNode */\n")
        fp.write("#define GRAPH_RUNTIME_NODE_MAX_INPUTS {}\n".format(node_max_inputs))
        fp.write("/*! Maximum supported nodes in a GraphRuntime */\n")
        fp.write("#define GRAPH_RUNTIME_MAX_NODES {}\n".format(max_nodes))
        fp.write("/*! Maximum input nodes in a GraphRuntime */\n")
        fp.write("#define GRAPH_RUNTIME_MAX_INPUT_NODES {}\n".format(max_input_nodes))
        fp.write("/*! Maximum nodes in a GraphRuntime for quick entry indexing */\n")
        fp.write("#define GRAPH_RUNTIME_MAX_NODE_ROW_PTR {}\n".format(max_node_row_ptr))
        fp.write("/*! Maximum output entries in a GraphRuntime */\n")
        fp.write("#define GRAPH_RUNTIME_MAX_OUTPUTS {}\n".format(max_outputs))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='')
    parser.add_argument('-o', '--output', default='')
    opts = parser.parse_args()

    graph = []
    with open(opts.input, 'rt') as fp:
        graph = fp.read()
    build_crt_config(opts.output, graph)
