#Author: zzk
"""Collect quantized weight or activation data"""
import tvm.ir
import tvm
import os
from tvm.runtime import Object
import numpy as np
import tqdm

from .quantize import QAnnotateKind, current_qconfig
from .. import op as _op
from .. import expr as _expr
from .. import analysis as _analysis
from .. import build_module as _build_module
from ...contrib import graph_executor
from .. import transform as _transform
from . import _quantize
from tvm import relay

def print_model(mod):
    quantize_op = _op.get("relay.op.annotation.simulated_quantize")
    annotate_cast_op = _op.get("annotation.cast_hint")
    stop_fusion_op = _op.get("annotation.stop_fusion")
    split_op = _op.get("split")
    concatenate_op = _op.get("concatenate")
    add_op = _op.get("add")
    multiply_op = _op.get("multiply")

    layer_counter = 0
    layer_dict = {}
    cfg = current_qconfig()
    input_expr = None
    
    root_dir_name = "./output_model/"
    if not os.path.exists(root_dir_name):
        os.mkdir(root_dir_name)
    model_file_name = root_dir_name + cfg.get_network_name() + ".prototxt"
    prototxt_file = open(model_file_name, 'w')
    prototxt_file.write('input: "data"\n')

    def visit_func(expr):
        nonlocal layer_counter
        nonlocal input_expr
        if isinstance(expr, _expr.Call):
            if(expr.op is not quantize_op and expr.op is not annotate_cast_op and expr.op is not stop_fusion_op):  
                if(expr.op == split_op):
                    prototxt_file.write('layer {\n')
                    prototxt_file.write('  name: "{}"\n'.format(expr.op.name  + "_" + str(layer_counter)))
                    prototxt_file.write('  type: "{}"\n'.format(expr.op.name))
                    if(isinstance(expr.args[0] , _expr.TupleGetItem)):
                        prototxt_file.write('  bottom: "{}"\n'.format(layer_dict[expr.args[0].tuple_value] + "." + str(expr.args[0].index)))
                    elif layer_counter == 0 or expr.args[0] == input_expr:    
                        prototxt_file.write('  bottom: "{}"\n'.format("data"))
                        input_expr = expr.args[0]
                    else:
                        prototxt_file.write('  bottom: "{}"\n'.format(layer_dict[expr.args[0]]))
                    split_len = expr.attrs.indices_or_sections
                    split_len_value = 0
                    if isinstance(split_len, int):
                        split_len_value = split_len
                    else:
                        split_len_value = len(split_len) + 1
                    for i in range(split_len_value):
                        prototxt_file.write('  top: "{}"\n'.format(expr.op.name + "_" + str(layer_counter) + "." + str(i)))
                    layer_dict[expr] = expr.op.name + "_" + str(layer_counter)
                    # prototxt_file.write("  hdf5_output_param {\n")
                    # if expr.attrs is not None:
                    #     for key in expr.attrs.keys():
                    #         if len(str(expr.attrs[key]).strip())>0:
                    #             prototxt_file.write("    {}: {}\n".format(key, str(expr.attrs[key])))
                    # prototxt_file.write("  }\n")
                    prototxt_file.write('}\n')
                    layer_counter = layer_counter + 1
                elif (expr.op == concatenate_op):
                    prototxt_file.write('layer {\n')
                    prototxt_file.write('  name: "{}"\n'.format(expr.op.name + "_" + str(layer_counter)))
                    prototxt_file.write('  type: "{}"\n'.format(expr.op.name))
                    concat_len = len(expr.args[0])
                    for i in range(concat_len):
                        expr_tmp = expr.args[0][i]
                        if(isinstance(expr_tmp , _expr.TupleGetItem)):
                            prototxt_file.write('  bottom: "{}"\n'.format(layer_dict[expr_tmp.tuple_value] + "." + str(expr_tmp.index)))
                        elif layer_counter == 0 or expr_tmp == input_expr:    
                            prototxt_file.write('  bottom: "{}"\n'.format("data"))
                            input_expr = expr.args[0]
                        else:
                            prototxt_file.write('  bottom: "{}"\n'.format(layer_dict[expr_tmp]))
                    prototxt_file.write('  top: "{}"\n'.format(expr.op.name + "_" + str(layer_counter)))
                    layer_dict[expr] = expr.op.name + "_" + str(layer_counter)
                    # prototxt_file.write("  hdf5_output_param {\n")
                    # if expr.attrs is not None:
                    #     for key in expr.attrs.keys():
                    #         if len(str(expr.attrs[key]).strip())>0:
                    #             prototxt_file.write("    {}: {}\n".format(key, str(expr.attrs[key])))
                    # prototxt_file.write("  }\n")
                    prototxt_file.write('}\n')
                    layer_counter = layer_counter + 1
                elif (expr.op == add_op or expr.op == multiply_op):
                    prototxt_file.write('layer {\n')
                    prototxt_file.write('  name: "{}"\n'.format(expr.op.name + "_" + str(layer_counter)))
                    prototxt_file.write('  type: "{}"\n'.format(expr.op.name))

                    if(isinstance(expr.args[0] , _expr.TupleGetItem)):
                        prototxt_file.write('  bottom: "{}"\n'.format(layer_dict[expr.args[0].tuple_value] + "." + str(expr.args[0].index)))
                    elif layer_counter == 0 or expr.args[0] == input_expr and not _analysis.check_constant(expr.args[0]):
                        prototxt_file.write('  bottom: "{}"\n'.format("data"))
                        input_expr = expr.args[0]
                    elif not _analysis.check_constant(expr.args[0]):
                        prototxt_file.write('  bottom: "{}"\n'.format(layer_dict[expr.args[0]]))
                    else:
                        pass

                    if(isinstance(expr.args[1] , _expr.TupleGetItem)):
                        prototxt_file.write('  bottom: "{}"\n'.format(layer_dict[expr.args[1].tuple_value] + "." + str(expr.args[1].index)))
                    elif layer_counter == 0 or expr.args[1] == input_expr and not _analysis.check_constant(expr.args[1]):
                        prototxt_file.write('  bottom: "{}"\n'.format("data"))
                        input_expr = expr.args[1]
                    elif not _analysis.check_constant(expr.args[1]):
                        prototxt_file.write('  bottom: "{}"\n'.format(layer_dict[expr.args[1]]))
                    else:
                        pass                    

                    prototxt_file.write('  top: "{}"\n'.format(expr.op.name + "_" + str(layer_counter)))
                    layer_dict[expr] = expr.op.name + "_" + str(layer_counter)
                    # prototxt_file.write("  hdf5_output_param {\n")
                    # if expr.attrs is not None:
                    #     for key in expr.attrs.keys():
                    #         if len(str(expr.attrs[key]).strip())>0:
                    #             prototxt_file.write("    {}: {}\n".format(key, str(expr.attrs[key])))
                    # prototxt_file.write("  }\n")
                    prototxt_file.write('}\n')
                    layer_counter = layer_counter + 1
                else:
                    prototxt_file.write('layer {\n')
                    prototxt_file.write('  name: "{}"\n'.format(expr.op.name + "_" + str(layer_counter)))
                    prototxt_file.write('  type: "{}"\n'.format(expr.op.name))
                    if(isinstance(expr.args[0] , _expr.TupleGetItem)):
                        prototxt_file.write('  bottom: "{}"\n'.format(layer_dict[expr.args[0].tuple_value] + "." + str(expr.args[0].index)))
                    elif layer_counter == 0 or expr.args[0] == input_expr:
                        prototxt_file.write('  bottom: "{}"\n'.format("data"))
                        input_expr = expr.args[0]
                    else:
                        prototxt_file.write('  bottom: "{}"\n'.format(layer_dict[expr.args[0]]))
                    prototxt_file.write('  top: "{}"\n'.format(expr.op.name + "_" + str(layer_counter)))
                    layer_dict[expr] = expr.op.name + "_" + str(layer_counter)
                    # prototxt_file.write("  hdf5_output_param {\n")
                    # if expr.attrs is not None:
                    #     for key in expr.attrs.keys():
                    #         if len(str(expr.attrs[key]).strip())>0:
                    #             prototxt_file.write("    {}: {}\n".format(key, str(expr.attrs[key])))
                    # prototxt_file.write("  }\n")
                    prototxt_file.write('}\n')
                    layer_counter = layer_counter + 1
            

    main_func = mod["main"]
    _analysis.post_order_visit(main_func, visit_func)
    prototxt_file.close()