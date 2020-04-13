# cost function for intelfocl 32*32 gemm version
def cal_cost(insn):
    """
    Cal the runtime cost statically

    Parameters
    ------------
    insn: the insn (json)

    Returns
    ------------
    the cost in s
    """
    factor = 1000000.0
    def alu_imm_cost(iter_out, iter_in, uop_bgn, uop_end):
        x = (uop_end - uop_bgn) * iter_out * iter_in
        cycles = x + 46
        return cycles / factor

    def alu_cost(iter_out, iter_in, uop_bgn, uop_end):
        x = (uop_end - uop_bgn) * iter_out * iter_in
        cycles = 2 * x + 46
        return cycles / factor

    def gemm_cost(iter_out, iter_in, uop_bgn, uop_end):
        x = (uop_end - uop_bgn) * iter_out * iter_in
        cycles = x + 80
        return cycles / factor

    def load_acc_cost(y_size, y_pad_0, y_pad_1, x_size, x_pad_0, x_pad_1):
        x = x_size * y_size
        cycles = x + 150
        return cycles / factor

    def load_acc8_cost(y_size, y_pad_0, y_pad_1, x_size, x_pad_0, x_pad_1):
        x = x_size * y_size
        cycles = 1.2 * x + 150
        return cycles / factor

    def load_inp_cost(y_size, y_pad_0, y_pad_1, x_size, x_pad_0, x_pad_1):
        x = (x_size + x_pad_0 + x_pad_1) * (y_size + y_pad_0 + y_pad_1)
        cycles = 1.1 * x + 150
        return cycles / factor

    def load_uop_cost(y_size, y_pad_0, y_pad_1, x_size, x_pad_0, x_pad_1):
        x = x_size * y_size
        cycles = 1.1 * x + 150
        return cycles / factor

    def load_wgt_cost(y_size, y_pad_0, y_pad_1, x_size, x_pad_0, x_pad_1):
        x = x_size * y_size
        cycles = 17 * x + 150
        return cycles / factor

    def store_cost(y_size, y_pad_0, y_pad_1, x_size, x_pad_0, x_pad_1):
        x = x_size * y_size
        cycles = x + 150
        return cycles / factor

    def nop_cost(name):
        if name == "NOP-COMPUTE-STAGE":
            return 38 / factor
        elif name == "NOP-MEMORY-STAGE":
            return 50 / factor
        elif name == "NOP-STORE-STAGE":
            return 39 / factor
        else:
            print("Unknown nop op {}".format(name))
            return 0

    if insn['type'] == "ALU":
        return alu_cost(insn['outer_loop'][0], insn['inner_loop'][0],
                        insn['range'][0], insn['range'][1])
    elif insn['type'] == "ALU IMM":
        return alu_imm_cost(insn['outer_loop'][0], insn['inner_loop'][0],
                        insn['range'][0], insn['range'][1])
    elif insn['type'] == "GEMM":
        return gemm_cost(insn['outer_loop'][0], insn['inner_loop'][0],
                        insn['range'][0], insn['range'][1])
    elif insn['name'] == "LOAD INP":
        return load_inp_cost(insn['y'][0], insn['y'][1], insn['y'][2],
                             insn['x'][0], insn['x'][1], insn['x'][2])
    elif insn['name'] == "LOAD WGT":
        return load_wgt_cost(insn['y'][0], insn['y'][1], insn['y'][2],
                             insn['x'][0], insn['x'][1], insn['x'][2])
    elif insn['name'] == "LOAD UOP":
        return load_uop_cost(insn['y'][0], insn['y'][1], insn['y'][2],
                             insn['x'][0], insn['x'][1], insn['x'][2])
    elif insn['name'] == "LOAD ACC":
        return load_acc_cost(insn['y'][0], insn['y'][1], insn['y'][2],
                             insn['x'][0], insn['x'][1], insn['x'][2])
    elif insn['name'] == "LOAD ACC 8":
        return load_acc8_cost(insn['y'][0], insn['y'][1], insn['y'][2],
                             insn['x'][0], insn['x'][1], insn['x'][2])
    elif insn['name'] == "STORE":
        return store_cost(insn['y'][0], insn['y'][1], insn['y'][2],
                          insn['x'][0], insn['x'][1], insn['x'][2])
    elif insn['type'] == "NOP":
        return nop_cost(insn['name'])
    else:
        print("Unknown op type: {}".format(insn['type']))
        return 0
