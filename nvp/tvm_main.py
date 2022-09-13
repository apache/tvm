from tvm_lower import *
from tvm_parse import *

def main():
    mod = gen_module()
    mod = get_lowered_tir(mod)
    keys = [key for key in mod.functions.keys()]
    print(mod.functions[keys[1]])
    # assert 0
    visit_stmts(mod.functions[keys[1]])

if __name__ == '__main__':
    main()
