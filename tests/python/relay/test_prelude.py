import tvm
from tvm import relay

def load_prelude():
    mod = relay.Module()
    return relay.prelude.Prelude(mod)

def test_list():
    prelude = load_prelude()

if __name__ == "__main__":
    test_list()
