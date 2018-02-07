import tvm

def test_config_string():
    a = tvm.build_module.BuildConfig()
    a.detect_global_barrier = True
    a.offset_factor = 20
    a.auto_unroll_max_depth = 2
    s = str(a)
    b = tvm.build_module.build_config_parse(s)
    for k in tvm.build_module.BuildConfig.defaults:
        assert getattr(a, k) == getattr(b, k)

if __name__ == "__main__":
    test_config_string()
