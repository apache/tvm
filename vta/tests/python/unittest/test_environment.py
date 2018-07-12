import vta


def test_env():
    env = vta.get_env()
    mock = env.mock
    assert mock.alu == "skip_alu"

def test_env_scope():
    env = vta.get_env()
    cfg = env.pkg_config().cfg_dict
    cfg["TARGET"] = "xyz"
    with vta.Environment(cfg):
        assert vta.get_env().TARGET == "xyz"
    assert vta.get_env().TARGET == env.TARGET


if __name__ == "__main__":
    test_env()
    test_env_scope()
