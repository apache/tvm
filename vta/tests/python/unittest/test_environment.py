import vta


def test_env():
    env = vta.get_env()
    mock = env.mock
    assert mock.alu == "skip_alu"


if __name__ == "__main__":
    test_env()
