def pytest_addoption(parser):
    parser.addoption(
        "--microtvm-platforms",
        default="host",
        help=(
            "Specify a comma-separated list of test models (i.e. as passed to tvm.target.micro()) "
            "for microTVM tests."
        ),
    )


def pytest_generate_tests(metafunc):
    if "platform" in metafunc.fixturenames:
        metafunc.parametrize("platform", metafunc.config.getoption("microtvm_platforms").split(","))
