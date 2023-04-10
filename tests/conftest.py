# content of conftest.py


def pytest_addoption(parser):
    parser.addoption("--global", action="store_true", help="run all combinations")


def pytest_generate_tests(metafunc):
    if "for_test" in metafunc.fixturenames:
        if metafunc.config.getoption("global"):
            path = './'
        else:
            path = '../'
        metafunc.parametrize("for_test", [path])
