def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: long-running solver benchmark (deselect with -m 'not slow')"
    )
